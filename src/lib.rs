#![allow(clippy::doc_markdown)]
//! DPDFNet LADSPA plugin (mono, single-model).
//!
//! One shared object per upstream model — the active model is selected
//! at build time by the `DPDFNET_MODEL` env var (see `build.rs`). All
//! per-model facts (sample rate, STFT geometry, recurrent state size,
//! LADSPA label / unique_id, embedded IR + initial state) materialise
//! as constants in the build-script-generated `model_const.rs`.
//!
//! Streaming inference of `ceva-ip/DPDFNet` via OpenVINO Runtime (Intel
//! oneDNN CPU plugin). The .so embeds the model's OpenVINO IR and a
//! precomputed initial recurrent state, so the only runtime dependency
//! is the system `openvino` package — no `libopenvino_onnx_frontend.so`
//! load, no on-disk model lookup, no network fetch on first frame.
//!
//! Real-time discipline: every per-frame buffer (FFT scratch, OpenVINO
//! tensor backing, output queue) lives on the struct so `run()` does
//! zero heap allocation in steady state. Output samples produced by
//! one call but exceeding the host buffer length are queued for the
//! next call instead of being dropped — dropping caused micro-stutters
//! and timing drift on small PW quantums.
//!
//! OpenVINO is configured for LATENCY mode with 1 inference thread /
//! 1 stream, matching the LADSPA host RT thread (FIFO 83). The CPU
//! plugin's vectorized GEMM kernels (AVX2/AVX-512 oneDNN) cut per-frame
//! inference cost vs. ONNX Runtime's default MlasGemm path on the
//! Intel hybrid silicon we ship to.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};

use ladspa::{
    DefaultValue, Plugin, PluginDescriptor, Port, PortConnection, PortDescriptor, Properties,
};
use openvino::{CompiledModel, Core, ElementType, InferRequest, RwPropertyKey, Shape, Tensor};
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

mod model_const {
    // `MODEL_NAME` ships for diagnostics / log lines; the LADSPA hot
    // path doesn't read it. The unique_id is generated as a single
    // hash literal — splitting it with underscores would make the
    // generated file harder to read, so the readability lint is
    // also suppressed here.
    #![allow(dead_code, clippy::unreadable_literal)]
    include!(concat!(env!("OUT_DIR"), "/model_const.rs"));
}

const PORT_INPUT: usize = 0;
const PORT_OUTPUT: usize = 1;
const PORT_ATTEN_LIMIT_DB: usize = 2;

fn build_init_state() -> Vec<f32> {
    assert_eq!(model_const::INIT_STATE.len(), model_const::STATE_SIZE * 4);
    let mut state = vec![0.0_f32; model_const::STATE_SIZE];
    for (i, chunk) in model_const::INIT_STATE.chunks_exact(4).enumerate() {
        state[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    state
}

/// Shared OpenVINO inference engine: one `InferRequest` plus the
/// `spec`/`state_in` input tensors and the `CompiledModel` that owns
/// them. Used by every LADSPA instance of this plugin in the process
/// (mic and output filter chains both run inside the same `pwloader`
/// process, and the data-loop calls plugins serially, so a single
/// `InferRequest` is correct and saves the per-request oneDNN /
/// TBB scratchpad — typically 10-15 MB).
///
/// Per-stream GRU state lives in `DpdfnetPlugin::state` (a plain
/// `Vec<f32>`). Each `process_frame()` copies it into the shared
/// `state_tensor` before `infer()` and copies `state_out` back into
/// the plugin's vec after — that's how distinct mic and output
/// streams keep independent recurrent context while sharing the
/// inference engine.
struct SharedEngine {
    infer: InferRequest,
    spec_tensor: Tensor,
    state_tensor: Tensor,
    /// Held to keep the model + weights alive for the program's
    /// lifetime; the bound tensors above reference its memory.
    _compiled: CompiledModel,
}

// SAFETY: the C API for `ov_infer_request_*` is callable from any
// thread as long as concurrent calls are externally synchronized
// (oneDNN/TBB primitives backing it are themselves thread-safe).
// We synchronize via the surrounding `Mutex<SharedEngine>`. The
// openvino crate marks `InferRequest` as `Send` already; this impl
// extends that to `Sync` for storage in `OnceLock<Mutex<…>>`.
unsafe impl Sync for SharedEngine {}

fn build_engine() -> Mutex<SharedEngine> {
    let mut core = Core::new().expect("OpenVINO Core::new failed");

    // Configure the CPU plugin BEFORE compile so the optimization
    // pipeline picks up our hints:
    //
    // * `SNIPPETS_MODE=DISABLE` — DPDFNet's GRU subgraph trips a bug in
    //   the Snippets tokenizer ("Topological order is required, but not
    //   set." raised from `pass/tokenization.cpp`) on OpenVINO 2026.0.0.
    //   Disabling the Snippets pass routes those ops through the regular
    //   oneDNN kernels — slightly less fused but functional.
    // * `PERFORMANCE_HINT=LATENCY` + `NUM_STREAMS=1` +
    //   `INFERENCE_NUM_THREADS=1` keep every op on the calling thread.
    //   The LADSPA host already calls us on a FIFO 83 RT thread;
    //   spawning OpenVINO worker threads at TBB defaults would cause
    //   priority inversion under load.
    // * `ENABLE_CPU_PINNING=NO` — `pwloader` already pins to P-cores via
    //   `sched_setaffinity`. Letting the CPU plugin add its own pinning
    //   on top can race with our mask.
    let cpu = openvino::DeviceType::CPU;
    for (key, val) in [
        (RwPropertyKey::Other("SNIPPETS_MODE".into()), "DISABLE"),
        (RwPropertyKey::HintPerformanceMode, "LATENCY"),
        (RwPropertyKey::NumStreams, "1"),
        (RwPropertyKey::InferenceNumThreads, "1"),
        (RwPropertyKey::HintEnableCpuPinning, "NO"),
    ] {
        let _ = core.set_property(&cpu, &key, val);
    }

    // IR weights must be passed as a U8 Tensor wrapping the .bin bytes.
    // Once compile_model finishes, the resulting CompiledModel holds
    // its own copy of the weights so this scratch tensor can drop.
    let weights_shape = Shape::new(&[model_const::IR_BIN.len() as i64])
        .expect("OpenVINO weights Shape::new failed");
    let mut weights_tensor = Tensor::new(openvino::ElementType::U8, &weights_shape)
        .expect("OpenVINO weights Tensor::new failed");
    weights_tensor
        .get_data_mut::<u8>()
        .expect("weights tensor get_data_mut failed")
        .copy_from_slice(model_const::IR_BIN);

    let model = core
        .read_model_from_buffer(model_const::IR_XML, Some(&weights_tensor))
        .expect("OpenVINO failed to read embedded DPDFNet IR");
    let mut compiled = core
        .compile_model(&model, cpu)
        .expect("OpenVINO compile_model(CPU) failed");
    let mut infer = compiled
        .create_infer_request()
        .expect("OpenVINO create_infer_request failed");

    let spec_shape = Shape::new(&[1, 1, model_const::FREQ_BINS as i64, 2])
        .expect("OpenVINO spec shape construction failed");
    let state_shape = Shape::new(&[model_const::STATE_SIZE as i64])
        .expect("OpenVINO state shape construction failed");
    let spec_tensor =
        Tensor::new(ElementType::F32, &spec_shape).expect("OpenVINO spec tensor allocation failed");
    let state_tensor = Tensor::new(ElementType::F32, &state_shape)
        .expect("OpenVINO state tensor allocation failed");
    infer
        .set_tensor("spec", &spec_tensor)
        .expect("OpenVINO set_tensor(\"spec\") failed");
    infer
        .set_tensor("state_in", &state_tensor)
        .expect("OpenVINO set_tensor(\"state_in\") failed");

    Mutex::new(SharedEngine {
        infer,
        spec_tensor,
        state_tensor,
        _compiled: compiled,
    })
}

fn shared_engine() -> &'static Mutex<SharedEngine> {
    static SHARED: OnceLock<Mutex<SharedEngine>> = OnceLock::new();
    SHARED.get_or_init(build_engine)
}

fn vorbis_window(win_len: usize) -> Vec<f32> {
    let half = win_len as f32 / 2.0;
    (0..win_len)
        .map(|n| {
            let s = (0.5 * std::f32::consts::PI * (n as f32 + 0.5) / half).sin();
            (0.5 * std::f32::consts::PI * s * s).sin()
        })
        .collect()
}

struct DpdfnetPlugin {
    /// Per-stream GRU state. Copied into the shared `state_tensor`
    /// before every `infer()` and refreshed from `state_out` after —
    /// that's how mic and output streams keep independent recurrent
    /// context while sharing the per-model `InferRequest`.
    state: Vec<f32>,
    /// Sliding analysis window. We keep the most recent WIN_LEN samples
    /// and advance by HOP_SIZE per frame, so iteration n reads
    /// `[n*HOP .. n*HOP + WIN]` from the original input. A `VecDeque`
    /// would force a discontiguous read; a plain `Vec` with `copy_within`
    /// keeps the slice contiguous for FFT scratch fill.
    in_buf: Vec<f32>,
    /// Carry-over OLA accumulator (samples 0..WIN_LEN are accumulated;
    /// 0..HOP_SIZE are flushed to `out_queue` per frame).
    ola_buf: Vec<f32>,
    /// Output produced ahead of the host's current buffer length —
    /// drained in subsequent `run()` calls. Sized once, never grows
    /// past WIN_LEN since one frame produces HOP_SIZE samples.
    out_queue: VecDeque<f32>,
    window: Vec<f32>,
    fft_fwd: Arc<dyn RealToComplex<f32>>,
    fft_inv: Arc<dyn ComplexToReal<f32>>,
    fft_real: Vec<f32>,
    fft_complex: Vec<Complex<f32>>,
    spec_in: Vec<f32>,
    spec_out: Vec<f32>,
}

impl DpdfnetPlugin {
    fn new(sample_rate: u64) -> Self {
        let sr = sample_rate as usize;
        assert!(
            sr == model_const::SAMPLE_RATE,
            "DPDFNet plugin '{}' requires {} Hz host sample rate (got {sr}); \
             set `audio.rate = {}` on the filter-chain node",
            model_const::LADSPA_LABEL,
            model_const::SAMPLE_RATE,
            model_const::SAMPLE_RATE
        );

        // Engine warm-up is deferred to the first `process_frame()` so
        // a pwloader instance whose graph has no consumer (passive
        // node, never called by the data-loop) doesn't pay the ~85 MB
        // OpenVINO Core + JIT cost up front. The first frame after
        // audio starts flowing pays a one-shot latency hit (engine
        // build + IR compile, ~150-300 ms on Intel hybrid silicon)
        // which is acceptable on stream startup.
        let mut planner = RealFftPlanner::<f32>::new();
        let fft_fwd = planner.plan_fft_forward(model_const::WIN_LEN);
        let fft_inv = planner.plan_fft_inverse(model_const::WIN_LEN);

        Self {
            state: build_init_state(),
            // Pre-fill the analysis buffer with WIN_LEN zeros so the
            // first run() call can already process ceil(n/HOP)+1 frames
            // and produce >= n samples. Without this priming the queue
            // would stay one HOP short of the host buffer on every call,
            // leaking ~64 zero samples per 1024 sample call (audible as
            // periodic clicks / robotic timbre at the host quantum
            // rate). Latency cost: WIN_LEN/sr ≈ 20 ms.
            in_buf: vec![0.0; model_const::WIN_LEN],
            ola_buf: vec![0.0; model_const::WIN_LEN],
            out_queue: VecDeque::with_capacity(model_const::WIN_LEN * 2),
            window: vorbis_window(model_const::WIN_LEN),
            fft_real: vec![0.0; model_const::WIN_LEN],
            fft_complex: vec![Complex::new(0.0, 0.0); model_const::FREQ_BINS],
            fft_fwd,
            fft_inv,
            spec_in: vec![0.0; model_const::FREQ_BINS * 2],
            spec_out: vec![0.0; model_const::FREQ_BINS * 2],
        }
    }

    /// Process exactly one analysis frame: window + FFT + ONNX +
    /// spectral blend + iFFT + windowed OLA + flush HOP samples to
    /// `out_queue`. Caller guarantees `in_buf.len() >= WIN_LEN`.
    fn process_frame(&mut self, alpha: f32) {
        for j in 0..model_const::WIN_LEN {
            self.fft_real[j] = self.in_buf[j] * self.window[j];
        }
        let _ = self
            .fft_fwd
            .process(&mut self.fft_real, &mut self.fft_complex);

        for (k, c) in self.fft_complex.iter().enumerate() {
            self.spec_in[k * 2] = c.re;
            self.spec_in[k * 2 + 1] = c.im;
        }

        // Hold the shared-engine lock for the duration of one
        // inference. The data-loop calls plugins serially so this
        // never contends; the lock just satisfies Rust's `&mut`
        // requirement on global state.
        {
            let mut engine = shared_engine()
                .lock()
                .expect("shared engine mutex poisoned");

            engine
                .spec_tensor
                .get_data_mut::<f32>()
                .expect("spec tensor get_data_mut failed")
                .copy_from_slice(&self.spec_in);
            engine
                .state_tensor
                .get_data_mut::<f32>()
                .expect("state tensor get_data_mut failed")
                .copy_from_slice(&self.state);

            engine.infer.infer().expect("OpenVINO infer() failed");

            let spec_e = engine
                .infer
                .get_tensor("spec_e")
                .expect("get_tensor(\"spec_e\") failed");
            let spec_e_data = spec_e.get_data::<f32>().expect("spec_e get_data failed");
            debug_assert_eq!(spec_e_data.len(), model_const::FREQ_BINS * 2);
            self.spec_out[..spec_e_data.len()].copy_from_slice(spec_e_data);

            let state_out = engine
                .infer
                .get_tensor("state_out")
                .expect("get_tensor(\"state_out\") failed");
            let state_out_data = state_out
                .get_data::<f32>()
                .expect("state_out get_data failed");
            debug_assert_eq!(state_out_data.len(), model_const::STATE_SIZE);
            self.state[..state_out_data.len()].copy_from_slice(state_out_data);
        }

        // alpha = 0 → fully enhanced; alpha = 1 → passthrough.
        if alpha > 0.0 {
            for k in 0..model_const::FREQ_BINS * 2 {
                self.spec_out[k] = alpha * self.spec_in[k] + (1.0 - alpha) * self.spec_out[k];
            }
        }

        for k in 0..model_const::FREQ_BINS {
            self.fft_complex[k] = Complex::new(self.spec_out[k * 2], self.spec_out[k * 2 + 1]);
        }
        let _ = self
            .fft_inv
            .process(&mut self.fft_complex, &mut self.fft_real);

        // realfft inverse leaves a 1/N scaling — fold into the synthesis
        // window so OLA gets the correct amplitude.
        let scale = 1.0 / model_const::WIN_LEN as f32;
        for j in 0..model_const::WIN_LEN {
            self.ola_buf[j] += self.fft_real[j] * scale * self.window[j];
        }

        // First HOP samples of the OLA accumulator are now stable.
        for j in 0..model_const::HOP_SIZE {
            self.out_queue.push_back(self.ola_buf[j]);
        }
        self.ola_buf
            .copy_within(model_const::HOP_SIZE..model_const::WIN_LEN, 0);
        for j in (model_const::WIN_LEN - model_const::HOP_SIZE)..model_const::WIN_LEN {
            self.ola_buf[j] = 0.0;
        }

        // Slide analysis window forward by HOP samples.
        self.in_buf.copy_within(model_const::HOP_SIZE.., 0);
        self.in_buf
            .truncate(self.in_buf.len() - model_const::HOP_SIZE);
    }
}

impl Plugin for DpdfnetPlugin {
    fn run<'a>(&mut self, sample_count: usize, ports: &[&'a PortConnection<'a>]) {
        let input = ports[PORT_INPUT].unwrap_audio();
        let mut output = ports[PORT_OUTPUT].unwrap_audio_mut();
        let atten_db = (*ports[PORT_ATTEN_LIMIT_DB].unwrap_control()).clamp(0.0, 100.0);

        let alpha = if atten_db <= 0.0 {
            1.0
        } else if atten_db >= 100.0 {
            0.0
        } else {
            10.0_f32.powf(-atten_db / 20.0)
        };

        let n = sample_count.min(input.len()).min(output.len());

        self.in_buf.extend_from_slice(&input[..n]);

        while self.in_buf.len() >= model_const::WIN_LEN {
            self.process_frame(alpha);
        }

        // Drain whatever the queue has; pad with zeros while the
        // first WIN_LEN samples of warm-up latency build up.
        for slot in output.iter_mut().take(n) {
            *slot = self.out_queue.pop_front().unwrap_or(0.0);
        }
    }
}

// `Option<PluginDescriptor>` is what the upstream `ladspa` crate's host
// dlsym shim reads for `_ladspa_descriptor` enumeration; the FFI-safety
// lint flags it because the enum lacks a `#[repr]`, but the discriminant
// is internal to the Rust caller so the shape doesn't cross the C ABI.
#[allow(improper_ctypes_definitions)]
#[no_mangle]
pub extern "C" fn get_ladspa_descriptor(index: u64) -> Option<PluginDescriptor> {
    if index != 0 {
        return None;
    }
    Some(PluginDescriptor {
        unique_id: model_const::LADSPA_UNIQUE_ID,
        label: model_const::LADSPA_LABEL,
        properties: Properties::PROP_REALTIME,
        name: model_const::LADSPA_NAME,
        maker: "BigLinux (DPDFNet by ceva-ip)",
        copyright: "MIT",
        ports: vec![
            Port {
                name: "Audio In",
                desc: PortDescriptor::AudioInput,
                ..Default::default()
            },
            Port {
                name: "Audio Out",
                desc: PortDescriptor::AudioOutput,
                ..Default::default()
            },
            Port {
                name: "Attenuation Limit (dB)",
                desc: PortDescriptor::ControlInput,
                hint: None,
                default: Some(DefaultValue::Value0),
                lower_bound: Some(0.0),
                upper_bound: Some(100.0),
            },
        ],
        new: |_, sample_rate| Box::new(DpdfnetPlugin::new(sample_rate)),
    })
}
