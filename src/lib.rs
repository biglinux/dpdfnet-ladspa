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
use std::sync::Arc;

use ladspa::{
    DefaultValue, Plugin, PluginDescriptor, Port, PortConnection, PortDescriptor, Properties,
};
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

mod engine;
use engine::EngineLoader;

mod model_const {
    // `MODEL_NAME` ships for diagnostics / log lines; the LADSPA hot
    // path doesn't read it. The unique_id is generated as a single
    // hash literal — splitting it with underscores would make the
    // generated file harder to read, so the readability lint is
    // also suppressed here.
    #![allow(dead_code, clippy::unreadable_literal)]
    include!(concat!(env!("OUT_DIR"), "/model_const.rs"));
}

/// The host sample rate this build's model requires. A filter chain running
/// at anything else gets unprocessed audio, so the value belongs in the
/// public surface rather than only in the failure message.
pub const MODEL_SAMPLE_RATE: usize = model_const::SAMPLE_RATE;

/// Smallest PipeWire block, in milliseconds, at which this build's model keeps
/// every callback inside its deadline. Measured by `tests/callback_deadline.rs`.
///
/// A quality tier that offers this model must negotiate at least this block,
/// and pay it as microphone latency.
pub const MIN_BLOCK_MS: u32 = model_const::MIN_BLOCK_MS;

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
    /// Builds the engine on a worker thread and hands it back when the
    /// host deactivates the node. `None` from it means "pass through".
    loader: EngineLoader,
    /// False when the host runs at a rate this model cannot serve. The
    /// engine is then never asked for and audio passes through.
    rate_ok: bool,
    /// True once the unusable-runtime line has been written, so it is
    /// written once instead of on every callback.
    failure_logged: bool,
    /// Per-stream GRU state, read and rewritten by every inference.
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
        // A wrong host rate used to abort the process. LADSPA cannot refuse
        // an instantiation, so refuse the model instead: `rate_ok` false
        // means the engine is never asked for and audio passes through.
        let rate_ok = sr == model_const::SAMPLE_RATE;
        if !rate_ok {
            eprintln!(
                "[{}] host sample rate is {sr} Hz, this plugin needs {}; \
                 passing audio through unprocessed. Set `audio.rate = {}` \
                 on the filter-chain node.",
                model_const::LADSPA_LABEL,
                model_const::SAMPLE_RATE,
                model_const::SAMPLE_RATE
            );
        }

        // The engine is built on a worker thread, not here and not in the
        // callback. An instance whose graph has no consumer never asks for
        // one, so an idle chain does not pay the OpenVINO Core and JIT cost;
        // when audio starts, the first frames pass through clean until the
        // build lands (170-460 ms depending on the model).
        let mut planner = RealFftPlanner::<f32>::new();
        let fft_fwd = planner.plan_fft_forward(model_const::WIN_LEN);
        let fft_inv = planner.plan_fft_inverse(model_const::WIN_LEN);

        Self {
            loader: EngineLoader::new(),
            rate_ok,
            failure_logged: false,
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

        // No engine yet (still compiling on the worker), or the model
        // refused the frame: carry the input spectrum through unchanged.
        // The synthesis rings below still advance, so one unprocessed
        // frame costs a little noise where a skipped one would put the
        // overlap-add permanently out of step.
        let enhanced = if self.rate_ok {
            match self.loader.engine() {
                Some(engine) => engine.run(&self.spec_in, &mut self.state, &mut self.spec_out),
                None => false,
            }
        } else {
            false
        };
        if !enhanced {
            self.spec_out.copy_from_slice(&self.spec_in);
        }
        if self.loader.is_failed() && !self.failure_logged {
            self.failure_logged = true;
            // The step that failed is already on the journal, one line up.
            eprintln!(
                "[{}] no inference engine; passing the microphone through \
                 unprocessed",
                model_const::LADSPA_LABEL
            );
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
    /// The host stopped this node. Hand the compiled model back so an idle
    /// chain stops holding it; the next activation rebuilds on the worker.
    fn deactivate(&mut self) {
        self.loader.retire();
    }

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
                // 0 dB is full passthrough, so a host that never writes this
                // control would get a denoiser that denoises nothing. Default
                // to the top of the range: unlimited suppression, which is
                // what someone loading a noise-suppression plugin asked for.
                default: Some(DefaultValue::Maximum),
                lower_bound: Some(0.0),
                upper_bound: Some(100.0),
            },
        ],
        new: |_, sample_rate| Box::new(DpdfnetPlugin::new(sample_rate)),
    })
}
