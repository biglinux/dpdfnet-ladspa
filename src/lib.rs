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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use ladspa::{
    DefaultValue, Plugin, PluginDescriptor, Port, PortConnection, PortDescriptor, Properties,
};
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

mod engine;

use engine::{Inference, MAX_DEPTH};

/// Hops one 40 ms callback carries — the block the BigLinux chain negotiates,
/// and therefore the handoff depth the reported latency assumes. A graph that
/// settles on a different block gets a different real latency; this is the one
/// worth declaring.
const SHIPPED_DEPTH_HOPS: usize = 40 * model_const::SAMPLE_RATE / 1000 / model_const::HOP_SIZE;

mod model_const {
    // `MODEL_NAME` ships for diagnostics / log lines; the LADSPA hot
    // path doesn't read it. The unique_id is generated as a single
    // hash literal — splitting it with underscores would make the
    // generated file harder to read, so the readability lint is
    // also suppressed here.
    #![allow(dead_code, clippy::unreadable_literal)]
    include!(concat!(env!("OUT_DIR"), "/model_const.rs"));
}

/// Hops the plugin asked the worker for, and how many came back enhanced.
///
/// Process-global, because the reader is `bigaudioimprove-pwloader` and the
/// plugin runs *inside* it: same address space, so a number crosses between
/// them for the price of an atomic store. Two relaxed stores per hop is the
/// whole audio-thread cost.
///
/// The first design published these on LADSPA control output ports. That is
/// correct LADSPA and useless here: `module-filter-chain` surfaces control
/// inputs as node properties and keeps outputs inside the graph, so nothing
/// outside could ever read them. Verified against the running chain, which
/// lists `ai:Attenuation Limit (dB)` and neither output.
///
/// Global and monotonic on purpose. The reader asks whether *this service* is
/// keeping up, so several instances summing is the right arithmetic, and a
/// counter that only ever rises means the reader can difference two samples
/// without caring what reset in between.
static HOPS_TOTAL: AtomicU64 = AtomicU64::new(0);
static HOPS_ENHANCED: AtomicU64 = AtomicU64::new(0);

/// Read the counters above.
///
/// Exported with the C ABI so the loader can `dlsym` it out of the same
/// `.so` the filter chain already loaded — the second `dlopen` of a path
/// returns the same link map, hence the same statics the audio thread is
/// writing.
///
/// # Safety
///
/// Both pointers must be non-null and aligned; nothing is retained.
#[no_mangle]
pub unsafe extern "C" fn dpdfnet_hops(total: *mut u64, enhanced: *mut u64) {
    if total.is_null() || enhanced.is_null() {
        return;
    }
    // SAFETY: the caller guarantees both pointers are valid for one write.
    unsafe {
        *total = HOPS_TOTAL.load(Ordering::Relaxed);
        *enhanced = HOPS_ENHANCED.load(Ordering::Relaxed);
    }
}

/// Samples of latency this plugin adds, for the host to declare to its graph.
///
/// Exported with the C ABI on purpose: the chain that has to publish this
/// number is built from a different repository, and a copy in a JSON file or
/// a second constant would drift from the binary at the first partial
/// upgrade. Reading it out of the `.so` cannot drift, because it is the
/// `.so`. A caller that finds no such symbol is looking at an older plugin.
#[no_mangle]
pub extern "C" fn dpdfnet_added_latency_frames() -> u32 {
    // The handoff is as deep as the hops one callback carries, so the number
    // depends on the block the graph settled on. Reported for the shipped
    // 40 ms block; the analysis window the constructor primes is on top.
    (SHIPPED_DEPTH_HOPS * model_const::HOP_SIZE + model_const::WIN_LEN) as u32
}

/// The host sample rate this build's model requires. A filter chain running
/// at anything else gets unprocessed audio, so the value belongs in the
/// public surface rather than only in the failure message.
pub const MODEL_SAMPLE_RATE: usize = model_const::SAMPLE_RATE;

/// Largest host block we reserve for, so a realtime `run()` never grows a
/// buffer. Matches PipeWire's `default.clock.quantum-limit` default of 8192;
/// an offline host such as ffmpeg can exceed it and the buffers still grow,
/// which is why the growth path stays.
const MAX_HOST_BLOCK: usize = 8192;

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
    /// Runs the model on a worker thread, `depth` hops behind the audio.
    inference: Inference,
    /// The noisy spectrum of every hop still waiting for its answer, oldest
    /// first, tagged with the hop it came from. The head is what the worker
    /// owes us now: either the enhanced version arrives, or this is what goes
    /// out instead — the same instant either way, so the output never jumps.
    dry_delay: VecDeque<(u64, Vec<f32>)>,
    /// Hops submitted and not yet collected, oldest first. The worker answers
    /// in order, so this is how a returned frame is matched to its hop
    /// without carrying a tag through the buffers.
    submitted: VecDeque<u64>,
    /// Buffers retired from the delay line, waiting to carry the next hop.
    dry_spares: Vec<Vec<f32>>,
    /// The noisy spectrum of the hop currently being emitted, kept for the
    /// attenuation blend. Not the same hop as `spec_in`, which is `depth`
    /// ahead of it.
    blend_ref: Vec<f32>,
    /// Hops seen since this stream started.
    hop: u64,
    /// How far behind the worker is allowed to run: the hops one callback
    /// carries. Recomputed whenever the host changes the block size.
    depth: usize,
    /// False when the host runs at a rate this model cannot serve. The
    /// engine is then never asked for and audio passes through.
    rate_ok: bool,
    /// True once the unusable-runtime line has been written, so it is
    /// written once instead of on every callback.
    failure_logged: bool,
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
            inference: Inference::new(),
            // Sized for the deepest handoff so the audio thread never grows
            // it; the entries themselves are reused, never reallocated.
            dry_delay: VecDeque::with_capacity(MAX_DEPTH + 2),
            submitted: VecDeque::with_capacity(MAX_DEPTH + 2),
            dry_spares: (0..MAX_DEPTH + 2)
                .map(|_| vec![0.0; model_const::FREQ_BINS * 2])
                .collect(),
            blend_ref: vec![0.0; model_const::FREQ_BINS * 2],
            hop: 0,
            depth: 1,
            rate_ok,
            failure_logged: false,
            // Pre-fill the analysis buffer with WIN_LEN zeros so the
            // first run() call can already process ceil(n/HOP)+1 frames
            // and produce >= n samples. Without this priming the queue
            // would stay one HOP short of the host buffer on every call,
            // leaking ~64 zero samples per 1024 sample call (audible as
            // periodic clicks / robotic timbre at the host quantum
            // rate). Latency cost: WIN_LEN/sr ≈ 20 ms.
            in_buf: {
                // Primed with WIN_LEN zeros but reserved for a whole block on
                // top: the first `run()` appends before it consumes, and that
                // first callback is exactly the one this plugin must not
                // stall in.
                let mut buf = Vec::with_capacity(MAX_HOST_BLOCK + model_const::WIN_LEN);
                buf.resize(model_const::WIN_LEN, 0.0);
                buf
            },
            ola_buf: vec![0.0; model_const::WIN_LEN],
            out_queue: VecDeque::with_capacity(MAX_HOST_BLOCK + model_const::WIN_LEN),
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

        HOPS_TOTAL.fetch_add(1, Ordering::Relaxed);
        let hop = self.hop;
        self.hop += 1;

        // Collect before handing over. The queue is exactly `depth` deep, so
        // a submit before the collect always finds it full and is refused —
        // that mistake cost 76 % of the hops at a 1920-sample block.
        //
        // What goes out now is the hop from `depth` back: far enough that the
        // worker has had a whole callback of wall clock to answer it, which
        // is the only interval it gets, since one callback carries several
        // hops with no time between them.
        let due = hop.saturating_sub(self.depth as u64);
        self.emit_hop(due);

        // Then remember this hop's noisy spectrum and hand it over.
        self.stash_dry(hop);
        if self.rate_ok && self.inference.submit(&self.spec_in) {
            self.submitted.push_back(hop);
        }
        if self.inference.is_failed() && !self.failure_logged {
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
                self.spec_out[k] = alpha * self.blend_ref[k] + (1.0 - alpha) * self.spec_out[k];
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

    /// Fill `spec_out` with the answer for hop `due`: the enhanced spectrum if
    /// the worker produced it, otherwise the noisy one from that same hop.
    ///
    /// Both queues run in submission order, so pairing is a matter of
    /// discarding whatever is older than the hop being emitted. That only
    /// happens when the worker falls behind, and its late answer is out of
    /// time by then — one raw hop beats an output that jumps.
    fn emit_hop(&mut self, due: u64) {
        let mut answered = false;
        while let Some(frame) = self.inference.take() {
            let matched = self.submitted.pop_front() == Some(due);
            if matched {
                self.spec_out.copy_from_slice(&frame);
                answered = true;
            }
            self.inference.recycle(frame);
            if matched {
                HOPS_ENHANCED.fetch_add(1, Ordering::Relaxed);
                break;
            }
        }

        // Retire every noisy frame up to and including `due`, keeping the
        // buffers. The one tagged `due` is the fallback when nothing enhanced
        // arrived; anything older is the worker running behind.
        while let Some(&(hop, _)) = self.dry_delay.front() {
            if hop > due {
                break;
            }
            let Some((_, buf)) = self.dry_delay.pop_front() else {
                break;
            };
            if hop == due {
                // The noisy spectrum of the hop being emitted. The blend below
                // needs this and not the hop just captured: they are `depth`
                // hops apart, 40 ms at the shipped block, and summing two
                // different instants combs the spectrum instead of blending it.
                self.blend_ref.copy_from_slice(&buf);
                if !answered {
                    self.spec_out.copy_from_slice(&buf);
                    answered = true;
                }
            }
            self.dry_spares.push(buf);
        }

        // Nothing for this hop at all: the delay line is still filling, which
        // lasts `depth` hops at the start of a stream. Silence, like the
        // analysis priming the constructor does.
        if !answered {
            self.spec_out.fill(0.0);
        }
    }

    /// Put this hop's noisy spectrum on the delay line, on a buffer that has
    /// already been emitted rather than a fresh one.
    fn stash_dry(&mut self, hop: u64) {
        let mut slot = self
            .dry_spares
            .pop()
            .unwrap_or_else(|| vec![0.0; model_const::FREQ_BINS * 2]);
        slot.copy_from_slice(&self.spec_in);
        self.dry_delay.push_back((hop, slot));
    }
}

impl Plugin for DpdfnetPlugin {
    /// The host is about to start this node. Ask the worker to compile now,
    /// so the build overlaps the graph coming up rather than the first words
    /// the user speaks.
    fn activate(&mut self) {
        self.inference.prime();
    }

    /// The host stopped this node. Forget the recurrent state so the next
    /// stream starts clean. The compiled model stays for a minute in case
    /// this was a mute rather than a hang-up.
    fn deactivate(&mut self) {
        self.inference.reset();
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

        // One callback carries this many analysis hops, back to back with no
        // wall clock between them, so that is exactly how far behind the
        // worker has to be allowed to run. Pinning it at one let only the
        // first hop of each callback come back enhanced: measured 24 % at a
        // 1920-sample quantum, 48 % at 960, 90 % at 480.
        let hops_per_callback = n.div_ceil(model_const::HOP_SIZE).max(1);
        if hops_per_callback != self.depth {
            self.depth = hops_per_callback;
            self.inference.set_depth(hops_per_callback);
        }

        // One callback carries this many analysis hops, back to back with no
        // wall clock between them, so that is exactly how far behind the
        // worker has to be allowed to run.
        let hops_per_callback = n.div_ceil(model_const::HOP_SIZE).max(1);
        if hops_per_callback != self.depth {
            self.depth = hops_per_callback;
            self.inference.set_depth(hops_per_callback);
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use ladspa::PortData;
    use std::cell::RefCell;

    /// A realtime callback must not touch the allocator, and the first one is
    /// the whole point of this plugin's design. `Vec::capacity` is the
    /// observation: if any buffer grows, it reallocated.
    ///
    /// Drives the concrete type, not the boxed trait object, so the buffers
    /// being watched are the ones the callback actually wrote to.
    #[test]
    fn no_callback_grows_a_buffer() {
        let descriptor = get_ladspa_descriptor(0).expect("descriptor");
        let ports = descriptor.ports.clone();
        let mut plugin = DpdfnetPlugin::new(MODEL_SAMPLE_RATE as u64);
        plugin.activate();

        // The largest block we reserve for, driven at once so the first
        // callback is also the worst callback.
        let quantum = MAX_HOST_BLOCK;
        let controls = [100.0_f32];
        let input = vec![0.1_f32; quantum];
        let mut output = vec![0.0_f32; quantum];

        let reserved = (
            plugin.in_buf.capacity(),
            plugin.ola_buf.capacity(),
            plugin.out_queue.capacity(),
        );

        for call in 0..4 {
            {
                let mut output_slot = Some(&mut output[..]);
                let mut connections: Vec<PortConnection> = Vec::with_capacity(ports.len());
                for (i, port) in ports.iter().enumerate() {
                    let data = match i {
                        0 => PortData::AudioInput(&input),
                        1 => PortData::AudioOutput(RefCell::new(
                            output_slot.take().expect("one output port"),
                        )),
                        _ => PortData::ControlInput(&controls[0]),
                    };
                    connections.push(PortConnection { port: *port, data });
                }
                let refs: Vec<&PortConnection> = connections.iter().collect();
                plugin.run(quantum, &refs);
            }

            let now = (
                plugin.in_buf.capacity(),
                plugin.ola_buf.capacity(),
                plugin.out_queue.capacity(),
            );
            assert_eq!(
                now, reserved,
                "callback {call} reallocated: {reserved:?} became {now:?}"
            );
        }
    }
}
