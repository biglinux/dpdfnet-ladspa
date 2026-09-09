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
use std::time::{Duration, Instant};

use ladspa::{
    DefaultValue, Plugin, PluginDescriptor, Port, PortConnection, PortDescriptor, Properties,
};
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

mod engine;
mod highband;

use engine::{Inference, MAX_DEPTH};

/// Hops the network's answer is already behind the frame it was asked about.
///
/// The model returns the enhanced spectrum of a frame four hops older than the
/// one submitted. Measured on the shipped graph, driven frame by frame through
/// a minimal host: the reconstruction correlates with its input at exactly
/// 1920 samples at 48 kHz and 640 at 16 kHz — 4.00 hops either way. It is also
/// what the OBS filter built on the same ONNX documents ("four hops (40 ms) of
/// internal signal delay"), which it reads from an `output_delay_hops`
/// metadata key — a key neither that ONNX nor this IR actually carries, which
/// is why the figure is a constant here.
///
/// The delay itself is not avoidable: the enhanced audio of a frame cannot
/// leave before the network has seen four more. What it must not do is go
/// unaccounted, which is what made the dry fallback jump 40 ms ahead of the
/// enhanced path and the attenuation blend sum two different instants.
const MODEL_DELAY_HOPS: usize = 4;

/// Hops between the frame handed to the worker and the frame going out.
///
/// One, so the delay this plugin adds no longer depends on the host's block.
/// It used to be the whole callback: the same model cost 80 ms in a chain at a
/// 960-sample quantum, 140 ms at 3840, and 150 ms in ffmpeg's converter.
///
/// A lag this short means the answer is not there yet when the hop comes due —
/// a callback carries its hops back to back with no wall clock between them,
/// so lowering the lag *alone* took the pause floor from −15.91 dB to
/// −2.94 dB, nearly every hop coming back raw. It works only together with
/// [`WAIT_BUDGET`]: the callback waits for the worker instead of giving up on
/// it.
const EMIT_LAG_HOPS: usize = 1;

/// Share of a callback's own duration the plugin may spend waiting for the
/// worker before it emits noisy hops for the rest of that callback.
///
/// Inference costs 3.3 ms per 10 ms hop for the heaviest model, so half the
/// period covers a whole callback of hops with margin and still leaves the
/// other half of the quantum to the rest of the graph. Running out is not an
/// error: the hop goes out noisy but time-aligned, which is the degradation
/// this plugin was built around.
const WAIT_BUDGET: f64 = 0.5;

/// Noisy frames the delay line holds: the handoff lag, the model's own delay,
/// and one being retired. Fixed, now that the lag is — the pool used to carry
/// one buffer per possible callback depth, 66 of them, for a line that is
/// never more than six deep.
const DRY_LINE: usize = EMIT_LAG_HOPS + MODEL_DELAY_HOPS + 2;

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
/// Three terms, none of them the host's: the network's own four hops, the one
/// hop of handoff lag, and the analysis window the constructor primes — 70 ms
/// at 48 kHz. The model's share used to be missing and the handoff was the
/// host's whole block, so the published number was 40 ms short of the audio
/// delivered and wrong again by however far the block sat from 20 ms.
#[no_mangle]
pub extern "C" fn dpdfnet_added_latency_frames() -> u32 {
    // Expressed in host samples: on a split-band build the hop and window are
    // the 48 kHz STFT's, not the 16 kHz model's, so the published figure is the
    // audio the graph actually sees delayed.
    ((MODEL_DELAY_HOPS + EMIT_LAG_HOPS) * model_const::STFT_HOP + model_const::STFT_WIN_LEN) as u32
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

/// How long an offline callback may wait for the worker. Generous, because
/// there is no deadline; bounded, so a wedged worker cannot hang a converter.
const OFFLINE_WAIT: Duration = Duration::from_millis(500);

/// Audio to observe before judging whether the host has a deadline. Long
/// enough that the engine build is not mistaken for a slow host, short enough
/// that a converted file loses only its first fifth of a second.
const OFFLINE_WARMUP_S: f64 = 0.2;

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

/// Raised-cosine blend across the four bins straddling the model / high-band
/// boundary, so the enhanced low band and the reconstructed high band do not
/// meet at a hard edge. A no-op when there is no high band (`low_bins ==
/// spectrum.len()`, the non-split build).
fn crossfade_boundary(spectrum: &mut [Complex<f32>], low_bins: usize) {
    const WIDTH: usize = 4;
    if low_bins < WIDTH / 2 || low_bins + WIDTH / 2 > spectrum.len() {
        return;
    }
    let base = low_bins - WIDTH / 2;
    let snapshot: [Complex<f32>; WIDTH] = std::array::from_fn(|k| spectrum[base + k]);
    let low = [snapshot[0], snapshot[1]]; // pure model side
    let high = [snapshot[2], snapshot[3]]; // pure high-band side
    for k in 0..WIDTH {
        let t = (k as f32 + 0.5) / WIDTH as f32;
        let w = 0.5 * (1.0 - (std::f32::consts::PI * t).cos());
        let model_val = if k < WIDTH / 2 { low[k] } else { low[1] };
        let hf_val = if k >= WIDTH / 2 {
            high[k - WIDTH / 2]
        } else {
            high[0]
        };
        spectrum[base + k] = Complex::new(
            model_val.re * (1.0 - w) + hf_val.re * w,
            model_val.im * (1.0 - w) + hf_val.im * w,
        );
    }
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
    /// Submissions the engine will accept before refusing: the hops one
    /// callback carries. Recomputed whenever the host changes the block size.
    depth: usize,
    /// Wall clock the current callback may wait until before it stops asking
    /// the worker for answers. [`WAIT_BUDGET`] of the callback's own duration,
    /// set once per callback so the hops inside it share one budget.
    wait_until: Instant,
    /// True once the host has been seen feeding audio faster than real time,
    /// which means it is a file converter and has no deadline to miss. Then
    /// the callback may wait for the worker instead of emitting raw audio.
    offline: bool,
    /// Audio handed over since the first callback, in seconds.
    audio_seen: f64,
    /// When the first callback ran, for that comparison.
    last_run: Option<Instant>,
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
    /// Forward-FFT result: the original noisy spectrum, `STFT_FREQ_BINS` long.
    /// For a split-band build this is wider than the model band, and the bins
    /// above the model cutoff are the high band the model never sees.
    fft_complex: Vec<Complex<f32>>,
    /// Enhanced spectrum assembled for the inverse FFT: the model's low band
    /// scaled back up, plus the reconstructed high band. Same width as
    /// `fft_complex`; equal to it bin-for-bin on a non-split build.
    out_complex: Vec<Complex<f32>>,
    spec_in: Vec<f32>,
    spec_out: Vec<f32>,
    /// High-band reconstruction and its speech gate — only a split-band build
    /// carries them; a normal build's high band is empty.
    highband: Option<highband::HighBand>,
    speech_gate: highband::SpeechGate,
}

impl DpdfnetPlugin {
    fn new(sample_rate: u64) -> Self {
        let sr = sample_rate as usize;
        // A wrong host rate used to abort the process. LADSPA cannot refuse
        // an instantiation, so refuse the model instead: `rate_ok` false
        // means the engine is never asked for and audio passes through. The
        // rate the plugin accepts is the STFT/host rate — 48 kHz for a
        // split-band build, whose model band stays 16 kHz internally.
        let rate_ok = sr == model_const::HOST_SAMPLE_RATE;
        if !rate_ok {
            eprintln!(
                "[{}] host sample rate is {sr} Hz, this plugin needs {}; \
                 passing audio through unprocessed. Set `audio.rate = {}` \
                 on the filter-chain node.",
                model_const::LADSPA_LABEL,
                model_const::HOST_SAMPLE_RATE,
                model_const::HOST_SAMPLE_RATE
            );
        }

        // The engine is built on a worker thread, not here and not in the
        // callback. An instance whose graph has no consumer never asks for
        // one, so an idle chain does not pay the OpenVINO Core and JIT cost;
        // when audio starts, the first frames pass through clean until the
        // build lands (170-460 ms depending on the model).
        // The STFT runs at the host geometry (== the model geometry on a
        // normal build, wider on a split-band one). The model band stays
        // `FREQ_BINS` and is extracted from the low end of this spectrum.
        let mut planner = RealFftPlanner::<f32>::new();
        let fft_fwd = planner.plan_fft_forward(model_const::STFT_WIN_LEN);
        let fft_inv = planner.plan_fft_inverse(model_const::STFT_WIN_LEN);
        let hf_bins = model_const::STFT_FREQ_BINS - model_const::FREQ_BINS;
        let highband = model_const::SPLIT_BAND.then(|| {
            highband::HighBand::new(
                hf_bins,
                model_const::STFT_HOP,
                model_const::HOST_SAMPLE_RATE,
            )
        });

        Self {
            inference: Inference::new(),
            // The dry line is exactly as deep as the handoff; `submitted` has
            // to hold whatever the engine accepted, which is a callback's
            // worth. Neither grows on the audio thread, and the entries
            // themselves are reused rather than reallocated.
            dry_delay: VecDeque::with_capacity(DRY_LINE),
            submitted: VecDeque::with_capacity(MAX_DEPTH + 2),
            dry_spares: (0..DRY_LINE)
                .map(|_| vec![0.0; model_const::FREQ_BINS * 2])
                .collect(),
            blend_ref: vec![0.0; model_const::FREQ_BINS * 2],
            hop: 0,
            depth: 1,
            wait_until: Instant::now(),
            offline: false,
            audio_seen: 0.0,
            last_run: None,
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
                // Primed with STFT_WIN_LEN zeros but reserved for a whole block
                // on top: the first `run()` appends before it consumes, and
                // that first callback is exactly the one this plugin must not
                // stall in.
                let mut buf = Vec::with_capacity(MAX_HOST_BLOCK + model_const::STFT_WIN_LEN);
                buf.resize(model_const::STFT_WIN_LEN, 0.0);
                buf
            },
            ola_buf: vec![0.0; model_const::STFT_WIN_LEN],
            out_queue: VecDeque::with_capacity(MAX_HOST_BLOCK + model_const::STFT_WIN_LEN),
            window: vorbis_window(model_const::STFT_WIN_LEN),
            fft_real: vec![0.0; model_const::STFT_WIN_LEN],
            fft_complex: vec![Complex::new(0.0, 0.0); model_const::STFT_FREQ_BINS],
            out_complex: vec![Complex::new(0.0, 0.0); model_const::STFT_FREQ_BINS],
            fft_fwd,
            fft_inv,
            spec_in: vec![0.0; model_const::FREQ_BINS * 2],
            spec_out: vec![0.0; model_const::FREQ_BINS * 2],
            highband,
            speech_gate: highband::SpeechGate::new(),
        }
    }

    /// Process exactly one analysis frame: window + FFT + ONNX +
    /// spectral blend + high-band reconstruction + iFFT + windowed OLA + flush
    /// HOP samples to `out_queue`. Caller guarantees
    /// `in_buf.len() >= STFT_WIN_LEN`.
    fn process_frame(&mut self, alpha: f32) {
        for j in 0..model_const::STFT_WIN_LEN {
            self.fft_real[j] = self.in_buf[j] * self.window[j];
        }
        let _ = self
            .fft_fwd
            .process(&mut self.fft_real, &mut self.fft_complex);

        // The model only ever sees the low `FREQ_BINS` bins. On a split-band
        // build the STFT is wider, so scale those bins to the magnitude the
        // 16 kHz network was trained on (`SPECTRUM_SCALE` is 1.0 otherwise).
        for k in 0..model_const::FREQ_BINS {
            self.spec_in[k * 2] = self.fft_complex[k].re * model_const::SPECTRUM_SCALE;
            self.spec_in[k * 2 + 1] = self.fft_complex[k].im * model_const::SPECTRUM_SCALE;
        }

        HOPS_TOTAL.fetch_add(1, Ordering::Relaxed);
        let hop = self.hop;
        self.hop += 1;

        // Collect before handing over. The engine's window is exactly `depth`
        // deep, so a submit before the collect always finds it full and is
        // refused — that mistake cost 76 % of the hops at a 1920-sample block.
        //
        // What goes out now is the hop right behind this one, not the hop a
        // whole callback back: a callback carrying nine hops used to delay the
        // audio by nine, and every host got a different latency. Nothing is
        // due until the delay line has filled. Saturating this to zero instead
        // made the first hops all claim hop 0, and each one popped a different
        // entry off `submitted` — after which the tag the collector expected
        // never matched a tag the worker sent again, for the rest of the
        // stream.
        let due = hop.checked_sub(EMIT_LAG_HOPS as u64);
        match due {
            Some(due) => self.emit_hop(due),
            None => self.spec_out.fill(0.0),
        }

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

        // Assemble the enhanced full-width spectrum for the inverse FFT. The
        // model's low band is scaled back up to the host magnitude; on a
        // split-band build the band above the model cutoff is reconstructed
        // from the original high band and the clean low band.
        let inv_scale = 1.0 / model_const::SPECTRUM_SCALE;
        for k in 0..model_const::FREQ_BINS {
            self.out_complex[k] = Complex::new(
                self.spec_out[k * 2] * inv_scale,
                self.spec_out[k * 2 + 1] * inv_scale,
            );
        }
        if let Some(highband) = self.highband.as_mut() {
            let input_energy: f32 = (0..model_const::FREQ_BINS)
                .map(|k| self.spec_in[k * 2].powi(2) + self.spec_in[k * 2 + 1].powi(2))
                .sum();
            let enhanced_energy: f32 = (0..model_const::FREQ_BINS)
                .map(|k| self.spec_out[k * 2].powi(2) + self.spec_out[k * 2 + 1].powi(2))
                .sum();
            let speech = self.speech_gate.update(input_energy, enhanced_energy);

            highband.process(
                &self.fft_complex[model_const::FREQ_BINS..],
                speech,
                &mut self.out_complex[model_const::FREQ_BINS..],
            );
            crossfade_boundary(&mut self.out_complex, model_const::FREQ_BINS);
        }

        let _ = self
            .fft_inv
            .process(&mut self.out_complex, &mut self.fft_real);

        // realfft inverse leaves a 1/N scaling — fold into the synthesis
        // window so OLA gets the correct amplitude.
        let scale = 1.0 / model_const::STFT_WIN_LEN as f32;
        for j in 0..model_const::STFT_WIN_LEN {
            self.ola_buf[j] += self.fft_real[j] * scale * self.window[j];
        }

        // First HOP samples of the OLA accumulator are now stable.
        for j in 0..model_const::STFT_HOP {
            self.out_queue.push_back(self.ola_buf[j]);
        }
        self.ola_buf
            .copy_within(model_const::STFT_HOP..model_const::STFT_WIN_LEN, 0);
        for j in (model_const::STFT_WIN_LEN - model_const::STFT_HOP)..model_const::STFT_WIN_LEN {
            self.ola_buf[j] = 0.0;
        }

        // Slide analysis window forward by HOP samples.
        self.in_buf.copy_within(model_const::STFT_HOP.., 0);
        self.in_buf
            .truncate(self.in_buf.len() - model_const::STFT_HOP);
    }

    /// Fill `spec_out` with the answer for hop `due`: the enhanced spectrum if
    /// the worker produced it, otherwise the noisy frame that sits at the same
    /// instant.
    ///
    /// Both queues run in submission order, so pairing is a matter of
    /// discarding whatever is older than the hop being emitted. That only
    /// happens when the worker falls behind, and its late answer is out of
    /// time by then — one raw hop beats an output that jumps.
    ///
    /// "The same instant" is what makes the raw fallback usable, and it is not
    /// hop `due`: the network answers about a frame [`MODEL_DELAY_HOPS`] older
    /// than the one submitted, so the noisy spectrum that lines up with the
    /// enhanced one is `due - MODEL_DELAY_HOPS`. Taking hop `due` instead put
    /// the dry path 40 ms ahead of the enhanced path — a jump of exactly that
    /// size whenever the worker missed a hop, and an attenuation blend that
    /// summed two instants 40 ms apart instead of mixing one.
    fn emit_hop(&mut self, due: u64) {
        let mut answered = false;
        // The dry line cannot line up before it holds the model's own delay.
        let Some(dry_due) = due.checked_sub(MODEL_DELAY_HOPS as u64) else {
            self.spec_out.fill(0.0);
            self.blend_ref.fill(0.0);
            return;
        };

        // Answers come back in submission order, so the front of `submitted`
        // names the next frame the worker will hand over. Comparing that tag
        // against `due` before collecting is what keeps the two streams in
        // step: hops the worker refused are simply absent from this queue, so
        // `due` is not `hop - depth` of anything it holds, and a collector
        // that popped unconditionally consumed the wrong tag and never lined
        // up again — measured as every later hop coming back raw.
        while let Some(&next) = self.submitted.front() {
            if next > due {
                break;
            }
            // At a one-hop lag the answer is normally still in flight, so this
            // is where the callback spends its budget. Recomputed per
            // iteration: a stale frame discarded here already consumed part of
            // it.
            let waiting = self.wait_timeout();
            let Some(frame) = (if waiting.is_zero() {
                self.inference.take()
            } else {
                self.inference.take_waiting(waiting)
            }) else {
                break;
            };
            self.submitted.pop_front();
            let matched = next == due;
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

        // Retire every noisy frame up to and including `dry_due`, keeping the
        // buffers. The one tagged `dry_due` is the fallback when nothing
        // enhanced arrived; anything older is the worker running behind.
        while let Some(&(hop, _)) = self.dry_delay.front() {
            if hop > dry_due {
                break;
            }
            let Some((_, buf)) = self.dry_delay.pop_front() else {
                break;
            };
            if hop == dry_due {
                // The noisy spectrum at the instant the enhanced one describes,
                // which is neither the hop just captured nor the hop submitted.
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

    /// How long this hop may block waiting for the worker.
    ///
    /// Whatever is left of the callback's budget, which is what makes a
    /// one-hop lag possible at all. A host with no deadline — a file converter
    /// — gets [`OFFLINE_WAIT`] instead, because there is nothing to be late
    /// for and every hop should come back enhanced.
    fn wait_timeout(&self) -> Duration {
        if self.offline {
            return OFFLINE_WAIT;
        }
        self.wait_until.saturating_duration_since(Instant::now())
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

        // A host that has handed over far more audio than time has passed is
        // reading a file, not a microphone. Compared cumulatively and latched
        // once true: the moment the callback starts waiting for the worker the
        // two rates converge, so a test that keeps re-deciding would switch
        // itself back off and sit at a few per cent enhanced.
        let now = Instant::now();
        let started = *self.last_run.get_or_insert(now);
        self.audio_seen += n as f64 / model_const::HOST_SAMPLE_RATE as f64;
        if !self.offline && self.audio_seen > OFFLINE_WARMUP_S {
            let elapsed = now.duration_since(started).as_secs_f64();
            self.offline = self.audio_seen > elapsed * 4.0;
        }

        // The hops in this callback share one waiting budget, measured from
        // the callback's own start: a hop that spends it all leaves the rest of
        // the block emitting noisy frames rather than pushing the whole graph
        // past its deadline.
        self.wait_until = now
            + Duration::from_secs_f64(
                n as f64 / model_const::HOST_SAMPLE_RATE as f64 * WAIT_BUDGET,
            );

        // One callback carries this many analysis hops, back to back with no
        // wall clock between them, so that is how many submissions the engine
        // has to accept before refusing. It used to be the emitted hop's lag
        // too, which is what made the latency the host's block.
        let hops_per_callback = n.div_ceil(model_const::STFT_HOP).max(1);
        if hops_per_callback != self.depth {
            self.depth = hops_per_callback;
            self.inference.set_depth(hops_per_callback);
        }
        self.in_buf.extend_from_slice(&input[..n]);

        while self.in_buf.len() >= model_const::STFT_WIN_LEN {
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
        let mut plugin = DpdfnetPlugin::new(model_const::HOST_SAMPLE_RATE as u64);
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
