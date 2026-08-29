//! OpenVINO inference, run on a worker thread rather than on the audio thread.
//!
//! Two costs used to land inside `run()`. Building the engine takes 170 ms for
//! the 16 kHz baseline and 460 ms for `dpdfnet8_48khz_hr`, against a 40 ms
//! callback budget. And inference itself, at 3.3 ms per 10 ms hop for the
//! heaviest model, is cheap on average but bursty: when several analysis hops
//! fall into one wake-up the callback overruns, and the whole graph xruns.
//!
//! Making the model faster does not fix the second one. Measured on this
//! CPU: INT8 through NNCF gave 3.29 ms against 3.30 ms for f32 — with 657
//! genuinely `i8` operations in the compiled graph — two inference threads
//! made it *slower* (3.79 to 5.30 ms), and the `f16` hint likewise. The cost
//! is dispatching a long chain of small recurrent operations, not arithmetic,
//! so no precision or parallelism lever moves it.
//!
//! What works is not requiring the answer inside the same callback.
//! [`Inference`] ships each analysis frame to a worker and reads back the
//! frame submitted one hop earlier. A late result is not an xrun: the caller
//! emits the delayed dry spectrum for that hop instead, time-aligned, and the
//! recurrent state stays with the worker so nothing desynchronizes. The price
//! is one callback of extra latency, which the host must declare.
//!
//! Nothing here panics. A machine without a usable OpenVINO runtime gets a
//! plugin that passes audio through, not a `pwloader` process that aborts and
//! takes the user's microphone with it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use openvino::{CompiledModel, Core, ElementType, InferRequest, RwPropertyKey, Shape, Tensor};

use crate::model_const;

/// One compiled model with its bound input tensors.
///
/// Owned by the worker thread, one per plugin instance. A shared engine would
/// need a mutex, and PipeWire can run more than one data loop — two graphs on
/// different loops would then block each other. The cost of not sharing is one
/// oneDNN scratchpad per instance, 10-15 MB, and in practice each chain has
/// its own `pwloader` process anyway.
struct Engine {
    infer: InferRequest,
    spec_tensor: Tensor,
    state_tensor: Tensor,
    /// Keeps the model and its weights alive; the bound tensors point into it.
    _compiled: CompiledModel,
}

impl Engine {
    /// Run one frame. `state` is read and then overwritten with the recurrent
    /// state the model produced.
    ///
    /// Returns `false` when OpenVINO refused the frame. The caller must still
    /// finish the frame — leaving `spec_out` as the unmodified input keeps the
    /// overlap-add rings in step, and one unprocessed frame is inaudible where
    /// a skipped one desynchronizes the synthesis for good.
    fn run(&mut self, spec_in: &[f32], state: &mut [f32], spec_out: &mut [f32]) -> bool {
        let Some(spec_slot) = self.spec_tensor.get_data_mut::<f32>().ok() else {
            return false;
        };
        spec_slot.copy_from_slice(spec_in);
        let Some(state_slot) = self.state_tensor.get_data_mut::<f32>().ok() else {
            return false;
        };
        state_slot.copy_from_slice(state);

        if self.infer.infer().is_err() {
            return false;
        }

        let Ok(spec_e) = self.infer.get_tensor("spec_e") else {
            return false;
        };
        let Ok(spec_e_data) = spec_e.get_data::<f32>() else {
            return false;
        };
        if spec_e_data.len() != spec_out.len() {
            return false;
        }
        spec_out.copy_from_slice(spec_e_data);

        let Ok(state_out) = self.infer.get_tensor("state_out") else {
            return false;
        };
        let Ok(state_out_data) = state_out.get_data::<f32>() else {
            return false;
        };
        if state_out_data.len() != state.len() {
            return false;
        }
        state.copy_from_slice(state_out_data);
        true
    }
}

/// Report why a build step failed, once per attempt. A silent `None` here
/// costs an afternoon: the chain comes up, passes audio through, and nothing
/// says which of the eleven fallible steps gave up.
fn step<T, E: std::fmt::Display>(what: &str, result: Result<T, E>) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(error) => {
            eprintln!(
                "[{}] engine build failed at {what}: {error}",
                model_const::LADSPA_LABEL
            );
            None
        }
    }
}

/// Compile the embedded IR and bind its input tensors. `None` on any failure.
fn build() -> Option<Engine> {
    let mut core = step("Core::new", Core::new())?;

    // Configure the CPU plugin BEFORE compile so the optimization pipeline
    // picks up our hints:
    //
    // * `SNIPPETS_MODE=DISABLE` — DPDFNet's GRU subgraph trips a bug in the
    //   Snippets tokenizer ("Topological order is required, but not set."
    //   raised from `pass/tokenization.cpp`) on OpenVINO 2026.0.0. Disabling
    //   the pass routes those ops through the regular oneDNN kernels —
    //   slightly less fused but functional.
    // * `PERFORMANCE_HINT=LATENCY` + `NUM_STREAMS=1` +
    //   `INFERENCE_NUM_THREADS=1` keep every op on the calling thread. The
    //   LADSPA host already calls us on a FIFO 83 RT thread; spawning
    //   OpenVINO worker threads at TBB defaults would cause priority
    //   inversion under load.
    // * `ENABLE_CPU_PINNING=NO` — `pwloader` already pins to P-cores via
    //   `sched_setaffinity`. Letting the CPU plugin add its own pinning on
    //   top can race with our mask.
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

    // IR weights must be passed as a U8 Tensor wrapping the .bin bytes. Once
    // compile_model finishes, the CompiledModel holds its own copy of the
    // weights, so this scratch tensor can drop.
    let weights_shape = step(
        "weights shape",
        Shape::new(&[model_const::IR_BIN.len() as i64]),
    )?;
    let mut weights_tensor = step(
        "weights tensor",
        Tensor::new(ElementType::U8, &weights_shape),
    )?;
    step("weights buffer", weights_tensor.get_data_mut::<u8>())?
        .copy_from_slice(model_const::IR_BIN);

    let model = step(
        "read_model_from_buffer",
        core.read_model_from_buffer(model_const::IR_XML, Some(&weights_tensor)),
    )?;
    let mut compiled = step("compile_model(CPU)", core.compile_model(&model, cpu))?;
    let mut infer = step("create_infer_request", compiled.create_infer_request())?;

    let spec_shape = step(
        "spec shape",
        Shape::new(&[1, 1, model_const::FREQ_BINS as i64, 2]),
    )?;
    let state_shape = step("state shape", Shape::new(&[model_const::STATE_SIZE as i64]))?;
    let spec_tensor = step("spec tensor", Tensor::new(ElementType::F32, &spec_shape))?;
    let state_tensor = step("state tensor", Tensor::new(ElementType::F32, &state_shape))?;
    step("set_tensor(spec)", infer.set_tensor("spec", &spec_tensor))?;
    step(
        "set_tensor(state_in)",
        infer.set_tensor("state_in", &state_tensor),
    )?;

    Some(Engine {
        infer,
        spec_tensor,
        state_tensor,
        _compiled: compiled,
    })
}

/// Deepest handoff the caller may ask for.
///
/// One callback carries `quantum / HOP_SIZE` analysis hops and the worker gets
/// no wall clock between them, so the queue has to be that deep or the extra
/// hops go out unprocessed. Measured with the queue pinned at one, driving
/// `dpdfnet8_48khz_hr`: 90 % of hops enhanced at a 480-sample quantum, 48 % at
/// 960, 24 % at 1920 — one per callback, exactly.
///
/// The ceiling is PipeWire's `default.clock.quantum-limit` of 8192 over the
/// smallest hop we ship, 160 samples at 16 kHz, rounded up.
pub const MAX_DEPTH: usize = 64;

/// Spectra in circulation: the deepest handoff, plus the one being read and a
/// spare, so a submit never fails for want of a buffer.
const POOL: usize = MAX_DEPTH + 2;

/// A spectrum buffer travelling between the two threads. Buffers are recycled
/// rather than allocated, so the audio thread never touches the allocator.
type Frame = Box<[f32]>;

/// How long an idle worker keeps the compiled model before releasing it.
///
/// A call that mutes and unmutes is the common case, and rebuilding costs
/// 170-460 ms of unprocessed audio each time. A minute covers that while a
/// chain nobody is recording from still gives its ~100 MB back.
const LINGER: Duration = Duration::from_secs(60);

enum Job {
    /// Interleaved re/im spectrum to enhance. Comes back through `done`.
    Run(Frame),
    /// Start compiling now, before audio arrives. Sent from `activate()`, so
    /// the build overlaps the graph coming up instead of the first speech.
    Prime,
    /// A new stream. Forget the recurrent state and clear a previous failure,
    /// so each activation gets one fresh attempt. The engine stays: releasing
    /// it is what [`LINGER`] is for.
    Reset,
}

/// The worker thread and the two queues to it.
///
/// Dropping this closes the job channel, which ends the worker.
pub struct Inference {
    /// Joined on drop. Without it the process can tear down while the worker
    /// is inside OpenVINO — observed as an intermittent SIGSEGV when a test
    /// binary exits mid-build, and the same race exists when `pwloader`
    /// stops.
    worker: Option<thread::JoinHandle<()>>,
    /// `Option` only so `Drop` can close it before joining; a live sender
    /// would leave the worker waiting on a queue nobody will fill.
    jobs: Option<SyncSender<Job>>,
    done: Receiver<Frame>,
    /// Buffers not currently in flight.
    pool: Vec<Frame>,
    /// Frames handed over and not yet collected, and the ceiling the caller
    /// set for it. The ceiling is what keeps the handoff exactly as deep as
    /// the caller's own delay line, so the two stay paired.
    outstanding: usize,
    depth: usize,
    /// The engine could not be built. Submitting again would re-run a
    /// 170-460 ms compile on the worker every hop.
    failed: Arc<AtomicBool>,
    /// When this side first saw `failed`. The latch expires after
    /// [`RETRY_AFTER`], so a transient failure — a half-installed runtime
    /// during an upgrade, a moment without memory — heals itself instead of
    /// leaving the microphone dry until the process restarts.
    failed_since: Option<Instant>,
}

/// How long a failed build is trusted to stay failed.
const RETRY_AFTER: Duration = Duration::from_secs(5);

impl Inference {
    #[must_use]
    pub fn new() -> Self {
        let (jobs, job_rx) = sync_channel::<Job>(POOL);
        let (done_tx, done) = sync_channel::<Frame>(POOL);
        let failed = Arc::new(AtomicBool::new(false));
        let worker_failed = Arc::clone(&failed);

        let worker = thread::spawn(move || {
            let mut engine: Option<Engine> = None;
            let mut state = crate::build_init_state();
            // Allocated once; the worker is not the audio thread but a
            // per-frame allocation is still pointless.
            let mut scratch: Vec<f32> = Vec::new();
            loop {
                let job = match job_rx.recv_timeout(LINGER) {
                    Ok(job) => job,
                    // Nothing for a whole minute: give the model back.
                    Err(RecvTimeoutError::Timeout) => {
                        engine = None;
                        continue;
                    }
                    Err(RecvTimeoutError::Disconnected) => return,
                };
                let ensure = |engine: &mut Option<Engine>| {
                    if engine.is_none() && !worker_failed.load(Ordering::Relaxed) {
                        *engine = build();
                        if engine.is_none() {
                            worker_failed.store(true, Ordering::Relaxed);
                        }
                    }
                };
                match job {
                    Job::Prime => ensure(&mut engine),
                    Job::Reset => {
                        state = crate::build_init_state();
                        worker_failed.store(false, Ordering::Relaxed);
                    }
                    Job::Run(mut frame) => {
                        ensure(&mut engine);
                        // On failure the frame goes back unchanged, which the
                        // caller treats exactly like a late result: dry audio,
                        // time-aligned, rings still advancing.
                        if let Some(engine) = engine.as_mut() {
                            if scratch.len() != frame.len() {
                                scratch = vec![0.0_f32; frame.len()];
                            }
                            if engine.run(&frame, &mut state, &mut scratch) {
                                frame.copy_from_slice(&scratch);
                            }
                        }
                        // `done` holds POOL slots and never more than the
                        // caller's depth is in play, so this cannot block;
                        // drop the frame rather than the worker if that ever
                        // stops being true.
                        let _ = done_tx.try_send(frame);
                    }
                }
            }
        });

        Self {
            worker: Some(worker),
            jobs: Some(jobs),
            done,
            pool: (0..POOL)
                .map(|_| vec![0.0_f32; model_const::FREQ_BINS * 2].into_boxed_slice())
                .collect(),
            outstanding: 0,
            depth: 1,
            failed,
            failed_since: None,
        }
    }

    /// Post a job, never blocking. False when the queue is full or the worker
    /// has already gone.
    fn send(&self, job: Job) -> bool {
        self.jobs
            .as_ref()
            .is_some_and(|jobs| jobs.try_send(job).is_ok())
    }

    /// How many frames the worker may hold at once.
    ///
    /// The caller sets this to the hops one of its callbacks carries, so a
    /// frame submitted in one callback comes back in the next. Clamped to
    /// [`MAX_DEPTH`], which the pool is sized for.
    pub fn set_depth(&mut self, hops_per_callback: usize) {
        self.depth = hops_per_callback.clamp(1, MAX_DEPTH);
    }

    /// Ask the worker to compile now. Called from `activate()`, before any
    /// audio, so the build overlaps the graph starting rather than speech.
    pub fn prime(&mut self) {
        self.send(Job::Prime);
    }

    /// Hand one analysis frame to the worker.
    ///
    /// Returns whether it was accepted. The caller pairs answers with hops by
    /// position, so it must not record a hop the worker never received — a
    /// refused submit that still got recorded shifts every later pairing and
    /// nothing matches again.
    ///
    /// Non-blocking: a full queue, an empty pool or a recent build failure
    /// simply skips this frame, which the caller hears as one raw hop.
    pub fn submit(&mut self, spectrum: &[f32]) -> bool {
        if self.expired_failure() || self.outstanding >= self.depth {
            return false;
        }
        let Some(mut frame) = self.pool.pop() else {
            return false;
        };
        frame.copy_from_slice(spectrum);
        if self.send(Job::Run(frame)) {
            self.outstanding += 1;
            return true;
        }
        false
    }

    /// The enhanced spectrum for an earlier frame, if the worker is done with
    /// it. Non-blocking. `None` means the caller should emit its delayed dry
    /// spectrum for this hop.
    ///
    /// Call [`recycle`](Self::recycle) with the frame once it has been read.
    pub fn take(&mut self) -> Option<Frame> {
        match self.done.try_recv() {
            Ok(frame) if self.outstanding > 0 => {
                self.outstanding -= 1;
                Some(frame)
            }
            // A frame nobody is waiting for: it was in the worker's hands when
            // the node reset, so it belongs to the previous stream.
            Ok(frame) => {
                self.pool.push(frame);
                None
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                // The worker is gone. Without this the plugin would emit dry
                // audio forever with `is_failed()` still reporting false.
                self.failed.store(true, Ordering::Relaxed);
                None
            }
        }
    }

    /// Return a collected buffer to the pool.
    pub fn recycle(&mut self, frame: Frame) {
        self.pool.push(frame);
    }

    /// True while the engine is known unbuildable. Clears itself after
    /// [`RETRY_AFTER`] so one bad moment does not last the session.
    #[must_use]
    pub fn is_failed(&self) -> bool {
        self.failed.load(Ordering::Relaxed)
    }

    /// Whether to skip this hop because the build failed recently. Lets the
    /// latch go once it is older than [`RETRY_AFTER`].
    fn expired_failure(&mut self) -> bool {
        if !self.failed.load(Ordering::Relaxed) {
            self.failed_since = None;
            return false;
        }
        match self.failed_since {
            None => {
                self.failed_since = Some(Instant::now());
                true
            }
            Some(since) if since.elapsed() >= RETRY_AFTER => {
                self.failed.store(false, Ordering::Relaxed);
                self.failed_since = None;
                false
            }
            Some(_) => true,
        }
    }

    /// The node stopped: drop the engine and the recurrent state so an idle
    /// chain holds neither.
    pub fn reset(&mut self) {
        self.send(Job::Reset);
        self.failed_since = None;
        // Whatever the worker already finished belongs to the stream that just
        // ended; handing it to the next one would start it with stale audio.
        while let Ok(frame) = self.done.try_recv() {
            self.pool.push(frame);
        }
        self.outstanding = 0;
    }
}

impl Default for Inference {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Inference {
    fn drop(&mut self) {
        // Close the queue first, so the worker's `recv_timeout` returns at
        // once instead of waiting out the linger. A worker already inside
        // `build()` finishes it — bounded by the compile, and this runs on
        // the host's teardown path, never on the data loop.
        self.jobs = None;
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One callback at 1920/48000. Every audio-thread call must be orders of
    /// magnitude below this.
    const DEADLINE: Duration = Duration::from_millis(40);

    fn spectrum(seed: f32) -> Vec<f32> {
        (0..model_const::FREQ_BINS * 2)
            .map(|i| ((i as f32 + seed) * 0.017).sin() * 0.05)
            .collect()
    }

    /// One analysis hop of wall clock, so the worker is fed at the rate real
    /// audio would feed it rather than flat out.
    fn hop_period() -> Duration {
        Duration::from_secs_f64(model_const::HOP_SIZE as f64 / model_const::SAMPLE_RATE as f64)
    }

    /// Drives `hops` hops at the real rate. Returns (worst audio-side call,
    /// how many hops came back enhanced) counted after `warmup` hops, which
    /// the engine build occupies.
    fn drive(inference: &mut Inference, warmup: usize, hops: usize) -> (Duration, usize) {
        let spec = spectrum(0.0);
        let mut worst = Duration::ZERO;
        let mut enhanced = 0;
        for hop in 0..warmup + hops {
            let start = Instant::now();
            let got = inference.take();
            inference.submit(&spec);
            let elapsed = start.elapsed();
            if let Some(frame) = got {
                if hop >= warmup {
                    enhanced += 1;
                }
                inference.recycle(frame);
            }
            if hop >= warmup {
                worst = worst.max(elapsed);
            }
            if let Some(idle) = hop_period().checked_sub(elapsed) {
                thread::sleep(idle);
            }
        }
        (worst, enhanced)
    }

    /// The handoff must stay exactly as deep as the caller asked, through the
    /// engine build and through a reset. Nothing bounded it before: the build
    /// parked four frames, the queue never drained, and every enhanced frame
    /// was afterwards blended against a noisy reference three hops newer than
    /// itself. Measured lag was 3.00 hops while the code advertised 1.
    #[test]
    fn the_handoff_never_runs_deeper_than_asked() {
        /// Four hops, the depth a 1920-sample callback needs at 48 kHz.
        const DEPTH: usize = 4;
        let mut inference = Inference::new();
        inference.set_depth(DEPTH);
        let spec = spectrum(0.0);
        let mut worst = 0_usize;

        let hop = |inference: &mut Inference| {
            let start = Instant::now();
            if let Some(frame) = inference.take() {
                inference.recycle(frame);
            }
            inference.submit(&spec);
            if let Some(idle) = hop_period().checked_sub(start.elapsed()) {
                thread::sleep(idle);
            }
        };

        // Across the build, which is where the queue used to fill up.
        for _ in 0..300 {
            hop(&mut inference);
            worst = worst.max(inference.outstanding);
        }
        assert!(
            worst <= DEPTH,
            "{worst} frames in flight during the build, cap is {DEPTH}"
        );

        // And across a reset, which used to leave the previous stream's
        // finished frames waiting for the next one.
        inference.reset();
        assert_eq!(inference.outstanding, 0, "reset must clear the handoff");
        assert_eq!(
            inference.pool.len(),
            POOL,
            "reset must return every buffer to the pool"
        );
        for _ in 0..300 {
            hop(&mut inference);
            worst = worst.max(inference.outstanding);
        }
        assert!(worst <= DEPTH, "{worst} frames in flight after reset");
    }

    #[test]
    fn the_audio_side_calls_are_far_below_one_callback() {
        let mut inference = Inference::new();
        // The heaviest model takes 460 ms to compile; 100 hops is a second.
        let (worst, enhanced) = drive(&mut inference, 100, 200);

        eprintln!("worst audio-side call={worst:?} enhanced={enhanced}/200");
        assert!(worst < DEADLINE / 10, "audio-side call cost {worst:?}");
        assert!(
            enhanced > 190,
            "only {enhanced} of 200 paced hops came back enhanced"
        );
    }

    #[test]
    fn the_worker_actually_enhances_the_frames_it_returns() {
        let mut inference = Inference::new();
        let spec = spectrum(0.0);
        let mut returned = 0;
        let mut differing = 0;

        for hop in 0..300 {
            let start = Instant::now();
            let got = inference.take();
            inference.submit(&spec);
            if let Some(frame) = got {
                // The first hops are the engine build; only judge after it.
                if hop >= 100 {
                    returned += 1;
                    if frame
                        .iter()
                        .zip(spec.iter())
                        .any(|(a, b)| (a - b).abs() > 1e-6)
                    {
                        differing += 1;
                    }
                }
                inference.recycle(frame);
            }
            if let Some(idle) = hop_period().checked_sub(start.elapsed()) {
                thread::sleep(idle);
            }
        }

        assert!(returned > 150, "only {returned} frames came back");
        assert!(
            differing > returned / 2,
            "the worker handed back the input unchanged {} of {returned} times",
            returned - differing
        );
    }

    #[test]
    fn resetting_frees_the_engine_and_a_later_submit_rebuilds() {
        let mut inference = Inference::new();
        drive(&mut inference, 100, 20);

        let start = Instant::now();
        inference.reset();
        let reset_cost = start.elapsed();
        assert!(reset_cost < DEADLINE / 10, "reset cost {reset_cost:?}");

        let (_, enhanced) = drive(&mut inference, 100, 100);
        assert!(enhanced > 0, "a reset worker must rebuild and answer again");
    }
}
