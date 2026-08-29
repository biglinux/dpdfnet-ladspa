//! OpenVINO inference engine and its off-audio-thread lifecycle.
//!
//! Building the engine reads the embedded IR, compiles it for the CPU plugin
//! and allocates the bound tensors. Measured on an i5-13400 that costs 170 ms
//! for the 16 kHz baseline and 460 ms for `dpdfnet8_48khz_hr`, against a 40 ms
//! callback budget — so it can never happen inside `run()`.
//!
//! [`EngineLoader`] moves it to a worker thread. The audio thread only asks,
//! polls, and hands the engine back when the host deactivates the node. Every
//! audio-side call is non-blocking; until the engine arrives the caller passes
//! audio through unprocessed, which is a clean start rather than a stall.
//!
//! Nothing here panics. A machine without a usable OpenVINO runtime gets a
//! plugin that passes audio through, not a `pwloader` process that aborts and
//! takes the user's microphone with it.

use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::thread;

use openvino::{CompiledModel, Core, ElementType, InferRequest, RwPropertyKey, Shape, Tensor};

use crate::model_const;

/// Queue depth. Deeper than the single build the audio thread can have in
/// flight, so retiring an engine never displaces a pending build request.
const QUEUE_DEPTH: usize = 4;

/// One compiled model with its bound input tensors.
///
/// Per plugin instance rather than per process. A shared engine would need a
/// mutex on the audio thread, and PipeWire can run more than one data loop —
/// two graphs on different loops would then block each other in real time.
/// The cost of not sharing is one oneDNN scratchpad per instance, 10-15 MB,
/// and in practice each chain has its own `pwloader` process anyway.
pub struct Engine {
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
    pub fn run(&mut self, spec_in: &[f32], state: &mut [f32], spec_out: &mut [f32]) -> bool {
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

/// Compile the embedded IR and bind its input tensors. `None` on any failure.
fn build() -> Option<Engine> {
    let mut core = Core::new().ok()?;

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
    let weights_shape = Shape::new(&[model_const::IR_BIN.len() as i64]).ok()?;
    let mut weights_tensor = Tensor::new(ElementType::U8, &weights_shape).ok()?;
    weights_tensor
        .get_data_mut::<u8>()
        .ok()?
        .copy_from_slice(model_const::IR_BIN);

    let model = core
        .read_model_from_buffer(model_const::IR_XML, Some(&weights_tensor))
        .ok()?;
    let mut compiled = core.compile_model(&model, cpu).ok()?;
    let mut infer = compiled.create_infer_request().ok()?;

    let spec_shape = Shape::new(&[1, 1, model_const::FREQ_BINS as i64, 2]).ok()?;
    let state_shape = Shape::new(&[model_const::STATE_SIZE as i64]).ok()?;
    let spec_tensor = Tensor::new(ElementType::F32, &spec_shape).ok()?;
    let state_tensor = Tensor::new(ElementType::F32, &state_shape).ok()?;
    infer.set_tensor("spec", &spec_tensor).ok()?;
    infer.set_tensor("state_in", &state_tensor).ok()?;

    Some(Engine {
        infer,
        spec_tensor,
        state_tensor,
        _compiled: compiled,
    })
}

enum Request {
    Build,
    /// Dropping a compiled model frees the oneDNN arenas — off the audio
    /// thread, like building one.
    Retire(Box<Engine>),
}

/// Owns the worker thread that builds and drops engines.
///
/// Dropping the loader closes the request channel, which ends the worker.
pub struct EngineLoader {
    requests: SyncSender<Request>,
    ready: Receiver<Box<Engine>>,
    engine: Option<Box<Engine>>,
    /// A build is queued and has not been collected yet.
    in_flight: bool,
    /// A build already came back empty. Asking again every callback would
    /// re-run a 170 ms compile on the worker forever on a machine with no
    /// usable OpenVINO.
    failed: bool,
}

impl EngineLoader {
    #[must_use]
    pub fn new() -> Self {
        let (requests, request_rx) = sync_channel::<Request>(QUEUE_DEPTH);
        let (ready_tx, ready) = sync_channel::<Box<Engine>>(QUEUE_DEPTH);

        thread::spawn(move || {
            for request in request_rx {
                match request {
                    Request::Build => {
                        if let Some(engine) = build() {
                            // A full queue means the audio thread stopped
                            // collecting; drop the engine here, never block.
                            let _ = ready_tx.try_send(Box::new(engine));
                        } else {
                            // Closing the channel is how the audio thread
                            // learns the build failed without waiting on it.
                            return;
                        }
                    }
                    Request::Retire(engine) => drop(engine),
                }
            }
        });

        Self {
            requests,
            ready,
            engine: None,
            in_flight: false,
            failed: false,
        }
    }

    /// The engine, if one is ready. Asks for a build the first time, and every
    /// time the node is reactivated after [`retire`](Self::retire).
    ///
    /// Non-blocking. `None` means the caller should pass audio through.
    pub fn engine(&mut self) -> Option<&mut Engine> {
        if self.in_flight {
            match self.ready.try_recv() {
                Ok(engine) => {
                    self.engine = Some(engine);
                    self.in_flight = false;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.in_flight = false;
                    self.failed = true;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
        if self.engine.is_none() && !self.in_flight && !self.failed {
            self.in_flight = self.requests.try_send(Request::Build).is_ok();
        }
        self.engine.as_deref_mut()
    }

    /// True once a build has come back empty — the runtime is unusable and no
    /// further attempt will be made.
    #[must_use]
    pub fn is_failed(&self) -> bool {
        self.failed
    }

    /// Hand the engine back for the worker to drop, so an idle node stops
    /// holding the compiled model. The next [`engine`](Self::engine) call
    /// starts a fresh build.
    pub fn retire(&mut self) {
        if let Some(engine) = self.engine.take() {
            let _ = self.requests.try_send(Request::Retire(engine));
        }
    }
}

impl Default for EngineLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// One callback at 1920/48000. Every audio-thread call into the loader
    /// must be orders of magnitude below this.
    const DEADLINE: Duration = Duration::from_millis(40);

    fn wait_ready(loader: &mut EngineLoader) -> (bool, Duration, Duration) {
        let start = Instant::now();
        let mut worst = Duration::ZERO;
        loop {
            let poll = Instant::now();
            let ready = loader.engine().is_some();
            worst = worst.max(poll.elapsed());
            if ready || loader.is_failed() {
                return (ready, start.elapsed(), worst);
            }
            assert!(
                start.elapsed() < Duration::from_secs(30),
                "build never ended"
            );
            thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn polling_never_costs_a_callback() {
        let mut loader = EngineLoader::new();
        let (ready, arrival, worst_poll) = wait_ready(&mut loader);
        assert!(ready, "the embedded IR must compile in a test environment");
        eprintln!("arrival={arrival:?} worst_poll={worst_poll:?}");
        assert!(worst_poll < DEADLINE / 10, "poll cost {worst_poll:?}");
    }

    #[test]
    fn retiring_frees_the_engine_and_a_later_call_rebuilds() {
        let mut loader = EngineLoader::new();
        assert!(wait_ready(&mut loader).0);

        let start = Instant::now();
        loader.retire();
        let retire_cost = start.elapsed();
        eprintln!("retire={retire_cost:?}");
        assert!(retire_cost < DEADLINE / 10, "retire cost {retire_cost:?}");

        assert!(wait_ready(&mut loader).0, "a retired loader must rebuild");
    }

    #[test]
    fn one_frame_advances_the_recurrent_state() {
        let mut loader = EngineLoader::new();
        assert!(wait_ready(&mut loader).0);

        let spec_in = vec![0.01_f32; model_const::FREQ_BINS * 2];
        let mut spec_out = vec![0.0_f32; model_const::FREQ_BINS * 2];
        let mut state = crate::build_init_state();
        let before = state.clone();

        let engine = loader.engine().expect("engine ready");
        assert!(engine.run(&spec_in, &mut state, &mut spec_out));
        assert_ne!(state, before, "inference must advance the recurrent state");
    }
}
