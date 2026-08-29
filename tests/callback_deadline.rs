//! Measures how long a single `run()` callback takes, the way a `PipeWire`
//! data-loop would call it.
//!
//! The engine used to be compiled inside the first callback: 170 ms for the
//! 16 kHz baseline, 460 ms for `dpdfnet8_48khz_hr`, against budgets of 40 ms
//! and below. Every stream start was an audible cut. This drives the real
//! plugin through the LADSPA trait and fails if any callback — the first one
//! included — runs past its deadline.

use std::cell::RefCell;
use std::time::{Duration, Instant};

use dpdfnet_ladspa::{get_ladspa_descriptor, MIN_BLOCK_MS, MODEL_SAMPLE_RATE};
use ladspa::{Data, PortConnection, PortData};

/// Blocks a `PipeWire` graph realistically negotiates for this chain.
const QUANTUM_MS: [usize; 4] = [10, 20, 40, 80];
/// Audio driven per configuration.
const SECONDS: f32 = 3.0;

/// Speech-like input, so the model has real work rather than silence.
fn fill_input(buffer: &mut [Data], rate: usize, quantum: usize, block: usize) {
    for (i, sample) in buffer.iter_mut().enumerate() {
        let n = block * quantum + i;
        let t = n as f32 / rate as f32;
        let hiss = ((n as f32 * 12.9898).sin() * 43758.547).fract() - 0.5;
        *sample = 0.25 * (t * 130.0 * std::f32::consts::TAU).sin()
            + 0.12 * (t * 390.0 * std::f32::consts::TAU).sin()
            + 0.04 * hiss;
    }
}

/// Drives one configuration. Returns (first, worst, overruns, callbacks).
fn drive(quantum: usize) -> (Duration, Duration, usize, usize) {
    let descriptor = get_ladspa_descriptor(0).expect("descriptor");
    let rate = MODEL_SAMPLE_RATE;
    let ports = descriptor.ports.clone();
    let mut plugin = (descriptor.new)(&descriptor, rate as u64);
    plugin.activate();

    // Unlimited suppression: the heaviest path through the plugin.
    let controls: [Data; 1] = [100.0];
    let mut input = vec![0.0; quantum];
    let mut output = vec![0.0; quantum];

    let deadline = Duration::from_secs_f32(quantum as f32 / rate as f32);
    let blocks = (SECONDS * rate as f32 / quantum as f32) as usize;
    let mut first = Duration::ZERO;
    let mut worst = Duration::ZERO;
    let mut over = 0;

    for block in 0..blocks {
        fill_input(&mut input, rate, quantum, block);

        let elapsed = {
            // The output slice is handed out once, so it moves through an
            // `Option` rather than being captured by a closure.
            let mut output_slot = Some(&mut output[..]);
            let mut connections: Vec<PortConnection> = Vec::with_capacity(ports.len());
            for (i, port) in ports.iter().enumerate() {
                let data = match i {
                    0 => PortData::AudioInput(&input),
                    1 => PortData::AudioOutput(RefCell::new(
                        output_slot.take().expect("one output port"),
                    )),
                    _ => PortData::ControlInput(&controls[i - 2]),
                };
                connections.push(PortConnection { port: *port, data });
            }
            let refs: Vec<&PortConnection> = connections.iter().collect();
            let start = Instant::now();
            plugin.run(quantum, &refs);
            start.elapsed()
        };

        if block == 0 {
            first = elapsed;
        }
        if elapsed > deadline {
            over += 1;
        }
        worst = worst.max(elapsed);

        // Pace like a real data-loop, so the worker thread gets wall clock
        // to deliver the engine and the measurement is not a flat-out run.
        if let Some(idle) = deadline.checked_sub(elapsed) {
            std::thread::sleep(idle);
        }
    }

    plugin.deactivate();
    (first, worst, over, blocks)
}

#[test]
fn the_first_callback_never_stalls_and_the_shipped_block_never_overruns() {
    for ms in QUANTUM_MS {
        let quantum = MODEL_SAMPLE_RATE * ms / 1000;
        let deadline = Duration::from_millis(ms as u64);
        let (first, worst, over, blocks) = drive(quantum);
        eprintln!(
            "{ms} ms (quantum {quantum}): first={first:?} worst={worst:?} over={over}/{blocks}"
        );

        // The whole point of building the engine off the audio thread: no
        // stream start costs a callback, whatever the block size.
        assert!(
            first < deadline,
            "the first callback took {first:?} of a {deadline:?} budget at {ms} ms"
        );

        // Smaller blocks are reported for comparison. The promise starts at
        // the model's measured minimum: below it the heavier models overrun,
        // which is exactly why the registry records the number.
        // One overrun in a hundred is a scheduler hiccup on a shared build
        // machine, not a DSP regression. A budget rather than zero keeps the
        // gate meaningful instead of flaky — a real regression moves the
        // whole distribution, not one sample.
        if ms as u32 >= MIN_BLOCK_MS {
            assert!(
                over * 100 <= blocks,
                "{over} of {blocks} callbacks exceeded {deadline:?} at {ms} ms; worst {worst:?}"
            );
        }
    }
}
