//! The plugin must never take the process down.
//!
//! It used to `assert!` on a host sample rate it could not serve, and to
//! `.expect(...)` at twenty-one points while building the `OpenVINO` engine.
//! Both abort across the FFI boundary, which kills `pwloader` and with it the
//! user's microphone. The contract now is: refuse the model, pass the audio.

use std::cell::RefCell;

use dpdfnet_ladspa::{get_ladspa_descriptor, MODEL_SAMPLE_RATE};
use ladspa::{Data, PortConnection, PortData};

const QUANTUM: usize = 512;

fn drive(rate: u64, blocks: usize) -> Vec<Data> {
    let descriptor = get_ladspa_descriptor(0).expect("descriptor");
    let ports = descriptor.ports.clone();
    let mut plugin = (descriptor.new)(&descriptor, rate);
    plugin.activate();

    let controls: [Data; 1] = [100.0];
    let mut input = vec![0.0; QUANTUM];
    let mut output = vec![0.0; QUANTUM];
    let (mut hops_total, mut hops_enhanced) = (0.0, 0.0);
    let mut collected = Vec::with_capacity(blocks * QUANTUM);

    for block in 0..blocks {
        for (i, sample) in input.iter_mut().enumerate() {
            let n = (block * QUANTUM + i) as f32;
            *sample = 0.3 * (n * 0.05).sin();
        }

        {
            // The connections borrow `output`, so they have to go out of
            // scope before the block is collected.
            let mut output_slot = Some(&mut output[..]);
            let mut report_slots = [Some(&mut hops_total), Some(&mut hops_enhanced)];
            let mut connections: Vec<PortConnection> = Vec::with_capacity(ports.len());
            for (i, port) in ports.iter().enumerate() {
                let data = match i {
                    0 => PortData::AudioInput(&input),
                    1 => PortData::AudioOutput(RefCell::new(
                        output_slot.take().expect("one output port"),
                    )),
                    2 => PortData::ControlInput(&controls[0]),
                    // The reporting ports. Each cell is handed out once, the
                    // same way the output slice is.
                    _ => PortData::ControlOutput(RefCell::new(
                        report_slots[i - 3].take().expect("one cell per port"),
                    )),
                };
                connections.push(PortConnection { port: *port, data });
            }
            let refs: Vec<&PortConnection> = connections.iter().collect();
            plugin.run(QUANTUM, &refs);
        }
        collected.extend_from_slice(&output);
    }

    plugin.deactivate();
    collected
}

/// A host rate the model cannot serve is a configuration mistake, not a
/// reason to abort. The node must come up and carry audio.
#[test]
fn an_unsupported_sample_rate_passes_audio_through() {
    let wrong = if MODEL_SAMPLE_RATE == 48_000 {
        16_000
    } else {
        48_000
    };
    let out = drive(wrong as u64, 20);

    assert!(out.iter().all(|s| s.is_finite()), "output must stay finite");

    // Skip the analysis warm-up, then check the signal is actually there.
    let tail = &out[out.len() / 2..];
    let energy: f32 = tail.iter().map(|s| s * s).sum::<f32>() / tail.len() as f32;
    assert!(
        energy > 0.001,
        "passthrough must carry the signal, got mean square {energy}"
    );
}

/// Deactivating hands the engine back to the worker. Running again must
/// rebuild rather than fault on the engine that is no longer there.
#[test]
fn reactivating_after_deactivate_keeps_working() {
    let first = drive(MODEL_SAMPLE_RATE as u64, 10);
    let second = drive(MODEL_SAMPLE_RATE as u64, 10);
    assert!(first.iter().all(|s| s.is_finite()));
    assert!(second.iter().all(|s| s.is_finite()));
}
