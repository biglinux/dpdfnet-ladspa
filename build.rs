#![allow(clippy::doc_markdown)]
//! Build-time model selection for the DPDFNet LADSPA crate.
//!
//! The LADSPA host loads exactly one model per shared object. We
//! produce one .so per upstream DPDFNet model by re-running this
//! crate's build with the `DPDFNET_MODEL` env var set to the
//! upstream registry name (`baseline`, `dpdfnet2`, `dpdfnet4`,
//! `dpdfnet8`, `dpdfnet2_48khz_hr`, `dpdfnet8_48khz_hr`). The script
//! reads the matching `model/<name>/{model.xml,model.bin,init_state.bin}`
//! files and emits `$OUT_DIR/model_const.rs`, which `src/lib.rs`
//! includes via `include!()`.
//!
//! Per-model facts that vary (sample rate, STFT geometry, recurrent
//! state size, LADSPA label / unique_id) all materialise as `pub
//! const`s in that file, so the runtime path stays a thin pointer
//! chase with no dispatch overhead.

use std::env;
use std::fs;
use std::path::PathBuf;

/// All models documented in `ceva-ip/DPDFNet@main:package/src/dpdfnet/models.py`.
/// Entries: `(registry_name, sample_rate_hz, min_block_ms, description)`.
///
/// `min_block_ms` is the smallest PipeWire block at which the plugin keeps
/// every callback inside its deadline, measured by
/// `tests/callback_deadline.rs`.
///
/// It is 10 ms for every model now that inference runs on a worker thread:
/// the callback only does the FFT, the overlap-add and two buffer copies, so
/// its cost no longer depends on the model. Before that change the heaviest
/// model spent 69.9 ms in a single callback against a 40 ms budget; it now
/// spends 0.6 ms against 80 ms and 0.13 ms against 10 ms.
///
/// What still varies per model is whether the worker keeps up — 3.3 ms of
/// inference per 10 ms hop for DPDFNet-8, a third of a core. When it falls
/// behind the caller emits time-aligned dry audio for that hop, which is a
/// quality loss and not an xrun. `engine.rs` measures that ratio.
const REGISTRY: &[(&str, usize, Option<u32>, &str)] = &[
    (
        "baseline",
        16_000,
        Some(10),
        "DPDFNet 16 kHz baseline (fastest, lowest compute)",
    ),
    (
        "dpdfnet2",
        16_000,
        Some(10),
        "DPDFNet-2 16 kHz (balanced quality/speed)",
    ),
    (
        "dpdfnet4",
        16_000,
        Some(10),
        "DPDFNet-4 16 kHz (higher quality)",
    ),
    (
        "dpdfnet8",
        16_000,
        Some(10),
        "DPDFNet-8 16 kHz (highest quality 16 kHz, offline only)",
    ),
    (
        "dpdfnet2_48khz_hr",
        48_000,
        Some(10),
        "DPDFNet-2 48 kHz hi-res (full-band, balanced)",
    ),
    (
        "dpdfnet8_48khz_hr",
        48_000,
        Some(10),
        "DPDFNet-8 48 kHz hi-res (full-band, highest quality, offline only)",
    ),
];

fn fnv1a64(s: &str) -> u64 {
    // 64-bit FNV-1a. Used to derive a stable LADSPA unique_id from the
    // model name. Range chosen to live above the IDs we already use
    // for GTCRN / DFN3 plugins (those are < 6000) without colliding.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in s.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    // Mask into a band that LADSPA hosts treat as private (>= 1<<31)
    // and stays well under the 32-bit truncation some hosts apply.
    (hash & 0x7fff_ffff) | 0x4000_0000
}

fn main() {
    println!("cargo:rerun-if-env-changed=DPDFNET_MODEL");

    let model_name = env::var("DPDFNET_MODEL")
        .expect("DPDFNET_MODEL env var must be set (e.g. baseline, dpdfnet2_48khz_hr)");

    let entry = REGISTRY
        .iter()
        .find(|(name, _, _, _)| *name == model_name)
        .unwrap_or_else(|| {
            let known: Vec<&str> = REGISTRY.iter().map(|(n, _, _, _)| *n).collect();
            panic!(
                "unknown DPDFNet model `{model_name}`; valid choices: {}",
                known.join(", ")
            );
        });
    let (name, sample_rate, min_block_ms, description) = (entry.0, entry.1, entry.2, entry.3);

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR unset");
    let model_dir = PathBuf::from(&manifest_dir).join("model").join(name);
    let xml_path = model_dir.join("model.xml");
    let bin_path = model_dir.join("model.bin");
    let state_path = model_dir.join("init_state.bin");

    for path in [&xml_path, &bin_path, &state_path] {
        assert!(
            path.exists(),
            "missing {} — run scripts/prepare_model.py --model {name} first",
            path.display()
        );
        println!("cargo:rerun-if-changed={}", path.display());
    }

    // STFT geometry follows directly from sample_rate at frame_ms=20.0.
    let win_len = sample_rate / 50;
    let hop_size = win_len / 2;
    let freq_bins = win_len / 2 + 1;

    // Recurrent state size = init_state.bin length / 4 (f32 little-endian).
    let state_bytes = fs::metadata(&state_path)
        .expect("init_state.bin metadata read failed")
        .len();
    assert!(
        state_bytes % 4 == 0,
        "init_state.bin size {state_bytes} is not a multiple of 4"
    );
    let state_size = state_bytes as usize / 4;

    let unique_id = fnv1a64(name);
    let label = format!("{name}_mono");
    let display = format!("DPDFNet {name} mono speech enhancement — {description}");

    let xml_path_str = xml_path.to_string_lossy().replace('\\', "\\\\");
    let bin_path_str = bin_path.to_string_lossy().replace('\\', "\\\\");
    let state_path_str = state_path.to_string_lossy().replace('\\', "\\\\");

    let generated = format!(
        r#"// AUTO-GENERATED by build.rs. Do not edit.

pub const MODEL_NAME: &str = "{name}";
pub const SAMPLE_RATE: usize = {sample_rate};
pub const WIN_LEN: usize = {win_len};
pub const HOP_SIZE: usize = {hop_size};
pub const MIN_BLOCK_MS: Option<u32> = {min_block_ms:?};
pub const FREQ_BINS: usize = {freq_bins};
pub const STATE_SIZE: usize = {state_size};
pub const LADSPA_LABEL: &str = "{label}";
pub const LADSPA_NAME: &str = "{display}";
pub const LADSPA_UNIQUE_ID: u64 = {unique_id};

pub static IR_XML: &[u8] = include_bytes!(r"{xml_path_str}");
pub static IR_BIN: &[u8] = include_bytes!(r"{bin_path_str}");
pub static INIT_STATE: &[u8] = include_bytes!(r"{state_path_str}");
"#
    );

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR unset");
    let out_path = PathBuf::from(out_dir).join("model_const.rs");
    fs::write(&out_path, generated).expect("failed to write model_const.rs");
}
