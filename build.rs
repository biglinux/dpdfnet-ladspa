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

/// All models documented in `ceva-ip/DPDFNet@main:package/src/dpdfnet/models.py`,
/// plus the split-band variants of the 16 kHz networks.
///
/// Entries: `(registry_name, model_dir, model_rate_hz, stft_rate_hz, description)`.
///
/// `model_dir` is where the IR files live under `model/`; it differs from
/// `registry_name` only for the split-band variants, which reuse the base
/// 16 kHz model's IR unchanged. `model_rate` is the rate the network was
/// trained at — it fixes the tensor geometry and the magnitude the model
/// expects. `stft_rate` is the rate the wrapper runs its STFT at, which is also
/// the host rate the plugin accepts.
///
/// A split-band variant pairs a 48 kHz `stft_rate` with a 16 kHz `model_rate`:
/// the STFT runs at 48 kHz (nfft 960), the network is fed the first 161 bins
/// (0–8 kHz — the exact frequencies it was trained on, once scaled to 16 kHz
/// magnitudes), and the wrapper reconstructs the band above 8 kHz. The result
/// is full-band output with no resampler, instead of the telephone-band output
/// the pure 16 kHz plugin gives inside a 48 kHz graph.
///
/// There is deliberately no per-model block size here. Inference runs on a
/// worker thread, so the audio callback does the FFT, the overlap-add and two
/// buffer copies and nothing else — its cost no longer depends on the model.
/// `tests/callback_deadline.rs` holds every model to every block from 10 ms up.
const REGISTRY: &[(&str, &str, usize, usize, &str)] = &[
    (
        "baseline",
        "baseline",
        16_000,
        16_000,
        "DPDFNet 16 kHz baseline (fastest, lowest compute)",
    ),
    (
        "dpdfnet2",
        "dpdfnet2",
        16_000,
        16_000,
        "DPDFNet-2 16 kHz (balanced quality/speed)",
    ),
    (
        "dpdfnet4",
        "dpdfnet4",
        16_000,
        16_000,
        "DPDFNet-4 16 kHz (higher quality)",
    ),
    (
        "dpdfnet8",
        "dpdfnet8",
        16_000,
        16_000,
        "DPDFNet-8 16 kHz (highest quality 16 kHz, offline only)",
    ),
    (
        "dpdfnet2_48khz_hr",
        "dpdfnet2_48khz_hr",
        48_000,
        48_000,
        "DPDFNet-2 48 kHz hi-res (full-band, balanced)",
    ),
    (
        "dpdfnet8_48khz_hr",
        "dpdfnet8_48khz_hr",
        48_000,
        48_000,
        "DPDFNet-8 48 kHz hi-res (full-band, highest quality, offline only)",
    ),
    (
        "baseline_sb",
        "baseline",
        16_000,
        48_000,
        "DPDFNet baseline split-band (full-band, lowest compute)",
    ),
    (
        "dpdfnet2_sb",
        "dpdfnet2",
        16_000,
        48_000,
        "DPDFNet-2 split-band (full-band, balanced, no resampler)",
    ),
    (
        "dpdfnet4_sb",
        "dpdfnet4",
        16_000,
        48_000,
        "DPDFNet-4 split-band (full-band, higher quality)",
    ),
    (
        "dpdfnet8_sb",
        "dpdfnet8",
        16_000,
        48_000,
        "DPDFNet-8 split-band (full-band, highest quality, offline only)",
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
        .find(|(name, ..)| *name == model_name)
        .unwrap_or_else(|| {
            let known: Vec<&str> = REGISTRY.iter().map(|(n, ..)| *n).collect();
            panic!(
                "unknown DPDFNet model `{model_name}`; valid choices: {}",
                known.join(", ")
            );
        });
    let (name, model_subdir, model_rate, stft_rate, description) =
        (entry.0, entry.1, entry.2, entry.3, entry.4);

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR unset");
    let model_dir = PathBuf::from(&manifest_dir)
        .join("model")
        .join(model_subdir);
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

    // Two geometries. The MODEL band is fixed by the rate the network was
    // trained at (frame_ms = 20.0): its win/hop/bins set the ONNX tensor shape
    // and never change for a given IR. The STFT the wrapper actually runs is at
    // `stft_rate` — equal to the model rate for a normal model, 48 kHz for a
    // split-band variant. When they differ, the audio callback runs the bigger
    // FFT, feeds the model the low `freq_bins`, and reconstructs the rest.
    let win_len = model_rate / 50;
    let hop_size = win_len / 2;
    let freq_bins = win_len / 2 + 1;
    let split_band = stft_rate != model_rate;
    let stft_win_len = stft_rate / 50;
    let stft_hop = stft_win_len / 2;
    let stft_freq_bins = stft_win_len / 2 + 1;
    // `SPECTRUM_SCALE` is `win_len / stft_win_len`: the magnitude the model
    // expects vs. what a `stft_win_len` FFT produces for the same tone, since
    // the window sum scales with its length. The low bins are divided by it
    // going in and multiplied coming out. Emitted as a literal `1.0` for a
    // normal build (a `320.0 / 320.0` would trip `clippy::eq_op`) and as the
    // exact division for a split-band build (1/3 has no finite decimal form).
    // Verified against the ONNX: the scaled 48 kHz low bins match native 16 kHz
    // to −39 dB.
    let spectrum_scale = if split_band {
        format!("{win_len}.0 / {stft_win_len}.0")
    } else {
        "1.0".to_string()
    };

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
pub const SAMPLE_RATE: usize = {model_rate};
pub const WIN_LEN: usize = {win_len};
pub const HOP_SIZE: usize = {hop_size};
pub const FREQ_BINS: usize = {freq_bins};
pub const STATE_SIZE: usize = {state_size};

// Host / STFT geometry. Equal to the model geometry for a normal build; for a
// split-band variant the host runs at 48 kHz while FREQ_BINS stays the model's.
pub const HOST_SAMPLE_RATE: usize = {stft_rate};
pub const STFT_WIN_LEN: usize = {stft_win_len};
pub const STFT_HOP: usize = {stft_hop};
pub const STFT_FREQ_BINS: usize = {stft_freq_bins};
pub const SPLIT_BAND: bool = {split_band};
pub const SPECTRUM_SCALE: f32 = {spectrum_scale};
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
