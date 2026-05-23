# dpdfnet-ladspa

Packaging of the **DPDFNet** models ([ceva-ip/DPDFNet](https://github.com/ceva-ip/DPDFNet)) as **LADSPA** plugins for Linux. One `.so` per upstream model — users install only the variant they want.

Not a fork of DPDFNet: this repository is just the packaging layer (Rust + PKGBUILD + preparation scripts). Weights come from Hugging Face at build time; inference code uses **OpenVINO Runtime**.

## Packaged models

Each entry becomes an independent Arch/BigLinux package (`pkgname=dpdfnet-ladspa-<variant>`) and a `.so` in `/usr/lib/ladspa/`.

| Variant               | Sample rate | Notes                                      |
|-----------------------|-------------|--------------------------------------------|
| `baseline`            | 16 kHz      | Fastest, 0–8 kHz band                      |
| `dpdfnet2`            | 16 kHz      | Upstream default, balanced quality/CPU     |
| `dpdfnet4`            | 16 kHz      | Deeper DPRNN stack                         |
| `dpdfnet8`            | 16 kHz      | Highest quality in the 16 kHz range        |
| `dpdfnet2_48khz_hr`   | 48 kHz      | Full-band, balanced                        |
| `dpdfnet8_48khz_hr`   | 48 kHz      | Full-band, highest quality                 |

## Layout

```
build.rs                   # model selection via DPDFNET_MODEL, emits model_const.rs
Cargo.toml / Cargo.lock    # cdylib crate, links to libopenvino_c via pkg-config
src/lib.rs                 # LADSPA implementation (STFT, OpenVINO inference, iSTFT)
scripts/prepare_model.py   # ONNX → OpenVINO IR + init_state extraction
pkgbuild/PKGBUILD          # split package, one package_*() per variant
```

Generated artefacts (gitignored): `model/<name>/{model.xml,model.bin,init_state.bin}`, `target-<model>/`, `pkgbuild/{src,pkg}/`, `pkgbuild/*.onnx`.

## How the build works

1. `prepare_model.py` downloads/reads the upstream ONNX, converts it to OpenVINO FP32 IR (`model.xml` + `model.bin`) and extracts the initial recurrent state (`init_state.bin`).
2. `build.rs` reads `model/<DPDFNET_MODEL>/*` and generates Rust constants (sample rate, STFT geometry, `state_size`, LADSPA label, `unique_id`) embedded into the `.so`.
3. `cargo build --release` produces `libdpdfnet_ladspa.so`; the `PKGBUILD` repeats the cycle per variant, isolating `CARGO_TARGET_DIR=target-<model>`, and renames the output to `libdpdfnet_<model>_ladspa.so` at install time.

Result: each `.so` carries a single model, no runtime dispatch and no external weight files.

## Manual build (without makepkg)

```bash
# 1. download upstream ONNX
mkdir -p /tmp/dpdfnet-models
curl -L -o /tmp/dpdfnet-models/dpdfnet2.onnx \
    https://huggingface.co/Ceva-IP/DPDFNet/resolve/main/onnx/dpdfnet2.onnx

# 2. convert to IR
python scripts/prepare_model.py --model dpdfnet2 \
    --src-onnx /tmp/dpdfnet-models/dpdfnet2.onnx

# 3. compile the .so for that model
DPDFNET_MODEL=dpdfnet2 CARGO_TARGET_DIR=target-dpdfnet2 \
    cargo build --release
```

Dependencies: `rustc >= 1.85`, `cargo`, `openvino` (runtime + headers), `python` with `numpy`, `onnx`, `openvino`.

## Build via PKGBUILD (Arch/BigLinux/Manjaro)

```bash
cd pkgbuild
makepkg -si
```

Builds all six split packages. `sha256sums` is set to `SKIP` while the Hugging Face registry does not pin a revision.

> **Warning:** `Cargo.toml` carries a `[patch.crates-io]` pointing at a patched `ladspa` crate in `../gtcrn-ladspa/ladspa/ladspa-patched`. For reproducible builds outside the maintainer's environment, check out `gtcrn-ladspa` next to this repo, or drop the patch and use upstream `ladspa = "0.3"` if it already carries the needed fix.

## Usage

Once any package is installed, the plugin shows up in LADSPA hosts (Ardour, Carla, EasyEffects via wrapper, etc.) under the model's label. For PipeWire filter-chain, reference `libdpdfnet_<model>_ladspa.so` and the label exposed by the plugin.

## License

- Code in this repository: **MIT** (see `Cargo.toml`).
- DPDFNet models and ONNX weights: **Apache-2.0** ([ceva-ip/DPDFNet](https://github.com/ceva-ip/DPDFNet)). The `PKGBUILD` installs a copy of the upstream `LICENSE` in `/usr/share/licenses/<pkg>/` of each split, per Apache-2.0 §4(a).

## Citation

Rika, Sapir, Gus (2025). *DPDFNet: Boosting DeepFilterNet2 via Dual-Path RNN*.
