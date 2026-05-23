"""Convert one upstream DPDFNet ONNX → OpenVINO IR + extract init_state.

Usage::

    python prepare_model.py --model <name> [--src-onnx <path>]

`<name>` is one of the entries in `ceva-ip/DPDFNet@main:package/src/dpdfnet/models.py`:
``baseline``, ``dpdfnet2``, ``dpdfnet4``, ``dpdfnet8``, ``dpdfnet2_48khz_hr``,
``dpdfnet8_48khz_hr``. ``--src-onnx`` defaults to
``/tmp/dpdfnet-models/<name>.onnx``; download the .onnx ahead of time
with::

    DPDFNET_MODEL_DIR=/tmp/dpdfnet-models python -c \
        "from dpdfnet.models import download_model; download_model(model='<name>')"

or fetch directly from Hugging Face::

    curl -L -o /tmp/dpdfnet-models/<name>.onnx \
        https://huggingface.co/Ceva-IP/DPDFNet/resolve/main/onnx/<name>.onnx

Outputs (in ``model/<name>/``):

* ``model.xml`` + ``model.bin`` — OpenVINO FP32 IR (Snippets-disabled at
  runtime to dodge the OV 2026.0.0 GRU tokenizer bug).
* ``init_state.bin`` — recurrent state initial values, packed as
  little-endian f32 of length ``state_size`` (zero-padded after
  ``erb_norm_init`` + ``spec_norm_init``).

The Rust ``build.rs`` reads each of these files when invoked with
``DPDFNET_MODEL=<name>`` and embeds them into the resulting .so.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import onnx
import openvino as ov

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
DEFAULT_CALIB_DIR = Path.home() / ".cache/biglinux-noise-reduction-pipewire/calibration/datasets/voicebank_demand/noisy_testset_wav"

# Mirrors `REGISTRY` in build.rs and upstream models.py.
KNOWN_MODELS = {
    "baseline",
    "dpdfnet2",
    "dpdfnet4",
    "dpdfnet8",
    "dpdfnet2_48khz_hr",
    "dpdfnet8_48khz_hr",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--model",
        required=True,
        choices=sorted(KNOWN_MODELS),
        help="upstream DPDFNet registry name",
    )
    p.add_argument(
        "--src-onnx",
        type=Path,
        default=None,
        help="source .onnx path (default /tmp/dpdfnet-models/<model>.onnx)",
    )
    p.add_argument(
        "--quantize",
        choices=("none", "fp16", "int8"),
        default="none",
        help="post-conversion compression: none=FP32 (default), fp16=halve "
             "weights, int8=NNCF PTQ (requires --calib-dir audio)",
    )
    p.add_argument(
        "--calib-dir",
        type=Path,
        default=DEFAULT_CALIB_DIR,
        help="directory of .wav files for INT8 calibration (default: VBD noisy testset)",
    )
    p.add_argument(
        "--calib-files",
        type=int,
        default=64,
        help="how many .wav files to stream through for calibration (default 64)",
    )
    return p.parse_args()


def _stft_frames(wav_path: Path, sr: int, n_fft: int, hop: int) -> np.ndarray:
    """Return [n_frames, freq_bins, 2] real/imag spectrogram resampled to sr."""
    import soundfile as sf
    import scipy.signal as sps

    audio, src_sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if src_sr != sr:
        n_out = int(round(audio.size * sr / src_sr))
        audio = sps.resample(audio, n_out).astype(np.float32)
    win = np.hanning(n_fft).astype(np.float32)
    n_frames = max(0, 1 + (audio.size - n_fft) // hop)
    out = np.zeros((n_frames, n_fft // 2 + 1, 2), dtype=np.float32)
    for f in range(n_frames):
        s = f * hop
        frame = audio[s : s + n_fft] * win
        spec = np.fft.rfft(frame).astype(np.complex64)
        out[f, :, 0] = spec.real
        out[f, :, 1] = spec.imag
    return out


def _calib_iter(
    compiled_fp32: ov.CompiledModel,
    calib_dir: Path,
    n_files: int,
    sr: int,
    n_fft: int,
    hop: int,
    state_size: int,
    init_state: np.ndarray,
) -> Iterator[dict]:
    """Yield {spec, state_in} dicts. State propagates frame-to-frame so the
    distribution NNCF sees matches the runtime distribution; reset to init
    at every file boundary so it never drifts into garbage."""
    files = sorted(calib_dir.glob("*.wav"))[:n_files]
    if not files:
        raise FileNotFoundError(f"no .wav under {calib_dir}")
    state = init_state.copy()
    infer_req = compiled_fp32.create_infer_request()
    for wp in files:
        state[:] = init_state
        frames = _stft_frames(wp, sr, n_fft, hop)
        for f in range(frames.shape[0]):
            spec = frames[f].reshape(1, 1, -1, 2).astype(np.float32)
            yield {"spec": spec, "state_in": state.copy()}
            res = infer_req.infer({"spec": spec, "state_in": state})
            state = list(res.values())[1].astype(np.float32).reshape(state_size)


def _quantize_int8(
    model: ov.Model,
    calib_dir: Path,
    n_files: int,
    sr: int,
    n_fft: int,
    hop: int,
    state_size: int,
    init_state: np.ndarray,
) -> ov.Model:
    import nncf

    # OpenVINO 2026.0.0 has a tokenizer bug that crashes Snippets on this
    # model's GRU stack. Disable globally for the duration of NNCF, which
    # would otherwise spawn its own Core inside statistics collection.
    _orig_compile = ov.Core.compile_model
    def _patched(self, model_, device_name="CPU", config=None, *a, **kw):
        cfg = dict(config or {})
        cfg.setdefault("SNIPPETS_MODE", "DISABLE")
        return _orig_compile(self, model_, device_name, cfg, *a, **kw)
    ov.Core.compile_model = _patched

    print(f"[quantize] streaming {n_files} files from {calib_dir}")
    core = ov.Core()
    compiled_fp32 = core.compile_model(
        model,
        "CPU",
        {"SNIPPETS_MODE": "DISABLE"},
    )
    samples = list(_calib_iter(compiled_fp32, calib_dir, n_files, sr, n_fft, hop, state_size, init_state))
    print(f"[quantize] collected {len(samples)} calibration frames")
    dataset = nncf.Dataset(samples)
    # Skip the spectral-mask subgraph and recurrent GRU stack:
    #   * mask/* — final output Reshape blows up the LPT
    #     `ReshapeTransformation` pass on OV 2026.0.0 because the
    #     dequantize-then-reshape rewrite produces an incompatible
    #     shape inference; keeping mask in FP32 sidesteps it.
    #   * *gru* / *grucell* — INT8 on recurrent ops causes drift +
    #     audible "metallic" artifacts for spectral denoisers.
    return nncf.quantize(
        model,
        dataset,
        preset=nncf.QuantizationPreset.MIXED,
        model_type=nncf.ModelType.TRANSFORMER,
        subset_size=min(len(samples), 1024),
        ignored_scope=nncf.IgnoredScope(
            patterns=[".*mask/.*", ".*gru.*", ".*grucell.*"],
        ),
    )


def main() -> int:
    args = parse_args()
    model_name: str = args.model
    src_onnx: Path = args.src_onnx or Path(f"/tmp/dpdfnet-models/{model_name}.onnx")
    out_dir = MODEL_DIR / model_name
    out_xml = out_dir / "model.xml"
    out_bin = out_dir / "model.bin"
    out_state = out_dir / "init_state.bin"

    if not src_onnx.exists():
        print(
            f"missing {src_onnx}; download with the dpdfnet package or "
            f"from huggingface.co/Ceva-IP/DPDFNet/resolve/main/onnx/{model_name}.onnx",
            file=sys.stderr,
        )
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{model_name}] reading {src_onnx}")
    m = onnx.load(str(src_onnx))
    md = {p.key: p.value for p in m.metadata_props}

    state_size = int(md["state_size"])
    erb_init = np.fromstring(md["erb_norm_init"], dtype=np.float32, sep=",")
    spec_init = np.fromstring(md["spec_norm_init"], dtype=np.float32, sep=",")
    print(
        f"[{model_name}] state_size={state_size} "
        f"erb_norm({len(erb_init)}) + spec_norm({len(spec_init)})"
    )

    init_state = np.zeros(state_size, dtype=np.float32)
    init_state[: erb_init.size] = erb_init
    init_state[erb_init.size : erb_init.size + spec_init.size] = spec_init
    out_state.write_bytes(init_state.tobytes())
    print(f"[{model_name}] wrote {out_state} ({out_state.stat().st_size} bytes)")

    print(f"[{model_name}] converting ONNX → IR")
    model = ov.convert_model(str(src_onnx))

    if args.quantize == "int8":
        n_fft = int(md["n_fft"])
        hop = int(md["hop_length"])
        sr_hz = int(md["sample_rate"])
        model = _quantize_int8(
            model,
            args.calib_dir,
            args.calib_files,
            sr_hz,
            n_fft,
            hop,
            state_size,
            init_state,
        )
        compress_fp16 = False
    else:
        compress_fp16 = args.quantize == "fp16"

    ov.save_model(model, str(out_xml), compress_to_fp16=compress_fp16)
    # ov.save_model writes <stem>.bin alongside the xml; if the upstream
    # filename differs from `model.bin` (e.g. older ov versions emit
    # `<src>.bin`), normalize it here so build.rs can include_bytes!
    # the canonical path.
    expected_bin = out_xml.with_suffix(".bin")
    if expected_bin != out_bin and expected_bin.exists():
        expected_bin.replace(out_bin)
    print(f"[{model_name}] wrote {out_xml} ({out_xml.stat().st_size} bytes)")
    print(f"[{model_name}] wrote {out_bin} ({out_bin.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
