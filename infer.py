"""
infer.py — Single-file inference for SEMamba++.

Usage:
    python infer.py \
        --input_wav  /path/to/degraded.wav \
        --output_wav /path/to/restored.wav \
        --checkpoint /path/to/semambapp.pth \
        --config     config.yaml
"""

import argparse
import logging
import os
import time
import torch
import librosa
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

from model.stfts import mag_phase_stft, mag_phase_istft
from model.semambapp import SEMambapp
from utils import load_config


def infer(args):
    log = logging.getLogger("SEMamba++")

    # ---- device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # ---- config ----
    log.info(f"Loading config from: {args.config}")
    cfg = load_config(args.config)
    n_fft = cfg["stft_cfg"]["n_fft"]
    hop_size = cfg["stft_cfg"]["hop_size"]
    win_size = cfg["stft_cfg"]["win_size"]
    compress_factor = cfg["model_cfg"]["compress_factor"]
    sr = cfg["stft_cfg"]["sampling_rate"]
    log.info(f"STFT params — n_fft: {n_fft}, hop: {hop_size}, win: {win_size}, "
             f"compress: {compress_factor}, sr: {sr}")

    # ---- model ----
    log.info("Initializing SEMamba++ model...")
    model = SEMambapp(cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {num_params:,}")

    log.info(f"Loading checkpoint from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["generator"])
    model.eval()
    log.info("Model loaded and set to eval mode.")

    # ---- load audio ----
    log.info(f"Loading audio: {args.input_wav}")
    noisy_wav, orig_sr = librosa.load(args.input_wav, sr=sr)
    duration = len(noisy_wav) / sr
    noisy_wav = torch.FloatTensor(noisy_wav).to(device)
    log.info(f"Audio loaded — duration: {duration:.2f}s, samples: {len(noisy_wav)}")

    # ---- RMS normalization (same as training) ----
    norm_factor = torch.sqrt(
        torch.tensor(len(noisy_wav), dtype=torch.float32, device=device)
        / torch.sum(noisy_wav ** 2.0).clamp(min=1e-8)
    )
    noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)  # [1, T]

    # ---- STFT → model → iSTFT ----
    log.info("Inference started...")
    t0 = time.time()
    with torch.no_grad():
        noisy_mag, noisy_pha, _ = mag_phase_stft(
            noisy_wav, n_fft, hop_size, win_size, compress_factor
        )
        log.info(f"STFT complete — mag shape: {list(noisy_mag.shape)}")

        mag_g, pha_g, com_g = model(noisy_mag, noisy_pha)
        log.info("Model forward pass complete.")

        audio_g = mag_phase_istft(
            mag_g, pha_g, n_fft, hop_size, win_size, compress_factor
        )
        log.info("iSTFT complete.")

    elapsed = time.time() - t0
    log.info(f"Inference finished in {elapsed:.3f}s (RTF: {elapsed / duration:.3f})")

    # ---- undo normalization ----
    audio_g = audio_g / norm_factor

    # ---- save ----
    os.makedirs(os.path.dirname(args.output_wav) or ".", exist_ok=True)
    sf.write(
        args.output_wav,
        audio_g.squeeze().cpu().numpy(),
        sr,
        subtype="PCM_16",
    )
    log.info(f"Restored audio saved to: {args.output_wav}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEMamba++ single-file inference")
    parser.add_argument("--input_wav", required=True, help="Path to degraded .wav")
    parser.add_argument("--output_wav", default="restored.wav", help="Output path")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    infer(args)
