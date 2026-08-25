"""
infer_folder.py — Batch inference for SEMamba++ over all .wav files in a folder.

Loads the model once and reuses it for every file (unlike calling infer.py
per file, which reloads the model each time). Uses the exact same
load_model()/restore_file() code path as infer.py, so results are identical.

Usage:
    python infer_folder.py \
        --input_dir  /path/to/degraded_wavs \
        --output_dir /path/to/restored_wavs \
        --checkpoint /path/to/semambapp.pth \
        --config     config.yaml
"""

import argparse
import glob
import logging
import os
import torch

from infer import load_model, restore_file

log = logging.getLogger("SEMamba++")


def infer_folder(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    model, cfg = load_model(args.config, args.checkpoint, device)

    os.makedirs(args.output_dir, exist_ok=True)
    wav_paths = sorted(
        p for ext in ("*.wav", "*.flac")
        for p in glob.glob(os.path.join(args.input_dir, ext))
    )
    log.info(f"Found {len(wav_paths)} audio files in {args.input_dir}")

    for i, input_wav in enumerate(wav_paths, 1):
        fname = os.path.basename(input_wav)
        output_wav = os.path.join(args.output_dir, fname)
        log.info(f"[{i}/{len(wav_paths)}] {fname}")
        restore_file(model, cfg, device, input_wav, output_wav)

    log.info(f"Done — {len(wav_paths)} files restored to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEMamba++ folder inference")
    parser.add_argument("--input_dir", required=True, help="Folder of degraded .wav files")
    parser.add_argument("--output_dir", required=True, help="Folder to save restored .wav files")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    infer_folder(args)
