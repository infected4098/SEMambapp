# SEMamba++ (Interspeech 2026 · Long Paper Track)

Official code repository for SEMamba++. [[Demo]](https://sites.google.com/view/semambapp) [[Paper (arXiv)]](https://arxiv.org/abs/2603.11669)

SEMamba++ is a general speech restoration (GSR) framework that leverages global, local, and periodic spectral patterns via a Mamba-based architecture. It handles a range of degradation conditions including noise, reverberation, and clipping.

---

## Prerequisites

Install all required dependencies:

```bash
pip install -r requirements.txt
```

For the Mamba backbone, follow the installation guide from [SEMamba](https://github.com/RoyChao19477/SEMamba), which resolves CUDA-specific build issues.

---

## Datasets

SEMamba++ can be trained on any dataset that provides speech, noise, and room impulse response (RIR) samples. Point each split to the corresponding JSON manifest file:

| Split | File |
|---|---|
| Training speech | `data/train_speech.json` |
| Training noise | `data/train_noise.json` |
| Training RIR | `data/train_rir.json` |
| Validation (clean) | `data/val_clean.json` |
| Validation (degraded) | `data/val_degraded.json` |

### Download sources

- **Speech:** [VCTK](https://datashare.ed.ac.uk/handle/10283/2950), LibriTTS
- **Noise:** [DNS Challenge 2020](https://github.com/microsoft/DNS-Challenge), [WHAM!](http://wham.whisper.ai/)
- **RIR:** [Arni](https://github.com/AaltoAcousticsLab/aalto-datasets), [DNS5](https://github.com/microsoft/DNS-Challenge)

---

## Pretrained weights

Pretrained weights will be released on [HuggingFace](https://huggingface.co/yongjoonlee/semambapp/resolve/main/semambapp.pth).

The released model was trained on VCTK and LibriTTS (~500 hours of speech combined).

---

## References

- [SEMamba](https://github.com/RoyChao19477/SEMamba) — Mamba-based speech enhancement backbone
- [BigVGAN](https://github.com/NVIDIA/BigVGAN) — Neural vocoder (NVIDIA)
- [MPSENet](https://github.com/yxlu-0102/MP-SENet) — Multi-scale phase-aware speech enhancement

---

## Citation

If you find SEMamba++ useful in your work, please cite:

```bibtex
@misc{lee2026semambageneralspeechrestoration,
  title         = {SEMamba++: A General Speech Restoration Framework
                   Leveraging Global, Local, and Periodic Spectral Patterns},
  author        = {Yongjoon Lee and Jung-Woo Choi},
  year          = {2026},
  eprint        = {2603.11669},
  archivePrefix = {arXiv},
  primaryClass  = {eess.AS},
  url           = {https://arxiv.org/abs/2603.11669}
}
```
