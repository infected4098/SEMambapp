# :speech_balloon: SEMamba++ (Interspeech 2026 · Long Paper Track)

Official code repository for SEMamba++. [Demo](https://sites.google.com/view/semambapp) [Paper (arXiv)](https://arxiv.org/abs/2603.11669) [Hugging Face](https://huggingface.co/yongjoonlee/semambapp)

SEMamba++ is a general speech restoration (GSR) framework that leverages global, local, and periodic spectral patterns via a Mamba-based architecture. It handles a range of degradation conditions including noise, reverberation, and clipping.

---

## Prerequisites

Create a conda environment with Python>=3.10 (CUDA>=13.0 required, otherwise you should install through [SEMamba](https://github.com/RoyChao19477/SEMamba))

```bash
conda create -n semambapp python==3.10
conda activate semambapp
```

Install all required dependencies:

```bash
git clone https://github.com/infected4098/SEMambapp.git
cd SEMambapp
pip install -r requirements.txt
```

For Mamba, use the below command to install:

```bash
MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall git+https://github.com/state-spaces/mamba.git --no-build-isolation
```

Or alternatively, you should follow the installation guide from [SEMamba](https://github.com/RoyChao19477/SEMamba), which resolves CUDA-specific build issues.

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


## Pretrained weights

Pretrained weights are released on [HuggingFace](https://huggingface.co/yongjoonlee/semambapp/resolve/main/semambapp.pth).

The released model was trained on VCTK and LibriTTS (~500 hours of speech combined).

---

## Inference
```python
python infer.py \
    --input_wav degraded.wav \
    --output_wav restored.wav \
    --checkpoint semambapp.pth \
    --config config.yaml
```

---


## References

- [SEMamba](https://github.com/RoyChao19477/SEMamba) 
- [BigVGAN](https://github.com/NVIDIA/BigVGAN) 
- [MPSENet](https://github.com/yxlu-0102/MP-SENet) 

---

## :notebook: Citation

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
