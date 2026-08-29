# :speech_balloon: SEMamba++ (Interspeech 2026 · Long Paper Track)

Official code repository for SEMamba++. [Demo](https://sites.google.com/view/semambapp) [Paper](https://arxiv.org/abs/2603.11669) [Hugging Face 🤗](https://huggingface.co/yongjoonlee/semambapp)

SEMamba++ is a general speech restoration (GSR) framework that leverages global, local, and periodic spectral patterns via a Mamba-based architecture. It handles a range of degradation conditions including noise, reverberation, and clipping.


---

## News

- **Aug 2026:**
  - Pretrained checkpoints have been updated to reflect a recent bug fix in the codebase.


---

## Prerequisites

Create a conda environment with Python>=3.10 required, otherwise you should install through [SEMamba](https://github.com/RoyChao19477/SEMamba))

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
pip install --no-cache-dir --no-build-isolation --no-deps causal-conv1d==1.7.0 mamba-ssm==2.3.2.post1
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

Pretrained weights are released on [Hugging Face🤗](https://huggingface.co/yongjoonlee/semambapp/resolve/main/semambapp.pth).

The checkpoint used in the paper was trained with code that contained a code bug. We fixed the bug and retrained the model under the same configuration. The released checkpoint (Aug 2026) is the result of that retraining.

<details>
<summary>Code bug</summary>

The channel FFN module for the third (last) resolution was not being utilized for that resolution. 
It was used for the second resolution.

**Wrong version** <br>
(Second resolution) Time Mamba &rarr; Frequency GLP &rarr; Channel FFN1 &rarr; Channel FFN2  <br>

(Third resolution) TIme Mamba &rarr; Frequency GLP 

**Corrected version** <br>
(Second resolution) Time Mamba &rarr; Frequency GLP &rarr; Channel FFN <br>

(Third resolution) TIme Mamba &rarr; Frequency GLP &rarr; Channel FFN

</details>

The tables below report both scores from the paper and retrained model so that the released weights can be compared against
the paper. Note in particular that the released checkpoint scores slightly higher on intrusive metrics (PESQ and LPS) across different datasets but scores slightly lower than the paper on the URGENT 2025 blind test.


### AATC 2025

|Name|SIG|BAK|OVRL|PESQ
|---|---|---|---|---|
|Degraded|2.89|3.02|2.47|1.91|
|SEMamba++(paper)|3.45|4.01|3.18|1.84|
|SEMamba++(retrained)|3.46|4.07|3.20|1.95|

### URGENT 2025 val
|Name|SCOREQ|UTMOS|OVRL|PESQ|LPS|
|---|---|---|---|---|---|
|Degraded|1.20|1.51|1.78|1.26|0.60
|SEMamba++(paper)|2.67|2.82|3.20|1.51|0.61|
|SEMamba++(retrained)|2.67|2.80|3.19|1.60|0.65|

### URGENT 2025 blind test

|Name|SCOREQ|UTMOS|OVRL
|---|---|---|---|
|Degraded|1.25|1.55|1.90|
|SEMamba++(paper)|2.49|2.61|3.13|
|SEMamba++(retrained)|2.35|2.47|3.07|

### DNS 2020 test

|Name|SIG|BAK|OVRL
|---|---|---|---|
|Degraded|3.053|2.509|2.255|
|SEMamba++(paper)|3.487|4.020|3.206|
|SEMamba++(retrained)|3.472|4.086|3.208|
---


## Inference
```bash
python infer.py \
    --input_wav degraded.wav \
    --output_wav restored.wav \
    --checkpoint semambapp.pth \
    --config config.yaml
```



```bash
python infer_folder.py \
    --input_dir  /path/to/degraded_wavs \
    --output_dir /path/to/restored_wavs \
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
