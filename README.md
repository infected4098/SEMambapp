# SEMambapp
(Submitted to Interspeech 2026) An official code repository for SEMamba++.


This repository provides the official codebase and resources for SEMamba++ as described in our research. This repository is currently anonymous and will remain so until the publication process is complete, after which it will be de-anonymized with full author and project details.
## Prerequisites
1. Install the dependencies.
```
pip install -r requirements.txt
```
2. For Mamba, we recommend installing through [SEMamba](https://github.com/RoyChao19477/SEMamba)'s implementation.

## Datasets

You can try GSR on arbitrary dataset but we list all the dataset sources used in our experiments. 
You can list the filepaths to `data/train_speech.json`, `data/train_noise.json`, `data/train_rir.json`, `data/val_clean.json`, `data/val_degraded.json`. 


## Link to datasets

1. Download [VCTK](https://datashare.ed.ac.uk/handle/10283/2950) for speech.
2. Download [DNS Challenge 2020](https://github.com/microsoft/DNS-Challenge) and [WHAM!](http://wham.whisper.ai/) for noise.
3. Download [Arni](https://github.com/AaltoAcousticsLab/aalto-datasets) and [DNS5](https://github.com/microsoft/DNS-Challenge) for reverberation.


## Notices

Pretrained models will be made publicly available upon completion of the publication process.

## References

SEMamba: [SEMamba](https://github.com/RoyChao19477/SEMamba)
BigVGAN: [BigVGAN](https://github.com/NVIDIA/BigVGAN)
MP-SENet: [MPSENet](https://github.com/yxlu-0102/MP-SENet)

