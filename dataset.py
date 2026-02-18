import os
import json
import random
import torch
import torch.utils.data
import librosa
import numpy as np
import scipy
from simulate_degradation import *
import yaml
from utils.util import load_config
from scipy.signal import stft
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from tqdm import tqdm
from model.stfts import mag_phase_stft, mag_phase_istft
import re



def list_files_in_directory(directory_path):
    files = []
    for root, dirs, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith('.wav'):   # only add .wav files
                files.append(os.path.join(root, filename))
    return files

def load_json_file(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def extract_identifier(file_path):
    return os.path.basename(file_path)

def get_clean_path_for_noisy(noisy_file_path, clean_path_dict):
    identifier = extract_identifier(noisy_file_path)
    return clean_path_dict.get(identifier, None)

def peak_normalize(audio_tensor, eps=1e-9):
    peak = audio_tensor.abs().max()
    return audio_tensor / (peak + eps)


class GSRDataset(torch.utils.data.Dataset):
    """
    Dataset for loading clean and noisy audio files.

    Args:
        clean_wavs_json (str): Directory containing clean audio files.
        noisy_wavs_json (str): Directory containing noisy audio files.
        audio_index_file (str): File containing audio indexes.
        sampling_rate (int, optional): Sampling rate of the audio files. Defaults to 16000.
        segment_size (int, optional): Size of the audio segments. Defaults to 32000.
        n_fft (int, optional): FFT size. Defaults to 400.
        hop_size (int, optional): Hop size. Defaults to 100.
        win_size (int, optional): Window size. Defaults to 400.
        compress_factor (float, optional): Magnitude compression factor. Defaults to 1.0.
        split (bool, optional): Whether to split the audio into segments. Defaults to True.
        n_cache_reuse (int, optional): Number of times to reuse cached audio. Defaults to 1.
        device (torch.device, optional): Target device. Defaults to None
        pcs (bool, optional): Use PCS in training period. Defaults to False
        mode: "Train or Validation, or Test"
    """
    def __init__(
        self, 
        cfg,
        clean_json, 
        noise_json, 
        rir_json, 
        clean_val_json,
        degraded_val_json,
        n_cache_reuse=1, 
        shuffle=True, 
        device=None, 
        pcs=False,
        seed=None,
        mode="Train",
        selected_degrads = None
    ):

        sampling_rate = cfg["stft_cfg"]["sampling_rate"]
        segment_size = cfg["training_cfg"]["segment_size"]
        n_fft = cfg["stft_cfg"]["n_fft"]
        hop_size = cfg["stft_cfg"]["hop_size"]
        win_size = cfg["stft_cfg"]["win_size"]
        compress_factor = cfg["model_cfg"]["compress_factor"]

        self.clean_wavs_path = load_json_file(clean_json)
        self.noise_wavs_path = load_json_file(noise_json)
        self.rir_wavs_path = load_json_file(rir_json)
        self.clean_val_wavs_path = load_json_file(clean_val_json)
        self.degraded_val_wavs_path = load_json_file(degraded_val_json)

        # Validation file arranging
        self.clean_val_path = sorted(self.clean_val_wavs_path, key=lambda p: int(p.rsplit('_', 1)[1].split('.')[0]))[:100]
        self.degraded_val_path = sorted(self.degraded_val_wavs_path, key=lambda p: int(p.rsplit('_', 1)[1].split('.')[0]))[:100]

        self.clean_test_path = sorted(self.clean_val_wavs_path, key=lambda p: int(p.rsplit('_', 1)[1].split('.')[0]))[100:]
        self.degraded_test_path = sorted(self.degraded_val_wavs_path, key=lambda p: int(p.rsplit('_', 1)[1].split('.')[0]))[100:]              

        if seed:
            random.seed(seed)

        if shuffle:
            random.shuffle(self.clean_wavs_path)
            random.shuffle(self.noise_wavs_path)
            random.shuffle(self.rir_wavs_path)

        #self.clean_path_dict = {extract_identifier(clean_path): clean_path for clean_path in self.clean_wavs_path}

        self.cfg = cfg
        self.seed = seed
        self.sampling_rate = sampling_rate
        self.segment_size = segment_size
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.compress_factor = compress_factor
        self.n_cache_reuse = n_cache_reuse

        self.cached_clean_wav = None
        self.cached_noise_wav = None
        self.cached_rir_wav = None
        self._cache_ref_count = 0
        self.device = device
        self.pcs = pcs
        self.mode = mode
        self.selected_degrads = selected_degrads



    def __getitem__(self, index):
        """
        Get an audio sample by index.

        Args:
            index (int): Index of the audio sample.

        Returns:
            tuple: clean audio, clean magnitude, clean phase, clean complex, noisy magnitude, noisy phase
        """

        if self.mode == "Train":
            clean_path = self.clean_wavs_path[index]
            # noise and rir has different length 
            noise_path = random.choice(list(self.noise_wavs_path))
            rir_path = random.choice(list(self.rir_wavs_path))
            #clean_path = get_clean_path_for_noisy(noisy_path, self.clean_path_dict)
            noise_audio, noise_sr = librosa.load(noise_path, sr=self.sampling_rate)
            clean_audio, clean_sr = librosa.load(clean_path, sr=self.sampling_rate)
            rir_audio, rir_sr = librosa.load(rir_path, sr=self.sampling_rate)

            # Resample if needed
            if noise_sr != self.sampling_rate:
                noise_audio = librosa.resample(noise_audio, noise_sr, self.sampling_rate)
            if clean_sr != self.sampling_rate:
                clean_audio = librosa.resample(clean_audio, clean_sr, self.sampling_rate)
            if rir_sr != self.sampling_rate:
                rir_audio = librosa.resample(rir_audio, rir_sr, self.sampling_rate)






            clean_audio = clean_audio.reshape(1, -1)
            noise_audio = noise_audio.reshape(1, -1)
            rir_audio = rir_audio.reshape(1, -1)


            degrad_cfgs, selected_degrads = random_select_and_order(self.cfg, seed=self.seed)
            if self.selected_degrads is None:

                degrad_cfgs, selected_degrads = random_select_and_order(self.cfg, seed=self.seed)

            else:
                degrad_cfgs, _ = random_select_and_order(self.cfg, seed=self.seed)
                selected_degrads = self.selected_degrads

            clean_audio, degraded_audio = apply_degradation(self.cfg, clean_audio, noise_audio, rir_audio, degrad_cfgs, selected_degrads, seed=self.seed)
            # except:
            #     noise_path = random.choice(list(self.noise_wavs_path))
            #     noise_audio, noise_sr = librosa.load(noise_path, sr=self.sampling_rate)
            #     noise_audio = noise_audio.reshape(1, -1)

            #     clean_audio, degraded_audio = apply_degradation(self.cfg, clean_audio, noise_audio, rir_audio, degrad_cfgs, selected_degrads, seed=self.seed)

            assert clean_audio.shape[-1] == degraded_audio.shape[-1]




            clean_audio, degraded_audio = torch.FloatTensor(clean_audio), torch.FloatTensor(degraded_audio)

            clean_audio = peak_normalize(clean_audio)
            degraded_audio = peak_normalize(degraded_audio)


            assert clean_audio.size(1) == degraded_audio.size(1)

            if clean_audio.size(1) >= self.segment_size:
                max_audio_start = clean_audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                clean_audio = clean_audio[:, audio_start:audio_start + self.segment_size]
                degraded_audio = degraded_audio[:, audio_start:audio_start + self.segment_size]
            else:
                clean_audio = torch.nn.functional.pad(clean_audio, (0, self.segment_size - clean_audio.size(1)), 'constant')
                degraded_audio = torch.nn.functional.pad(degraded_audio, (0, self.segment_size - degraded_audio.size(1)), 'constant')


            clean_mag, clean_pha, clean_com = mag_phase_stft(clean_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor)
            degraded_mag, degraded_pha, degraded_com = mag_phase_stft(degraded_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor)

            return (clean_audio.squeeze(), clean_mag.squeeze(), clean_pha.squeeze(), clean_com.squeeze(), degraded_audio.squeeze(), degraded_mag.squeeze(), degraded_pha.squeeze())



        elif self.mode == "Validation": # Validation
            clean_path = self.clean_val_path[index]
            degraded_path = self.degraded_val_path[index]

            clean_audio, clean_sr = librosa.load(clean_path, sr=self.sampling_rate)
            degraded_audio, degraded_sr = librosa.load(degraded_path, sr=self.sampling_rate)




            clean_audio, degraded_audio = torch.FloatTensor(clean_audio), torch.FloatTensor(degraded_audio)


            clean_audio = peak_normalize(clean_audio)
            degraded_audio = peak_normalize(degraded_audio)

            assert clean_audio.shape[-1] == degraded_audio.shape[-1]

            # norm_factor = torch.sqrt(len(degraded_audio) / torch.sum(degraded_audio ** 2.0))
            # clean_audio = (clean_audio * norm_factor).unsqueeze(0)
            # degraded_audio = (degraded_audio * norm_factor).unsqueeze(0)

            clean_mag, clean_pha, clean_com = mag_phase_stft(clean_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor)
            degraded_mag, degraded_pha, degraded_com = mag_phase_stft(degraded_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor)
            return (clean_audio.squeeze(), clean_mag.squeeze(), clean_pha.squeeze(), clean_com.squeeze(), degraded_audio.squeeze(), degraded_mag.squeeze(), degraded_pha.squeeze())

        elif self.mode == "Test": # Test
            clean_path = self.clean_test_path[index]
            degraded_path = self.degraded_test_path[index]

            clean_audio, clean_sr = librosa.load(clean_path, sr=self.sampling_rate)
            degraded_audio, degraded_sr = librosa.load(degraded_path, sr=self.sampling_rate)



            clean_audio, degraded_audio = torch.FloatTensor(clean_audio), torch.FloatTensor(degraded_audio)

            clean_audio = peak_normalize(clean_audio)
            degraded_audio = peak_normalize(degraded_audio)

            assert clean_audio.shape[-1] == degraded_audio.shape[-1]

            # norm_factor = torch.sqrt(len(degraded_audio) / torch.sum(degraded_audio ** 2.0))
            # clean_audio = (clean_audio * norm_factor).unsqueeze(0)
            # degraded_audio = (degraded_audio * norm_factor).unsqueeze(0)


            clean_mag, clean_pha, clean_com = mag_phase_stft(clean_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor)
            degraded_mag, degraded_pha, degraded_com = mag_phase_stft(degraded_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor)


            return (clean_audio.squeeze(), clean_mag.squeeze(), clean_pha.squeeze(), clean_com.squeeze(), degraded_audio.squeeze(), degraded_mag.squeeze(), degraded_pha.squeeze())

    def __len__(self):
        if self.mode == "Train":
            return len(self.clean_wavs_path)
        elif self.mode == "Validation":
            return len(self.clean_val_path)
        elif self.mode == "Test":
            return len(self.clean_test_path)
