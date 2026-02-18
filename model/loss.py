# Reference: https://github.com/yxlu-0102/MP-SENet/blob/main/models/generator.py

import torch
import torch.nn as nn
import numpy as np
from pesq import pesq
from joblib import Parallel, delayed
from typing import Optional, List, Union, Dict, Tuple
from collections import namedtuple
import typing
from librosa.filters import mel as librosa_mel_fn
import functools
from scipy import signal
import math


def phase_losses(phase_r, phase_g, cfg):
    """
    Calculate phase losses including in-phase loss, gradient delay loss, 
    and integrated absolute frequency loss between reference and generated phases.
    
    Args:
        phase_r (torch.Tensor): Reference phase tensor of shape (batch, freq, time).
        phase_g (torch.Tensor): Generated phase tensor of shape (batch, freq, time).
        h (object): Configuration object containing parameters like n_fft.
    
    Returns:
        tuple: Tuple containing in-phase loss, gradient delay loss, and integrated absolute frequency loss.
    """
    dim_freq = cfg['stft_cfg']['n_fft'] // 2 + 1  # Calculate frequency dimension
    dim_time = phase_r.size(-1)  # Calculate time dimension
    
    # Construct gradient delay matrix
    gd_matrix = (torch.triu(torch.ones(dim_freq, dim_freq), diagonal=1) - 
                 torch.triu(torch.ones(dim_freq, dim_freq), diagonal=2) - 
                 torch.eye(dim_freq)).to(phase_g.device)
    
    # Apply gradient delay matrix to reference and generated phases
    gd_r = torch.matmul(phase_r.permute(0, 2, 1), gd_matrix)
    gd_g = torch.matmul(phase_g.permute(0, 2, 1), gd_matrix)
    
    # Construct integrated absolute frequency matrix
    iaf_matrix = (torch.triu(torch.ones(dim_time, dim_time), diagonal=1) - 
                  torch.triu(torch.ones(dim_time, dim_time), diagonal=2) - 
                  torch.eye(dim_time)).to(phase_g.device)
    
    # Apply integrated absolute frequency matrix to reference and generated phases
    iaf_r = torch.matmul(phase_r, iaf_matrix)
    iaf_g = torch.matmul(phase_g, iaf_matrix)
    
    # Calculate losses
    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(gd_r - gd_g))
    iaf_loss = torch.mean(anti_wrapping_function(iaf_r - iaf_g))
    
    return ip_loss, gd_loss, iaf_loss

def anti_wrapping_function(x):
    """
    Anti-wrapping function to adjust phase values within the range of -pi to pi.
    
    Args:
        x (torch.Tensor): Input tensor representing phase differences.
    
    Returns:
        torch.Tensor: Adjusted tensor with phase values wrapped within -pi to pi.
    """
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

def compute_stft(y: torch.Tensor, n_fft: int, hop_size: int, win_size: int, center: bool, compress_factor: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the Short-Time Fourier Transform (STFT) and return magnitude, phase, and complex components.

    Args:
        y (torch.Tensor): Input signal tensor.
        n_fft (int): Number of FFT points.
        hop_size (int): Hop size for STFT.
        win_size (int): Window size for STFT.
        center (bool): Whether to pad the input on both sides.
        compress_factor (float, optional): Compression factor for magnitude. Defaults to 1.0.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Magnitude, phase, and complex components.
    """
    eps = torch.finfo(y.dtype).eps
    hann_window = torch.hann_window(win_size).to(y.device)
    
    stft_spec = torch.stft(
        y, 
        n_fft=n_fft, 
        hop_length=hop_size, 
        win_length=win_size, 
        window=hann_window, 
        center=center, 
        pad_mode='reflect', 
        normalized=False, 
        return_complex=True
    )
    
    real_part = stft_spec.real
    imag_part = stft_spec.imag

    mag = torch.sqrt( real_part.pow(2) * imag_part.pow(2) + eps )
    pha = torch.atan2( real_part + eps, imag_part + eps )

    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
    
    return mag, pha, com



class MultiScaleMelSpectrogramLoss(nn.Module):
    """Compute distance between mel spectrograms. Can be used
    in a multi-scale way.

    Parameters
    ----------
    n_mels : List[int]
        Number of mels per STFT, by default [5, 10, 20, 40, 80, 160, 320],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [32, 64, 128, 256, 512, 1024, 2048]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 0.0 (no ampliciation on mag part)
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 1.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    Additional code copied and modified from https://github.com/descriptinc/audiotools/blob/master/audiotools/core/audio_signal.py
    """

    def __init__(
        self,
        sampling_rate: int,
        n_mels: List[int] = [40, 80, 160, 320],
        window_lengths: List[int] = [256, 512, 1024, 2048],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 0.0,
        log_weight: float = 1.0,
        pow: float = 1.0,
        weight: float = 1.0,
        match_stride: bool = False,
        mel_fmin: List[float] = [0, 0, 0, 0],
        mel_fmax: List[float] = [None, None, None, None],
        window_type: str = "hann",
    ):
        super().__init__()
        self.sampling_rate = sampling_rate

        STFTParams = namedtuple(
            "STFTParams",
            ["window_length", "hop_length", "window_type", "match_stride"],
        )

        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow

    @staticmethod
    @functools.lru_cache(None)
    def get_window(
        window_type,
        window_length,
    ):
        return signal.get_window(window_type, window_length)

    @staticmethod
    @functools.lru_cache(None)
    def get_mel_filters(sr, n_fft, n_mels, fmin, fmax):
        return librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

    def mel_spectrogram(
        self,
        wav,
        n_mels,
        fmin,
        fmax,
        window_length,
        hop_length,
        match_stride,
        window_type,
    ):
        """
        Mirrors AudioSignal.mel_spectrogram used by BigVGAN-v2 training from: 
        https://github.com/descriptinc/audiotools/blob/master/audiotools/core/audio_signal.py
        """
        B, C, T = wav.shape

        if match_stride:
            assert (
                hop_length == window_length // 4
            ), "For match_stride, hop must equal n_fft // 4"
            right_pad = math.ceil(T / hop_length) * hop_length - T
            pad = (window_length - hop_length) // 2
        else:
            right_pad = 0
            pad = 0

        wav = torch.nn.functional.pad(wav, (pad, pad + right_pad), mode="reflect")

        #window = self.get_window(window_type, window_length)
        #window = torch.from_numpy(window).to(wav.device).float()

        stft = torch.stft(
            wav.reshape(-1, T),
            n_fft=window_length,
            hop_length=hop_length,
            window=torch.hann_window(window_length).to(wav.device),
            return_complex=True,
            center=True,
        )
        _, nf, nt = stft.shape
        stft = stft.reshape(B, C, nf, nt)
        if match_stride:
            """
            Drop first two and last two frames, which are added, because of padding. Now num_frames * hop_length = num_samples.
            """
            stft = stft[..., 2:-2]
        magnitude = torch.abs(stft)

        nf = magnitude.shape[2]
        mel_basis = self.get_mel_filters(
            self.sampling_rate, 2 * (nf - 1), n_mels, fmin, fmax
        )
        mel_basis = torch.from_numpy(mel_basis).to(wav.device)
        mel_spectrogram = magnitude.transpose(2, -1) @ mel_basis.T
        mel_spectrogram = mel_spectrogram.transpose(-1, 2)

        return mel_spectrogram

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes mel loss between an estimate and a reference
        signal.

        Parameters
        ----------
        x : torch.Tensor
            Estimate signal
        y : torch.Tensor
            Reference signal

        Returns
        -------
        torch.Tensor
            Mel loss.
        """

        loss = 0.0
        for n_mels, fmin, fmax, s in zip(
            self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params
        ):
            kwargs = {
                "n_mels": n_mels,
                "fmin": fmin,
                "fmax": fmax,
                "window_length": s.window_length,
                "hop_length": s.hop_length,
                "match_stride": s.match_stride,
                "window_type": s.window_type,
            }

            x_mels = self.mel_spectrogram(x, **kwargs)
            y_mels = self.mel_spectrogram(y, **kwargs)
            x_logmels = torch.log(
                x_mels.clamp(min=self.clamp_eps).pow(self.pow)
            ) / torch.log(torch.tensor(10.0))
            y_logmels = torch.log(
                y_mels.clamp(min=self.clamp_eps).pow(self.pow)
            ) / torch.log(torch.tensor(10.0))

            loss += self.log_weight * self.loss_fn(x_logmels, y_logmels)
            loss += self.mag_weight * self.loss_fn(x_logmels, y_logmels)

        return loss
