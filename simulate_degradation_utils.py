import os
import json
import random
import torch
import torch.utils.data
import librosa
from torchaudio.io import AudioEffector, CodecConfig
import numpy as np
import scipy
from scipy import signal
import pyroomacoustics as pra
# Directly copied from https://github.com/espnet/espnet/blob/master/espnet2/train/preprocessor.py


def framing(
    x,
    frame_length: int = 512,
    frame_shift: int = 256,
    centered: bool = True,
    padded: bool = True,
):
    if x.size == 0:
        raise ValueError("Input array size is zero")
    if frame_length < 1:
        raise ValueError("frame_length must be a positive integer")
    if frame_length > x.shape[-1]:
        raise ValueError("frame_length is greater than input length")
    if 0 >= frame_shift:
        raise ValueError("frame_shift must be greater than 0")

    if centered:
        pad_shape = [(0, 0) for _ in range(x.ndim - 1)] + [
            (frame_length // 2, frame_length // 2)
        ]
        x = np.pad(x, pad_shape, mode="constant", constant_values=0)

    if padded:
        # Pad to integer number of windowed segments
        # I.e make x.shape[-1] = frame_length + (nseg-1)*nstep,
        #  with integer nseg
        nadd = (-(x.shape[-1] - frame_length) % frame_shift) % frame_length
        pad_shape = [(0, 0) for _ in range(x.ndim - 1)] + [(0, nadd)]
        x = np.pad(x, pad_shape, mode="constant", constant_values=0)

    # Created strided array of data segments
    if frame_length == 1 and frame_length == frame_shift:
        result = x[..., None]
    else:
        shape = x.shape[:-1] + (
            (x.shape[-1] - frame_length) // frame_shift + 1,
            frame_length,
        )
        strides = x.strides[:-1] + (frame_shift * x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return result


def detect_non_silence(
    x: np.ndarray,
    threshold: float = 0.01,
    frame_length: int = 1024,
    frame_shift: int = 512,
    window: str = "boxcar",
) -> np.ndarray:
    """Power based voice activity detection.

    Args:
        x: (Channel, Time)
    >>> x = np.random.randn(1000)
    >>> detect = detect_non_silence(x)
    >>> assert x.shape == detect.shape
    >>> assert detect.dtype == np.bool
    """
    if x.shape[-1] < frame_length:
        return np.full(x.shape, fill_value=True, dtype=bool)

    if x.dtype.kind == "i":
        x = x.astype(np.float64)
    # framed_w: (C, T, F)
    framed_w = framing(
        x,
        frame_length=frame_length,
        frame_shift=frame_shift,
        centered=False,
        padded=True,
    )
    framed_w *= scipy.signal.get_window(window, frame_length).astype(framed_w.dtype)
    # power: (C, T)
    power = (framed_w**2).mean(axis=-1)
    # mean_power: (C, 1)
    mean_power = np.mean(power, axis=-1, keepdims=True)
    if np.all(mean_power == 0):
        return np.full(x.shape, fill_value=True, dtype=bool)
    # detect_frames: (C, T)
    detect_frames = power / mean_power > threshold
    # detects: (C, T, F)
    detects = np.broadcast_to(
        detect_frames[..., None], detect_frames.shape + (frame_shift,)
    )
    # detects: (C, TF)
    detects = detects.reshape(*detect_frames.shape[:-1], -1)
    # detects: (C, TF)
    return np.pad(
        detects,
        [(0, 0)] * (x.ndim - 1) + [(0, x.shape[-1] - detects.shape[-1])],
        mode="edge",
    )



# Directly copied from https://github.com/urgent-challenge/urgent2025_challenge/blob/main/simulation/simulate_data_from_param.py 
def mix_noise(speech_sample, noise_sample, snr=5.0, rng=None):
    """Mix the speech sample with an additive noise sample at a given SNR.

    Args:
        speech_sample (np.ndarray): a single speech sample (Channel, Time)
        noise_sample (np.ndarray): a single noise sample (Channel, Time)
        snr (float): signal-to-nosie ratio (SNR) in dB
        rng (np.random.Generator): random number generator
    Returns:
        noisy_sample (np.ndarray): output noisy sample (Channel, Time)
        noise (np.ndarray): scaled noise sample (Channel, Time)
    """
    len_speech = speech_sample.shape[-1]
    len_noise = noise_sample.shape[-1]
    if len_noise < len_speech:
        offset = rng.integers(0, len_speech - len_noise)
        # Repeat noise
        noise_sample = np.pad(
            noise_sample,
            [(0, 0), (offset, len_speech - len_noise - offset)],
            mode="wrap",
        )
    elif len_noise > len_speech:
        offset = rng.integers(0, len_noise - len_speech)
        noise_sample = noise_sample[:, offset : offset + len_speech]


    power_speech = (speech_sample[detect_non_silence(speech_sample)] ** 2).mean()
    power_noise = (noise_sample[detect_non_silence(noise_sample)] ** 2).mean()
    scale = 10 ** (-snr / 20) * np.sqrt(power_speech) / np.sqrt(max(power_noise, 1e-10))
    noise = scale * noise_sample
    noisy_speech = speech_sample + noise
    return noisy_speech, noise

def simulate_rir_pyroomacoustics(
    fs,
    room_dim=(7.0, 7.0, 3.0),
    rt60=0.4,
    src_pos=(2.0, 2.0, 1.5),
    mic_pos=(4.0, 3.0, 1.5),
    max_order=10,
):
    """Simulate a mono RIR using pyroomacoustics (shoebox + ISM)."""
    # wall absorption from RT60
    e_absorption, max_order_rt = pra.inverse_sabine(rt60, room_dim)
    if max_order is None:
        max_order = max_order_rt

    room = pra.ShoeBox(
        room_dim,
        fs=fs,
        materials=pra.Material(e_absorption),
        max_order=max_order,
    )

    room.add_source(np.array(src_pos)[:, None])

    mic_array = np.array(mic_pos)[:, None]
    room.add_microphone_array(pra.MicrophoneArray(mic_array, fs))

    room.compute_rir()

    # room.rir[mic_index][src_index] -> (rir_time,)
    rir = np.array(room.rir[0][0])[None, :]  # (1, T)
    return rir


def add_reverberation(speech_sample, rir_sample):
    """Mix the speech sample with an additive noise sample at a given SNR.

    Args:
        speech_sample (np.ndarray): a single speech sample (1, Time)
        rir_sample (np.ndarray): a single room impulse response (RIR) (Channel, Time)
    Returns:
        reverberant_sample (np.ndarray): output noisy sample (Channel, Time)
    """
    reverberant_sample = scipy.signal.convolve(speech_sample, rir_sample, mode="full")
    return reverberant_sample[:, : speech_sample.shape[1]]

def add_reverberation_with_arni_or_sim(
    speech_sample,
    arni_rir_sample,
    fs,
    rt60=None,
    sim_prob=0.5,
    rng=None,
    **sim_kwargs,
):
    """
    With probability `sim_prob`, ignore `arni_rir_sample` and simulate an RIR
    using pyroomacoustics; otherwise use the given Arni RIR.

    Args:
        speech_sample: np.ndarray, shape (1, T)
        arni_rir_sample: np.ndarray, shape (C, Trir) from Arni
        fs: sampling rate (Hz)
        sim_prob: probability of using simulated RIR
        rng: np.random.Generator or None (for reproducibility)
        **sim_kwargs: extra args for simulate_rir_pyroomacoustics
    """
    if rng is None:
        rng = np.random.default_rng()

    use_sim = rng.random() < sim_prob

    if use_sim:
        rir = simulate_rir_pyroomacoustics(fs=fs, rt60=rt60, **sim_kwargs)  # (1, T)
    else:
        rir = arni_rir_sample  # (C, T)

    # If Arni has multiple channels and you want mono, you can pick or average:
    # rir = rir[[0], :]   # first channel
    # or:
    # rir = rir.mean(axis=0, keepdims=True)

    reverberant = add_reverberation(speech_sample, rir)
    return reverberant, rir, use_sim





def bandwidth_limitation(speech_sample, fs: int, fs_new: int, res_type="kaiser_best", 
                         lowpass_type="chebyshev", order=8, ripple=1, stop_atten=40):
    """Apply the bandwidth limitation distortion to the input signal.

    Args:
        speech_sample (np.ndarray): a single speech sample (1, Time)
        fs (int): sampling rate in Hz
        fs_new (int): effective sampling rate in Hz
        res_type (str): resampling method

    Returns:
        ret (np.ndarray): bandwidth-limited speech sample (1, Time)
    """
    opts = {"res_type": res_type}
    if fs == fs_new:
        return speech_sample
    assert fs > fs_new, (fs, fs_new)

    # Design lowpass filter (My modifications)
    nyq_orig = fs / 2
    nyq_new = fs_new / 2
    norm_cutoff = nyq_new / nyq_orig  # Normalize the frequency
    if lowpass_type == "butter":
        b, a = signal.butter(order, norm_cutoff, btype="low", analog=False)
    elif lowpass_type == "cheby1":
        b, a = signal.cheby1(order, ripple, norm_cutoff, btype="low")
    elif lowpass_type == "cheby2":
        b, a = signal.cheby2(order, ripple, norm_cutoff, btype="low")
    elif lowpass_type == "bessel":
        # Use 'norm=' option to keep gain ~1 at DC
        b, a = signal.bessel(order, norm_cutoff, btype="low", norm='phase')
    elif lowpass_type == "ellip":
        b, a = signal.ellip(order, ripple, stop_atten, norm_cutoff, btype="low")
    else:
        raise ValueError("filter_type must be butter, cheby1, bessel, or ellip")

    # Zero-phase filtering to avoid phase distortion
    ret = signal.filtfilt(b, a, speech_sample)

    # resample back to the original sampling rate


    ret = librosa.resample(ret, orig_sr=fs, target_sr=fs_new, **opts)

    ret = librosa.resample(ret, orig_sr=fs_new, target_sr=fs, **opts)
    return ret[:, : speech_sample.shape[1]]


def clipping(speech_sample, min_quantile: float = 0.0, max_quantile: float = 0.5):
    """Apply the clipping distortion to the input signal.

    Args:
        speech_sample (np.ndarray): a single speech sample (1, Time)
        min_quantile (float): lower bound on the quantile of samples to be clipped
        max_quantile (float): upper bound on the quantile of samples to be clipped

    Returns:
        ret (np.ndarray): clipped speech sample (1, Time)
    """
    q = np.array([min_quantile, max_quantile])
    min_, max_ = np.quantile(speech_sample, q, axis=-1, keepdims=False)
    # per-channel clipping
    ret = np.stack(
        [
            np.clip(speech_sample[i], min_[i], max_[i])
            for i in range(speech_sample.shape[0])
        ],
        axis=0,
    )
    return ret


def get_packet_loss_idx(
    speech_length, fs, packet_duration_ms, packet_loss_rate, max_continuous_packet_loss
):
    """Returns a list of indices (of packets) that are zeroed out."""

    # speech duration in ms and the number of packets
    speech_duration_ms = speech_length / fs * 1000
    num_packets = int(speech_duration_ms // packet_duration_ms)

    # randomly select the packet loss rate and calculate the packet loss duration
    #packet_loss_rate = np.random.uniform(*packet_loss_rate)
    packet_loss_duration_ms = packet_loss_rate * speech_duration_ms

    # calculate the number of packets to be zeroed out
    num_packet_loss = int(round(packet_loss_duration_ms / packet_duration_ms, 0))

    # list of length of each packet loss
    packet_loss_lengths = []
    for _ in range(num_packet_loss):
        num_continuous_packet_loss = np.random.randint(1, max_continuous_packet_loss)
        packet_loss_lengths.append(num_continuous_packet_loss)

        if num_packet_loss - sum(packet_loss_lengths) <= max_continuous_packet_loss:
            packet_loss_lengths.append(num_packet_loss - sum(packet_loss_lengths))
            break

    packet_loss_start_indices = np.random.choice(
        range(num_packets), len(packet_loss_lengths), replace=False
    )
    packet_loss_indices = []
    for idx, length in zip(packet_loss_start_indices, packet_loss_lengths):
        packet_loss_indices += list(range(idx, idx + length))

    return list(set(packet_loss_indices))


def packet_loss(
    speech_sample, fs: int, packet_duration_ms: int = 20, packet_loss_rate: int = 0.05, max_continuous_packet_loss: int = 5
):
    speech_length = speech_sample.shape[-1]
    packet_loss_indices = get_packet_loss_idx(
        speech_length, fs, packet_duration_ms, packet_loss_rate, max_continuous_packet_loss
    )

    for idx in packet_loss_indices:
        start = idx * packet_duration_ms * fs // 1000
        end = (idx + 1) * packet_duration_ms * fs // 1000
        speech_sample[:, start:end] = 0

    return speech_sample


def codec_compression(
    speech_sample,
    fs: int,
    format: str,
    encoder: str = None,
    qscale: int = None,
):
    assert format in ["mp3", "ogg"], format
    assert encoder in [None, "None", "vorbis", "opus"], encoder

    encoder = None if encoder == "None" else encoder
    if speech_sample.ndim == 2:
        speech_sample = speech_sample.T  # (channel, sample) -> (sample, channel)
    try:
        module = AudioEffector(
            format=format,
            encoder=encoder,
            codec_config=CodecConfig(qscale=qscale),
            pad_end=True,
        )
        output = module.apply(torch.from_numpy(speech_sample), fs).numpy()
    except Exception as e:
        print(format, encoder, qscale, flush=True)
        print(e, flush=True)

    if output.shape[0] < speech_sample.shape[0]:
        zeros = np.zeros((speech_sample.shape[0] - output.shape[0], output.shape[1]))
        output = np.concatenate((output, zeros), axis=0)
    elif output.shape[0] > speech_sample.shape[0]:
        output = output[: speech_sample.shape[0]]

    assert speech_sample.shape == output.shape, (speech_sample.shape, output.shape)
    return (
        output.T if output.ndim == 2 else output
    )  # (sample, channel) -> (channel, sample)
