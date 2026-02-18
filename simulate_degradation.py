import os
import json
import random
import torch
import torch.utils.data
import librosa
import numpy as np
import scipy
from simulate_degradation_utils import *
import yaml


def random_select_and_order(cfg, seed=None):
    '''
    Randomly select and order the degradation configurations.
    '''
    if seed:
        random.seed(seed)
    degrad_configs = {
        "snr": random.choice(cfg["degradation_cfg"]["snr"]),
        "bandwidth_sr": random.choice(cfg["degradation_cfg"]["bandwidth_sr"]),
        "bandwidth_type": random.choice(cfg["degradation_cfg"]["bandwidth_type"]),
        "lowpass_type": random.choice(cfg["degradation_cfg"]["lowpass_type"]),
        "lowpass_order": random.choice(cfg["degradation_cfg"]["lowpass_order"]),
        "clipping_min": np.random.uniform(*cfg["degradation_cfg"]["clipping_min"]),
        "clipping_max": np.random.uniform(*cfg["degradation_cfg"]["clipping_max"]),
        "packet_duration": random.choice(cfg["degradation_cfg"]["packet_duration"]),
        "packet_loss_rate": np.random.uniform(*cfg["degradation_cfg"]["packet_loss_rate"]),
    }
    # degrad_types = ["noise", "reverb", "bandwidth", "clipping", "packet_loss"]   
    # #degrad_types = ["noise", "reverb"]
    # degrad_probs = [cfg["degradation_cfg"]["noise_prob"], cfg["degradation_cfg"]["reverb_prob"], cfg["degradation_cfg"]["bandwidth_prob"], 
    #                   cfg["degradation_cfg"]["clipping_prob"], cfg["degradation_cfg"]["packet_loss_prob"]]
    # degrad_probs = np.array(degrad_probs) / np.sum(degrad_probs)  # Normalize probabilities to sum up to 1

    degrad_types = ["noise", "reverb", "bandwidth", "clipping"]   
    #degrad_types = ["noise", "reverb"]
    degrad_probs = [cfg["degradation_cfg"]["noise_prob"], cfg["degradation_cfg"]["reverb_prob"], cfg["degradation_cfg"]["bandwidth_prob"], 
                      cfg["degradation_cfg"]["clipping_prob"], cfg["degradation_cfg"]["packet_loss_prob"]]
    #degrad_probs = np.array(degrad_probs) / np.sum(degrad_probs)  # Normalize probabilities to sum up to 1
    degrad_probs = {"noise": degrad_probs[0], "reverb": degrad_probs[1], "bandwidth": degrad_probs[2], "clipping": degrad_probs[3], "packet_loss": degrad_probs[4]}

    selected_degradations = [x for x in degrad_types if random.random() < degrad_probs[x]]

    if len(selected_degradations) == 0:
        degrad_probs  = [degrad_probs[x] for x in degrad_types]
        selected_degradations = np.random.choice(degrad_types, p=degrad_probs, size=1).tolist()


    #selected_degradations = ['noise']
    #selected_degradations = np.random.choice(degrad_types, p=degrad_probs, size=random.randint(1, len(degrad_types)), replace=False).tolist()
    degrad_order_map = {
        "noise": 2,
        "reverb": 1,
        "bandwidth": 5,
        "clipping": 3,
        "packet_loss": 4
    }
    # Shuffle the order of selected degradations
    selected_degradations = sorted(selected_degradations, key=lambda x: degrad_order_map[x])

    return degrad_configs, selected_degradations

# degrad_configs = {"snr": 20, "bandwidth_freq": 1000, ...}, selected_degradations = ["noise", "reverb", ...]


def apply_degradation(cfg, speech_sample, noise_sample, rir_sample, degradation_configs, 
                      selected_degradations: list, seed: int = None):
    '''Apply degradations to speech sample
    Args:
        speech_sample: The original speech signal. (np.ndarray): a single speech sample (1, T)
        noise_sample: The noise signal to be added. a single noise sample (1, T)
        rir_sample: The room impulse response for reverberation. a single room impulse response (RIR) (1, T)
        degradation_configs: The configuration parameters for each degradation type.
        selected_degradations: The list of selected degradation types to apply.
    '''
    assert type(selected_degradations) == list, "selected_degradations must be a list."
    assert len(selected_degradations) >= 1, "At least one degradation type must be selected."

    sr = cfg["stft_cfg"]["sampling_rate"]
    rng = np.random.default_rng(seed=seed)
    # Start with speech sample

    if selected_degradations[0] == "noise":
        degraded_sample, _ = mix_noise(speech_sample, noise_sample, degradation_configs["snr"], rng)
    elif selected_degradations[0] == "reverb":
        degraded_sample = add_reverberation(speech_sample, rir_sample)
    elif selected_degradations[0] == "bandwidth":
        degraded_sample = bandwidth_limitation(speech_sample, sr, degradation_configs["bandwidth_sr"], degradation_configs["bandwidth_type"], 
                                               degradation_configs["lowpass_type"], degradation_configs["lowpass_order"])
    elif selected_degradations[0] == "clipping":
        degraded_sample = clipping(speech_sample, degradation_configs["clipping_min"], degradation_configs["clipping_max"])
    elif selected_degradations[0] == "packet_loss":
        degraded_sample = packet_loss(speech_sample, sr, degradation_configs["packet_duration"], degradation_configs["packet_loss_rate"], max_continuous_packet_loss=5)    

    if len(selected_degradations) > 1:

        for degrad in selected_degradations[1:]:
            if degrad == "noise":
                degraded_sample, _ = mix_noise(degraded_sample, noise_sample, degradation_configs["snr"], rng)
            elif degrad == "reverb":
                degraded_sample = add_reverberation(degraded_sample, rir_sample)
            elif degrad == "bandwidth":
                degraded_sample = bandwidth_limitation(degraded_sample, sr, degradation_configs["bandwidth_sr"], degradation_configs["bandwidth_type"], 
                                                       degradation_configs["lowpass_type"], degradation_configs["lowpass_order"])
            elif degrad == "clipping":
                degraded_sample = clipping(degraded_sample, degradation_configs["clipping_min"], degradation_configs["clipping_max"])
            elif degrad == "packet_loss":
                degraded_sample = packet_loss(degraded_sample, sr, degradation_configs["packet_duration"], degradation_configs["packet_loss_rate"], max_continuous_packet_loss=5)
    
    return speech_sample, degraded_sample


