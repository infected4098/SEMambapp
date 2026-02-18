import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import torch
import torch.nn.functional as F
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
import easydict
from dataset import GSRDataset
from model.stfts import mag_phase_stft, mag_phase_istft
from model.discriminator import MultiScaleSubbandCQTDiscriminator, MultiResolutionDiscriminator, \
    feature_loss, generator_loss, discriminator_loss
from model.loss import phase_losses, MultiScaleMelSpectrogramLoss
from utils import load_config, load_ckpts, load_optimizer_states, save_checkpoint, build_env, load_json_file, scan_checkpoint
import random
import torchaudio
import torch.distributed as dist
from evaluation_metrics import load_modules, compute_val_metrics
from model.semambapp import SEMambapp

os.environ['MASTER_ADDR'] = 'localhost'
torch.backends.cudnn.benchmark = True
steps = 0

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def setup_schedulers(optimizers, cfg, last_epoch):
    """Set up learning rate schedulers."""
    optim_g, optim_d = optimizers
    lr_decay = cfg['training_cfg']['lr_decay']

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=lr_decay, last_epoch=last_epoch)

    return scheduler_g, scheduler_d

def load_ckpts_spec(args, device, prefix):
    """Load checkpoints if available."""
    if os.path.isdir(args.exp_path):
        cp_g = scan_checkpoint(args.exp_path, f'{prefix}g_')
        cp_do = scan_checkpoint(args.exp_path, f'{prefix}do_')
        if cp_g is None or cp_do is None:
            return None, None, 0, -1
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        return state_dict_g, state_dict_do, state_dict_do['steps'] + 1, state_dict_do['epoch']
    return None, None, 0, -1


def run(rank, n_gpus, a, hps):


    global steps

    # Initialize distributed training if using multiple GPUs
    if n_gpus > 1:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    
    torch.manual_seed(hps["env_setting"]["seed"])
    torch.cuda.set_device(rank)
    device = torch.device('cuda:{:d}'.format(rank))

    # Collecting filelists for training and validation

    # Training dataset configuration
    trainset = GSRDataset(
        hps,
        a.clean_train_json,
        a.noise_train_json,
        a.rir_train_json,
        a.clean_valid_json,
        a.degraded_valid_json,
        n_cache_reuse=1, 
        shuffle=True, 
        device=device, 
        pcs=False,
        seed=None,
        mode="Train"
    )
    
    train_sampler = DistributedSampler(trainset, rank = rank) if n_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=hps["env_setting"]["num_workers"], shuffle=False,
                              sampler=train_sampler,
                              batch_size=hps["training_cfg"]["batch_size"], pin_memory=True, drop_last=True)

    # Validation dataset configuration
    if rank == 0:
        validset = GSRDataset(
        hps,
        a.clean_train_json,
        a.noise_train_json,
        a.rir_train_json,
        a.clean_valid_json,
        a.degraded_valid_json,
        n_cache_reuse=1, 
        shuffle=True, 
        device=device, 
        pcs=False,
        seed=None,
        mode="Validation"
        )
        validation_loader = DataLoader(
            validset,
            num_workers=1,
            shuffle=False,
            sampler=None,
            batch_size=1,
            pin_memory=True,
            drop_last=True,
        )

        # Initialize Weights & Biases logging
        wandb.init(project=f"{a.experiment_name}", resume="allow")
        wandb.run.name = ""
        wandb.config.update(hps)


    # Initializing modules
    univsemamba = SEMambapp(hps).to(device)
    mssbcqtd = MultiScaleSubbandCQTDiscriminator().to(device)
    mrd = MultiResolutionDiscriminator().to(device)

    fn_mel_loss_multiscale = MultiScaleMelSpectrogramLoss(
                sampling_rate=16000
            ).to(device) 
    mrstft, pesq, utmos = load_modules(hps, device)
    # Print model parameter counts
    if rank == 0:
        print('Number of Parameters for SEMambapp:', get_param_num(univsemamba))
        print("Number of Parameters for MSSBCQTD:  ", get_param_num(mssbcqtd))
        print("Number of Parameters for MRD:  ", get_param_num(mrd))





    state_dict_g, state_dict_do, steps, last_epoch = load_ckpts_spec(a, device, prefix='ln_')
    if state_dict_g is not None:
        univsemamba.load_state_dict(state_dict_g['generator'], strict=False)
        mssbcqtd.load_state_dict(state_dict_do['mssbcqtd'], strict=False)
        mrd.load_state_dict(state_dict_do['mrd'], strict=False)


    optim_g = torch.optim.AdamW(univsemamba.parameters(), hps["training_cfg"]["learning_rate"], betas=[hps["training_cfg"]["adam_b1"], hps["training_cfg"]["adam_b2"]])
    optim_d = torch.optim.AdamW(itertools.chain(mrd.parameters(), mssbcqtd.parameters()),
                                hps["training_cfg"]["learning_rate"], betas=[hps["training_cfg"]["adam_b1"], hps["training_cfg"]["adam_b2"]])

    # Load optimizer states
    if state_dict_do is not None:
        print("Loading Optimizer States...")
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g, scheduler_d = setup_schedulers([optim_g, optim_d], hps, last_epoch)



    # Setup distributed data parallel if using multiple GPUs
    if n_gpus > 1:
        univsemamba = DDP(univsemamba, device_ids=[rank])
        mssbcqtd = DDP(mssbcqtd, device_ids=[rank]).to(device)
        mrd = DDP(mrd, device_ids=[rank]).to(device)

    # Set models to training mode
    univsemamba.train()
    mssbcqtd.train()
    mrd.train()

    # Initialize learning rate schedulers
    raf_scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps["training_cfg"]["lr_decay"], last_epoch=last_epoch)
    raf_scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps["training_cfg"]["lr_decay"], last_epoch=last_epoch)

    # Main training loop
    for epoch in range(max(0, last_epoch), a.training_epochs):
        start = time.time()
        
        if rank == 0:
            print("Epoch: {:d}".format(epoch))
            print('Learning Rate : {:.6f}'.format(optim_g.param_groups[0]['lr']))
            train(a, rank, epoch, hps, univsemamba, [mssbcqtd, mrd], [fn_mel_loss_multiscale, mrstft, pesq, utmos], [optim_g, optim_d],
                     [scheduler_g, scheduler_d], [train_loader, validation_loader], n_gpus, device)
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))
        else:
            train(a, rank, epoch, hps, univsemamba, [mssbcqtd, mrd], [fn_mel_loss_multiscale, mrstft, pesq, utmos], [optim_g, optim_d],
                     [scheduler_g, scheduler_d], [train_loader, None], n_gpus, device)

def train(a, rank, epoch, hps, nets, discs, aux, optims, schedulers, loaders, n_gpus, device=None):

    generator = nets
    mssbcqtd, mrd = discs
    fn_mel_loss_multiscale, mrstft, pesq, utmos = aux
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, validation_loader = loaders

    global steps

    # Set epoch for distributed sampler
    if n_gpus > 1:
        train_loader.sampler.set_epoch(epoch)

    # Set models to training mode
    generator.train()
    mssbcqtd.train()
    mrd.train()

    # Training loop over batches
    for i, batch in enumerate(train_loader):
        if rank == 0:
            start_b = time.time()
        clean_audio, clean_mag, clean_pha, clean_com, _, noisy_mag, noisy_pha = batch # [B, 1, F, T], F = nfft // 2+ 1, T = nframes
        del _
        clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))
        clean_mag = torch.autograd.Variable(clean_mag.to(device, non_blocking=True))
        clean_pha = torch.autograd.Variable(clean_pha.to(device, non_blocking=True))
        clean_com = torch.autograd.Variable(clean_com.to(device, non_blocking=True))
        noisy_mag = torch.autograd.Variable(noisy_mag.to(device, non_blocking=True))
        noisy_pha = torch.autograd.Variable(noisy_pha.to(device, non_blocking=True))

        mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)

        audio_g = mag_phase_istft(mag_g, pha_g, hps["stft_cfg"]["n_fft"], hps["stft_cfg"]["hop_size"], hps["stft_cfg"]["win_size"], hps["model_cfg"]["compress_factor"])

        # Discriminator
        # ------------------------------------------------------- #
        optim_d.zero_grad()
        y_dq_hat_r, y_dq_hat_g, _, _ = mssbcqtd(clean_audio.unsqueeze(1), audio_g.unsqueeze(1).detach())
        loss_disc_q, losses_disc_q_r, losses_disc_q_g = discriminator_loss(
            y_dq_hat_r, y_dq_hat_g
        )
        # MRD
        y_dr_hat_r, y_dr_hat_g, _, _ = mrd(clean_audio.unsqueeze(1), audio_g.unsqueeze(1).detach())
        loss_disc_r, losses_disc_r_r, losses_disc_r_g = discriminator_loss(
            y_dr_hat_r, y_dr_hat_g
        )

        loss_disc_all = loss_disc_q + loss_disc_r
        
        loss_disc_all.backward()
        optim_d.step()
        # ------------------------------------------------------- #
        
        # Generator
        # ------------------------------------------------------- #
        optim_g.zero_grad()

        y_dq_hat_r, y_dq_hat_g, fmap_q_r, fmap_q_g = mssbcqtd(clean_audio.unsqueeze(1), audio_g.unsqueeze(1))
        loss_fm_q = feature_loss(fmap_q_r, fmap_q_g)
        loss_gen_q, losses_gen_q = generator_loss(y_dq_hat_g)

        # MRD loss
        y_dr_hat_r, y_dr_hat_g, fmap_r_r, fmap_r_g = mrd(clean_audio.unsqueeze(1), audio_g.unsqueeze(1))
        loss_fm_r = feature_loss(fmap_r_r, fmap_r_g)
        loss_gen_r, losses_gen_r = generator_loss(y_dr_hat_g)

        adv_g_loss = loss_gen_q + loss_gen_r
        fm_g_loss = loss_fm_q + loss_fm_r
        # Reference: https://github.com/yxlu-0102/MP-SENet/blob/main/train.py
        # L2 Magnitude Loss
        loss_mag = F.mse_loss(clean_mag, mag_g)
        # Anti-wrapping Phase Loss
        loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g, hps)
        loss_pha = loss_ip + loss_gd + loss_iaf
        # L2 Complex Loss
        loss_com = F.mse_loss(clean_com, com_g) * 2
        # Time Loss
        #loss_time = F.l1_loss(clean_audio, audio_g)

        # Consistency Loss
        _, _, rec_com = mag_phase_stft(audio_g, hps["stft_cfg"]["n_fft"], hps["stft_cfg"]["hop_size"], hps["stft_cfg"]["win_size"], hps["model_cfg"]["compress_factor"], addeps=True)
        loss_con = F.mse_loss(com_g, rec_com) * 2


        mel_loss = fn_mel_loss_multiscale(clean_audio.unsqueeze(1), audio_g.unsqueeze(1))


        loss_gen_all = (
            adv_g_loss * hps['training_cfg']['loss']['adv_g'] +
            fm_g_loss * hps['training_cfg']['loss']['fm_g'] +
            mel_loss * hps['training_cfg']['loss']['mel'] +
            loss_mag * hps['training_cfg']['loss']['magnitude'] +
            loss_pha * hps['training_cfg']['loss']['phase'] +
            loss_com * hps['training_cfg']['loss']['complex'] +
            loss_con * hps['training_cfg']['loss']['consistancy']
        )

        loss_gen_all.backward()
        optim_g.step()
        

        if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        adv_g_loss = adv_g_loss.item()
                        fm_g_loss = fm_g_loss.item()
                        mag_error = F.mse_loss(clean_mag, mag_g).item()
                        ip_error, gd_error, iaf_error = phase_losses(clean_pha, pha_g, hps)
                        pha_error = (ip_error+gd_error+iaf_error).item()
                        com_error = F.mse_loss(clean_com, com_g).item()
                        time_error = F.l1_loss(clean_audio, audio_g).item()
                        con_error = F.mse_loss( com_g, rec_com ).item()
                        mel_error = fn_mel_loss_multiscale(clean_audio.unsqueeze(1), audio_g.unsqueeze(1)).item()
                        print(
                            'Steps : {:d}, Gen Loss: {:4.3f}, Disc Loss: {:4.3f}, adv_g_loss Loss: {:4.3f}, '
                            'fm_g_loss: {:4.3f}, Mag Loss: {:4.3f}, Pha Loss: {:4.3f}, Com Loss: {:4.3f}, Time Loss: {:4.3f}, Mel Loss: {:4.3f}, Cons Loss: {:4.3f}, s/b : {:4.3f}'.format(
                                steps, loss_gen_all, loss_disc_all, adv_g_loss, fm_g_loss, mag_error, pha_error, com_error, time_error, mel_error, con_error, time.time() - start_b
                            )
                        )

                # Checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    exp_name = f"{a.exp_path}/ln_g_{steps:08d}.pth"
                    save_checkpoint(
                        exp_name,
                        {
                            'generator': (generator.module if n_gpus > 1 else generator).state_dict()
                        }
                    )
                    exp_name = f"{a.exp_path}/ln_do_{steps:08d}.pth"

                    save_checkpoint(
                        exp_name,
                        {
                            'mssbcqtd': (mssbcqtd.module if n_gpus > 1 else mssbcqtd).state_dict(),
                            'mrd': (mrd.module if n_gpus > 1 else mrd).state_dict(),
                            'optim_g': optim_g.state_dict(),
                            'optim_d': optim_d.state_dict(),
                            'steps': steps,
                            'epoch': epoch
                        }
                    )

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    wandb.log({"Training/adv_g_loss" :adv_g_loss, "steps":steps})
                    wandb.log({"Training/loss_gen_all" :loss_gen_all, "steps":steps})
                    wandb.log({"Training/adv_d_loss" :loss_disc_all, "steps":steps})
                    wandb.log({"Training/fm_g_loss" :fm_g_loss, "steps":steps})
                    wandb.log({"Training/mag_error" :mag_error, "steps":steps})
                    wandb.log({"Training/pha_error" :pha_error, "steps":steps})
                    wandb.log({"Training/com_error" :com_error, "steps":steps})
                    wandb.log({"Training/time_error" :time_error, "steps":steps})
                    wandb.log({"Training/con_error" :con_error, "steps":steps})
                    wandb.log({"Training/mel_error" :mel_loss, "steps":steps})

                # If NaN happend in training period, RaiseError
                if torch.isnan(loss_gen_all).any():
                    raise ValueError("NaN values found in loss_gen_all")

                # Validation
                if steps % a.validation_interval == 0 and steps != 0:
                    print("Validation Started...")
                    generator.eval()
                    torch.cuda.empty_cache()
                    audios_r, audios_g = [], []
                    val_mag_err_tot = 0
                    val_pha_err_tot = 0
                    val_com_err_tot = 0
                    val_mrstft_score = 0
                    val_pesq_score = 0
                    val_utmos = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            clean_audio, clean_mag, clean_pha, clean_com, noisy_audio, noisy_mag, noisy_pha = batch # [B, 1, F, T], F = nfft // 2+ 1, T = nframes
                            clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))
                            noisy_audio = torch.autograd.Variable(noisy_audio.to(device, non_blocking=True))
                            clean_mag = torch.autograd.Variable(clean_mag.to(device, non_blocking=True))
                            clean_pha = torch.autograd.Variable(clean_pha.to(device, non_blocking=True))
                            clean_com = torch.autograd.Variable(clean_com.to(device, non_blocking=True))

                            mag_g, pha_g, com_g = generator(noisy_mag.to(device), noisy_pha.to(device))

                            audio_g = mag_phase_istft(mag_g, pha_g, hps["stft_cfg"]["n_fft"], hps["stft_cfg"]["hop_size"], hps["stft_cfg"]["win_size"], hps["model_cfg"]["compress_factor"])
                            
                            #audios_r += torch.split(clean_audio, 1, dim=0) # [1, T] * B
                            #audios_g += torch.split(audio_g, 1, dim=0)

                            val_mag_err_tot += F.mse_loss(clean_mag, mag_g).item()
                            val_ip_err, val_gd_err, val_iaf_err = phase_losses(clean_pha, pha_g, hps)
                            val_pha_err_tot += (val_ip_err + val_gd_err + val_iaf_err).item()
                            val_com_err_tot += F.mse_loss(clean_com, com_g).item()

                            # Trim audio_g to fit clean_audio length
                            if audio_g.size(1) > clean_audio.size(1):
                                audio_g = audio_g[:, :clean_audio.size(1)]
                            elif audio_g.size(1) < clean_audio.size(1):
                                clean_audio = clean_audio[:, :audio_g.size(1)]

                            score_metrics = compute_val_metrics(mrstft, pesq, utmos, clean_audio, audio_g, hps) 

                            val_mrstft_score += score_metrics["mrstft_score"].item()
                            val_pesq_score += score_metrics["pesq_score"].item()
                            val_utmos += score_metrics["utmos_score"].item()

                        val_mag_err = val_mag_err_tot / (j+1)
                        val_pha_err = val_pha_err_tot / (j+1)
                        val_com_err = val_com_err_tot / (j+1)
                        val_mrstft_score = val_mrstft_score / (j+1)
                        val_pesq_score = val_pesq_score / (j+1)
                        val_utmos = val_utmos / (j+1)

                        print('Steps : {:d}, PESQ Score: {:4.3f}, UTMOS: {:4.3f}, MRSTFT Score: {:4.3f}, s/b : {:4.3f}'.
                                format(steps, val_pesq_score, val_utmos, val_mrstft_score, time.time() - start_b))

                        wandb.log({"Validation/PESQ Score" : val_pesq_score, "steps":steps})
                        wandb.log({"Validation/UTMOS" : val_utmos, "steps":steps})
                        wandb.log({"Validation/mrstft_score" : val_mrstft_score, "steps":steps})
                        wandb.log({"Validation/Magnitude Loss" : val_mag_err, "steps":steps})
                        wandb.log({"Validation/Phase Loss" : val_pha_err, "steps":steps})

                    generator.train()

        steps += 1

    scheduler_g.step()
    scheduler_d.step()




def main():

    print('Initializing Training Process..')

    # RAF training configuration
    a = easydict.EasyDict({
    'config': 'semambapp_cfg.yaml',
    'clean_train_json': 'data/train_speech.json',
    'noise_train_json': 'data/train_noise.json',
    'rir_train_json': 'data/train_rir.json',
    'clean_valid_json': 'data/val_clean.json',
    'degraded_valid_json': 'data/val_degraded.json',
    "training_epochs": 100,
    "stdout_interval": 1250,
    "checkpoint_interval": 5000,
    "summary_interval": 1250,
    "validation_interval": 5000,
    "exp_path": "exp",
    "fine_tuning": False,
    "experiment_name": "train_semambapp",
    
    })

    a.exp_path = os.path.join(a.exp_path, a.experiment_name)

    # Ensure CUDA availability for RAF training
    assert torch.cuda.is_available(), "GSR training requires CUDA."

    # Load RAF configuration
    hps = load_config(a.config)

    # Setup multi-GPU RAF training
    n_gpus = torch.cuda.device_count()
    hps["training_cfg"]["batch_size"] = hps["training_cfg"]["batch_size"] // n_gpus  # Divide batch size by number of GPUs
    print("The number of GPUs used for GSR training is:", n_gpus)
    print("GSR Batch size per GPU is set to:", hps["training_cfg"]["batch_size"])

    port = 50000 + random.randint(0, 100)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    hps["env_setting"]["num_gpus"] = n_gpus
    
    # Launch RAF training
    if n_gpus > 1:
        mp.spawn(run, nprocs=n_gpus, args=(n_gpus, a, hps,))
    else:
        run(0, n_gpus, a, hps)

if __name__ == "__main__":
    main()
