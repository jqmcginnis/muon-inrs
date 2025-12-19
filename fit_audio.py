import argparse
import soundfile as sf
import os
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
import torchaudio  
import torchaudio.transforms as T 
import torchmetrics  
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from muon import SingleDeviceMuonWithAuxAdam
from models import (GaussFFN, GaussMLP, ReluFFN, ReluMLP,
                    ReluPosEncoding, SirenMLP,
                    WireMLP, FinerMLP, WireRealMLP)

def set_seed(seed):
    """Set seed for reproducibility across all random number generators"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_audio(path):
    """Loads an audio file and normalizes it to [-1, 1]"""
    if sf is None:
        raise ImportError("soundfile library is required for FFmpeg-free audio loading. Please install it with 'pip install soundfile'")
        
    # soundfile.read returns (data, samplerate). data is a numpy array (samples, channels) 
    # BUT for mono files, it returns (samples,)
    waveform_np, sample_rate = sf.read(path, dtype='float32') 
    
    # === CRITICAL FIX: Ensure 2D (samples, channels) even for mono files ===
    # soundfile.read returns 1D for mono, but we need 2D (samples, 1)
    if waveform_np.ndim == 1:
        waveform_np = waveform_np[:, np.newaxis] # Reshape from (samples,) to (samples, 1)

    # Convert numpy array (samples, channels) to torch tensor
    # No .T needed here, as the numpy array is already (samples, channels)
    waveform = torch.from_numpy(waveform_np) 
    # ===================================================================

    
    # Clamp to ensure it's within range
    waveform = torch.clamp(waveform, -1.0, 1.0)
    
    # waveform is now (samples, channels). We can return it directly as 'audio'.
    audio = waveform
    return audio, sample_rate

def parse_list(param_str, num_layers):
    """
    Parse the parameter from command line.
    
    Args:
        param_str: String from command line (e.g., "0.05" or "0.1,0.05,0.02,0.01")
        num_layers: Number of layers with activations (excludes final output layer)
    
    Returns:
        Single float or list of floats
    """
    if ',' in param_str:
        param_values = [float(x.strip()) for x in param_str.split(',')]
        if len(param_values) != (num_layers - 1):
            raise ValueError(f"Number of parameter values ({len(param_values)}) must match num_layers -1 ({num_layers-1}).")
        return param_values
    else:
        param_value = float(param_str)
        param_values = [param_value] * (num_layers - 1)
        return param_values

def get_coordinates(num_samples, scale=1.0):
    """Generate 1D coordinates from -1 to 1"""
    coords = torch.linspace(-scale, scale, num_samples)
    return coords.view(-1, 1) # Shape (num_samples, 1)

def calculate_snr(target, pred):
    """
    Calculates the Signal-to-Noise Ratio (SNR) between two audio signals.
    
    Args:
        target (torch.Tensor): The ground truth audio tensor.
        pred (torch.Tensor): The predicted audio tensor.
    """
    noise_power = torch.mean((target - pred) ** 2)
    
    if noise_power == 0:
        return float('inf')
        
    signal_power = torch.mean(target ** 2)
    
    if signal_power == 0:
        # This case is ambiguous (e.g., predicting silence perfectly)
        # We can return -inf or 0, let's return 0 as a convention
        if noise_power == 0:
             return float('inf') # Perfect prediction
        else:
             return -float('inf') # Predicting noise for silence

    snr_val = 10 * torch.log10(signal_power / noise_power)
    return snr_val

def create_comparison_plot(gt_audio, recon_audio, epoch, snr_val, si_snr_val, sample_rate):
    """Create side-by-side plot of ground truth and reconstructed audio (channel 0)"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    num_samples = len(gt_audio)
    time = np.linspace(0, num_samples / sample_rate, num_samples)
    
    # Use channel 0 for plotting
    gt_channel = gt_audio[:, 0]
    recon_channel = recon_audio[:, 0]
    
    axes[0].plot(time, gt_channel)
    axes[0].set_title('Ground Truth (Channel 0)', fontsize=12)
    axes[0].set_ylabel('Amplitude')
    
    axes[1].plot(time, np.clip(recon_channel, -1, 1)) # Clip for display
    axes[1].set_title(f'Reconstruction (Channel 0)\nEpoch {epoch}', fontsize=12)
    axes[1].set_ylabel('Amplitude')
    
    diff = gt_channel - recon_channel # Use unclipped for diff
    axes[2].plot(time, diff)
    
    title = f'Difference (Channel 0)\nSNR: {snr_val:.2f}dB\nSI-SNR: {si_snr_val:.2f}dB'
    
    axes[2].set_title(title, fontsize=12)
    axes[2].set_ylabel('Amplitude')
    axes[2].set_xlabel('Time (s)')
    
    plt.tight_layout()
    return fig

def plot_spectrogram(waveform_np, sample_rate, title, vmin=None, vmax=None):
    """Create a log-mel spectrogram plot for (channel 0)"""
    
    n_fft = 1024
    hop_length = 512
    n_mels = 64
    
    # Shape (samples, channels) -> (channels, samples)
    # The input waveform_np is (samples, channels)
    # We want to convert it to (channels, samples) for torchaudio transforms,
    # so we use .T or .transpose(0, 1)
    waveform_tensor = torch.from_numpy(waveform_np).T
    
    # Select channel 0, ensure shape (1, samples)
    waveform_ch_0 = waveform_tensor[0:1, :] 
    
    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    ).to('cpu')

    try:
        mel_spec = mel_spectrogram_transform(waveform_ch_0)
    except RuntimeError as e:
        print(f"Warning: Could not create spectrogram. Error: {e}")
        # Return an empty figure and placeholder min/max
        return plt.figure(), -80.0, 0.0
        
    log_mel_spec = T.AmplitudeToDB()(mel_spec)
    log_spec_np = log_mel_spec[0].numpy() # Get the numpy data
    
    # Use provided vmin/vmax if they exist, otherwise calculate from data
    plot_vmin = vmin if vmin is not None else log_spec_np.min()
    plot_vmax = vmax if vmax is not None else log_spec_np.max()
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    num_samples = waveform_np.shape[0]
    duration_sec = num_samples / sample_rate
    
    # Plot with time on x-axis and mel bins (or freq) on y-axis
    img = ax.imshow(
        log_spec_np, # Use the numpy data
        aspect='auto', 
        origin='lower', 
        extent=[0, duration_sec, 0, n_mels], # x-axis in seconds, y-axis as mel bins
        vmin=plot_vmin,  # <-- APPLY FIXED vmin
        vmax=plot_vmax   # <-- APPLY FIXED vmax
    )
    
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Mel Bins', fontsize=12) # More accurate than "Frequency"
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    
    plt.tight_layout()
    return fig, plot_vmin, plot_vmax


def train_model(args):
    """Train the neural field model on the given audio with specified parameters."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    si_snr_metric = torchmetrics.audio.ScaleInvariantSignalNoiseRatio().to(device)
    
    audio, sample_rate = load_audio(args.audio)
    num_samples, c = audio.shape # c = number of channels
 
    coords = get_coordinates(num_samples, scale=args.coordinate_scale)
    target = audio # Target is already (num_samples, channels)
    
    is_inpainting_task = args.inpainting_ratio < 1.0
    
    if is_inpainting_task:
        print(f"Performing inpainting task. Training on {args.inpainting_ratio * 100:.2f}% of samples.")
        # num_pixels -> num_samples
        num_train_samples = int(num_samples * args.inpainting_ratio)
        
        indices = torch.randperm(num_samples)
        train_indices = indices[:num_train_samples]
        test_indices = indices[num_train_samples:]
        
        train_coords = coords[train_indices].to(device)
        train_target = target[train_indices].to(device)
        
        test_coords = coords[test_indices].to(device)
        test_target = target[test_indices].to(device)
 
    else:
        print("Performing overfitting task on all samples.")
        train_coords = coords.to(device)
        train_target = target.to(device)

    if args.model == 'relu_ffn':
        model = ReluFFN(input_dim=1, mapping_size=args.mapping_size, hidden_dim=args.hidden_dim,
                       output_dim=c, num_layers=args.num_layers, sigma=args.fourier_sigma)
    elif args.model == 'relu_mlp':
        model = ReluMLP(input_dim=1, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers)
    elif args.model == 'relu_pos_enc':
        model = ReluPosEncoding(input_dim=1, mapping_size=args.mapping_size, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers)        
    elif args.model == 'gauss_ffn':
        model = GaussFFN(input_dim=1, mapping_size=args.mapping_size, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers, 
                        sigma=args.fourier_sigma, a=args.gauss_scale)
    elif args.model == 'gauss_mlp':
        model = GaussMLP(input_dim=1, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers, a=args.gauss_scale)
    elif args.model == 'siren_mlp':
        model = SirenMLP(input_dim=1, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers, omega=args.siren_omega)
    elif args.model == 'wire_mlp':
        model = WireMLP(input_dim=1, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers, omega=args.wire_omega, sigma=args.wire_sigma)
    elif args.model == 'finer_mlp':
        model = FinerMLP(input_dim=1, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers, omega=args.finer_omega, 
                        init_bias=args.finer_init_bias, bias_scale=args.finer_bias_scale)
    elif args.model == 'real_wire':
        model = WireRealMLP(input_dim=1, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers, omega=args.wire_omega, sigma=args.wire_sigma)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")
    
    print(f"Using model: {args.model} with {sum(p.numel() for p in model.parameters())} parameters.")
    print(model)

    model = model.to(device)
    if args.optimizer == 'muon':
        print("INFO: Setting up Muon optimizer.")
        muon_params = []
        other_params = [] # Biases, gains, first and last layer

        # Define models where the first layer is preceded by an embedding.
        # For these, we might want to optimize the first MLP layer with Muon.
        first_layer_muon_models = {'relu_ffn', 'gauss_ffn', 'relu_pos_enc'}
        is_special_model = args.model in first_layer_muon_models

        if hasattr(model, 'mlp') and isinstance(model.mlp, torch.nn.Sequential):
            num_mlp_layers = len(model.mlp)
            
            for name, param in model.named_parameters():
                is_muon_target = False
                
                # Check if the parameter is a weight matrix within the MLP
                if 'mlp' in name and 'weight' in name and param.ndim >= 2:
                    try:
                        # e.g., name is 'mlp.2.weight', extract '2'
                        layer_idx = int(name.split('.')[1])
                        
                        # Rule 1 (Default): Always optimize hidden layers (not first, not last) with Muon.
                        is_hidden_layer = 0 < layer_idx < num_mlp_layers - 1
                        
                        # Rule 2 (Conditional): Optimize the first layer (idx 0) if the flag is set for a special model.
                        is_first_layer_for_muon = (
                            is_special_model and 
                            args.optimize_first_layer_with_muon and 
                            layer_idx == 0
                        )

                        if is_hidden_layer or is_first_layer_for_muon:
                            is_muon_target = True

                    except (ValueError, IndexError):
                        pass # Not a standard mlp layer name, treat as 'other'
                
                if is_muon_target:
                    muon_params.append(param)
                else:
                    # Everything else (biases, embedding matrices, first/last layers by default) goes to Adam.
                    other_params.append(param)
        else:
            print("WARNING: Model does not have a standard 'mlp' attribute. Cannot separate params for Muon.")
            other_params = list(model.parameters())

        if muon_params: # Proceed with Muon only if we found params to optimize
            if is_special_model and args.optimize_first_layer_with_muon:
                print("INFO: --optimize_first_layer_with_muon=True. The first MLP layer will also be optimized by Muon.")

            param_groups = [
                dict(params=muon_params, use_muon=True, lr=args.muon_lr, weight_decay=args.muon_weight_decay),
                dict(params=other_params, use_muon=False, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.muon_aux_weight_decay),
            ]
            optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
            print(f"INFO: Muon optimizer configured. Muon params: {len(muon_params)}, Other params: {len(other_params)}.")
            wandb.config.update({"optimizer_type": "muon"}, allow_val_change=True)
        
        else: # If no params for Muon were found, raise an error instead of falling back
            raise ValueError(f"Muon optimizer was selected (model: {args.model}), but no suitable "
                             f"parameters were identified for Muon.")
    
    elif args.optimizer == 'lbfgs':
        print("INFO: Using L-BFGS optimizer.")
        optimizer = optim.LBFGS(
            model.parameters(),
            lr=args.lr,
            max_iter=args.lbfgs_max_iter,
            history_size=args.lbfgs_history_size
        )
        wandb.config.update({"optimizer_type": "lbfgs"}, allow_val_change=True)

    elif args.optimizer == 'adam':
        print("INFO: Using Adam optimizer.")
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)
        wandb.config.update({"optimizer_type": "adam"}, allow_val_change=True)

    else:
        raise ValueError(f"Unsupported optimizer type: {args.optimizer}")

    scheduler = None
    if args.optimizer != 'lbfgs': 
        if args.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max)
            print(f"Using CosineAnnealingLR scheduler with T_max={args.T_max}")
        elif args.scheduler == 'step':
            scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
            print(f"Using StepLR scheduler with step_size={args.step_size} and gamma={args.gamma}")
    else:
        print("INFO: Schedulers are disabled for L-BFGS optimizer.")
    
    metrics = {
        'epochs': [], 'full_snr': [], 'full_si_snr': [],
        'train_snr': [], 'test_snr': []
    }
    layer_metrics = {}
    
    gt_audio_np = audio.cpu().numpy()
    
    # Log ground truth audio
    # wandb.Audio internaly uses soundfile, which expects (samples, channels)
    wandb.log({"audio/ground_truth": wandb.Audio(gt_audio_np, sample_rate=sample_rate, caption="")})

    # Log ground truth spectrogram and capture its dB range
    gt_spec_fig, spec_vmin, spec_vmax = plot_spectrogram(
        gt_audio_np, 
        sample_rate, 
        ""
    )
    print(f"INFO: Setting fixed spectrogram range from GT: vmin={spec_vmin:.2f}, vmax={spec_vmax:.2f}")
    
    wandb.log({"plots/spectrogram_ground_truth": wandb.Image(gt_spec_fig, caption="")})
    plt.close(gt_spec_fig)

    # stable rank etc 
    layer_metrics = {}
    
    for epoch in range(args.epochs):
        model.train()
        
        if args.optimizer == 'lbfgs':
            def closure():
                optimizer.zero_grad()
                pred = model(train_coords)
                loss = F.mse_loss(pred, train_target)
                loss.backward()
                return loss
            
            optimizer.step(closure)
            with torch.no_grad():
                pred = model(train_coords)
                loss = F.mse_loss(pred, train_target)

        else: # Adam and Muon
            optimizer.zero_grad()
            pred = model(train_coords)
            loss = F.mse_loss(pred, train_target)
            loss.backward()
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        if epoch % args.log_n_epochs == 0:
            model.eval()
            with torch.no_grad():
                # A. Calculate metrics on the full audio
                full_pred = model(coords.to(device))
                pred_audio = full_pred.view(num_samples, c)
                target_audio_gpu = target.to(device) # Full target on GPU for metric calc
                
                full_snr_val = calculate_snr(target_audio_gpu, pred_audio).item()
                
                full_si_snr_val = si_snr_metric(pred_audio.T, target_audio_gpu.T).item()
                
                # B. Calculate SNR on train/test subsets
                train_snr_val = calculate_snr(train_target, model(train_coords)).item()
                
                test_snr_val = 0.0
                if is_inpainting_task:
                    test_snr_val = calculate_snr(test_target, model(test_coords)).item()

                metrics['epochs'].append(epoch)
                metrics['full_snr'].append(full_snr_val)
                metrics['full_si_snr'].append(full_si_snr_val)
                metrics['train_snr'].append(train_snr_val)
                metrics['test_snr'].append(test_snr_val)
                
                if is_inpainting_task:
                    print(f"Epoch {epoch:4d}: Loss={loss.item():.6f}, TrainSNR={train_snr_val:.2f}, TestSNR={test_snr_val:.2f}, FullSNR={full_snr_val:.2f}, SI-SNR={full_si_snr_val:.4f}")
                else:
                    print(f"Epoch {epoch:4d}: Loss={loss.item():.6f}, SNR={full_snr_val:.2f}, SI-SNR={full_si_snr_val:.4f}")
                
                log_dict = {
                    'epoch': epoch,
                    'loss': loss.item(),
                    'full_snr': full_snr_val,
                    'full_si_snr': full_si_snr_val,
                    'train_snr': train_snr_val,
                }

                # Optimizer LR logging 
                if args.optimizer == 'muon' and wandb.config.optimizer_type == 'muon':
                    log_dict['learning_rate_muon'] = optimizer.param_groups[0]['lr']
                    log_dict['learning_rate_aux'] = optimizer.param_groups[1]['lr']
                else:
                    log_dict['learning_rate'] = optimizer.param_groups[0]['lr']

                if is_inpainting_task:
                    log_dict['test_snr'] = test_snr_val
                
                if args.log_audio_evolution: # Changed from log_image_evolution
                    pred_audio_np = pred_audio.cpu().numpy()
                    target_audio_np = target.cpu().numpy()
                    
                    comparison_fig = create_comparison_plot(
                        target_audio_np, pred_audio_np, epoch, 
                        full_snr_val, full_si_snr_val, sample_rate
                    )
                    
                    # Clip audio to [-1, 1] for wandb.Audio logging
                    recon_audio_clipped_np = np.clip(pred_audio_np, -1, 1)
                    
            
                    recon_spec_fig, _, _ = plot_spectrogram(
                        pred_audio_np, 
                        sample_rate, 
                        f"",
                        vmin=spec_vmin,
                        vmax=spec_vmax
                    )
            
                    
                    log_dict.update({
                        'plots/comparison': wandb.Image(comparison_fig, caption=f'Epoch {epoch}: SNR={full_snr_val:.2f}, SI-SNR={full_si_snr_val:.4f}'),
                        'audio/reconstruction': wandb.Audio(recon_audio_clipped_np, sample_rate=sample_rate, caption=f''),               
                        'plots/spectrogram_reconstruction': wandb.Image(recon_spec_fig, caption=f"")
                
                    })
                    plt.close(comparison_fig)
                    
            
                    plt.close(recon_spec_fig)

                if hasattr(model, 'get_detailed_matrix_info'):
                    info = model.get_detailed_matrix_info()
                    for i, layer_info in enumerate(info['layer_infos']):
                        # Initialize lists in dictionary if they don't exist
                        if f'stable_rank_layer_{i}' not in layer_metrics:
                            layer_metrics[f'stable_rank_layer_{i}'] = []
                            layer_metrics[f'spectral_norm_layer_{i}'] = []
                            layer_metrics[f'condition_number_layer_{i}'] = []

                        # Get scalar values
                        stable_rank_val = layer_info.get('stable_rank', 0)
                        spectral_norm_val = layer_info.get('linear_spectral_norm', 0)
                        condition_number_val = layer_info.get('spectral_condition_no', 0)

                        # Append to our history dictionary
                        layer_metrics[f'stable_rank_layer_{i}'].append(stable_rank_val)
                        layer_metrics[f'spectral_norm_layer_{i}'].append(spectral_norm_val)
                        layer_metrics[f'condition_number_layer_{i}'].append(condition_number_val)

                        # Add to the current wandb log dictionary
                        log_dict[f'stable_rank/layer_{i}'] = stable_rank_val
                        log_dict[f'spectral_norm/layer_{i}'] = spectral_norm_val
                        log_dict[f'condition_number/layer_{i}'] = condition_number_val
                    
                    if 'end_to_end_spectral_bound' in info:
                         if 'end_to_end_bound' not in layer_metrics:
                            layer_metrics['end_to_end_bound'] = []
                         end_to_end_val = info['end_to_end_spectral_bound']
                         layer_metrics['end_to_end_bound'].append(end_to_end_val)
                         log_dict['end_to_end_bound'] = end_to_end_val
                
                wandb.log(log_dict)

    return model, metrics, layer_metrics, audio, sample_rate, spec_vmin, spec_vmax

def plot_metrics_seaborn_separate(metrics, layer_metrics, args):
    """Plot training metrics using seaborn and log to wandb."""
    sns.set_style("darkgrid")
    
    is_inpainting_task = args.inpainting_ratio < 1.0
    
    main_df = pd.DataFrame({
        'Epoch': metrics['epochs'],
        'Full SNR': metrics['full_snr'],
        'Full SI-SNR': metrics['full_si_snr'],
        'Train SNR': metrics['train_snr']
    })
    if is_inpainting_task:
        main_df['Test SNR'] = metrics['test_snr']

    # 1. SNR Plot
    plt.figure(figsize=(10, 6))
    if is_inpainting_task:
        snr_df = main_df.melt(id_vars=['Epoch'], value_vars=['Train SNR', 'Test SNR', 'Full SNR'],
                               var_name='Metric', value_name='SNR (dB)')
        ax = sns.lineplot(data=snr_df, x='Epoch', y='SNR (dB)', hue='Metric')
        ax.set_title('SNR Over Training (Inpainting)', fontsize=16, fontweight='bold', pad=20)
    else:
        ax = sns.lineplot(data=main_df, x='Epoch', y='Full SNR')
        ax.set_title('SNR Over Training (Overfitting)', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('SNR (dB)', fontsize=14)
    ax.set_xlabel('Epoch', fontsize=14)
    plt.tight_layout()
    wandb.log({"plots/snr_comparison": wandb.Image(plt)})
    plt.close()

    # 2. SI-SNR Plot
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=main_df, x='Epoch', y='Full SI-SNR')
    ax.set_title('Full Audio SI-SNR Over Training', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('SI-SNR (dB)', fontsize=14)
    plt.tight_layout()
    wandb.log({"plots/si_snr": wandb.Image(plt)})
    plt.close()
    
    if not layer_metrics:
        print("No layer metrics found to plot.")
        sns.reset_defaults()
        return

    layer_df = pd.DataFrame({'Epoch': metrics['epochs']})
    for key, values in layer_metrics.items():
        layer_df[key] = values

    # 4. Stable Ranks Plot
    stable_rank_keys = [k for k in layer_df.columns if 'stable_rank_layer_' in k]
    if stable_rank_keys:
        plt.figure(figsize=(12, 7))
        stable_rank_df_melted = layer_df.melt(id_vars=['Epoch'], value_vars=stable_rank_keys, var_name='Layer', value_name='Stable Rank')
        stable_rank_df_melted['Layer'] = stable_rank_df_melted['Layer'].str.replace('stable_rank_layer_', 'Layer ')
        ax = sns.lineplot(data=stable_rank_df_melted, x='Epoch', y='Stable Rank', hue='Layer', linewidth=2)
        ax.set_title('Stable Rank Evolution', fontsize=16, fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        wandb.log({"plots/stable_ranks": wandb.Image(plt)})
        plt.close()

    # 5. Spectral Norms Plot
    spectral_norm_keys = [k for k in layer_df.columns if 'spectral_norm_layer_' in k]
    if spectral_norm_keys:
        plt.figure(figsize=(12, 7))
        spectral_norm_df_melted = layer_df.melt(id_vars=['Epoch'], value_vars=spectral_norm_keys, var_name='Layer', value_name='Spectral Norm')
        spectral_norm_df_melted['Layer'] = spectral_norm_df_melted['Layer'].str.replace('spectral_norm_layer_', 'Layer ')
        ax = sns.lineplot(data=spectral_norm_df_melted, x='Epoch', y='Spectral Norm', hue='Layer', linewidth=2)
        ax.set_title('Spectral Norm Evolution', fontsize=16, fontweight='bold', pad=20)
        ax.set_yscale('log')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        wandb.log({"plots/spectral_norms": wandb.Image(plt)})
        plt.close()

    # 6. Condition Numbers Plot
    condition_keys = [k for k in layer_df.columns if 'condition_number_layer_' in k]
    if condition_keys:
        plt.figure(figsize=(12, 7))
        condition_df_melted = layer_df.melt(id_vars=['Epoch'], value_vars=condition_keys, var_name='Layer', value_name='Condition Number')
        condition_df_melted['Layer'] = condition_df_melted['Layer'].str.replace('condition_number_layer_', 'Layer ')
        ax = sns.lineplot(data=condition_df_melted, x='Epoch', y='Condition Number', hue='Layer', linewidth=2)
        ax.set_title('Condition Number Evolution', fontsize=16, fontweight='bold', pad=20)
        ax.set_yscale('log')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        wandb.log({"plots/condition_numbers": wandb.Image(plt)})
        plt.close()

    sns.reset_defaults()
    print("All plots saved and logged to wandb under 'plots/' namespace")

def main():
    parser = argparse.ArgumentParser(description="Train a Neural Field for Audio Overfitting or Inpainting.")

    # General arguments
    parser.add_argument('--audio', default='data/gt_bach.wav', type=str, help='Path to the input audio file.') # Changed
    parser.add_argument('--epochs', default=5000, type=int, help='Number of training epochs.')
    parser.add_argument('--log_n_epochs', default=500, type=int, help='Frequency of logging metrics and audio plots.')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility.')
    parser.add_argument('--log_audio_evolution', action='store_true', help='Log intermediate audio reconstructions to wandb.') # Changed
    parser.add_argument('--project_name', type=str, default='cvpr-audio', help='Wandb project name.') # Changed default

    # Optimizer arguments 
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'muon', 'lbfgs'], help='Optimizer to use.')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate for Adam, Muon (aux), and L-BFGS (initial step size).')
    parser.add_argument('--adam_weight_decay', type=float, default=0.0, help='Weight decay for pure Adam optimizer.')

    # Muon specific 
    parser.add_argument('--muon_weight_decay', type=float, default=0.0, help='Weight decay for Muon optimizer (hidden weights).')
    parser.add_argument('--muon_aux_weight_decay', type=float, default=0.0, help='Weight decay for auxiliary Adam in Muon.')
    parser.add_argument('--muon_lr', type=float, default=1e-3, help='Learning rate for the Muon part (hidden weights).')
    parser.add_argument('--optimize_first_layer_with_muon', action='store_true', help='For models with embeddings (FFN, PosEnc), also optimize the first linear layer of the MLP with Muon.')
    
    # L-BFGS specific 
    parser.add_argument('--lbfgs_max_iter', type=int, default=20, help='Max iterations per epoch for L-BFGS.')
    parser.add_argument('--lbfgs_history_size', type=int, default=100, help='History size for L-BFGS.')

    # Scheduler arguments 
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['none', 'cosine', 'step'], help='Learning rate scheduler type (not used with L-BFGS).')
    parser.add_argument('--T_max', type=int, default=5000, help='T_max for CosineAnnealingLR scheduler.')
    parser.add_argument('--step_size', type=int, default=1000, help='Step size for StepLR scheduler.')
    parser.add_argument('--gamma', type=float, default=0.9, help='Gamma for StepLR scheduler.')

    # Model and architecture arguments 
    parser.add_argument('--model', choices=['relu_mlp', 'relu_ffn', 'gauss_ffn', 'gauss_mlp', 'real_wire', 'siren_mlp', 'wire_mlp', 'relu_pos_enc', 'finer_mlp'], default='siren_mlp')
    parser.add_argument('--num_layers', default=5, type=int, help='Number of hidden layers in the MLP.')
    parser.add_argument('--hidden_dim', default=256, type=int, help='Dimension of hidden layers.')
    parser.add_argument('--mapping_size', default=128, type=int, help='Mapping size for Fourier Feature mappings.')
    
    # Model-specific activation arguments 
    parser.add_argument('--fourier_sigma', default=10.0, type=float, help='Sigma for Fourier Feature mapping.')
    parser.add_argument('--siren_omega', default='30.0', type=str, help="Omega for SIREN layers.")
    parser.add_argument('--finer_omega', default='30.0', type=str, help="Omega for FINER layers.")
    parser.add_argument('--finer_init_bias', action="store_true",  help="Initial bias for FINER layers.")
    parser.add_argument('--finer_bias_scale', default=float(1/math.sqrt(2)), type=float, help="Bias scale for FINER layers.")
    parser.add_argument('--gauss_scale', default='0.05', type=str, help='Scale parameter for Gaussian activation.')
    parser.add_argument('--wire_sigma', default='10.0', type=str, help="Sigma for WIRE layers.")
    parser.add_argument('--wire_omega', default='30.0', type=str, help="Omega for WIRE layers.")
    parser.add_argument('--mfn_fourier_scale', default=256, type=float, help='Scale for MFN_FourierNet model.')
    
    # normalization scale
    parser.add_argument('--coordinate_scale', default=100.0, type=float, help='Normalization scale for Fourier features.')

    # Task-specific arguments
    parser.add_argument('--inpainting_ratio', default=1.0, type=float, help='Ratio of samples for training (1.0 for overfitting, <1.0 for inpainting).')
    
    # Plotting 
    parser.add_argument('--create_plots', action='store_true', help='Plot metrics separately using seaborn.')

    args = parser.parse_args()
    set_seed(args.seed)

    # Parse list-based arguments 
    args.gauss_scale = parse_list(args.gauss_scale, args.num_layers)
    args.siren_omega = parse_list(args.siren_omega, args.num_layers)
    args.finer_omega = parse_list(args.finer_omega, args.num_layers)
    args.wire_sigma = parse_list(args.wire_sigma, args.num_layers)
    args.wire_omega = parse_list(args.wire_omega, args.num_layers)

    if not 0.0 < args.inpainting_ratio <= 1.0:
        raise ValueError("inpainting_ratio must be between 0.0 and 1.0.")

    wandb.init(
        project=args.project_name,
        config=vars(args)
    )

    model, metrics, layer_metrics, original_audio, sample_rate, spec_vmin, spec_vmax = train_model(args)
    model.eval()
    device = next(model.parameters()).device
    si_snr_metric = torchmetrics.audio.ScaleInvariantSignalNoiseRatio().to(device)

    with torch.no_grad():
        num_samples, c = original_audio.shape
        coords = get_coordinates(num_samples, args.coordinate_scale).to(device)
        
        final_pred_gpu = model(coords).view(num_samples, c)
        final_pred_cpu = final_pred_gpu.cpu()
        original_audio_gpu = original_audio.to(device)
        
        final_snr = calculate_snr(original_audio_gpu, final_pred_gpu).item()       
        final_si_snr = si_snr_metric(final_pred_gpu.T, original_audio_gpu.T).item()

        print(f"Final Full Audio SNR: {final_snr:.2f} dB")
        print(f"Final Full Audio SI-SNR: {final_si_snr:.4f} dB")
        
        final_comparison_fig = create_comparison_plot(
            original_audio.numpy(), final_pred_cpu.numpy(), args.epochs, 
            final_snr, final_si_snr, sample_rate
        )
        
        final_recon_clipped_np = np.clip(final_pred_cpu.numpy(), -1, 1)
        

        final_spec_fig, _, _ = plot_spectrogram(
            final_pred_cpu.numpy(), 
            sample_rate, 
            "",
            vmin=spec_vmin,
            vmax=spec_vmax
        )

        
        log_dict = {
            'final_full_snr': final_snr,
            'final_si_snr': final_si_snr,
            'plots/final_comparison': wandb.Image(final_comparison_fig, 
                                                 caption=f'Final Result: SNR={final_snr:.2f}, SI-SNR={final_si_snr:.4f}'),                                
            'audio/final_reconstruction': wandb.Audio(final_recon_clipped_np, sample_rate=sample_rate),
            'plots/final_spectrogram': wandb.Image(final_spec_fig, caption=""),
        }
            
        wandb.log(log_dict)
        plt.close(final_comparison_fig)
        plt.close(final_spec_fig)

    plot_metrics_seaborn_separate(metrics, layer_metrics, args)
    
    wandb.finish()
    print("Training finished and results logged to wandb.")

if __name__ == "__main__":
    main()
