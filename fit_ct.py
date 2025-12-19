import argparse
import os
import random
import math 
import lpips
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
import cv2
import kornia
from muon import SingleDeviceMuonWithAuxAdam
from models import (GaussFFN, GaussMLP, ReluFFN, ReluMLP,
                    ReluPosEncoding, SirenMLP,
                    WireMLP, FinerMLP, WireRealMLP)

def normalize(x, fullnormalize=False):
    '''
    Normalize input to lie between 0, 1. (Used for CT image loading)
    '''
    if x.sum() == 0:
        return x
    xmax = x.max()
    if fullnormalize:
        xmin = x.min()
    else:
        xmin = 0
    xnormalized = (x - xmin)/(xmax - xmin)
    return xnormalized

def radon(imten, angles):
    '''
        Compute forward radon operation
        
        Inputs:
            imten: (1, 1, H, W) image tensor
            angles: (nangles) angles tensor
        Outputs:
            sinogram: (nangles, W) sinogram
    '''
    nangles = len(angles)
    imten_rep = torch.repeat_interleave(imten, nangles, 0) # Shape: (nangles, 1, H, W)
    imten_rot = kornia.geometry.rotate(imten_rep, angles) # Shape: (nangles, 1, H, W)
    sinogram = imten_rot.sum(2).squeeze() # Shape: (nangles, W)
    return sinogram

def measure(x, noise_snr=40, tau=100):
    ''' Realistic sensor measurement with readout and photon noise
    '''
    x_meas = np.copy(x)
    noise = np.random.randn(x_meas.size).reshape(x_meas.shape)*noise_snr
    if tau != float('Inf'):
        x_meas = x_meas*tau
        x_meas[x > 0] = np.random.poisson(x_meas[x > 0])
        x_meas[x <= 0] = -np.random.poisson(-x_meas[x <= 0])
        x_meas = (x_meas + noise)/tau
    else:
        x_meas = x_meas + noise
    return x_meas

def set_seed(seed):
    """Set seed for reproducibility across all random number generators"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_ct_image(path):
    """
    Loads a grayscale CT phantom image as (H, W, 1) float tensor [0, 1].
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    if img is None:
        raise FileNotFoundError(f"Could not load grayscale image from {path}")
    img_normalized = normalize(img, fullnormalize=True)
    return torch.tensor(img_normalized, dtype=torch.float32).unsqueeze(-1) # (H, W, 1)

def parse_list(param_str, num_layers):
    """Parse the parameter from command line."""
    if ',' in param_str:
        param_values = [float(x.strip()) for x in param_str.split(',')]
        if len(param_values) != (num_layers - 1):
            raise ValueError(f"Number of parameter values ({len(param_values)}) must match num_layers -1 ({num_layers-1}).")
        return param_values
    else:
        param_value = float(param_str)
        param_values = [param_value] * (num_layers - 1)
        return param_values

def get_coordinates(h, w):
    x = torch.linspace(-1, 1, w)
    y = torch.linspace(-1, 1, h)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
    coords = torch.stack([grid_x.flatten(),grid_y.flatten()], dim=-1) 
    return coords

def psnr(img1, img2, max_val=1.0):
    """Calculates the Peak Signal-to-Noise Ratio (PSNR)"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr_val = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr_val

def compute_lpips(img1, img2, lpips_model, device):
    """
    Compute LPIPS for grayscale images.
    Grayscale images are converted to 3-channel for LPIPS model.
    """
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2).float()
    
    # Both imgs are (H, W) or (H, W, 1)
    img1 = img1.squeeze().unsqueeze(0).repeat(3, 1, 1) # (3, H, W)
    img2 = img2.squeeze().unsqueeze(0).repeat(3, 1, 1) # (3, H, W)
    img1_batch = (img1.unsqueeze(0) * 2.0 - 1.0).to(device) # (1, 3, H, W)
    img2_batch = (img2.unsqueeze(0) * 2.0 - 1.0).to(device) # (1, 3, H, W)

    with torch.no_grad():
        lpips_val = lpips_model(img1_batch, img2_batch)
    
    return lpips_val.item()

def create_comparison_image(gt_img, recon_img, epoch, psnr_val, ssim_val, lpips_val):
    """Create side-by-side comparison for grayscale images."""
    
    gt_img = gt_img.squeeze()
    recon_img = recon_img.squeeze()
        
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(gt_img, cmap='gray')
    axes[0].set_title('Ground Truth', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(np.clip(recon_img, 0, 1), cmap='gray')
    axes[1].set_title(f'Reconstruction\nEpoch {epoch}', fontsize=12)
    axes[1].axis('off')
    
    diff = np.abs(gt_img - recon_img)
    diff_amplified = np.clip(diff, 0, 1)
    axes[2].imshow(diff_amplified, cmap='viridis') 
    
    title = f'Difference \nPSNR: {psnr_val:.2f}dB\nSSIM: {ssim_val:.4f}\nLPIPS: {lpips_val:.4f}'
    
    axes[2].set_title(title, fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig

def train_model(args, device):
    """Train the neural field model on the CT reconstruction task."""
    
    lpips_model = lpips.LPIPS(net='alex').to(device)
    
    print("Setting up CT Reconstruction Task.")
    
    # Load grayscale phantom
    img = load_ct_image(args.image)
    h, w, c = img.shape # c will always be 1
    print(f"Loaded phantom with resolution: {h}x{w}")
    
    # Get coordinates
    coords = get_coordinates(h, w).to(device)

    # gt_image_tensor is the (H, W, 1) ground truth phantom
    # We need it as [1, 1, H, W] for radon
    gt_image_tensor = img.permute(2, 0, 1).unsqueeze(0).to(device)
    
    thetas = torch.linspace(0, 180, args.num_projections, device=device)
    
    print("Generating clean sinogram...")
    target_sinogram_clean = radon(gt_image_tensor, thetas) # (nangles, W)
    
    if args.use_noise:
        print(f"Applying noise: SNR={args.noise_snr}, Tau={args.noise_tau}")
        sinogram_np = target_sinogram_clean.detach().cpu().numpy()
        sinogram_noisy = measure(sinogram_np, noise_snr=args.noise_snr, tau=args.noise_tau)
        train_target = torch.tensor(sinogram_noisy, device=device, dtype=torch.float32)
    else:
        print("Training on clean (noise-free) data.")
        train_target = target_sinogram_clean
        
    if args.model == 'relu_ffn':
        model = ReluFFN(input_dim=2, mapping_size=args.mapping_size, hidden_dim=args.hidden_dim,
                       output_dim=c, num_layers=args.num_layers, sigma=args.fourier_sigma)
    elif args.model == 'relu_mlp':
        model = ReluMLP(input_dim=2, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers)
    elif args.model == 'relu_pos_enc':
        model = ReluPosEncoding(input_dim=2, mapping_size=args.mapping_size, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers)        
    elif args.model == 'gauss_ffn':
        model = GaussFFN(input_dim=2, mapping_size=args.mapping_size, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers, 
                        sigma=args.fourier_sigma, a=args.gauss_scale)
    elif args.model == 'gauss_mlp':
        model = GaussMLP(input_dim=2, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers, a=args.gauss_scale)
    elif args.model == 'siren_mlp':
        model = SirenMLP(input_dim=2, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers, omega=args.siren_omega)
    elif args.model == 'wire_mlp':
        model = WireMLP(input_dim=2, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers, omega=args.wire_omega, sigma=args.wire_sigma)
    elif args.model == 'finer_mlp':
        model = FinerMLP(input_dim=2, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers, omega=args.finer_omega, 
                        init_bias=args.finer_init_bias, bias_scale=args.finer_bias_scale)
    elif args.model == 'real_wire':
        model = WireRealMLP(input_dim=2, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers, omega=args.wire_omega, sigma=args.wire_sigma)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")
    
    print(f"Using model: {args.model} with {sum(p.numel() for p in model.parameters())} parameters.")
    print(model)
    model = model.to(device)

    if args.optimizer == 'muon':
        print("INFO: Setting up Muon optimizer.")
        muon_params = []
        other_params = [] 
        first_layer_muon_models = {'relu_ffn', 'gauss_ffn', 'relu_pos_enc'}
        is_special_model = args.model in first_layer_muon_models

        if hasattr(model, 'mlp') and isinstance(model.mlp, torch.nn.Sequential):
            num_mlp_layers = len(model.mlp)
            for name, param in model.named_parameters():
                is_muon_target = False
                if 'mlp' in name and 'weight' in name and param.ndim >= 2:
                    try:
                        layer_idx = int(name.split('.')[1])
                        is_hidden_layer = 0 < layer_idx < num_mlp_layers - 1
                        is_first_layer_for_muon = (
                            is_special_model and 
                            args.optimize_first_layer_with_muon and 
                            layer_idx == 0
                        )
                        if is_hidden_layer or is_first_layer_for_muon:
                            is_muon_target = True
                    except (ValueError, IndexError):
                        pass 
                if is_muon_target:
                    muon_params.append(param)
                else:
                    other_params.append(param)
        else:
            print("WARNING: Model does not have a standard 'mlp' attribute. Cannot separate params for Muon.")
            other_params = list(model.parameters())

        if muon_params: 
            if is_special_model and args.optimize_first_layer_with_muon:
                print("INFO: --optimize_first_layer_with_muon=True. The first MLP layer will also be optimized by Muon.")
            param_groups = [
                dict(params=muon_params, use_muon=True, lr=args.muon_lr, weight_decay=args.muon_weight_decay),
                dict(params=other_params, use_muon=False, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.muon_aux_weight_decay),
            ]
            optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
            print(f"INFO: Muon optimizer configured. Muon params: {len(muon_params)}, Other params: {len(other_params)}.")
            wandb.config.update({"optimizer_type": "muon"}, allow_val_change=True)
        else: 
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
        'epochs': [], 'full_psnr': [], 'ssim': [], 'lpips': [],
    }
    layer_metrics = {}
    
    # Log ground truth image. `img` is (H, W, 1) tensor.
    gt_img_np = img.cpu().numpy()
    wandb.log({"images/ground_truth": wandb.Image(gt_img_np, caption="Ground Truth")})
    # Log the target sinogram
    wandb.log({"images/target_sinogram": wandb.Image(
        train_target.cpu().numpy(), 
        caption="Target Sinogram (Noisy)" if args.use_noise else "Target Sinogram (Clean)"
    )})

    print("Starting training loop...")
    for epoch in tqdm(range(args.epochs), desc="Training"):
        model.train()
        
        if args.optimizer == 'lbfgs':
            def closure():
                optimizer.zero_grad()
                reconstructed_pixels = model(coords) # (H*W, 1)
                reconstructed_image = reconstructed_pixels.view(1, 1, h, w) # (1, 1, H, W)
                reconstructed_sinogram = radon(reconstructed_image, thetas) # (nangles, W)
                loss = F.mse_loss(reconstructed_sinogram, train_target)
                loss.backward()
                return loss
            optimizer.step(closure)
            with torch.no_grad():
                reconstructed_pixels = model(coords)
                reconstructed_image = reconstructed_pixels.view(1, 1, h, w)
                reconstructed_sinogram = radon(reconstructed_image, thetas)
                loss = F.mse_loss(reconstructed_sinogram, train_target)
        else: # Adam and Muon
            optimizer.zero_grad()
            reconstructed_pixels = model(coords) # (H*W, 1)
            reconstructed_image = reconstructed_pixels.view(1, 1, h, w) # (1, 1, H, W)
            reconstructed_sinogram = radon(reconstructed_image, thetas) # (nangles, W)
            loss = F.mse_loss(reconstructed_sinogram, train_target)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
        
        if epoch % args.log_n_epochs == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                log_dict = {'epoch': epoch, 'loss': loss.item()}
                
                # Get full predicted image
                full_pred_pixels = model(coords)
                pred_img_tensor = full_pred_pixels.view(h, w, c) # (H, W, 1)
                
                # Clamp the reconstruction to [0, 1] as the GT image is in this range
                clamped_pred_img_tensor = torch.clamp(pred_img_tensor, min=0.0, max=1.0)
                pred_img_np = clamped_pred_img_tensor.cpu().numpy()

                # gt_image_tensor is (1, 1, H, W)
                target_img_np = gt_image_tensor.cpu().numpy().squeeze() # (H, W)

                # calculate full-image metrics (PSNR, SSIM, LPIPS)
                # clamped_pred_img_tensor is (H, W, 1), gt_image_tensor is (1, 1, H, W)
                full_psnr_val = psnr(clamped_pred_img_tensor.squeeze(), gt_image_tensor.squeeze()).item()
                # ssim needs (H, W) vs (H, W)
                ssim_val = ssim(target_img_np, pred_img_np.squeeze(), data_range=1.0)
                # lpips needs (H, W, 1) vs (H, W)
                lpips_val = compute_lpips(target_img_np, pred_img_np, lpips_model, device)
                
                metrics['epochs'].append(epoch)
                metrics['full_psnr'].append(full_psnr_val)
                metrics['ssim'].append(ssim_val)
                metrics['lpips'].append(lpips_val)
                
                log_dict.update({
                    'full_psnr': full_psnr_val,
                    'ssim': ssim_val,
                    'lpips': lpips_val,
                })

                print(f"Epoch {epoch:4d}: Loss={loss.item():.6f}, PSNR={full_psnr_val:.2f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}")
                
                if args.optimizer == 'muon' and wandb.config.optimizer_type == 'muon':
                    log_dict['learning_rate_muon'] = optimizer.param_groups[0]['lr']
                    log_dict['learning_rate_aux'] = optimizer.param_groups[1]['lr']
                else:
                    log_dict['learning_rate'] = optimizer.param_groups[0]['lr']
                
                if args.log_image_evolution:
                    # create_comparison_image expects (H, W) or (H, W, 1)
                    comparison_fig = create_comparison_image(target_img_np, pred_img_np, epoch, full_psnr_val, ssim_val, lpips_val)
                    log_dict.update({
                        'images/comparison': wandb.Image(comparison_fig, caption=f'Epoch {epoch}: PSNR={full_psnr_val:.2f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}'),
                        'images/reconstruction': wandb.Image(np.clip(pred_img_np, 0, 1), caption=f'Reconstruction at epoch {epoch}')
                    })
                    plt.close(comparison_fig)

                if hasattr(model, 'get_detailed_matrix_info'):
                    info = model.get_detailed_matrix_info()
                    for i, layer_info in enumerate(info['layer_infos']):
                        # Initialize lists for new metrics if this is the first log step
                        if f'stable_rank_layer_{i}' not in layer_metrics:
                            layer_metrics[f'stable_rank_layer_{i}'] = []
                            layer_metrics[f'effective_rank_layer_{i}'] = [] 
                            layer_metrics[f'spectral_norm_layer_{i}'] = []
                            layer_metrics[f'condition_number_layer_{i}'] = []
                        
                        # Get values from the model
                        stable_rank_val = layer_info.get('stable_rank', 0)
                        effective_rank_val = layer_info.get('effective_rank', 0) 
                        spectral_norm_val = layer_info.get('linear_spectral_norm', 0)
                        condition_number_val = layer_info.get('spectral_condition_no', 0)
                        
                        # Append values for plotting
                        layer_metrics[f'stable_rank_layer_{i}'].append(stable_rank_val)
                        layer_metrics[f'effective_rank_layer_{i}'].append(effective_rank_val) 
                        layer_metrics[f'spectral_norm_layer_{i}'].append(spectral_norm_val)
                        layer_metrics[f'condition_number_layer_{i}'].append(condition_number_val)
                        
                        # Log values to wandb for this step
                        log_dict[f'stable_rank/layer_{i}'] = stable_rank_val
                        log_dict[f'effective_rank/layer_{i}'] = effective_rank_val 
                        log_dict[f'spectral_norm/layer_{i}'] = spectral_norm_val
                        log_dict[f'condition_number/layer_{i}'] = condition_number_val

                    if 'end_to_end_spectral_bound' in info:
                         if 'end_to_end_bound' not in layer_metrics:
                            layer_metrics['end_to_end_bound'] = []
                         end_to_end_val = info['end_to_end_spectral_bound']
                         layer_metrics['end_to_end_bound'].append(end_to_end_val)
                         log_dict['end_to_end_bound'] = end_to_end_val
                
                wandb.log(log_dict)

    # Return the original image tensor (H, W, 1) for final eval
    return model, metrics, layer_metrics, img, lpips_model

def plot_metrics_seaborn_separate(metrics, layer_metrics, args):
    """Plot training metrics using seaborn and log to wandb."""
    sns.set_style("darkgrid")
    
    main_df = pd.DataFrame({
        'Epoch': metrics['epochs'],
        'Full PSNR': metrics['full_psnr'],
        'SSIM': metrics['ssim'],
        'LPIPS': metrics['lpips'],
    })
    
    # 1. PSNR Plot
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=main_df, x='Epoch', y='Full PSNR')
    ax.set_title('PSNR Over Training (CT Reconstruction)', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('PSNR (dB)', fontsize=14)
    ax.set_xlabel('Epoch', fontsize=14)
    plt.tight_layout()
    wandb.log({"plots/psnr_comparison": wandb.Image(plt)})
    plt.close()

    # 2. SSIM and LPIPS plots
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=main_df, x='Epoch', y='SSIM')
    ax.set_title('Full Image SSIM Over Training', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    wandb.log({"plots/ssim": wandb.Image(plt)})
    plt.close()
    
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=main_df, x='Epoch', y='LPIPS')
    ax.set_title('Full Image LPIPS Over Training', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    wandb.log({"plots/lpips": wandb.Image(plt)})
    plt.close()
    
    # 3. Linear Algebra Metrics
    if not layer_metrics:
        print("No layer metrics found to plot.")
        sns.reset_defaults()
        return

    layer_df = pd.DataFrame({'Epoch': metrics['epochs']})
    for key, values in layer_metrics.items():
        layer_df[key] = values

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
    
    effective_rank_keys = [k for k in layer_df.columns if 'effective_rank_layer_' in k]
    if effective_rank_keys:
        plt.figure(figsize=(12, 7))
        effective_rank_df_melted = layer_df.melt(id_vars=['Epoch'], value_vars=effective_rank_keys, var_name='Layer', value_name='Effective Rank')
        effective_rank_df_melted['Layer'] = effective_rank_df_melted['Layer'].str.replace('effective_rank_layer_', 'Layer ')
        ax = sns.lineplot(data=effective_rank_df_melted, x='Epoch', y='Effective Rank', hue='Layer', linewidth=2)
        ax.set_title('Effective Rank Evolution', fontsize=16, fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        wandb.log({"plots/effective_ranks": wandb.Image(plt)})
        plt.close()

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
    parser = argparse.ArgumentParser(description="Train a Neural Field for CT Reconstruction.")

    parser.add_argument('--image', default='data/chest.png', type=str, 
                        help='Path to the grayscale phantom image.')
    parser.add_argument('--epochs', default=5000, type=int, 
                        help='Number of training epochs (equivalent to TRAINING_STEPS).')
    parser.add_argument('--log_n_epochs', default=100, type=int, 
                        help='Frequency of logging metrics and images.')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility.')
    parser.add_argument('--log_image_evolution', action='store_true', help='Log intermediate image reconstructions to wandb.')
    parser.add_argument('--project_name', type=str, default='cvpr-ct', help='Wandb project name.')
    parser.add_argument('--create_plots', action='store_true', help='Plot metrics separately using seaborn.')

    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'muon', 'lbfgs'], help='Optimizer to use.')
    parser.add_argument('--lr', default=1e-2, type=float, help='Learning rate for Adam, Muon (aux), and L-BFGS (initial step size).')
    parser.add_argument('--adam_weight_decay', type=float, default=0.0, help='Weight decay for pure Adam optimizer.')
    parser.add_argument('--muon_weight_decay', type=float, default=0.0, help='Weight decay for Muon optimizer (hidden weights).')
    parser.add_argument('--muon_aux_weight_decay', type=float, default=0.0, help='Weight decay for auxiliary Adam in Muon.')
    parser.add_argument('--muon_lr', type=float, default=5e-2, help='Learning rate for the Muon part (hidden weights).')
    parser.add_argument('--optimize_first_layer_with_muon', action='store_true', help='For models with embeddings (FFN, PosEnc), also optimize the first linear layer of the MLP with Muon.')
    parser.add_argument('--lbfgs_max_iter', type=int, default=20, help='Max iterations per epoch for L-BFGS.')
    parser.add_argument('--lbfgs_history_size', type=int, default=100, help='History size for L-BFGS.')

    parser.add_argument('--scheduler', type=str, default='none', choices=['none', 'cosine', 'step'], help='Learning rate scheduler type (not used with L-BFGS).')
    parser.add_argument('--T_max', type=int, default=5000, help='T_max for CosineAnnealingLR scheduler (matched to default epochs).')
    parser.add_argument('--step_size', type=int, default=500, help='Step size for StepLR scheduler.')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma for StepLR scheduler.')

    parser.add_argument('--model', choices=['relu_mlp', 'relu_ffn', 'real_wire','gauss_ffn', 'gauss_mlp', 'siren_mlp', 'wire_mlp', 'relu_pos_enc', 'replicate_mlp', 'finer_mlp', 'fourier_net'], default='relu_ffn')
    parser.add_argument('--num_layers', default=5, type=int, help='Number of hidden layers in the MLP.')
    parser.add_argument('--hidden_dim', default=256, type=int, help='Dimension of hidden layers.')
    parser.add_argument('--mapping_size', default=64, type=int, help='Mapping size for Fourier Feature mappings.')
    
    parser.add_argument('--fourier_sigma', default=4.0, type=float, help='Sigma for Fourier Feature mapping.')
    parser.add_argument('--siren_omega', default='30.0', type=str, help="Omega for SIREN layers.")
    parser.add_argument('--finer_omega', default='30.0', type=str, help="Omega for FINER layers.")
    parser.add_argument('--finer_init_bias', action="store_true",  help="Initial bias for FINER layers.")
    parser.add_argument('--finer_bias_scale', default=float(1/math.sqrt(2)), type=float, help="Bias scale for FINER layers.")
    parser.add_argument('--gauss_scale', default='0.05', type=str, help='Scale parameter for Gaussian activation.')
    parser.add_argument('--wire_sigma', default='10.0', type=str, help="Sigma for WIRE layers.")
    parser.add_argument('--wire_omega', default='10.0', type=str, help="Omega for WIRE layers.")
    parser.add_argument('--mfn_fourier_scale', default=256, type=float, help='Scale for MFN_FourierNet model.')
    
    parser.add_argument('--num_projections', default=100, type=int, 
                        help='Number of projection angles for the sinogram.')
    parser.add_argument('--use_noise', action='store_true', 
                        help='Add realistic noise to the sinogram.')
    parser.add_argument('--noise_snr', default=2.0, type=float, 
                        help='Readout noise SNR.')
    parser.add_argument('--noise_tau', default=3e1, type=float, 
                        help='Photon noise tau parameter.')

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    args.gauss_scale = parse_list(args.gauss_scale, args.num_layers)
    args.siren_omega = parse_list(args.siren_omega, args.num_layers)
    args.finer_omega = parse_list(args.finer_omega, args.num_layers)
    args.wire_sigma = parse_list(args.wire_sigma, args.num_layers)
    args.wire_omega = parse_list(args.wire_omega, args.num_layers)

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Phantom file not found: {args.image}")

    wandb.init(
        project=args.project_name,
        config=vars(args)
    )

    model, metrics, layer_metrics, original_img, lpips_model = train_model(args, device)
    model.eval()
    with torch.no_grad():
        h, w, c = original_img.shape
        coords = get_coordinates(h, w).to(device) # Use device here
        final_pred_tensor = model(coords).view(h, w, c) # (H, W, 1)
        
        final_clamped_pred_tensor = torch.clamp(final_pred_tensor, min=0.0, max=1.0)
        final_pred_np = final_clamped_pred_tensor.cpu().numpy()
        original_img_np = original_img.cpu().numpy() # (H, W, 1)
        
        # Calculate final metrics
        final_psnr = psnr(final_clamped_pred_tensor, original_img.to(device)).item()
        # ssim needs (H, W)
        final_ssim = ssim(original_img_np.squeeze(), final_pred_np.squeeze(), data_range=1.0)
        # lpips needs (H, W, 1) vs (H, W, 1)
        final_lpips = compute_lpips(original_img_np, final_pred_np, 
                                  lpips_model, device)
        
        print("\n--- Final Results ---")
        print(f"Final Full Image PSNR: {final_psnr:.2f}")
        print(f"Final Full Image SSIM: {final_ssim:.4f}")
        print(f"Final Full Image LPIPS: {final_lpips:.4f}")
        
        final_comparison_fig = create_comparison_image(
            original_img_np, final_pred_np, args.epochs, final_psnr, final_ssim, final_lpips)
        
        log_dict = {
            'final_full_psnr': final_psnr,
            'final_ssim': final_ssim,
            'final_lpips': final_lpips,
            'images/final_comparison': wandb.Image(final_comparison_fig, 
                                                 caption=f'Final Result: PSNR={final_psnr:.2f}, SSIM={final_ssim:.4f}, LPIPS={final_lpips:.4f}'),                                
            'images/final_reconstruction': wandb.Image(np.clip(final_pred_np, 0, 1))
        }
            
        wandb.log(log_dict)
        plt.close(final_comparison_fig)
        
    if args.create_plots:
        plot_metrics_seaborn_separate(metrics, layer_metrics, args)
    
    wandb.finish()
    print("Training finished and results logged to wandb.")

if __name__ == "__main__":
    main()
