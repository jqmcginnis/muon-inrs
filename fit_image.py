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

def load_image(path):
    img = Image.open(path).convert('RGB')
    # just for debugging with smaller images, as in SIREN demo code
    # Resize image to 256x256, for checking quality metrics against official SIREN implementation
    # img = img.resize((256, 256))#, Image.LANCZOS)
    img = np.array(img) / 255.0
    return torch.tensor(img, dtype=torch.float32)

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

def get_coordinates(h, w):
    x = torch.linspace(-1, 1, w)
    y = torch.linspace(-1, 1, h)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
    coords = torch.stack([grid_x.flatten(),grid_y.flatten()], dim=-1) 
    return coords

def psnr(img1, img2, max_val=1.0):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        img1 (torch.Tensor): The first image tensor.
        img2 (torch.Tensor): The second image tensor.
        max_val (float): The maximum possible pixel value of the images 
                         (e.g., 1.0 for normalized floats, 255.0 for 8-bit images).
    """
    # Calculate Mean Squared Error (MSE)
    mse = torch.mean((img1 - img2) ** 2)
    
    # Handle the case where images are identical (MSE=0)
    if mse == 0:
        return float('inf')
    
    psnr_val = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr_val

def compute_lpips(img1, img2, lpips_model, device):
    """Compute LPIPS between two images"""
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2).float()
    
    img1 = (img1.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0).to(device)
    img2 = (img2.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0).to(device)
    
    with torch.no_grad():
        lpips_val = lpips_model(img1, img2)
    
    return lpips_val.item()

def create_comparison_image(gt_img, recon_img, epoch, psnr_val, ssim_val, lpips_val):
    """Create side-by-side comparison of ground truth and reconstructed image"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(gt_img)
    axes[0].set_title('Ground Truth', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(np.clip(recon_img, 0, 1))
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

def train_model(args):
    """Train the neural field model on the given image with specified parameters."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    lpips_model = lpips.LPIPS(net='alex').to(device)
    
    img = load_image(args.image)
    h, w, c = img.shape
 
    coords = get_coordinates(h, w)
    target = img.view(-1, c)
    
    is_inpainting_task = args.inpainting_ratio < 1.0
    
    if is_inpainting_task:
        print(f"Performing inpainting task. Training on {args.inpainting_ratio * 100:.2f}% of pixels.")
        num_pixels = h * w
        num_train_pixels = int(num_pixels * args.inpainting_ratio)
        
        indices = torch.randperm(num_pixels)
        train_indices = indices[:num_train_pixels]
        test_indices = indices[num_train_pixels:]
        
        train_coords = coords[train_indices].to(device)
        train_target = target[train_indices].to(device)
        
        test_coords = coords[test_indices].to(device)
        test_target = target[test_indices].to(device)
        
        mask = torch.ones(num_pixels, 1) * 0.2 # just for visualization
        mask[train_indices] = 1.0
        mask_img = mask.view(h, w, 1).cpu().numpy()
        wandb.log({"images/training_mask": wandb.Image(mask_img, caption="Training Pixel Mask (White = Known)")})
    else:
        print("Performing overfitting task on all pixels.")
        train_coords = coords.to(device)
        train_target = target.to(device)

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
    elif args.model == 'real_wire':
        model = WireRealMLP(input_dim=2, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers, omega=args.wire_omega, sigma=args.wire_sigma)
    elif args.model == 'finer_mlp':
        model = FinerMLP(input_dim=2, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers, omega=args.finer_omega, 
                        init_bias=args.finer_init_bias, bias_scale=args.finer_bias_scale)
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
    # Schedulers are not typically used with second order optimizers such as L-BFGS
    if args.optimizer != 'lbfgs': 
        if args.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=1e-6)
            print(f"Using CosineAnnealingLR scheduler with T_max={args.T_max}")
        elif args.scheduler == 'step':
            scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
            print(f"Using StepLR scheduler with step_size={args.step_size} and gamma={args.gamma}")
    else:
        print("INFO: Schedulers are disabled for L-BFGS optimizer.")
    
    metrics = {
        'epochs': [], 'full_psnr': [], 'ssim': [], 'lpips': [],
        'train_psnr': [], 'test_psnr': []
    }
    layer_metrics = {}
    
    gt_img_np = img.numpy()
    # omit caption for ground truth
    wandb.log({"images/ground_truth": wandb.Image(gt_img_np)})

    # stable rank etc
    layer_metrics = {}
    
    for epoch in range(args.epochs):
        model.train()
        
        if args.optimizer == 'lbfgs':
            # L-BFGS requires a closure function
            def closure():
                optimizer.zero_grad()
                pred = model(train_coords)
                loss = F.mse_loss(pred, train_target)
                loss.backward()
                return loss
            
            optimizer.step(closure)
            # Re-calculate loss for logging after the step
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
                # calculate metrics on the full image for SSIM, LPIPS, etc.
                full_pred = model(coords.to(device))
                pred_img = full_pred.view(h, w, c).cpu().numpy()
                target_img = img.numpy()
                
                full_psnr_val = psnr(torch.tensor(pred_img), torch.tensor(target_img)).item()
                ssim_val = ssim(target_img, pred_img, channel_axis=-1, data_range=1.0)
                lpips_val = compute_lpips(target_img, pred_img, lpips_model, device)
                
                # B. Calculate PSNR on train/test subsets
                train_psnr_val = psnr(model(train_coords), train_target).item()
                
                test_psnr_val = 0.0
                if is_inpainting_task:
                    test_psnr_val = psnr(model(test_coords), test_target).item()

                metrics['epochs'].append(epoch)
                metrics['full_psnr'].append(full_psnr_val)
                metrics['ssim'].append(ssim_val)
                metrics['lpips'].append(lpips_val)
                metrics['train_psnr'].append(train_psnr_val)
                metrics['test_psnr'].append(test_psnr_val)
                
                if is_inpainting_task:
                    print(f"Epoch {epoch:4d}: Loss={loss.item():.6f}, TrainPSNR={train_psnr_val:.2f}, TestPSNR={test_psnr_val:.2f}, FullPSNR={full_psnr_val:.2f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}")
                else:
                    print(f"Epoch {epoch:4d}: Loss={loss.item():.6f}, PSNR={full_psnr_val:.2f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}")
                
                log_dict = {
                    'epoch': epoch,
                    'loss': loss.item(),
                    'full_psnr': full_psnr_val,
                    'ssim': ssim_val,
                    'lpips': lpips_val,
                    'train_psnr': train_psnr_val,
                }

                if args.optimizer == 'muon' and wandb.config.optimizer_type == 'muon': # Check it didn't fall back to Adam
                    log_dict['learning_rate_muon'] = optimizer.param_groups[0]['lr']
                    log_dict['learning_rate_aux'] = optimizer.param_groups[1]['lr']
                else:
                    log_dict['learning_rate'] = optimizer.param_groups[0]['lr']

                if is_inpainting_task:
                    log_dict['test_psnr'] = test_psnr_val
                
                if args.log_image_evolution:
                    comparison_fig = create_comparison_image(target_img, pred_img, epoch, full_psnr_val, ssim_val, lpips_val)
                    log_dict.update({
                        'images/comparison': wandb.Image(comparison_fig, caption=f'Epoch {epoch}: PSNR={full_psnr_val:.2f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}'),
                        'images/reconstruction': wandb.Image(np.clip(pred_img, 0, 1), caption=f'Reconstruction at epoch {epoch}')
                    })
                    plt.close(comparison_fig)
                
                if hasattr(model, 'get_detailed_matrix_info'):
                    info = model.get_detailed_matrix_info()
                    for i, layer_info in enumerate(info['layer_infos']):
                        # Initialize lists in dictionary if they don't exist
                        if f'stable_rank_layer_{i}' not in layer_metrics:
                            layer_metrics[f'stable_rank_layer_{i}'] = []
                            layer_metrics[f'effective_rank_layer_{i}'] = []  # <-- ADD THIS
                            layer_metrics[f'spectral_norm_layer_{i}'] = []
                            layer_metrics[f'condition_number_layer_{i}'] = []

                        # Get scalar values
                        stable_rank_val = layer_info.get('stable_rank', 0)
                        effective_rank_val = layer_info.get('effective_rank', 0)  # <-- ADD THIS
                        spectral_norm_val = layer_info.get('linear_spectral_norm', 0)
                        condition_number_val = layer_info.get('spectral_condition_no', 0)

                        # Append to our history dictionary
                        layer_metrics[f'stable_rank_layer_{i}'].append(stable_rank_val)
                        layer_metrics[f'effective_rank_layer_{i}'].append(effective_rank_val)  # <-- ADD THIS
                        layer_metrics[f'spectral_norm_layer_{i}'].append(spectral_norm_val)
                        layer_metrics[f'condition_number_layer_{i}'].append(condition_number_val)

                        # Add to the current wandb log dictionary
                        log_dict[f'stable_rank/layer_{i}'] = stable_rank_val
                        log_dict[f'effective_rank/layer_{i}'] = effective_rank_val  # <-- ADD THIS
                        log_dict[f'spectral_norm/layer_{i}'] = spectral_norm_val
                        log_dict[f'condition_number/layer_{i}'] = condition_number_val
                    
                    if 'end_to_end_spectral_bound' in info:
                         if 'end_to_end_bound' not in layer_metrics:
                            layer_metrics['end_to_end_bound'] = []
                         end_to_end_val = info['end_to_end_spectral_bound']
                         layer_metrics['end_to_end_bound'].append(end_to_end_val)
                         log_dict['end_to_end_bound'] = end_to_end_val

                wandb.log(log_dict)

    return model, metrics, layer_metrics, img, lpips_model

def plot_metrics_seaborn_separate(metrics, layer_metrics, args):
    """Plot training metrics using seaborn and log to wandb."""
    sns.set_style("darkgrid")
    
    is_inpainting_task = args.inpainting_ratio < 1.0
    
    main_df = pd.DataFrame({
        'Epoch': metrics['epochs'],
        'Full PSNR': metrics['full_psnr'],
        'SSIM': metrics['ssim'],
        'LPIPS': metrics['lpips'],
        'Train PSNR': metrics['train_psnr']
    })
    if is_inpainting_task:
        main_df['Test PSNR'] = metrics['test_psnr']

    plt.figure(figsize=(10, 6))
    if is_inpainting_task:
        psnr_df = main_df.melt(id_vars=['Epoch'], value_vars=['Train PSNR', 'Test PSNR', 'Full PSNR'],
                               var_name='Metric', value_name='PSNR (dB)')
        ax = sns.lineplot(data=psnr_df, x='Epoch', y='PSNR (dB)', hue='Metric')
        ax.set_title('PSNR Over Training (Inpainting)', fontsize=16, fontweight='bold', pad=20)
    else:
        ax = sns.lineplot(data=main_df, x='Epoch', y='Full PSNR')
        ax.set_title('PSNR Over Training (Overfitting)', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('PSNR (dB)', fontsize=14)
    ax.set_xlabel('Epoch', fontsize=14)
    plt.tight_layout()
    wandb.log({"plots/psnr_comparison": wandb.Image(plt)})
    plt.close()

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
    
    if not layer_metrics:
        print("No layer metrics found to plot.")
        sns.reset_defaults()
        return

    layer_df = pd.DataFrame({'Epoch': metrics['epochs']})
    for key, values in layer_metrics.items():
        layer_df[key] = values

    # 3. Stable Ranks Plot
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

    # 4. Effective Ranks Plot  
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
    parser = argparse.ArgumentParser(description="Train a Neural Field for Image Overfitting or Inpainting.")

    # General arguments
    parser.add_argument('--image', default='data/kodim01.png', type=str, help='Path to the input image.')
    parser.add_argument('--epochs', default=5000, type=int, help='Number of training epochs.')
    parser.add_argument('--log_n_epochs', default=500, type=int, help='Frequency of logging metrics and images.')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility.')
    parser.add_argument('--log_image_evolution', action='store_true', help='Log intermediate image reconstructions to wandb.')
    parser.add_argument('--project_name', type=str, default='cvpr-images', help='Wandb project name.')

    # Optimizer arguments
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'muon', 'lbfgs'], help='Optimizer to use.')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate for Adam, Muon (aux), and L-BFGS (initial step size).')
    parser.add_argument('--adam_weight_decay', type=float, default=0.0, help='Weight decay for pure Adam optimizer.')

    # Muon specific
    parser.add_argument('--muon_weight_decay', type=float, default=0.0, help='Weight decay for Muon optimizer (hidden weights).')
    parser.add_argument('--muon_aux_weight_decay', type=float, default=0.0, help='Weight decay for auxiliary Adam in Muon.')
    parser.add_argument('--muon_lr', type=float, default=1e-1, help='Learning rate for the Muon part (hidden weights).')
    parser.add_argument('--optimize_first_layer_with_muon', action='store_true', help='For models with embeddings (FFN, PosEnc), also optimize the first linear layer of the MLP with Muon.')
    
    # L-BFGS specific
    parser.add_argument('--lbfgs_max_iter', type=int, default=20, help='Max iterations per epoch for L-BFGS.')
    parser.add_argument('--lbfgs_history_size', type=int, default=100, help='History size for L-BFGS.')

    # Scheduler arguments
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['none', 'cosine', 'step'], help='Learning rate scheduler type (not used with L-BFGS).')
    parser.add_argument('--T_max', type=int, default=5000, help='T_max for CosineAnnealingLR scheduler.')
    parser.add_argument('--step_size', type=int, default=500, help='Step size for StepLR scheduler.')
    parser.add_argument('--gamma', type=float, default=0.9, help='Gamma for StepLR scheduler.')

    # Model and architecture arguments
    parser.add_argument('--model', choices=['relu_mlp', 'relu_ffn', 'relu_hash', 'gauss_ffn', 'gauss_mlp', 'siren_mlp', 'wire_mlp', 'real_wire', 'relu_pos_enc', 'replicate_mlp', 'finer_mlp', 'fourier_net'], default='relu_mlp')
    parser.add_argument('--num_layers', default=5, type=int, help='Number of hidden layers in the MLP.')
    parser.add_argument('--hidden_dim', default=300, type=int, help='Dimension of hidden layers.')
    parser.add_argument('--mapping_size', default=128, type=int, help='Mapping size for Fourier Feature mappings.')
    
    # Model-specific activation arguments
    parser.add_argument('--fourier_sigma', default=10.0, type=float, help='Sigma for Fourier Feature mapping.')
    parser.add_argument('--siren_omega', default='40.0,40.0,40.0,40.0', type=str, help="Omega for SIREN layers.")
    parser.add_argument('--finer_omega', default='40.0,40.0,40.0,40.0', type=str, help="Omega for FINER layers.")
    parser.add_argument('--finer_init_bias', action="store_true",  help="Initial bias for FINER layers.")
    parser.add_argument('--finer_bias_scale', default=float(1/math.sqrt(2)), type=float, help="Bias scale for FINER layers.")
    parser.add_argument('--gauss_scale', default='0.0236', type=str, help='Scale parameter for Gaussian activation.')
    parser.add_argument('--wire_sigma', default='10.0', type=str, help="Sigma for WIRE layers.")
    parser.add_argument('--wire_omega', default='20.0', type=str, help="Omega for WIRE layers.")
    parser.add_argument('--mfn_fourier_scale', default=256, type=float, help='Scale for MFN_FourierNet model.')
    
    # HashGrid specific arguments
    parser.add_argument('--hash_n_levels', default=16, type=int, help='Number of levels in HashGrid.')
    parser.add_argument('--hash_n_features_per_level', default=2, type=int, help='Number of features per level in HashGrid.')
    parser.add_argument('--hash_log2_hashmap_size', default=15, type=int, help='Log2 of hashmap size for HashGrid.')
    parser.add_argument('--hash_base_resolution', default=16, type=int, help='Base resolution for HashGrid.')
    parser.add_argument('--hash_finest_resolution', default=512, type=int, help='Finest resolution for HashGrid.')
    parser.add_argument('--inpainting_ratio', default=1.0, type=float, help='Ratio of pixels for training (1.0 for overfitting, <1.0 for inpainting).')

    args = parser.parse_args()
    set_seed(args.seed)

    # Parse model-specific parameters that can be lists
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

    model, metrics, layer_metrics, original_img, lpips_model = train_model(args)
    
    # Final evaluation and logging
    model.eval()
    with torch.no_grad():
        h, w, c = original_img.shape
        coords = get_coordinates(h, w).to(next(model.parameters()).device)
        final_pred = model(coords).view(h, w, c).cpu()
        
        final_psnr = psnr(final_pred, original_img).item()
        final_ssim = ssim(original_img.numpy(), final_pred.numpy(), channel_axis=-1, data_range=1.0)
        final_lpips = compute_lpips(original_img.numpy(), final_pred.numpy(), 
                                  lpips_model, next(model.parameters()).device)
        
        print("\n--- Final Results ---")
        print(f"Final Full Image PSNR: {final_psnr:.2f}")
        print(f"Final Full Image SSIM: {final_ssim:.4f}")
        print(f"Final Full Image LPIPS: {final_lpips:.4f}")
        
        final_comparison_fig = create_comparison_image(
            original_img.numpy(), final_pred.numpy(), args.epochs, final_psnr, final_ssim, final_lpips)
        
        log_dict = {
            'final_full_psnr': final_psnr,
            'final_ssim': final_ssim,
            'final_lpips': final_lpips,
            'images/final_comparison': wandb.Image(final_comparison_fig, 
                                                 caption=f'Final Result: PSNR={final_psnr:.2f}, SSIM={final_ssim:.4f}, LPIPS={final_lpips:.4f}'),                                
            'images/final_reconstruction': wandb.Image(np.clip(final_pred.numpy(), 0, 1))
        }
            
        wandb.log(log_dict)
        plt.close(final_comparison_fig)

    plot_metrics_seaborn_separate(metrics, layer_metrics, args)
    
    wandb.finish()
    print("Training finished and results logged to wandb.")

if __name__ == "__main__":
    main()