#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import h5py
import argparse
from tqdm import tqdm

# =======================================================
# 1) Import shared utility functions
# =======================================================
from wcnc_utils import (
    load_system_params,
    ISACDatasetSionna,
    initial_rainbow_beam_ULA_YOLO,
    compute_uplink_signal_torch,
    pairwise_l2_torch,
    knn_predict_torch,
    count_parameters,
)

def calculate_knn_metrics(pos_est, x_gt, y_gt, phi_gt, r_gt):
    """Compute sum of squared errors for k-NN evaluation."""
    x_est, y_est = pos_est[:, 0], pos_est[:, 1]
    r_est = torch.sqrt(x_est**2 + y_est**2)
    phi_est = torch.rad2deg(torch.atan2(y_est, x_est))
    dist_sq_error = (x_est - x_gt)**2 + (y_est - y_gt)**2
    phi_sq_error = (phi_est - phi_gt)**2
    r_sq_error = (r_est - r_gt)**2
    return torch.sum(dist_sq_error), torch.sum(phi_sq_error), torch.sum(r_sq_error)




# =======================================================
# 2) Network model definitions
# =======================================================
class RainbowBeamModel(nn.Module):
    """Physical beamforming model for evaluation (non-trainable)."""
    def __init__(self, PS_init, TTD_init):
        super().__init__()
        self.PS = nn.Parameter(PS_init.clone().detach(), requires_grad=False)
        self.TTD = nn.Parameter(TTD_init.clone().detach(), requires_grad=False)

    def forward(self, H, fm_list, pt_scaling_factor, add_noise=False, noise_std_dev=None):
        B = H.shape[0]
        ps_reshaped = self.PS.view(1, -1)
        ttd_reshaped = self.TTD.view(1, -1)
        
        PS_expanded = ps_reshaped.expand(B, -1)
        TTD_expanded = 1e-9 * ttd_reshaped.expand(B, -1)
        
        clean_echo = compute_uplink_signal_torch(H, PS_expanded, TTD_expanded, fm_list)
        scaled_echo = clean_echo * pt_scaling_factor
        
        final_echo = scaled_echo
        if add_noise and noise_std_dev is not None and noise_std_dev.item() > 0:
            noise = (torch.randn_like(scaled_echo.real) + 1j * torch.randn_like(scaled_echo.imag)) * noise_std_dev.to(scaled_echo.device)
            final_echo = scaled_echo + noise

        mag_sq = torch.abs(final_echo)**2
        mag_sq_db = 10 * torch.log10(mag_sq + 1e-20)
        return mag_sq_db

# =======================================================
# 3) Main flow
# =======================================================
def main():
    # --- 1. Parameter and path setup ---
    parser = argparse.ArgumentParser(description='Evaluate position estimation using k-NN on GPU.')
    parser.add_argument('--data_root', type=str, default='sionna_1008/los', help='Root directory of the dataset.')
    parser.add_argument('--pt_dbm', type=float, default=13.0, help='Transmit power in dBm.')
    parser.add_argument('--k_neighbors', type=int, default=5, help='Number of neighbors for k-NN.')
    parser.add_argument('--add_noise', action='store_true', help='Flag to add thermal noise to the signal.')
    parser.add_argument('--load_weights', type=str, default=None, help='Path to a .pt file containing pre-trained PS and TTD weights.')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU index to use.')
    args = parser.parse_args()
    
    wall_type = os.path.basename(args.data_root)
    TRAIN_DATA_ROOT = os.path.join(args.data_root, 'train')
    FIGURE_DIR = './figure_knn'
    os.makedirs(os.path.join(FIGURE_DIR, wall_type), exist_ok=True)
    
    dis_max, batch_size = 200, 256
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load system parameters and data ---
    PARAM_FILE = os.path.join(TRAIN_DATA_ROOT, 'system_params.npz')
    Ns, Nt, num_subcarriers, f0, c, f_scs, phi_start, phi_end, user_height, BS_height = load_system_params(PARAM_FILE)
    BW = f_scs * num_subcarriers; fc = f0 + BW / 2; d = c / fc / 2
    fm_list = torch.from_numpy((f0 + f_scs * np.arange(num_subcarriers)).astype(np.float32)).to(device)

    full_dataset = ISACDatasetSionna(data_root=TRAIN_DATA_ROOT)
    
    n_samples = len(full_dataset)
    train_size = int(0.8 * n_samples)
    val_size = int(0.1 * n_samples)
    test_size = n_samples - train_size - val_size

    print(f"Dataset split: total={n_samples}, train={train_size}, val={val_size}, test={test_size}")
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # --- 3. Compute power and noise ---
    pt_scaling_factor = torch.tensor(np.sqrt(10**(args.pt_dbm / 10.0)), dtype=torch.float32, device=device)
    noise_std_dev = None
    noise_status_str = "Clean"
    if args.add_noise:
        noise_power_mw = 1.38e-23 * 290.0 * BW * 1000
        noise_std_dev = torch.tensor(np.sqrt(noise_power_mw / 2.0), dtype=torch.float32, device=device)
        noise_status_str = f"Noisy (Pt={args.pt_dbm}dBm)"
        print(f"Physical noise enabled. Transmit power: {args.pt_dbm} dBm.")
    else:
        print("No-noise evaluation.")

    # --- 4. Initialize or load beam weights ---
    if args.load_weights:
        print(f"Loading beamforming weights from: {args.load_weights}")
        if not os.path.exists(args.load_weights): raise FileNotFoundError(f"Weights not found: {args.load_weights}")
        weights = torch.load(args.load_weights, map_location=device, weights_only=True)
        PS_init, TTD_init = weights['PS'].to(device), weights['TTD'].to(device)
        print("Weights loaded.")
    else:
        print("No weight file provided, computing initial rainbow beam weights...")
        PS_init, TTD_init = initial_rainbow_beam_ULA_YOLO(Nt, d, BW, f_scs, fm_list, phi_start, phi_end)
        print("Weights initialized.")
    
    model = RainbowBeamModel(PS_init=PS_init, TTD_init=TTD_init).to(device)
    model.eval()

    # --- 5. Build k-NN codebook ---
    print(f"\n-- Building k-NN codebook ({noise_status_str}) --")
    train_features, train_labels = [], []
    with torch.no_grad():
        for batch_data in tqdm(train_loader, desc="Building codebook..."):
            H = batch_data['H'].to(device)
            mag_sq_db = model(H, fm_list, pt_scaling_factor, args.add_noise, noise_std_dev)
            train_features.append(mag_sq_db.cpu().numpy())
            train_labels.append(torch.stack([batch_data['x_gt'], batch_data['y_gt']], dim=1).numpy())
    X_train_np = np.concatenate(train_features, axis=0)
    y_train_np = np.concatenate(train_labels, axis=0)
    
    # Load codebook onto GPU
    print("Loading codebook to GPU...")
    X_train_gpu = torch.from_numpy(X_train_np).to(device)
    y_train_gpu = torch.from_numpy(y_train_np).to(device)
    
    codebook_bytes = X_train_gpu.element_size() * X_train_gpu.nelement() + y_train_gpu.element_size() * y_train_gpu.nelement()
    codebook_mb = codebook_bytes / (1024**2)
    print(f"Codebook ready: features {X_train_gpu.shape}, labels {y_train_gpu.shape}")
    print(f"Codebook GPU memory usage: {codebook_mb:.2f} MB")

    # --- 6. Evaluate on test set ---
    print(f"\n-- Evaluating k-NN on test set (k={args.k_neighbors}, {noise_status_str}) --")
    all_pos_est, all_pos_gt, all_phi_gt, all_r_gt = [], [], [], []
    total_dist_sq, total_phi_sq, total_r_sq = 0.0, 0.0, 0.0
    
    inference_start_time = time.perf_counter()
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Testing..."):
            H = batch_data['H'].to(device)
            x_gt = batch_data['x_gt'].to(device)
            y_gt = batch_data['y_gt'].to(device)
            phi_gt = batch_data['phi_gt'].to(device)
            r_gt = batch_data['r_gt'].to(device)
            
            # (1) compute query features on GPU
            mag_sq_db_query = model(H, fm_list, pt_scaling_factor, args.add_noise, noise_std_dev)
            
            # (2) perform k-NN on GPU
            pos_est = knn_predict_torch(X_train_gpu, y_train_gpu, mag_sq_db_query, k=args.k_neighbors)
            
            # (3) Compute metrics on GPU
            dist_sq, phi_sq, r_sq = calculate_knn_metrics(pos_est, x_gt, y_gt, phi_gt, r_gt)
            total_dist_sq += dist_sq.item()
            total_phi_sq += phi_sq.item()
            total_r_sq += r_sq.item()
            
            # (4) Move results back to CPU for plotting
            all_pos_est.append(pos_est.cpu())
            all_pos_gt.append(torch.stack([x_gt.cpu(), y_gt.cpu()], dim=1))
            all_phi_gt.append(phi_gt.cpu())
            all_r_gt.append(r_gt.cpu())
            
    inference_end_time = time.perf_counter()

    # --- 7. Compute and print final metrics ---
    num_test_samples = len(test_dataset)
    dist_rmse = np.sqrt(total_dist_sq / num_test_samples)
    phi_rmse = np.sqrt(total_phi_sq / num_test_samples)
    r_rmse = np.sqrt(total_r_sq / num_test_samples)
    
    total_inference_time_s = inference_end_time - inference_start_time
    avg_latency_ms = (total_inference_time_s * 1000) / num_test_samples

    print("\n" + "="*50)
    print("                   k-NN (GPU) Final Results")
    print("="*50)
    print(f" Dist RMSE : {dist_rmse:.4f} m")
    print(f" Angle RMSE: {phi_rmse:.4f} °")
    print(f" Range RMSE: {r_rmse:.4f} m")
    print("-" * 50)
    print(f" Codebook GPU memory: {codebook_mb:.2f} MB")
    print(f" Total inference time: {total_inference_time_s:.2f} s ({num_test_samples} samples)")
    print(f" Avg latency per sample: {avg_latency_ms:.4f} ms")
    print("="*50 + "\n")

    # --- 8. Plotting ---
    pos_est_all = torch.cat(all_pos_est).numpy()
    r_gt_all = torch.cat(all_r_gt).numpy()
    phi_gt_all = torch.cat(all_phi_gt).numpy()
    r_est_all = np.sqrt(pos_est_all[:, 0]**2 + pos_est_all[:, 1]**2)
    phi_est_all = np.rad2deg(np.arctan2(pos_est_all[:, 1], pos_est_all[:, 0]))

    weight_status = "loaded" if args.load_weights else "initial"
    # NOTE: update filename and title
    filename_suffix = f"GPU_{wall_type}_{noise_status_str}_k{args.k_neighbors}_{weight_status}_weights".replace(' ', '')
    title_suffix = f"(GPU k-NN, {wall_type.capitalize()}, {noise_status_str}, k={args.k_neighbors})"

    plt.figure(figsize=(8, 8))
    plt.scatter(r_gt_all, r_est_all, alpha=0.5, s=10)
    plt.plot([0, dis_max], [0, dis_max], 'r--', label='Ideal (y=x)')
    plt.title(f'Range Estimation\n{title_suffix}\nRMSE: {r_rmse:.4f} m')
    plt.xlabel('Ground Truth Range (m)'); plt.ylabel('Predicted Range (m)')
    plt.grid(True); plt.legend(); plt.axis('equal'); plt.xlim(0, dis_max); plt.ylim(0, dis_max); plt.tight_layout()
    range_fig_path = os.path.join(FIGURE_DIR, wall_type, f'range_{filename_suffix}.png')
    plt.savefig(range_fig_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.scatter(phi_gt_all, phi_est_all, alpha=0.5, s=10)
    plt.plot([-60, 60], [-60, 60], 'r--', label='Ideal (y=x)')
    plt.title(f'Angle Estimation\n{title_suffix}\nRMSE: {phi_rmse:.4f}°')
    plt.xlabel('Ground Truth Angle (°)'); plt.ylabel('Predicted Angle (°)')
    plt.grid(True); plt.legend(); plt.axis('equal'); plt.xlim(-60, 60); plt.ylim(-60, 60); plt.tight_layout()
    angle_fig_path = os.path.join(FIGURE_DIR, wall_type, f'angle_{filename_suffix}.png')
    plt.savefig(angle_fig_path, dpi=200)
    plt.close()
    
    print(f"Evaluation complete, figures saved to: '{os.path.join(FIGURE_DIR, wall_type)}'")

if __name__ == "__main__":
    main()