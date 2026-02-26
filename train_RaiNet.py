#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import h5py
import argparse
import random
import copy

# =======================================================
# 1) Import shared utility functions
# =======================================================
from wcnc_utils import (
    count_parameters,
    load_system_params,
    ISACDatasetSionna,
    initial_rainbow_beam_ULA_YOLO,
    compute_uplink_signal_torch,
)

def visualize_beam_patterns(model, fm_list, Nt, d, save_dir, tag):
    """
    Plot beam patterns:
    1. Frequency-Angle energy heatmap
    2. Histogram of beam pointing angles (beam squint)
    """
    model.eval()
    device = fm_list.device
    
    # 1. Prepare data
    PS = model.PS.detach() # [1, Nt]
    TTD = model.TTD.detach() # [1, Nt] (stored as ns)
    
    # Generate Angle Grid
    theta_deg = np.linspace(-90, 90, 361)
    theta_rad = np.deg2rad(theta_deg)
    theta_tensor = torch.tensor(theta_rad, dtype=torch.float32, device=device)
    
    # Frequency
    f0 = fm_list[0]
    freq_diff = fm_list - f0
    c = 3e8
    n_idx = torch.arange(Nt, device=device, dtype=torch.float32) - (Nt - 1)/2
    
    # Expand dimensions for broadcasting [Freq=M, Angle=K, Antenna=N]
    PS_exp = PS.view(1, 1, -1)
    TTD_exp = TTD.view(1, 1, -1) 
    freq_diff_exp = freq_diff.view(-1, 1, 1)
    
    # phase_w = -PS - 2*pi*df*TTD*1e-9
    phase_w = -PS_exp - 2 * np.pi * freq_diff_exp * (TTD_exp * 1e-9)
    
    f_exp = fm_list.view(-1, 1, 1)
    sin_theta = torch.sin(theta_tensor).view(1, -1, 1)
    n_exp = n_idx.view(1, 1, -1)
    
    # phase_ch = 2*pi * f * n * d * sin(theta) / c
    phase_ch = 2 * np.pi * f_exp * n_exp * d * sin_theta / c
    
    # Total Phase & Sum over Antennas
    total_phase = phase_w + phase_ch
    array_factor = torch.sum(torch.exp(1j * total_phase), dim=2) / np.sqrt(Nt)
    
    # Power: [M, K] (Freq, Angle)
    power_spectrum = torch.abs(array_factor)**2
    power_spectrum_np = power_spectrum.cpu().numpy() # [Freq, Angle]
    
    # 3. Plot 1: Heatmap (Frequency vs Angle)
    plt.figure(figsize=(10, 6))
    f_start_ghz = fm_list[0].item() / 1e9
    f_end_ghz = fm_list[-1].item() / 1e9
    
    plt.imshow(power_spectrum_np, aspect='auto', origin='lower', 
               extent=[theta_deg[0], theta_deg[-1], f_start_ghz, f_end_ghz],
               cmap='jet')
    plt.colorbar(label='Normalized Power')
    plt.title(f'Beam Pattern Heatmap (Freq vs Angle) - {tag}')
    plt.xlabel('Angle (deg)')
    plt.ylabel('Frequency (GHz)')
    plt.grid(False)
    
    heatmap_path = os.path.join(save_dir, f'beam_heatmap_{tag}.png')
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Heatmap saved to {heatmap_path}")
    
    # 4. Plot 2: Histogram of Peak Angles (Beam Squint Distribution)
    max_angle_indices = np.argmax(power_spectrum_np, axis=1) # [M]
    peak_angles = theta_deg[max_angle_indices]
    
    plt.figure(figsize=(8, 6))
    plt.hist(peak_angles, bins=60, range=(-60, 60), color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'Histogram of Beam Pointing Angles - {tag}')
    plt.xlabel('Angle (deg)')
    plt.ylabel('Count (Number of Subcarriers)')
    plt.grid(True, alpha=0.3)
    
    hist_path = os.path.join(save_dir, f'beam_angle_hist_{tag}.png')
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"Histogram saved to {hist_path}")

    # 5. Plot 3: Spatial Coverage Map (LoS Approximation) for selected frequencies
    # Select 3 frequencies: Start, Center, End
    f_indices = [0, len(fm_list)//2, len(fm_list)-1]
    f_labels = ['Start', 'Center', 'End']
    
    # Define Spatial Grid
    x_range = np.linspace(0, 100, 200) # 0 to 100m
    y_range = np.linspace(-50, 50, 200) # -50 to 50m
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    
    # Convert to Polar coords for array factor calculation
    R_grid = np.sqrt(X_grid**2 + Y_grid**2)
    Phi_grid = np.arctan2(Y_grid, X_grid) # radians
    
    # Mask out very close range to avoid singularity
    valid_mask = R_grid > 1.0
    
    for f_idx, f_label in zip(f_indices, f_labels):
        f_val = fm_list[f_idx]
        
        # Get weights for this frequency
        # w = exp(j * (-PS - 2*pi*(f-f0)*TTD)) / sqrt(Nt)
        # We need to compute Array Factor: AF(phi) = sum(w_n * exp(j * k * n * d * sin(phi)))
        
        # Calculate phase of weights [1, Nt]
        freq_diff_val = f_val - f0
        phase_w_f = -PS - 2 * np.pi * freq_diff_val * (TTD * 1e-9) # [1, Nt]
        
        # Calculate phase of channel for each grid point [Grid_H, Grid_W, Nt]
        # k * n * d * sin(phi)
        # k = 2*pi*f/c
        k = 2 * np.pi * f_val / c
        
        # n_idx: [Nt] -> [1, 1, Nt]
        n_idx_exp = n_idx.view(1, 1, -1)
        
        # sin(phi): [Grid_H, Grid_W] -> [Grid_H, Grid_W, 1]
        sin_phi_grid = torch.tensor(np.sin(Phi_grid), device=device, dtype=torch.float32).unsqueeze(-1)
        
        # Channel Phase
        phase_ch_grid = k * d * n_idx_exp * sin_phi_grid
        
        # Total Phase: [Grid_H, Grid_W, Nt]
        # phase_w_f: [1, Nt] -> broadcast
        total_phase_grid = phase_w_f.view(1, 1, -1) + phase_ch_grid
        
        # Sum over antennas
        af_grid = torch.sum(torch.exp(1j * total_phase_grid), dim=2) / np.sqrt(Nt)
        
        # Power
        power_grid = torch.abs(af_grid)**2
        
        # Apply Path Loss (1/r^2) for realistic coverage map
        # Or just plot Array Factor pattern projected on space?
        # User asked for "energy received at each point", which implies path loss.
        r_grid_tensor = torch.tensor(R_grid, device=device, dtype=torch.float32)
        received_power = power_grid / (r_grid_tensor**2 + 1e-6)
        
        # Convert to dB
        received_power_db = 10 * torch.log10(received_power + 1e-20)
        received_power_db_np = received_power_db.cpu().numpy()
        
        # Mask invalid
        received_power_db_np[~valid_mask] = np.nan
        
        plt.figure(figsize=(10, 8))
        plt.imshow(received_power_db_np, extent=[0, 100, -50, 50], origin='lower', cmap='jet', aspect='equal')
        plt.colorbar(label='Received Power (dB)')
        plt.title(f'Spatial Beam Coverage ({f_label} Freq: {f_val.item()/1e9:.2f} GHz) - {tag}')
        plt.xlabel('Range X (m)')
        plt.ylabel('Cross-Range Y (m)')
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Mark BS position
        plt.plot(0, 0, 'ko', markersize=8, label='BS')
        plt.legend()
        
        cov_path = os.path.join(save_dir, f'beam_coverage_{f_label}_{tag}.png')
        plt.savefig(cov_path, dpi=300)
        plt.close()
        print(f"Coverage map ({f_label}) saved to {cov_path}")

def loss_ann(pos_est, x_gt, y_gt, phi_gt, r_gt):
    """Localization loss: MSE in Cartesian space + auxiliary polar error stats."""
    x_est, y_est = pos_est[:, 0], pos_est[:, 1]
    dist_sq_error = (x_est - x_gt) ** 2 + (y_est - y_gt) ** 2
    total_loss = torch.mean(dist_sq_error)
    dist_sq_error_sum = torch.sum(dist_sq_error)

    r_est = torch.sqrt(x_est**2 + y_est**2)
    phi_est = torch.rad2deg(torch.atan2(y_est, x_est))
    phi_sq_error_sum = torch.sum((phi_est - phi_gt) ** 2)
    r_sq_error_sum = torch.sum((r_est - r_gt) ** 2)

    return total_loss, dist_sq_error_sum, phi_sq_error_sum, r_sq_error_sum

# =======================================================
# 2) Network model definitions
# =======================================================
class RainbowBeamModel(nn.Module):
    def __init__(self, f_scs, N_az, N_el, PS_init, TTD_init):
        super().__init__()
        self.PS = nn.Parameter(PS_init.clone().detach(), requires_grad=True)
        self.TTD = nn.Parameter(TTD_init.clone().detach(), requires_grad=True)
    def forward(self, H, fm_list, pt_scaling_factor, noise_std_dev):
        PS_expanded, TTD_expanded = self.PS.expand(H.shape[0], -1), 1e-9 * self.TTD.expand(H.shape[0], -1)
        clean_echo = compute_uplink_signal_torch(H, PS_expanded, TTD_expanded, fm_list)
        scaled_echo = clean_echo * pt_scaling_factor
        noise = (torch.randn_like(scaled_echo.real) + 1j * torch.randn_like(scaled_echo.imag)) * noise_std_dev.to(scaled_echo.device)
        # Increase epsilon from 1e-20 to 1e-10 to prevent gradient explosion at deep nulls
        # 10*log10(1e-10) = -100 dB, which is sufficient for noise floor
        mag_sq_db = 10 * torch.log10(torch.abs(scaled_echo + noise)**2 + 1e-10)
        return mag_sq_db, PS_expanded, TTD_expanded

class RaiNet(nn.Module):
    def __init__(self, input_len: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Dropout(0.1),
        )
        with torch.no_grad():
            dummy = torch.zeros(1,1,input_len)
            feat = self.features(dummy)
            flat_dim = feat.numel()

        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Linear(128, 84),
            nn.BatchNorm1d(84),
            nn.Tanh(),
            nn.Linear(84, 2),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.to(torch.float32)
        x = x.unsqueeze(1)
        z = self.features(x)
        z = z.view(z.size(0), -1)
        out = self.classifier(z)
        return out

# =======================================================
# 3) Training and evaluation functions
# =======================================================
def run_training_session(
    session_name, model1, model2, optimizer, scheduler,
    train_loader, val_loader, test_loader, epochs, device,
    fm_list, pt_scaling_factor, noise_std_dev, dis_max,
    train_model1, train_model2, args, training_objective='location'):
    
    print("\n" + "="*60)
    print(f"Starting training session: {session_name} (dataset: {args.wall_type})")
    print(f"Mode: {training_objective.upper()} | Train Model1: {train_model1} | Train Model2: {train_model2}")
    print("="*60 + "\n")

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 20
    dis_str = str(int(dis_max))
    
    for param in model1.parameters(): param.requires_grad = train_model1
    for param in model2.parameters(): param.requires_grad = train_model2

    for epoch in range(epochs):
        print("-" * 50)
        # Set train/eval mode based on whether the model is being updated
        # When frozen, use eval() to use fixed BN statistics and deterministic behavior
        if train_model1:
            model1.train()
        else:
            model1.eval()
            
        if train_model2:
            model2.train()
        else:
            model2.eval()
        
        running_loss, total_dist_sq_error_train, total_phi_sq_error_train, total_r_sq_error_train = 0.0, 0.0, 0.0, 0.0
        for batch_data in train_loader:
            H, x_gt, y_gt, phi_gt, r_gt = (d.to(device) for d in batch_data.values())
            optimizer.zero_grad()
            mag_sq_db, _, _ = model1(H, fm_list, pt_scaling_factor, noise_std_dev)
            
            if training_objective == 'energy':
                # Objective: Maximize total energy
                # mag_sq_db is in dB domain; convert to linear, sum, then take negative log as Loss
                power_linear = 10 ** (mag_sq_db / 10.0)
                total_energy = torch.sum(power_linear, dim=1)
                # Loss = -10 * log10(Avg Energy) -> Minimize this = Maximize Energy
                loss = -torch.mean(10 * torch.log10(total_energy + 1e-16))
                
                # Placeholder: localization errors are not computed in energy mode
                dist_sq_err_sum = torch.tensor(0.0)
                phi_sq_err_sum = torch.tensor(0.0)
                r_sq_err_sum = torch.tensor(0.0)
            else:
                # Objective: localization accuracy
                # [Norm] Normalize dB values to roughly [-1, 1] to stabilize gradients
                pos_coarse = model2(mag_sq_db / 100.0) * dis_max
                loss, dist_sq_err_sum, phi_sq_err_sum, r_sq_err_sum = loss_ann(pos_coarse, x_gt, y_gt, phi_gt, r_gt)
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients in strong LoS scenarios
            # Apply clipping to both Model1 and Model2 parameters with max_norm=1.0
            if train_model1:
                torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=1.0)
            if train_model2:
                torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0)

            optimizer.step()
            
            running_loss += loss.item() * H.size(0)
            total_dist_sq_error_train += dist_sq_err_sum.item()
            total_phi_sq_error_train += phi_sq_err_sum.item()
            total_r_sq_error_train += r_sq_err_sum.item()

        train_len = len(train_loader.dataset)
        avg_loss = running_loss / train_len
        
        if training_objective == 'energy':
            print(f"Epoch[{epoch+1}/{epochs}] [{session_name} Train] Energy Loss: {avg_loss:.4f} (Lower is Better/Higher Energy)")
        else:
            avg_dist_rmse = np.sqrt(total_dist_sq_error_train / train_len)
            avg_phi_rmse = np.sqrt(total_phi_sq_error_train / train_len)
            avg_r_rmse = np.sqrt(total_r_sq_error_train / train_len)
            print(f"Epoch[{epoch+1}/{epochs}] [{session_name} Train] Loss: {avg_loss:.4f}, Dist RMSE: {avg_dist_rmse:.4f}m, Angle RMSE: {avg_phi_rmse:.4f}°, Range RMSE: {avg_r_rmse:.4f}m")

        model1.eval(); model2.eval()
        val_loss, total_dist_sq_error_val, total_phi_sq_error_val, total_r_sq_error_val = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                H, x_gt, y_gt, phi_gt, r_gt = (d.to(device) for d in batch_data.values())
                mag_sq_db, _, _ = model1(H, fm_list, pt_scaling_factor, noise_std_dev)
                
                if training_objective == 'energy':
                    power_linear = 10 ** (mag_sq_db / 10.0)
                    total_energy = torch.sum(power_linear, dim=1)
                    loss = -torch.mean(10 * torch.log10(total_energy + 1e-16))
                    dist_sq_err_sum, phi_sq_err_sum, r_sq_err_sum = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
                else:
                    # [Norm]
                    pos_coarse = model2(mag_sq_db / 100.0) * dis_max
                    loss, dist_sq_err_sum, phi_sq_err_sum, r_sq_err_sum = loss_ann(pos_coarse, x_gt, y_gt, phi_gt, r_gt)
                
                val_loss += loss.item() * H.size(0)
                total_dist_sq_error_val += dist_sq_err_sum.item()
                total_phi_sq_error_val += phi_sq_err_sum.item()
                total_r_sq_error_val += r_sq_err_sum.item()
        
        val_len = len(val_loader.dataset)
        avg_val_loss = val_loss / val_len
        
        if training_objective == 'energy':
            print(f"Epoch[{epoch+1}/{epochs}] [{session_name} Val]   Energy Loss: {avg_val_loss:.4f}")
        else:
            avg_val_dist_rmse = np.sqrt(total_dist_sq_error_val / val_len)
            avg_val_phi_rmse = np.sqrt(total_phi_sq_error_val / val_len)
            avg_val_r_rmse = np.sqrt(total_r_sq_error_val / val_len)
            print(f"Epoch[{epoch+1}/{epochs}] [{session_name} Val]   Loss: {avg_val_loss:.4f}, Dist RMSE: {avg_val_dist_rmse:.4f}m, Angle RMSE: {avg_val_phi_rmse:.4f}°, Range RMSE: {avg_val_r_rmse:.4f}m")
        
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            print(f"New best model found (Val Loss: {best_val_loss:.4f}), saving models...")
            torch.save(model1.state_dict(), os.path.join(args.output_dir, f'best_model1_{session_name}_{args.wall_type}_{dis_str}.pt'))
            torch.save(model2.state_dict(), os.path.join(args.output_dir, f'best_model2_{session_name}_{args.wall_type}_{dis_str}.pt'))
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"\n[!] Early stopping triggered: {session_name} showed no improvement in {patience} epochs. Ending training session.")
            break
    
    return best_val_loss

def evaluate_and_plot(model1, model2, test_loader, device, fm_list, pt_scaling_factor, noise_std_dev, dis_max, args, final_session_name):
    print("\n" + "="*60)
    print(f"Starting final evaluation (model from {final_session_name})...")
    print("="*60 + "\n")

    model1.eval(); model2.eval()
    all_r_gt, all_r_est, all_phi_gt, all_phi_est = [], [], [], []
    total_dist_sq_error_test, total_phi_sq_error_test, total_r_sq_error_test = 0.0, 0.0, 0.0
    
    latencies = []
    is_cuda = device.type == 'cuda'

    with torch.no_grad():
        for batch_data in test_loader:
            H, x_gt, y_gt, phi_gt, r_gt = (d.to(device) for d in batch_data.values())

            if is_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_time = time.perf_counter()

            mag_sq_db, _, _ = model1(H, fm_list, pt_scaling_factor, noise_std_dev)
            # [Norm]
            pos_coarse = model2(mag_sq_db / 100.0) * dis_max
            
            if is_cuda:
                end_event.record()
                torch.cuda.synchronize()
                batch_latency_ms = start_event.elapsed_time(end_event)
            else:
                end_time = time.perf_counter()
                batch_latency_ms = (end_time - start_time) * 1000
            
            latencies.append(batch_latency_ms)

            x_est, y_est = pos_coarse[:, 0], pos_coarse[:, 1]
            r_est, phi_est = torch.sqrt(x_est**2 + y_est**2), torch.rad2deg(torch.atan2(y_est, x_est))

            all_r_gt.extend(r_gt.cpu().numpy()); all_r_est.extend(r_est.cpu().numpy())
            all_phi_gt.extend(phi_gt.cpu().numpy()); all_phi_est.extend(phi_est.cpu().numpy())

            dist_sq_error = (x_est - x_gt) ** 2 + (y_est - y_gt) ** 2
            total_dist_sq_error_test += torch.sum(dist_sq_error).item()
            total_phi_sq_error_test += torch.sum((phi_est - phi_gt)**2).item()
            total_r_sq_error_test += torch.sum((r_est - r_gt)**2).item()

    test_len = len(test_loader.dataset)
    test_dist_rmse = np.sqrt(total_dist_sq_error_test / test_len)
    test_phi_rmse = np.sqrt(total_phi_sq_error_test / test_len)
    test_r_rmse = np.sqrt(total_r_sq_error_test / test_len)
    
    avg_latency_batch = np.mean(latencies)
    avg_latency_sample = avg_latency_batch / test_loader.batch_size

    print("-" * 30)
    print("  Final performance metrics")
    print("-" * 30)
    print(f"  Dist RMSE        : {test_dist_rmse:.4f} m")
    print(f"  Angle RMSE       : {test_phi_rmse:.4f} °")
    print(f"  Range RMSE       : {test_r_rmse:.4f} m")
    print(f"  Avg Latency/Batch: {avg_latency_batch:.4f} ms")
    print(f"  Avg Latency/Sample: {avg_latency_sample:.4f} ms")
    print("-" * 30)

    return test_dist_rmse, test_phi_rmse, test_r_rmse

def analyze_param_change(model, prev_params, stage_name):
    """
    Analyze PS/TTD parameter changes and print statistics
    """
    current_ps = model.PS.detach().cpu().numpy()
    current_ttd = model.TTD.detach().cpu().numpy()
    
    print(f"\n>>> [{stage_name}] Parameter statistics and change analysis")
    print(f"{'Param':<5} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10} | {'Delta Mean':<12}")
    print("-" * 70)
    
    # PS Stats
    ps_mean, ps_std = np.mean(current_ps), np.std(current_ps)
    ps_min, ps_max = np.min(current_ps), np.max(current_ps)
    ps_delta = "N/A"
    if prev_params is not None:
        ps_delta = f"{np.mean(np.abs(current_ps - prev_params['PS'])):.4e}"
    print(f"{'PS':<5} | {ps_mean:<10.4f} | {ps_std:<10.4f} | {ps_min:<10.4f} | {ps_max:<10.4f} | {ps_delta:<12}")
    
    # TTD Stats
    ttd_mean, ttd_std = np.mean(current_ttd), np.std(current_ttd)
    ttd_min, ttd_max = np.min(current_ttd), np.max(current_ttd)
    ttd_delta = "N/A"
    if prev_params is not None:
        ttd_delta = f"{np.mean(np.abs(current_ttd - prev_params['TTD'])):.4e}"
    print(f"{'TTD':<5} | {ttd_mean:<10.4f} | {ttd_std:<10.4f} | {ttd_min:<10.4f} | {ttd_max:<10.4f} | {ttd_delta:<12}")
    print("-" * 70 + "\n")
    
    return {'PS': current_ps.copy(), 'TTD': current_ttd.copy()}

def recalibrate_bn(model, loader, device, fm_list, pt_scaling_factor, noise_std_dev, model1_frozen=None):
    """
    Run a forward pass over a portion of data to update BatchNorm running_mean and running_var.
    When Model 1 (Beamformer) is updated, its output distribution changes and frozen Model 2
    (Receiver) BatchNorm stats can become stale. Use this helper to quickly recalibrate BN stats.
    """
    model.train()  # ensure train mode to update BN statistics
    print(f"[Info] Recalibrating BatchNorm statistics...")

    with torch.no_grad():  # no gradients required
        # If recalibrating Model 2, Model 1 must provide inputs
        if model1_frozen is not None:
            model1_frozen.eval()

        for i, batch_data in enumerate(loader):
            # Only a subset is needed; 100 batches is sufficient for stable stats
            if i > 100:
                break

            # [Fix] Obtain data the same way as main loop (first value is H)
            H = next(iter(batch_data.values())).to(device)

            # Compute Model 1 output (input to Model 2)
            if model1_frozen is not None:
                mag_sq_db, _, _ = model1_frozen(H, fm_list, pt_scaling_factor, noise_std_dev)
                # Forward through Model 2 only to update its BN stats
                # [Norm] Keep normalization consistent with training
                _ = model(mag_sq_db / 100.0)
            else:
                # If recalibrating Model 1 (it typically has no BN), keep logic for completeness
                _ = model(H, fm_list, pt_scaling_factor, noise_std_dev)

    print(f"[Info] BN recalibration complete.")

# =======================================================
# 4) Main function
# =======================================================
def main():
    ### Global random seed setup
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[Info] Global random seed set to: {SEED}")

    print("="*60)
    print(">>> Training script version: 2026-01-12_Final_Optimized")
    print(">>> Features:")
    print("    1. [Stabilization] Gradient clipping (Clip Grad Norm = 1.0)")
    print("    2. [Stabilization] BN recalibration (Recalibrate BN at stage switch)")
    print("    3. [Stabilization] Increased log epsilon (1e-20 -> 1e-10)")
    print("    4. [Optimization] Aggressive LR decay for Stage 2 (LR / 5 for Cycle > 1)")
    print("    5. [New] Input normalization (Input / 100.0) to stabilize baseline")
    print("="*60)
    
    # --- 1. Parameter and path setup ---
    parser = argparse.ArgumentParser(description='Train the RaiNet model for position estimation.')
    parser.add_argument('--pt_dbm', type=float, default=23.0, help='Transmit power in dBm.')
    parser.add_argument('--wall_type', type=str, default='los_5_95', help='Type of wall geometry for naming.')
    parser.add_argument('--training_mode', type=str, default='iteration', choices=['joint', 'fixed', 'iteration', 'sequential'], help='Training mode.')
    parser.add_argument('--lr_model1', type=float, default=2e-2, help='Learning rate for model1 (RainbowBeamModel).')
    parser.add_argument('--lr_ttd', type=float, default=2e-2, help="Initial LR for TTD")
    parser.add_argument('--lr_ps', type=float, default=5e-3, help="Initial LR for PS")
    parser.add_argument('--lr_model2', type=float, default=1e-3, help='Learning rate for model2 (RaiNet).')
    parser.add_argument('--gpu_id', type=str, default='1', help='GPU index to use for training.')
    parser.add_argument('--iteration_cycles', type=int, default=3, help='Number of cycles for the iteration training mode.')
    parser.add_argument('--output_dir', type=str, default='./0116output/output', help='Directory to save models.')
    parser.add_argument('--figure_dir', type=str, default='./0116output/figure', help='Directory to save figures.')
    parser.add_argument('--data_root', type=str, default=None, help='Root directory for dataset.')
    args = parser.parse_args()
    
    # args.output_dir, args.figure_dir = './output', './figure'
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.figure_dir, 'range'), exist_ok=True)
    os.makedirs(os.path.join(args.figure_dir, 'angle'), exist_ok=True)
    
    # Visualization folders
    vis_process_dir = os.path.join(args.figure_dir, 'training_process')
    vis_final_dir = os.path.join(args.figure_dir, 'final_result')
    os.makedirs(vis_process_dir, exist_ok=True)
    os.makedirs(vis_final_dir, exist_ok=True)
    
    if args.data_root:
        DATA_ROOT = args.data_root
    else:
        DATA_ROOT = 'sionna_1109/los'
        
    TRAIN_DATA_ROOT = os.path.join(DATA_ROOT, 'train')
    
    dis_max, batch_size, epochs = 200, 256, 100
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 2. Load system parameters and data ---
    PARAM_FILE = os.path.join(TRAIN_DATA_ROOT, 'system_params.npz')
    Ns, Nt, num_subcarriers, f0, c, f_scs, phi_start, phi_end, user_height, BS_height = load_system_params(PARAM_FILE)
    BW = f_scs * num_subcarriers; fc = f0 + BW / 2; d = c / fc / 2
    fm_list = torch.from_numpy((f0 + f_scs * np.arange(num_subcarriers)).astype(np.float32)).to(device)
    
    full_dataset = ISACDatasetSionna(data_root=TRAIN_DATA_ROOT)
    train_size, val_size = int(0.8 * len(full_dataset)), int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
    train_loader, val_loader, test_loader = [DataLoader(ds, batch_size=batch_size, shuffle=is_train, num_workers=4, pin_memory=True) for ds, is_train in zip([train_dataset, val_dataset, test_dataset], [True, False, False])]

    # --- 3. Compute power and noise ---
    data_scaling_factor = full_dataset.scaling_factor
    pt_scaling_factor = torch.tensor(np.sqrt(10**(args.pt_dbm / 10.0)), dtype=torch.float32, device=device)
    noise_power_mw = 1.38e-23 * 290.0 * BW * 1000
    raw_noise_std = np.sqrt(noise_power_mw / 2.0)
    scaled_noise_std = raw_noise_std * data_scaling_factor
    noise_std_dev = torch.tensor(scaled_noise_std, dtype=torch.float32, device=device)
    print(f"[Info] Noise standard deviation (scaled): {scaled_noise_std:.4e} (raw: {raw_noise_std:.4e})")

    # --- 4. Build models ---
    # [Note] Use wide-range initialization (-60 ~ 60) to align with train_beam.py and allow energy-maximization focusing
    phi_init_start = -60.0
    phi_init_end = 60.0
    print(f"[Init] Beam initialization params: phi_start={phi_init_start}, phi_end={phi_init_end} (Wide Range Coverage)")
    
    PS_init, TTD_init = initial_rainbow_beam_ULA_YOLO(Nt, d, BW, f_scs, fm_list, phi_init_start, phi_init_end)
    model1 = RainbowBeamModel(f_scs, Nt, 1, PS_init.reshape(1, -1), TTD_init.reshape(1, -1)).to(device)
    model2 = RaiNet(input_len=num_subcarriers).to(device)

    print("\n" + "="*60)
    print("Model complexity:")
    print(f"  - Model 1 (RainbowBeamModel) Trainable Params: {count_parameters(model1)}")
    print(f"  - Model 2 (RaiNet) Trainable Params          : {count_parameters(model2)}")
    print("="*60)
    
    common_params = {
        'model1': model1, 'model2': model2,
        'train_loader': train_loader, 'val_loader': val_loader, 'test_loader': test_loader, 
        'epochs': epochs, 'device': device, 'fm_list': fm_list, 
        'pt_scaling_factor': pt_scaling_factor, 'noise_std_dev': noise_std_dev, 
        'dis_max': dis_max, 'args': args
    }
    
    final_session_name = "" 
    dis_str = str(int(dis_max))

    # --- 5. Execute according to training mode ---
    if args.training_mode == 'joint':
        final_session_name = "JOINT"
        optimizer = optim.Adam([
            {'params': [model1.PS], 'lr': args.lr_ps},
            {'params': [model1.TTD], 'lr': args.lr_ttd},
            {'params': model2.parameters(), 'lr': args.lr_model2}
        ])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
        run_training_session(session_name=final_session_name, optimizer=optimizer, scheduler=scheduler,
                             train_model1=True, train_model2=True, **common_params)

    elif args.training_mode == 'fixed':
        final_session_name = "FIXED"
        optimizer = optim.Adam(model2.parameters(), lr=args.lr_model2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
        run_training_session(session_name=final_session_name, optimizer=optimizer, scheduler=scheduler,
                             train_model1=False, train_model2=True, **common_params)

    elif args.training_mode == 'iteration':
        prev_params = None
        final_session_name = ""
        
        # Automatic iteration control
        best_cycle_val_loss = float('inf')
        loss_threshold = 1e-5  # threshold to consider loss as improved
        max_cycles = 20        # a large max cycles guard to prevent infinite loops

        print(f"[Info] Iteration mode: automatic early-stop enabled (Max Cycles: {max_cycles}, Threshold: {loss_threshold})")

        # [Added] 0. Initial evaluation and visualization
        print("\n" + "="*40)
        print(">>> [Init] Initial evaluation (Before Training)")
        print("="*40)
        visualize_beam_patterns(model1, fm_list, Nt, d, vis_process_dir, tag="INITIAL")
        evaluate_and_plot(model1, model2, test_loader, device, fm_list, pt_scaling_factor, noise_std_dev, dis_max, args, final_session_name="INITIAL")
        prev_params = analyze_param_change(model1, prev_params, "INITIAL")

        # ==========================================
        # [Added] Resume Check & Baseline Comparison
        # ==========================================
        resume_c1_path = os.path.join(args.output_dir, f'best_model2_ITER-C1-S2-NET-LOC_{args.wall_type}_{dis_str}.pt')
        
        if os.path.exists(resume_c1_path):
            print("\n" + "="*60)
            print(f">>> [Resume] Detected Cycle 1 model ({os.path.basename(resume_c1_path)})")
            print(">>> [Resume] Skipping baseline and initial reset; resuming iteration loop.")
            print("="*60)
        else:
            # 0.5 Baseline: freeze initial beam, train Model 2 only
            print("\n" + "="*60)
            print(">>> [Baseline] Starting baseline test: freeze initial beam, train receiver (Model2) only...")
            print("="*60)
            
            # Save initial weights so baseline can be reset afterwards, ensuring a fair starting point for iteration
            init_model1_state = copy.deepcopy(model1.state_dict())
            init_model2_state = copy.deepcopy(model2.state_dict())
            
            session_baseline = "BASELINE_FIXED_INIT"
            optimizer_baseline = optim.Adam(model2.parameters(), lr=args.lr_model2)
            scheduler_baseline = optim.lr_scheduler.ReduceLROnPlateau(optimizer_baseline, 'min', factor=0.5, patience=5)
            
            run_training_session(session_name=session_baseline, optimizer=optimizer_baseline, scheduler=scheduler_baseline,
                                train_model1=False, train_model2=True, 
                                training_objective='location',
                                **common_params)
            
            # Evaluate baseline performance
            best_base_path2 = os.path.join(args.output_dir, f'best_model2_{session_baseline}_{args.wall_type}_{dis_str}.pt')
            if os.path.exists(best_base_path2):
                model2.load_state_dict(torch.load(best_base_path2, map_location=device, weights_only=True))
            
            print(f"\n>>> [Baseline complete] Baseline performance with fixed initial beam:")
            evaluate_and_plot(model1, model2, test_loader, device, fm_list, pt_scaling_factor, noise_std_dev, dis_max, args, final_session_name=session_baseline)
            
            # [Reset] Restore initial state and begin official iteration training
            print("\n" + "="*60)
            print(">>> [Reset] Restoring models to initial state and starting iteration training...")
            print("="*60)
            model1.load_state_dict(init_model1_state)
            model2.load_state_dict(init_model2_state)

        for i in range(max_cycles):
            cycle_num = i + 1
            print("\n" + "#"*70)
            print(f"# Starting iterative training: Cycle {cycle_num} (Max: {max_cycles})")
            print("#"*70 + "\n")
            
            # [Added] Resume Logic: Check if this cycle is already completed
            resume_session_p2 = f"ITER-C{cycle_num}-S2-NET-LOC"
            resume_model2_path = os.path.join(args.output_dir, f'best_model2_{resume_session_p2}_{args.wall_type}_{dis_str}.pt')
            
            if i == 0:
                resume_session_p1 = f"ITER-C{cycle_num}-S1-3-JOINT-ENERGY"
            else:
                resume_session_p1 = f"ITER-C{cycle_num}-S1-BEAM-LOCATION"
            resume_model1_path = os.path.join(args.output_dir, f'best_model1_{resume_session_p1}_{args.wall_type}_{dis_str}.pt')

            if os.path.exists(resume_model2_path):
                print(f">>> [Resume] Found that Cycle {cycle_num} is already completed (Found {os.path.basename(resume_model2_path)}).")
                print(">>> [Resume] Loading models and skipping this cycle...")
                try:
                    state_dict2 = torch.load(resume_model2_path, map_location=device, weights_only=True)
                    model2.load_state_dict(state_dict2)
                    
                    # Try loading Model 1
                    if os.path.exists(resume_model1_path):
                        state_dict1 = torch.load(resume_model1_path, map_location=device, weights_only=True)
                        model1.load_state_dict(state_dict1)
                    else:
                        print(f"[Warning] Model 1 file not found: {resume_model1_path}, using current state.")
                    
                    # Update parameter change record
                    prev_params = analyze_param_change(model1, prev_params, f"Cycle{cycle_num}_Resume")
                    
                    # Skip the rest of the loop
                    continue
                except Exception as e:
                    print(f"[Error] Failed to load resume models: {e}. restarting cycle...")
            
            # --- Stage 1: Train Beamformer (Model 1) ---
            if i == 0:
                # === Special handling for first cycle: use a three-step energy-maximization strategy (based on log1214_total.txt) ===
                print(f">>> [Cycle {cycle_num}] Stage 1 (Special): Energy-maximization warmup (TTD -> PS -> Joint)")
                
                # 1.1 Train TTD Only (Energy)
                session_p1_1 = f"ITER-C{cycle_num}-S1-1-TTD-ENERGY"
                print(f"   -> Sub-stage 1.1: TTD Training (Energy)")
                optimizer_p1_1 = optim.Adam([{'params': [model1.TTD], 'lr': args.lr_ttd}])
                scheduler_p1_1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_p1_1, 'min', factor=0.5, patience=5)
                run_training_session(session_name=session_p1_1, optimizer=optimizer_p1_1, scheduler=scheduler_p1_1,
                                     train_model1=True, train_model2=False, training_objective='energy', **common_params)
                
                # [Log] record after TTD stage
                prev_params = analyze_param_change(model1, prev_params, "C1_S1.1_TTD")
                visualize_beam_patterns(model1, fm_list, Nt, d, vis_process_dir, tag="Cycle1_S1.1_TTD")

                # 1.2 Train PS Only (Energy)
                session_p1_2 = f"ITER-C{cycle_num}-S1-2-PS-ENERGY"
                print(f"   -> Sub-stage 1.2: PS Training (Energy)")
                optimizer_p1_2 = optim.Adam([{'params': [model1.PS], 'lr': args.lr_ps}])
                scheduler_p1_2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_p1_2, 'min', factor=0.5, patience=5)
                run_training_session(session_name=session_p1_2, optimizer=optimizer_p1_2, scheduler=scheduler_p1_2,
                                     train_model1=True, train_model2=False, training_objective='energy', **common_params)
                
                # [Log] record after PS stage
                prev_params = analyze_param_change(model1, prev_params, "C1_S1.2_PS")
                visualize_beam_patterns(model1, fm_list, Nt, d, vis_process_dir, tag="Cycle1_S1.2_PS")

                # 1.3 Joint Fine-tune (Energy)
                session_p1_3 = f"ITER-C{cycle_num}-S1-3-JOINT-ENERGY"
                print(f"   -> Sub-stage 1.3: Joint Fine-tune (Energy)")
                # Use a smaller learning rate for fine-tuning (reference log1214: lr=0.001)
                optimizer_p1_3 = optim.Adam([
                    {'params': [model1.PS], 'lr': 0.001},
                    {'params': [model1.TTD], 'lr': 0.001}
                ])
                scheduler_p1_3 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_p1_3, 'min', factor=0.5, patience=5)
                # Joint fine-tuning usually requires fewer epochs; set to 50 here (reference log1214)
                common_params_short = common_params.copy()
                common_params_short['epochs'] = 50
                run_training_session(session_name=session_p1_3, optimizer=optimizer_p1_3, scheduler=scheduler_p1_3,
                                     train_model1=True, train_model2=False, training_objective='energy', **common_params_short)
                
                session_p1 = session_p1_3 # mark as the latest session
                
                # [Vis] After Stage 1 energy training, plot beam patterns
                print(f">>> [Vis] Generating plots for Cycle 1 Energy Stage...")
                visualize_beam_patterns(model1, fm_list, Nt, d, vis_process_dir, tag="Cycle1_S1_Final")
                
            else:
                # === Subsequent cycles: standard joint training (objective: Location) ===
                obj = 'location'
                session_p1 = f"ITER-C{cycle_num}-S1-BEAM-{obj.upper()}"
                print(f">>> [Cycle {cycle_num}] Stage 1: Train Beamformer (objective: {obj})")
                
                optimizer_p1 = optim.Adam([
                    {'params': [model1.PS], 'lr': args.lr_ps},
                    {'params': [model1.TTD], 'lr': args.lr_ttd}
                ])
                scheduler_p1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_p1, 'min', factor=0.5, patience=5)
                
                run_training_session(session_name=session_p1, optimizer=optimizer_p1, scheduler=scheduler_p1,
                                     train_model1=True, train_model2=False, 
                                     training_objective=obj,
                                     **common_params)
                
                # [Vis] Visualize after subsequent beam training
                visualize_beam_patterns(model1, fm_list, Nt, d, vis_process_dir, tag=f"Cycle{cycle_num}_S1_BeamLoc")

            
            # Load best model for this stage
            best_model1_path = os.path.join(args.output_dir, f'best_model1_{session_p1}_{args.wall_type}_{dis_str}.pt')
            if os.path.exists(best_model1_path):
                model1.load_state_dict(torch.load(best_model1_path, map_location=device, weights_only=True))
            
            # Record parameter changes
            prev_params = analyze_param_change(model1, prev_params, session_p1)
            
            # [Eval] After Stage 1, evaluate Beamformer impact on localization (RaiNet not updated)
            print(f"\n>>> [Cycle {cycle_num} Stage 1 complete] Evaluate system performance after Beamformer update (RaiNet not updated):")
            evaluate_and_plot(model1, model2, test_loader, device, fm_list, pt_scaling_factor, noise_std_dev, dis_max, args, final_session_name=f"Cycle{cycle_num}_S1_End")

            # [Added] BN recalibration: after Beamformer updates, Model 2 input distribution changes. Recalibrate Model 2 BN stats before evaluation/training.
            # This helps mitigate loss spikes at the start of Stage 2.
            recalibrate_bn(model2, train_loader, device, fm_list, pt_scaling_factor, noise_std_dev, model1_frozen=model1)

            # --- Stage 2: Train RaiNet (Model 2) ---
            session_p2 = f"ITER-C{cycle_num}-S2-NET-LOC"
            print(f">>> [Cycle {cycle_num}] Stage 2: Train RaiNet (objective: location)")
            
            # [Optimization] Apply a mild exponential decay across cycles to avoid LR becoming too small
            # Strategy change: replace previous /5.0 step with multiplicative decay factor 0.7 per cycle
            decay_rate = 0.7
            current_lr_model2 = args.lr_model2 * (decay_rate ** (cycle_num - 1))
            
            # Set a lower bound to prevent LR from approaching zero
            min_lr = 1e-5
            current_lr_model2 = max(current_lr_model2, min_lr)

            if cycle_num > 1:
                print(f"[Auto-Adjustment] Cycle {cycle_num}, adjusted Model2 learning rate to {current_lr_model2:.1e} (Decay: {decay_rate}^k)")

            optimizer_p2 = optim.Adam(model2.parameters(), lr=current_lr_model2)
            scheduler_p2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_p2, 'min', factor=0.5, patience=5)
            
            current_cycle_val_loss = run_training_session(session_name=session_p2, optimizer=optimizer_p2, scheduler=scheduler_p2,
                                 train_model1=False, train_model2=True, 
                                 training_objective='location',
                                 **common_params)
            
            # Load best model for this stage
            best_model2_path = os.path.join(args.output_dir, f'best_model2_{session_p2}_{args.wall_type}_{dis_str}.pt')
            if os.path.exists(best_model2_path):
                model2.load_state_dict(torch.load(best_model2_path, map_location=device, weights_only=True))
            
            final_session_name = session_p2
            
            # Print performance after this cycle
            print(f"\n>>> [Cycle {cycle_num} complete] Current system evaluation:")
            evaluate_and_plot(model1, model2, test_loader, device, fm_list, pt_scaling_factor, noise_std_dev, dis_max, args, final_session_name)

            # --- Automatic stopping decision ---
            print(f"\n[Cycle {cycle_num} Check] Current Val Loss: {current_cycle_val_loss:.6f} vs Best Previous: {best_cycle_val_loss:.6f}")
            
            if current_cycle_val_loss < best_cycle_val_loss - loss_threshold:
                print(f"[Info] Performance improved (Loss: {best_cycle_val_loss:.6f} -> {current_cycle_val_loss:.6f}), proceeding to next cycle.")
                best_cycle_val_loss = current_cycle_val_loss
            else:
                print(f"[Info] No significant improvement (Diff < {loss_threshold}), stopping iteration.")
                break
        
        # --- [Added] Stage 3: Joint Training (Like Sequential P3) ---
        print("\n" + "="*60)
        print(">>> [Iteration Post-Processing] Starting joint fine-tune...")
        print("="*60 + "\n")
        
        session_final_joint = f"ITER-FINAL-JOINT"
        
        # Use smaller learning rates for fine-tuning
        optimizer_joint = optim.Adam([
            {'params': model1.parameters(), 'lr': args.lr_ps * 0.1}, 
            {'params': model2.parameters(), 'lr': args.lr_model2 * 0.1}
        ])
        scheduler_joint = optim.lr_scheduler.ReduceLROnPlateau(optimizer_joint, 'min', factor=0.5, patience=5)
        
        run_training_session(session_name=session_final_joint, optimizer=optimizer_joint, scheduler=scheduler_joint,
                             train_model1=True, train_model2=True, 
                             training_objective='location',
                             **common_params)
        
        final_session_name = session_final_joint


    elif args.training_mode == 'sequential':
        # Stage 1: Train Beamformer only with objective to maximize total energy
        session_p1 = "SEQ-P1-MAX_ENERGY"
        optimizer_p1 = optim.Adam([
            {'params': [model1.PS], 'lr': args.lr_ps},
            {'params': [model1.TTD], 'lr': args.lr_ttd}
        ])
        scheduler_p1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_p1, 'min', factor=0.5, patience=5)
        
        run_training_session(session_name=session_p1, optimizer=optimizer_p1, scheduler=scheduler_p1,
                             train_model1=True, train_model2=False, 
                             training_objective='energy',
                             **common_params)

        # Load best model1 from Stage 1
        best_model1_path_p1 = os.path.join(args.output_dir, f'best_model1_{session_p1}_{args.wall_type}_{dis_str}.pt')
        if os.path.exists(best_model1_path_p1):
             model1.load_state_dict(torch.load(best_model1_path_p1, map_location=device, weights_only=True))
        
        # Stage 2: Fix Beamformer, train RaiNet with localization objective
        session_p2 = "SEQ-P2-FIX_BEAM_LOC"
        optimizer_p2 = optim.Adam(model2.parameters(), lr=args.lr_model2)
        scheduler_p2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_p2, 'min', factor=0.5, patience=5)
        
        run_training_session(session_name=session_p2, optimizer=optimizer_p2, scheduler=scheduler_p2,
                             train_model1=False, train_model2=True, 
                             training_objective='location',
                             **common_params)
                             
        final_session_name = session_p2

    # --- 6. Final evaluation and plotting ---
    if final_session_name:
        print("\n" + "="*60)
        print("All training complete; loading final best models for testing and plotting...")
        best_model1_path = os.path.join(args.output_dir, f'best_model1_{final_session_name}_{args.wall_type}_{dis_str}.pt')
        best_model2_path = os.path.join(args.output_dir, f'best_model2_{final_session_name}_{args.wall_type}_{dis_str}.pt')

        if os.path.exists(best_model1_path) and os.path.exists(best_model2_path):
            model1.load_state_dict(torch.load(best_model1_path, map_location=device, weights_only=True))
            model2.load_state_dict(torch.load(best_model2_path, map_location=device, weights_only=True))
            evaluate_and_plot(model1, model2, test_loader, device, fm_list, pt_scaling_factor, noise_std_dev, dis_max, args, final_session_name)
            
            # [Vis] Plot beam patterns after final training
            print(f">>> [Vis] Generating final plots...")
            visualize_beam_patterns(model1, fm_list, Nt, d, vis_final_dir, tag="Final_Result")
        else:
            print(f"[Error] Final model files not found ({best_model1_path} or {best_model2_path}). Please ensure training completed successfully.")
    else:
        print("[Info] No training sessions were executed, skipping final evaluation.")

if __name__ == "__main__":
    main()
