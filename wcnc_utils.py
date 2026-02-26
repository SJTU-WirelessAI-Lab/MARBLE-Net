#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities for WCNC project.

This module centralizes common helpers used by the training and data
generation scripts to reduce duplication while preserving original
behavior.
"""
import os
import numpy as np
import torch
import h5py


def count_parameters(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if num_params >= 1e6:
        return f'{num_params / 1e6:.2f} M'
    else:
        return f'{num_params / 1e3:.2f} K'


def load_system_params(param_file):
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"Parameter file not found: {param_file}")
    data = np.load(param_file, allow_pickle=True)

    Nt = data.get('Nt', None)
    if Nt is None:
        raise KeyError(f"Missing required key 'Nt' in parameter file: {param_file}")
    Nt = int(np.array(Nt).item())

    if 'num_ofdm_slot' in data:
        Ns = int(np.array(data['num_ofdm_slot']).item())
    elif 'Ns' in data:
        Ns = int(np.array(data['Ns']).item())
    else:
        Ns = 1

    if 'num_subcarriers' in data:
        num_subcarriers = int(np.array(data['num_subcarriers']).item())
    elif 'M' in data:
        num_subcarriers = int(np.array(data['M']).item()) + 1
    else:
        num_subcarriers = 1585

    f0 = float(np.array(data.get('f0', 28e9)).item())
    c = 3e8
    f_scs = 240e3

    phi_start_deg = float(np.array(data.get('phi_start_deg', -60.0)).item())
    phi_end_deg = float(np.array(data.get('phi_end_deg', 60.0)).item())

    user_height = float(np.array(data.get('UE_HEIGHT', data.get('user_height', 25.0))).item())

    if 'BS_POSITION' in data:
        bs_pos = data['BS_POSITION']
        try:
            BS_height = float(np.array(bs_pos)[2])
        except Exception:
            BS_height = float(np.array(data.get('BS_height', 25.0)).item())
    else:
        BS_height = float(np.array(data.get('BS_height', 25.0)).item())

    print(f"Loaded system parameters: Ns={Ns}, Nt={Nt}, num_subcarriers={num_subcarriers}, f0={f0:.2e}, BS_height={BS_height:.2f}, user_height={user_height:.2f}")
    return Ns, Nt, num_subcarriers, f0, c, f_scs, phi_start_deg, phi_end_deg, user_height, BS_height


class ISACDatasetSionna(object):
    """Lightweight dataset helper that supports delayed HDF5 loading.

    Provides a minimal interface compatible with existing scripts. The
    class avoids loading the entire HDF5 into memory by opening it on
    demand in __getitem__.
    """
    def __init__(self, data_root):
        traj_path = os.path.join(data_root, 'trajectory_data.npz')
        h5_candidate = os.path.join(data_root, 'channel_data_full.h5')
        if not os.path.exists(h5_candidate):
            h5_candidate = os.path.join(data_root, 'channel_data.h5')
        if not os.path.exists(h5_candidate):
            raise FileNotFoundError(f"HDF5 channel file not found: {h5_candidate}")

        self.h5_path = h5_candidate
        print(f"Detected data file: {self.h5_path} (delayed loading mode)")

        with h5py.File(self.h5_path, 'r') as f:
            self.num_samples = f['h_matrices'].shape[0]
            subset_size = min(1000, self.num_samples)
            sample_data = f['h_matrices'][:subset_size]
            mean_amplitude = np.mean(np.abs(sample_data))
            self.scaling_factor = 1.0 / mean_amplitude if mean_amplitude > 0 else 1e7
            if 'position' in f:
                self.positions = f['position'][:]
            else:
                if not os.path.exists(traj_path):
                    raise FileNotFoundError(f"Trajectory data not found (H5 missing 'position' and no {traj_path})")
                traj_data = np.load(traj_path)
                self.positions = traj_data['position']

    def __len__(self):
        return int(self.num_samples)

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            h_matrix = f['h_matrices'][idx]
        if h_matrix.ndim == 3:
            h_matrix = h_matrix[0, :, :]
        pos = self.positions[idx]
        x_gt, y_gt = float(pos[0]), float(pos[1])
        r_gt = np.sqrt(x_gt**2 + y_gt**2)
        phi_gt = np.rad2deg(np.arctan2(y_gt, x_gt))
        return {
            'H': torch.from_numpy(h_matrix.astype(np.complex64)) * self.scaling_factor,
            'x_gt': torch.tensor(x_gt, dtype=torch.float32),
            'y_gt': torch.tensor(y_gt, dtype=torch.float32),
            'phi_gt': torch.tensor(phi_gt, dtype=torch.float32),
            'r_gt': torch.tensor(r_gt, dtype=torch.float32)
        }


def initial_rainbow_beam_ULA_YOLO(N, d, BW, f_scs, fm_list, phi_1, phi_M):
    device = fm_list.device if isinstance(fm_list, torch.Tensor) else torch.device('cpu')
    dtype = fm_list.dtype if isinstance(fm_list, torch.Tensor) else torch.float32
    c = 3e8
    antenna_idx = torch.arange(N, dtype=dtype, device=device) - (N - 1) / 2
    PS = -fm_list[0] * antenna_idx * d * torch.sin(torch.deg2rad(torch.tensor(phi_1, device=device, dtype=dtype))) / c
    TTD = -PS / BW - ((fm_list[0] + BW) * antenna_idx * d * torch.sin(torch.deg2rad(torch.tensor(phi_M, device=device, dtype=dtype)))) / (BW * c)
    PS, TTD = 2.0 * torch.pi * PS, 1e9 * TTD
    PS, TTD = torch.fmod(PS, 2*torch.pi), torch.fmod(TTD, 1e9/f_scs)
    return PS, TTD


def compute_uplink_signal_torch(H_oneway, PS_expanded, TTD_expanded, fm_list):
    device, dtype = H_oneway.device, H_oneway.dtype
    B, Nt, num_subcarriers = H_oneway.shape
    ps_b = PS_expanded.unsqueeze(1)
    ttd_b = TTD_expanded.unsqueeze(1)
    fm_b = fm_list.view(1, -1, 1)
    phase = -(ps_b + 2 * np.pi * (fm_b - fm_list[0]) * ttd_b)
    normalization_factor = 1.0 / torch.sqrt(torch.tensor(Nt, dtype=torch.float32, device=device))
    w_matrix = torch.exp(1j * phase).to(device=device, dtype=dtype) * normalization_factor
    H_transposed = H_oneway.transpose(1, 2)
    uplink_signal = torch.einsum('bmn,bmn->bm', w_matrix.conj(), H_transposed)
    return uplink_signal


def pairwise_l2_torch(a, b):
    a_sum_sq = torch.sum(a**2, dim=1, keepdim=True)
    b_sum_sq = torch.sum(b**2, dim=1, keepdim=True)
    dist_sq = a_sum_sq + b_sum_sq.T - 2.0 * torch.mm(a, b.T)
    return torch.sqrt(torch.clamp(dist_sq, min=0.0))


def knn_predict_torch(X_codebook_gpu, Y_positions_gpu, x_query_gpu, k=5):
    D = pairwise_l2_torch(x_query_gpu, X_codebook_gpu)
    k_safe = min(k, X_codebook_gpu.shape[0])
    _, nn_idx = torch.topk(D, k=k_safe, dim=1, largest=False)
    return Y_positions_gpu[nn_idx].mean(dim=1)
