#!/usr/bin/env python
# coding: utf-8

# GPU Configuration and Imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
import sionna.phy
import sionna.rt

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
print(f"Available GPUs: {len(physical_devices)}")
for gpu in physical_devices:
    print(f"  {gpu}")
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

tf.get_logger().setLevel('ERROR')

import numpy as np
import math
import random
from tqdm import tqdm
import h5py

# Import other Sionna components
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver
import time
# Configuration
OUTPUT_DIR = 'sionna_1109/los/train'
BASE_XML_FILE = 'walls/L/wall_L.xml'

# System parameters
Nt = 128
num_rx_ant = 1
c = 3e8
f0 = 28e9
f_scs = 240e3
M = 1584 - 1
num_subcarriers = M + 1
num_ofdm_slot = 1

# User trajectory parameters
RADIUS_MIN = 5.0
RADIUS_MAX = 200.0
UE_HEIGHT = 25.0
NUM_USERS = 100000
RADIAL_SPEED_MIN_MS = 0
RADIAL_SPEED_MAX_MS = 0

# Base station configuration
BS_POSITION = [0, 0, 25.0]

# Ray tracing parameters
max_depth = 0
subcarriers_per_batch = 32

def generate_random_users(num_users, radius_min, radius_max, z_pos):
    """Generate random user positions inside a sector.

    The angular span is -60 to 60 degrees and radial distance is
    sampled uniformly in radius squared between radius_min and radius_max.
    """
    print(f"\n-- Generating {num_users} random user positions (radius {radius_min}-{radius_max} m, angle -60..60 deg) --")
    user_positions = []
    np.random.seed(42)
    angle_min_rad = np.deg2rad(-60.0)
    angle_max_rad = np.deg2rad(60.0)

    for _ in range(num_users):
        r_sq = np.random.uniform(radius_min**2, radius_max**2)
        r = np.sqrt(r_sq)
        theta = np.random.uniform(angle_min_rad, angle_max_rad)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        user_positions.append([float(x), float(y), float(z_pos)])

    print(f"Generated {num_users} user positions.")
    return user_positions

def check_for_nan_complex(tensor):
    """Check whether a complex TensorFlow tensor contains NaNs."""
    real_part_nan = tf.reduce_any(tf.math.is_nan(tf.math.real(tensor)))
    imag_part_nan = tf.reduce_any(tf.math.is_nan(tf.math.imag(tensor)))
    return real_part_nan or imag_part_nan

def main():
    """Main: generate channel HDF5 and save trajectories and params."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")

    # Load scene
    try:
        scene = load_scene(BASE_XML_FILE, merge_shapes=True)
        print(f"Loaded scene: {BASE_XML_FILE}")
        for name, obj in scene.objects.items():
            print(f'{name:<15}{obj.radio_material.name}')
    except Exception as e:
        print(f"Failed to load scene: {e}")
        return

    # Configure scene and arrays
    scene.frequency = f0
    scene.tx_array = PlanarArray(num_rows=1, num_cols=Nt, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="tr38901", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=num_rx_ant, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V")
    tx = Transmitter(name="tx", position=BS_POSITION, orientation=[0, 0, 0])
    scene.add(tx)

    frequencies = subcarrier_frequencies(num_subcarriers, f_scs)
    p_solver = PathSolver()
    user_positions = generate_random_users(NUM_USERS, RADIUS_MIN, RADIUS_MAX, UE_HEIGHT)

    if not user_positions:
        print("No user positions generated, exiting.")
        return

    actual_num_users = len(user_positions)
    trajectories = {'position': [], 'velocity': []}
    bs_pos_np = np.array(BS_POSITION)

    # HDF5 setup
    h5_output_filename = os.path.join(OUTPUT_DIR, 'channel_data.h5')
    h_shape = (actual_num_users, num_ofdm_slot, Nt, num_subcarriers)
    h_dtype = np.complex64

    with h5py.File(h5_output_filename, 'w') as h5_file:
        h_dataset = h5_file.create_dataset('h_matrices', shape=h_shape, dtype=h_dtype)
        print(f"Created HDF5: {h5_output_filename}, dataset 'h_matrices' shape: {h_shape}")

        print(f"\n-- Generating and saving channels for {actual_num_users} users --")

        # Main loop with retry on failure
        for user_idx, ue_pos in tqdm(enumerate(user_positions), total=actual_num_users, desc="Generate and save channels"):
            generation_successful = False
            while not generation_successful:
                try:
                    # radial velocity (zero by default)
                    speed = random.uniform(RADIAL_SPEED_MIN_MS, RADIAL_SPEED_MAX_MS)
                    sign = random.choice([-1.0, 1.0])
                    direction_vec = np.array(ue_pos) - bs_pos_np
                    norm = np.linalg.norm(direction_vec)
                    radial_unit_vec = direction_vec / norm if norm > 1e-6 else np.array([0., 0., 0.])
                    ue_vel = [float(v) for v in (sign * speed * radial_unit_vec)]

                    # update receiver
                    if "rx" in scene.receivers:
                        scene.remove("rx")
                    rx = Receiver(name="rx", position=ue_pos, velocity=ue_vel)
                    scene.add(rx)

                    # compute paths and CIR
                    paths = p_solver(scene=scene,
                                     max_depth=max_depth,
                                     los=True,
                                     specular_reflection=True,
                                     diffuse_reflection=False,
                                     refraction=True,
                                     synthetic_array=False)

                    num_paths = tf.shape(paths.tau)[-1]
                    if tf.equal(num_paths, 0):
                        tqdm.write(f"Warning: user {user_idx+1} (pos: {ue_pos}) found no paths, retrying...")
                        continue

                    first_path_delay = tf.reduce_min(paths.tau, axis=-1, keepdims=True)

                    a_taps, tau_taps_relative = paths.cir(sampling_frequency=f_scs, num_time_steps=num_ofdm_slot, out_type='tf')
                    tau_taps_absolute = tau_taps_relative + tf.cast(first_path_delay, tf.float32)

                    a = tf.expand_dims(a_taps, axis=0)
                    tau = tf.expand_dims(tau_taps_absolute, axis=0)

                    # convert CIR to frequency domain in batches
                    h_freq_list = []
                    for i in range(0, num_subcarriers, subcarriers_per_batch):
                        batch_frequencies = frequencies[i : i + subcarriers_per_batch]
                        h_freq_batch = cir_to_ofdm_channel(batch_frequencies, a, tau, normalize=False)
                        h_freq_list.append(h_freq_batch)

                    h_time_freq = tf.concat(h_freq_list, axis=-1)

                    h_squeezed = tf.squeeze(h_time_freq, axis=1)
                    h_final_tensor = tf.reshape(h_squeezed, [num_ofdm_slot, Nt, num_subcarriers])

                    if check_for_nan_complex(h_final_tensor):
                        tqdm.write(f"Warning: user {user_idx+1} (pos: {ue_pos}) generated NaN, retrying...")
                        continue

                    h_final = h_final_tensor.numpy()

                    # write to HDF5 and record trajectory
                    h_dataset[user_idx, ...] = h_final
                    trajectories['position'].append(ue_pos)
                    trajectories['velocity'].append(ue_vel)
                    generation_successful = True

                except Exception as e:
                    tqdm.write(f"User {user_idx+1} failed (exception: {e}), retrying...")

    print(f"\nAll channel matrices saved to {h5_output_filename}.")

    # Save trajectories and system parameters
    print("\n-- Saving trajectories and system parameters --")

    traj_filename = os.path.join(OUTPUT_DIR, 'trajectory_data.npz')
    np.savez(traj_filename,
             position=np.array(trajectories['position']),
             velocity=np.array(trajectories['velocity']))
    print(f"Trajectory saved to: {traj_filename}")

    params_filename = os.path.join(OUTPUT_DIR, 'system_params.npz')
    system_params = {
        'Nt': Nt, 'num_rx_ant': num_rx_ant, 'M': M, 'num_subcarriers': num_subcarriers,
        'Ns': num_ofdm_slot, 'f0': f0, 'c': c,
        'RADIUS_MIN': RADIUS_MIN, 'RADIUS_MAX': RADIUS_MAX, 'UE_HEIGHT': UE_HEIGHT,
        'phi_start_deg': -60.0, 'phi_end_deg': 60.0,
        'RADIAL_SPEED_MIN_MS': RADIAL_SPEED_MIN_MS, 'RADIAL_SPEED_MAX_MS': RADIAL_SPEED_MAX_MS,
        'NUM_USERS': NUM_USERS,
        'BS_POSITION': bs_pos_np
    }
    np.savez(params_filename, **system_params)
    print(f"System parameters saved to: {params_filename}")

if __name__ == '__main__':
    main()