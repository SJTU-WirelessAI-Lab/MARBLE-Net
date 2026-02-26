# MARBLE-Net

Official code for the paper:

**MARBLE-Net: Learning to Localize in Multipath Environment with Adaptive Rainbow Beams**  
arXiv: [2511.06971](https://arxiv.org/abs/2511.06971)

---

## Overview

MARBLE-Net is a joint beamforming and positioning system for mmWave sensing. It co-designs a wideband rainbow beam (hardware beamformer with Phase Shifters + True-Time Delays) and a lightweight CNN localizer in an alternating, end-to-end trainable framework.

This repository provides:

| Script | Purpose |
|--------|---------|
| `generate_h5.py` | Ray-tracing dataset generation (Sionna + TensorFlow) |
| `wcnc_utils.py` | Shared utilities: data loading, beam initialisation, metrics |
| `train_knn_gpu.py` | GPU-accelerated k-NN positioning baseline |
| `train_RaiNet.py` | RaiNet baseline (CNN receiver, fixed or learned beam) |
| `train_MARBLE.py` | **Proposed MARBLE-Net** (joint beam + CNN, alternating training) |

---

## Requirements

> [!IMPORTANT]
> Dataset generation and model training use **two separate environments** because Sionna requires TensorFlow while the training scripts use PyTorch.

### Environment 1 — Data Generation (Sionna / TensorFlow)

```bash
conda create -n sionna1.2 python=3.10 -y
conda activate sionna1.2
pip install sionna==1.2.1 tensorflow h5py numpy tqdm
```

Tested with: **Sionna 1.2.1 · TensorFlow 2.19 · CUDA 12**

### Environment 2 — Training (PyTorch)

```bash
conda create -n torch2.9 python=3.10 -y
conda activate torch2.9
pip install torch numpy h5py matplotlib tqdm
```

Tested with: **PyTorch 2.9.1+cu130 · CUDA 13**

---

## Dataset Generation

> [!NOTE]
> A Mitsuba-format scene XML file (e.g. `walls/L/wall_L.xml`) is required.  
> You can also substitute any Sionna built-in scene via `sionna.rt.scene`.

Edit `generate_h5.py` to set your scene path (`BASE_XML_FILE`) and output directory (`OUTPUT_DIR`), then run:

```bash
conda activate sionna1.2
python generate_h5.py
```

**Output files** (under `OUTPUT_DIR/`):

| File | Description |
|------|-------------|
| `channel_data.h5` | HDF5 channel matrices, shape `[N, 1, Nt, M]` |
| `trajectory_data.npz` | UE positions and velocities |
| `system_params.npz` | System configuration (Nt, M, f0, BW, …) |

For a quick test, set `NUM_USERS = 200` in the script.

---

## Training & Evaluation

All training scripts share the same `--data_root` argument pointing to the parent directory that contains the `train/` subfolder generated above.

### k-NN Baseline

```bash
conda activate torch2.9
python train_knn_gpu.py \
    --data_root sionna_1109/los \
    --k_neighbors 5 \
    --gpu_id 0
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | `sionna_1008/los` | Path to dataset root |
| `--pt_dbm` | `13.0` | Transmit power (dBm) |
| `--k_neighbors` | `5` | k for k-NN |
| `--add_noise` | off | Enable thermal noise |
| `--load_weights` | `None` | Pre-trained PS/TTD `.pt` file |
| `--gpu_id` | `0` | GPU index |

### RaiNet Baseline

```bash
conda activate torch2.9
python train_RaiNet.py \
    --data_root sionna_1109/los \
    --training_mode iteration \
    --gpu_id 0
```

### MARBLE-Net (Proposed)

```bash
conda activate torch2.9
python train_MARBLE.py \
    --data_root sionna_1109/los \
    --training_mode iteration \
    --gpu_id 0
```

**Common training arguments** (apply to both `train_RaiNet.py` and `train_MARBLE.py`):

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | `sionna_1109/los` | Dataset root directory |
| `--training_mode` | `iteration` | `joint` / `fixed` / `iteration` / `sequential` |
| `--pt_dbm` | `23.0` | Transmit power (dBm) |
| `--lr_ps` | `5e-3` | Learning rate for Phase Shifters |
| `--lr_ttd` | `2e-2` | Learning rate for True-Time Delays |
| `--lr_model2` | `1e-3` | Learning rate for CNN receiver |
| `--gpu_id` | `1` | GPU index |
| `--output_dir` | `./0116output/output` | Model checkpoint directory |
| `--figure_dir` | `./0116output/figure` | Figure output directory |
| `--wall_type` | `los_5_95` | Tag for naming saved files |

Show all options:

```bash
python train_knn_gpu.py -h
python train_RaiNet.py -h
python train_MARBLE.py -h
```

---

## Training Modes

| Mode | Description |
|------|-------------|
| `joint` | Train beamformer + CNN simultaneously |
| `fixed` | Fix initial beam, train CNN only (ablation) |
| `iteration` | **Recommended** — alternate beam and CNN training across cycles |
| `sequential` | First maximize beam energy, then train CNN |

---

## Reproducibility

All training scripts fix random seeds (NumPy / PyTorch / CUDA) to `42` by default.  
Deterministic mode is enabled (`torch.backends.cudnn.deterministic = True`).

To reproduce results, ensure identical dataset splits by keeping the default seed.

---

## Citation

If you find this code useful, please cite:

```bibtex
@article{liang2025marblenet,
  title   = {MARBLE-Net: Learning to Localize in Multipath Environment with Adaptive Rainbow Beams},
  author  = {Liang, Qiushi and Cai, Yeyue and Mo, Jianhua and Tao, Meixia},
  journal = {arXiv preprint arXiv:2511.06971},
  year    = {2025}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgements

- [Sionna](https://github.com/NVlabs/sionna) — ray-tracing based channel simulation
- [PyTorch](https://pytorch.org/) — deep learning framework
