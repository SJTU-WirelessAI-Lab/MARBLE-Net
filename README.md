# WCNC Channel & Positioning

Minimal repo description:

- Generate ray-tracing channel data with Sionna (TensorFlow) using `generate_h5.py`.
- Train and evaluate position-estimation models with PyTorch using `train_knn_gpu.py`, `train_MARBLE.py`, and `train_RaiNet.py`.
- Common helpers are in `wcnc_utils.py`.

Quick start (assumes dependencies installed):

```bash
python generate_h5.py            # create HDF5 channel dataset (requires Sionna + TensorFlow)
python train_knn_gpu.py --data_root sionna_1109/los --k_neighbors 5 --gpu_id 0
python train_RaiNet.py --data_root sionna_1109/los --training_mode iteration --gpu_id 0
```

Dependencies are listed in `requirements.txt`.

License: MIT (see `LICENSE`).
