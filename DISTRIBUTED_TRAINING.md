# OpenDriveFM — Distributed Training Scaling Guide

## Current Setup
- Hardware: Apple Silicon MacBook (MPS backend)
- Batch size: 2
- Training: Single device
- Best model: v11 — ADE=2.457m, IoU=0.136

---

## Scaling Path — PyTorch DDP

OpenDriveFM's architecture is DDP-ready by design:
- **Shared CNN backbone**: processes B×V×T images in one batched conv — scales linearly
- **Multi-task heads**: BEV decoder + TrajHead — independent gradients, no sync bottleneck
- **Trust scorer**: per-camera, embarrassingly parallel across batch dimension

### Launch Command

```bash
# 4-GPU single node
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    scripts/train_nuscenes_mini_trust.py \
    --config configs/default.yaml \
    --distributed

# 2-node × 4-GPU
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --master_addr=<node0_ip> \
    --master_port=29500 \
    scripts/train_nuscenes_mini_trust.py \
    --config configs/default.yaml \
    --distributed
```

### DDP Wrapper

```python
# Add to train_nuscenes_mini_trust.py
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train_ddp(rank, world_size, cfg):
    setup_ddp(rank, world_size)

    model = OpenDriveFM(...).to(rank)
    model = DDP(model, device_ids=[rank],
                find_unused_parameters=True)  # trust scorer may skip

    # Distributed sampler — no data overlap across GPUs
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank)
    loader  = DataLoader(dataset, batch_size=CFG_BATCH,
                         sampler=sampler, num_workers=4)

    # Training loop unchanged — DDP handles gradient sync
    for batch in loader:
        occ, traj, trust, _ = model(x, K=K, T_ego_cam=T)
        loss = compute_loss(occ, traj, trust, ...)
        loss.backward()   # gradients synced across GPUs automatically
        optimizer.step()
```

### Batch Size Scaling

| GPUs | Batch/GPU | Effective Batch | Expected Speedup |
|------|-----------|----------------|-----------------|
| 1    | 2         | 2              | 1× (baseline)   |
| 4    | 2         | 8              | ~3.5×           |
| 8    | 4         | 32             | ~6.5×           |
| 16   | 4         | 64             | ~11×            |

**Linear scaling rule**: when effective batch increases N×, scale lr by √N:
```python
lr = base_lr * math.sqrt(world_size)  # e.g. 1e-4 * sqrt(4) = 2e-4
```

### Expected Training Time Scaling

| Setup | Epochs | Est. Time |
|-------|--------|-----------|
| 1× MacBook MPS (current) | 120 | ~4 hours |
| 4× A100 80GB | 120 | ~18 min |
| 8× A100 80GB | 120 | ~10 min |
| Full nuScenes (28K samples) × 8 A100 | 120 | ~6 hours |

### Memory Requirements

```
Per GPU minimum:
  Batch=2, V=6, T=4, H=90, W=160:  ~4GB VRAM
  Batch=4, V=6, T=4, H=90, W=160:  ~7GB VRAM
  Batch=8, V=6, T=4, H=90, W=160:  ~13GB VRAM

Recommended: A100 40GB or H100 80GB for batch=16+
```

### Gradient Sync Considerations

```python
# Trust contrastive loss — needs all-reduce across GPUs
# Each GPU sees different faulted cameras → gradients complement
# DDP all-reduce averages correctly: no special handling needed

# BEV decoder — standard cross-entropy, DDP handles automatically

# Trajectory head — SmoothL1, DDP handles automatically

# Only concern: find_unused_parameters=True needed if
# trust scorer is disabled (enable_trust=False)
```

### PyTorch Lightning DDP (Simpler)

```python
# In configs/default.yaml — change:
trainer:
  accelerator: gpu
  devices: 4                    # number of GPUs
  strategy: ddp                 # automatic DDP
  num_nodes: 1
  precision: 16-mixed           # AMP for speed
  max_epochs: 120
  batch_size: 2                 # per GPU
```

```bash
# Lightning handles everything automatically
python scripts/train_nuscenes_mini_trust.py \
    --config configs/ddp_4gpu.yaml
```

---

## Hardware Constraint Note

Current training was performed on Apple Silicon MacBook (MPS backend)
due to hardware constraints — a single-device setup. The architecture
is designed to scale to multi-GPU DDP without modification:

- No custom CUDA ops that break DDP
- No shared mutable state between workers  
- Stateless model forward pass
- Standard PyTorch losses compatible with gradient averaging

**The scaling path is clear and the architecture supports it.
Full nuScenes training (87× more data) on 8× A100 is the next step.**
