# GPU Setup

Oasis has been tested on NVIDIA A40 nodes with:

- NVIDIA driver exposing CUDA `12.8`
- `torch 2.11.0+cu128`

The learned PyTorch-backed families (`probe_gnn`, `gnn_direct`, and `moe` with
`gnn` or `schnet` gates) honor the top-level experiment config setting:

```toml
seed = 42
dev_run = true
device = "cuda"
```

## Reproducible Environment

Base environment:

```bash
uv sync --extra dev
```

Then replace the default PyTorch wheel with the tested CUDA 12.8 build:

```bash
uv pip uninstall torch
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch
```

If you also need the matching companion packages:

```bash
uv pip uninstall torch torchvision torchaudio
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

The helper script [scripts/bootstrap_gpu_env.sh](../scripts/bootstrap_gpu_env.sh)
automates that sequence.

## Verify CUDA

On a GPU compute node, verify the environment with:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Expected shape of output:

- `2.11.0+cu128`
- `12.8`
- `True`
- at least one visible GPU

## Live Smoke Test

Use `dev_run = true` to keep the experiment small:

- caps the learning-curve frame to 24 rows
- uses one repeat
- uses a single small sweep size

Then start an interactive GPU shell from
[interactive_gpu_dev.example.sh](../interactive_gpu_dev.example.sh), activate
the environment, and run:

```bash
python -m oasis mlip.toml experiment.toml
```

## Locking

For long-term reproducibility:

- keep `pyproject.toml` committed with the pinned `torch==2.11.0`
- keep `uv.lock` committed
- keep this document updated if the tested CUDA/PyTorch pairing changes
