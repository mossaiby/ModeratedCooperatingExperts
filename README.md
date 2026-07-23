# Moderated Cooperating Experts (MoCE)

A moderator LLM decomposes a user request into typed content blocks (text,
code, structured/JSON, image) with a dependency DAG. Each block type is
filled by its own specialist "expert" LLM under strict output constraints
(no extra commentary, only the requested content), then results are
assembled into a final document.

See [PLAN.md](PLAN.md) for the full design/architecture.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .[dev]
```

### GPU (CUDA) torch

PyPI only distributes CPU-only `torch` wheels, so installing the base
`torch` dependency above (or a plain `pip install torch`) gives you a
CPU-only build even on a machine with an NVIDIA GPU. Install the CUDA build
explicitly from the PyTorch index **after** the step above:

```powershell
.\.venv\Scripts\python.exe -m pip install "torch==2.6.0+cu124" --index-url https://download.pytorch.org/whl/cu124
```

Verify it picked up the GPU:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Pick the CUDA tag (`cu121`/`cu124`/`cu126`/`cu128`, etc.) matching your
driver from https://pytorch.org/get-started/locally/ — a newer NVIDIA
driver can generally run an older `cuXXX` torch build (backward compatible).

The example [configs/models.yaml](configs/models.yaml) is tuned for a
4GB-VRAM GPU (e.g. RTX 3050): small models, `float16`, and
`max_loaded_models: 2` so at most two models are resident at once.

## Usage

```powershell
.\.venv\Scripts\python.exe -m moce.cli run "Write a function to reverse a string, explain it, and summarize as JSON" --dry-run
.\.venv\Scripts\python.exe -m moce.cli run "..." --verbose
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/
```

Tests use scripted/mocked generators and do not require torch/transformers
or any model downloads.
