# Moderated Cooperating Experts (MoCE)

A moderator LLM decomposes a user request into typed content blocks (text,
code, structured/JSON, image) with a dependency DAG. Each block type is
filled by its own specialist "expert" LLM under strict output constraints
(no extra commentary, only the requested content), then results are
assembled into a final Markdown document.

See [PLAN.md](PLAN.md) for the full design/architecture, and
[REPORT.md](REPORT.md) for a detailed write-up of how the pipeline works,
the failure modes encountered in practice, and how each was addressed.

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
`max_loaded_models: 2` so at most two models are resident at once. The
`code` role uses a slightly larger 3B model (needs ~6-7GB VRAM) since code
quality/instruction-following matters most there; drop it back to the 1.5B
variant if you're VRAM-constrained. The `image` role uses `sd-turbo`, a
distilled single-step Stable Diffusion model, for low VRAM/fast generation.

## Usage

```powershell
.\.venv\Scripts\python.exe -m moce.cli run "Write a function to reverse a string, explain it, and summarize as JSON" --dry-run
.\.venv\Scripts\python.exe -m moce.cli run "..." --verbose
```

### Diagnostics flags

These flags are available both on the top-level `moce` group and on the
`run` subcommand (either works; they're merged if passed on both):

| Flag | Effect |
| --- | --- |
| `--verbose` | Prints the moderator's plan and each block's output (status + validated content). |
| `--show-plan` | Prints just the moderator's plan (JSON), without the rest of `--verbose`'s block-result output. |
| `--debug` | Implies `--verbose`. Enables DEBUG-level logging and prints each block's raw (pre-validation) model output plus its retry count. This is for diagnosing **this project's own logic** — plan generation, prompt substitution, validation/retries — and never escalates third-party library logging. |
| `--show-model-noise` | Un-suppresses `transformers`/`huggingface_hub`/`diffusers`/`accelerate`/`safetensors`/`httpx`/`httpcore`/etc. logging and progress bars, which are silenced by default (even under `--debug`) since they're rarely useful for diagnosing this project's logic and are extremely chatty. |
| `--max-workers N` | Number of blocks to run concurrently within a single dependency "generation" (default 1 = fully sequential). Real concurrency is still capped by GPU memory — see [configs/models.yaml](configs/models.yaml)'s `max_loaded_models`. |

`--debug` and `--show-model-noise` are intentionally independent: combining
them surfaces third-party logs at INFO level (never DEBUG), so `--debug`
alone stays readable and focused on the pipeline's own behavior.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/
```

Tests use scripted/mocked generators and do not require torch/transformers
or any model downloads.

