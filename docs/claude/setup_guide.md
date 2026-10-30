# LLM Inference Engine Setup Guide

Section 3 of Joe's handoff doc, reproduced verbatim except that Google Docs export escapes were removed and the full file listings in steps 6.3 to 6.8 and 7.1 were replaced by pointers to the repo (those files exist and have since changed, so the repo is the source of truth). Joe is now the solo developer, so Step 5 (partner access) is no longer in use.

Sep 24, 2026 · @Joe Zhuo

## Overview

This guide takes both of you from a blank machine to a GitHub repo called `llm-inference-engine` that builds, runs a first CUDA kernel on the RTX 5060, and passes a first unit test, in about one evening each (3 to 5 hours, most of it downloads).

The one constraint that shapes everything: **macOS cannot run CUDA at all**, so the Windows laptop is the team's only local GPU. Your partner develops pure C++ components natively on the Mac and runs GPU code remotely on your laptop (over SSH) or on Colab.

### Who does what

| Step | You (Windows + RTX 5060) | Partner (Mac) |
| --- | --- | --- |
| Windows host prep | Yes | No |
| Ubuntu + CUDA in WSL2 | Yes | No |
| Nsight tools and GPU counters | Yes | No (views reports you export) |
| Python environment + model download | Yes | Yes (CPU/MPS only, for reading the model) |
| Mac toolchain + remote GPU access | Host the SSH access | Yes |
| Create the GitHub repo and bootstrap files | Yes (repo owner) | Clones and verifies |
| CI and conventions | Together | Together |
| Day-1 smoke tests | All GPU tests | CPU tests locally, GPU tests over SSH |

### End-of-setup checklist

- [ ] `nvidia-smi` inside WSL2 shows the RTX 5060
- [ ] `nvcc --version` shows CUDA 12.8 or newer
- [ ] `deviceQuery` reports compute capability 12.0
- [ ] `ncu` profiles a toy kernel without a permissions error
- [ ] Python venv has PyTorch with CUDA and can generate text from TinyLlama on the GPU
- [ ] The repo builds with `cmake --preset debug` on your laptop and `cmake --preset cpu` on the Mac
- [ ] `ctest` passes on both machines
- [ ] Your partner can SSH into your WSL2 and run the GPU tests
- [ ] GitHub Actions is green on `main`

When every box is checked, you are ready for Phase 1 of the learning plan and milestone M1.

## Step 1: Windows host prep (you, ~45 min)

On Windows you install exactly three things that matter: the NVIDIA driver, WSL2 with Ubuntu, and VS Code. Everything else lives inside Ubuntu.

### 1.1 Update Windows and the NVIDIA driver

1. Run Windows Update until nothing is pending (Windows 11 22H2 or newer is needed for the networking mode in 1.3).
2. Install the latest **GeForce Game Ready or Studio driver** for the RTX 5060 Laptop GPU from nvidia.com or the NVIDIA app. Either works; Studio updates less often and is a bit more stable.
3. **Do not install any NVIDIA driver inside Ubuntu later.** The Windows driver is what exposes the GPU to WSL2.
4. In the NVIDIA app (or NVIDIA Control Panel), find the developer setting "Manage GPU Performance Counters" and set it to **allow access to all users**. This is required for Nsight Compute. Reboot afterward.

### 1.2 Install WSL2 and Ubuntu 24.04

Open PowerShell **as Administrator**:

```
wsl --install -d Ubuntu-24.04
wsl --update
wsl --set-default-version 2
```

Reboot if asked, open "Ubuntu 24.04" from the Start menu, and create a Linux username and password. Then confirm in PowerShell:

```
wsl -l -v
```

The `VERSION` column must say `2`.

### 1.3 Configure WSL2 resources and networking

Create the file `C:\Users\<you>\.wslconfig` with:

```
[wsl2]
memory=12GB
processors=8
swap=8GB
networkingMode=mirrored
```

- Set `memory` to about 75% of your RAM (use `8GB` if you have 16 GB total). Compiling CUDA and loading PyTorch are memory hungry.
- `processors` should be your physical core count or a bit less.
- `networkingMode=mirrored` makes WSL2 share Windows' network interfaces, which makes it much easier for your partner to SSH in later (Step 5).

Apply it with `wsl --shutdown` in PowerShell, then reopen Ubuntu.

### 1.4 Install the editor and terminal

1. **Windows Terminal** (preinstalled on Windows 11). Set Ubuntu 24.04 as the default profile.
2. **VS Code** for Windows, then these extensions: **WSL**, **C/C++** (Microsoft), **clangd**, **CMake Tools**, **Nsight Visual Studio Code Edition**, **Python**, **GitLens** (optional).
3. Open projects by running `code .` **from inside Ubuntu**. VS Code then runs its server inside WSL, so IntelliSense, the terminal, and the debugger all use the Linux toolchain.

### 1.5 Rules for working in WSL2

- **Keep the repo in the Linux filesystem** (`~/code/llm-inference-engine`), never under `/mnt/c/...`. Builds on `/mnt/c` are several times slower.
- Benchmark only when **plugged in** with Windows power mode set to **Best performance**, and the laptop vendor's performance mode on (if it has one).
- Optional: add a Windows Defender exclusion for the WSL virtual disk folder if builds feel slow.

## Step 2: Ubuntu and the CUDA toolkit in WSL2 (you, ~1 hour)

Inside Ubuntu you install the compiler toolchain, then the CUDA **toolkit only** from NVIDIA's WSL-Ubuntu repository. Every command in this step runs in the Ubuntu terminal.

### 2.1 Base packages

```
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential gcc-13 g++-13 gdb cmake ninja-build git git-lfs \
  clangd clang-format clang-tidy pkg-config wget curl unzip htop \
  python3 python3-venv python3-pip openssh-server
```

Ubuntu 24.04 ships GCC 13 and CMake 3.28, which is exactly what the project needs. Check with `gcc --version` and `cmake --version`.

### 2.2 CUDA toolkit (WSL-Ubuntu repo)

```
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-13-3
```

- Your GPU (compute capability 12.0) needs **CUDA 12.8 or newer**. The WSL repo currently offers up to 13.3; if `cuda-toolkit-13-3` is not found, run `apt search cuda-toolkit` and pick the newest `cuda-toolkit-13-x` (or `cuda-toolkit-12-9`).
- **Never install the `cuda`, `cuda-13-x`, or `cuda-drivers` packages** in WSL. Those pull in a Linux driver and break GPU access. Only `cuda-toolkit-*`.
- If `apt` complains about an old NVIDIA key, run `sudo apt-key del 7fa2af80` and retry.

### 2.3 Environment variables

Append to `~/.bashrc`:

```
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Then `source ~/.bashrc`. Do not touch `/usr/lib/wsl/lib`; that folder holds the driver libraries Windows provides.

### 2.4 Verify the GPU

```
nvidia-smi
nvcc --version
```

`nvidia-smi` must list the RTX 5060 Laptop GPU and `nvcc` must report 12.8 or newer. Then compile a tiny device query. Save as `~/scratch/query.cu`:

```cpp
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    std::printf("%s\n", p.name);
    std::printf("compute capability: %d.%d\n", p.major, p.minor);
    std::printf("SMs: %d\n", p.multiProcessorCount);
    std::printf("VRAM: %.2f GB\n", p.totalGlobalMem / 1e9);
    std::printf("L2: %d KB\n", p.l2CacheSize / 1024);
    std::printf("shared mem per SM: %zu KB\n", p.sharedMemPerMultiprocessor / 1024);
}
```

```
cd ~/scratch && nvcc -arch=sm_120 query.cu -o query && ./query
```

Expected: compute capability 12.0 and about 8 GB of VRAM. **Write the SM count, L2 size, and shared memory per SM into the repo's `docs/hardware.md` once the repo exists**; you will use these numbers when choosing kernel launch sizes.

### 2.5 GitHub access from WSL

```
type -p gh >/dev/null || sudo apt install -y gh
gh auth login
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch main
git config --global pull.rebase true
```

Choose HTTPS and "Login with a web browser" in `gh auth login`. If `gh` is not in Ubuntu's repos, install it from GitHub's official apt repository (cli.github.com).

## Step 3: Profiling tools (you, ~30 min)

The command-line profilers (`nsys` and `ncu`) run inside WSL2, and the GUI viewers run on Windows (and on the Mac) to open the report files.

### 3.1 Check the command-line tools

```
which nsys ncu compute-sanitizer
ncu --version
nsys --version
```

The CUDA toolkit normally installs all three. If one is missing: `apt search nsight` and install the newest `nsight-compute-*` / `nsight-systems-*` package from the same NVIDIA repo. NVIDIA's WSL guide notes that profiling under WSL2 needs Windows 11 with a recent driver, which you already have.

### 3.2 Install the GUI viewers

- **Windows:** download Nsight Systems and Nsight Compute installers from developer.nvidia.com (free NVIDIA developer account). Open reports straight from WSL via the path `\\wsl$\Ubuntu-24.04\home\<user>\code\llm-inference-engine\reports`.
- **Mac (partner):** both tools have macOS host builds that can open `.nsys-rep` and `.ncu-rep` files. They cannot profile on the Mac, only view.

### 3.3 First profile

Save as `~/scratch/vadd.cu`:

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void vadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int n = 1 << 26;
    float *a, *b, *c;
    cudaMalloc(&a, n * sizeof(float));
    cudaMalloc(&b, n * sizeof(float));
    cudaMalloc(&c, n * sizeof(float));
    cudaMemset(a, 0, n * sizeof(float));
    cudaMemset(b, 0, n * sizeof(float));
    for (int i = 0; i < 5; ++i) vadd<<<(n + 255) / 256, 256>>>(a, b, c, n);
    cudaDeviceSynchronize();
    std::printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}
```

```
nvcc -O3 -arch=sm_120 -lineinfo vadd.cu -o vadd
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis -k vadd -c 1 ./vadd
nsys profile --trace=cuda -o vadd_timeline ./vadd
compute-sanitizer ./vadd
```

- `ncu` should print a **DRAM Throughput** percentage. This kernel is pure memory traffic, so expect a high number. Write it down: it is your first "% of peak" measurement.
- If you see `ERR_NVGPUCTRPERM`, the performance counter setting from Step 1.1 is not active. Fix it on Windows, reboot, and run `wsl --shutdown`.
- Open `vadd_timeline.nsys-rep` in the Nsight Systems GUI and find the five kernel launches on the timeline.
- `compute-sanitizer` must report `0 errors`.

Always compile with `-lineinfo` in profiling builds so Nsight Compute can map stalls back to your source lines.

## Step 4: Python environment and the model (both, ~30 min)

Python is only for reference outputs, tests, and plots, never the engine itself. Both of you set it up, because the PyTorch forward-pass exercise in the learning plan runs fine on a Mac CPU.

### 4.1 Create the virtual environment

**You (WSL2):**

```
python3 -m venv ~/.venvs/llmie
source ~/.venvs/llmie/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers safetensors sentencepiece huggingface_hub numpy pandas matplotlib jupyter
```

The `cu128` wheel is built for CUDA 12.8, which includes support for your `sm_120` GPU. PyTorch wheels ship their own CUDA runtime, so it does not matter that your system toolkit is 13.x. If the PyTorch site lists a newer CUDA wheel (for example `cu129` or `cu130`), that works too.

**Partner (Mac):**

```
python3 -m venv ~/.venvs/llmie
source ~/.venvs/llmie/bin/activate
pip install --upgrade pip
pip install torch transformers safetensors sentencepiece huggingface_hub numpy pandas matplotlib jupyter
```

Add `source ~/.venvs/llmie/bin/activate` to your shell profile, or activate it each session.

### 4.2 Download TinyLlama

```
mkdir -p ~/models
hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir ~/models/tinyllama
```

On older `huggingface_hub` versions the command is `huggingface-cli download` with the same arguments. No login is needed; the model is not gated. You should end up with `model.safetensors` (~2.2 GB), `config.json`, `tokenizer.model`, `tokenizer.json`, and `generation_config.json`.

Models live **outside** the repo in `~/models`. The repo gets a git-ignored symlink to them in Step 6.

### 4.3 Verify PyTorch and the model

Save as `~/scratch/hf_check.py`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "/home/<user>/models/tinyllama"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, torch.cuda.get_device_capability() if device == "cuda" else "")

tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)

msgs = [{"role": "user", "content": "Explain what a KV cache is in one sentence."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(device)
out = model.generate(ids, max_new_tokens=60, do_sample=False)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

```
python ~/scratch/hf_check.py
```

On your laptop it must print `device: cuda (12, 0)` followed by a sensible sentence. If PyTorch warns that `sm_120` is not supported, you installed a wheel older than CUDA 12.8; reinstall with the `cu128` index URL. On the Mac it runs on the CPU and takes a few seconds longer, which is fine.

## Step 5: Partner's Mac setup and remote GPU access (~1.5 hours)

Your partner writes and tests all pure C++ code natively on the Mac, and runs anything that touches CUDA on your laptop over SSH (primary) or on Colab (fallback). The repo is structured so the Mac build simply skips CUDA targets.

### 5.1 Native toolchain on the Mac

```
xcode-select --install
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install cmake ninja clang-format git gh python@3.12 googletest nlohmann-json
gh auth login
```

- Apple Clang (from the Command Line Tools) handles C++20 fine for this project. Homebrew's `clang-format` keeps formatting identical to yours, and the VS Code clangd extension downloads its own `clangd` on first use.
- VS Code with the same extensions as Step 1.4, plus **Remote - SSH**.
- Set the same `git config --global` values as Step 2.5.

### 5.2 The CPU-only build

The project's CMake (Step 6) checks whether a CUDA compiler exists. On the Mac it doesn't, so only the pure C++ libraries and their tests build: the safetensors header parser, block manager, scheduler, tokenizer wrapper, and config code. That covers most of Person B's MVP work. The rule that makes this possible: **no `#include <cuda_runtime.h>` in any header outside `src/kernels/` and `src/runtime/`.** Plain C++ headers describe interfaces; CUDA stays in `.cu` files.

### 5.3 Remote access to your GPU (you set up, partner uses)

WSL2 with mirrored networking (Step 1.3) plus Tailscale gives your partner an SSH login into your Ubuntu from anywhere.

**On your laptop, in Ubuntu:**

1. Make sure systemd is on: `/etc/wsl.conf` should contain `[boot]` and `systemd=true` (the default for new Ubuntu 24.04 installs). If you add it, run `wsl --shutdown` in PowerShell.
2. Give your partner their own Linux user so your files stay separate: `sudo adduser partner` (use their name).
3. Configure SSH on port 2222 (port 22 may clash with Windows): edit `/etc/ssh/sshd_config` and set `Port 2222`, `PasswordAuthentication no`, `PubkeyAuthentication yes`. Ubuntu 24.04 uses socket activation, so apply with `sudo systemctl daemon-reload && sudo systemctl restart ssh.socket`, then `sudo systemctl enable ssh`.
4. Your partner runs `ssh-keygen -t ed25519` on the Mac and sends you the **public** key (`~/.ssh/id_ed25519.pub`). Add it to `/home/partner/.ssh/authorized_keys` (`chmod 700 ~/.ssh`, `chmod 600 authorized_keys`, owned by that user).

**On your laptop, in Windows PowerShell (Administrator):**

```
New-NetFirewallHyperVRule -Name "WSL-SSH" -DisplayName "WSL SSH 2222" -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -Protocol TCP -LocalPorts 2222
New-NetFirewallRule -DisplayName "WSL SSH 2222" -Direction Inbound -Protocol TCP -LocalPort 2222 -Action Allow
```

The first rule opens the WSL VM's Hyper-V firewall for port 2222 (from Microsoft's WSL networking docs); the second opens the Windows firewall.

**Tailscale (both):** install Tailscale on Windows and on the Mac, sign in, and either invite your partner to your tailnet or use Tailscale's "Share" feature to share just your laptop with them. Note the laptop's Tailscale name (for example `joe-laptop`).

**On the Mac,** add to `~/.ssh/config`:

```
Host gpu
    HostName joe-laptop
    Port 2222
    User partner
    ServerAliveInterval 30
```

Then `ssh gpu` should land in Ubuntu, and `nvidia-smi` should show your GPU. In VS Code, "Remote-SSH: Connect to Host" and pick `gpu` for a full editor on your machine.

**Keeping it reachable:**

- WSL shuts down when nothing is running in it. Create a Windows Task Scheduler task that runs at logon: program `wsl.exe`, arguments `-d Ubuntu-24.04 --exec sleep infinity`, with "run whether user is logged on or not" unchecked (it needs your session).
- Set Windows to never sleep while plugged in during agreed work hours.
- **Share the GPU politely:** wrap every benchmark in a lock so two people never benchmark at once: `flock /tmp/gpu.lock ./build/release/engine bench ...`. Unit tests are fine to run concurrently; benchmarks are not.

### 5.4 Fallback: Google Colab

When the laptop is off, your partner can build and test on a free Colab T4 (compute capability 7.5). In a notebook with a GPU runtime:

```
!nvidia-smi
!git clone https://<token>@github.com/<you>/llm-inference-engine.git
%cd llm-inference-engine
!cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
!cmake --build build -j
!cd build && ctest --output-on-failure
```

- Use a **fine-grained GitHub token** with read-only access to this one repo, stored in Colab's Secrets panel, never pasted into a shared notebook.
- T4 numbers are for correctness only. Never mix them into benchmark results.
- If Colab's `cmake` is too old, `!pip install cmake` gives a new one.

## Step 6: Create the repo and its structure (you, partner verifies, ~1 hour)

The bootstrap is **scaffolding only**: build system, configs, one CPU test, and one GPU smoke test. Every real engine component (buffers, loader, kernels, block manager) is written by hand during the milestones, as agreed in the build approach.

### 6.1 Create the GitHub repo

In Ubuntu:

```
mkdir -p ~/code && cd ~/code
gh repo create llm-inference-engine --public --clone --license mit --description "Single-GPU LLM inference engine in C++20 and CUDA"
cd llm-inference-engine
```

- **Public from day one** is recommended: free GitHub accounts only get branch protection on public repos, and a visible, steady commit history is itself a signal to recruiters. If you prefer private until the MVP ships, protect `main` by convention instead.
- Add your partner: repo Settings, Collaborators, add their GitHub username with Write access.
- If public: Settings, Branches (or Rules), add a rule for `main` that requires a pull request, one approval, and passing status checks (after Step 7 creates them).

### 6.2 Create the folder structure

```
mkdir -p include/engine/{core,kernels,loader,kv,model,runtime,sched,sampling,tokenizer} \
         src/{core,kernels,loader,kv,model,runtime,sched,sampling,tokenizer,app} \
         tests/{unit,gpu,golden} scripts docs results reports cmake .github/workflows
find include src tests scripts docs results reports -type d -empty -exec touch {}/.gitkeep \;
ln -s ~/models models
```

The resulting layout:

```
llm-inference-engine/
  CMakeLists.txt            top-level build, CUDA detection, dependencies
  CMakePresets.json         debug / release / cpu presets
  .clang-format             shared formatting
  .clangd                   editor settings for .cu files
  .gitignore
  README.md
  LICENSE
  include/engine/           public headers, plain C++ only (no CUDA includes)
    core/ kernels/ loader/ kv/ model/ runtime/ sched/ sampling/ tokenizer/
  src/
    CMakeLists.txt
    core/                   version, logging, tensor views
    kernels/                all .cu files + cuda_check.h (CUDA allowed here)
    loader/                 safetensors parsing and weight upload
    kv/                     block manager (pure C++)
    model/                  config + forward pass
    runtime/                model runner, CUDA Graphs (CUDA allowed here)
    sched/                  scheduler (pure C++)
    sampling/               sampler host code
    tokenizer/              SentencePiece wrapper
    app/                    main.cpp (CLI), bench tools
  tests/
    CMakeLists.txt
    unit/                   CPU-only tests, run everywhere including the Mac and CI
    gpu/                    tests that need a GPU, label "gpu"
    golden/                 HF reference tensors (git-ignored, regenerated by script)
  scripts/                  Python: dump_reference.py, bench.py, plot.py
  docs/                     hardware.md, design notes, results.md, learning log
  results/                  committed benchmark CSVs
  reports/                  nsys/ncu reports (git-ignored)
  models -> ~/models        symlink, git-ignored
  .github/workflows/ci.yml
```

**The one structural rule:** `#include <cuda_runtime.h>` (or any CUDA header) may appear only in `src/kernels/` and `src/runtime/`. Public headers in `include/engine/` expose plain C++ functions (for example `void launch_rmsnorm(...)` taking raw pointers and sizes). This is what lets the Mac build everything except the GPU parts.

### 6.3 Config files

The `.gitignore`, `.clang-format`, `.clangd` and `docs/hardware.md` contents from this step are already in the repo (and have been extended since); read the files in the repo rather than this guide.

### 6.4 to 6.8 Build files and bootstrap sources

The top-level `CMakeLists.txt`, `src/CMakeLists.txt`, `tests/CMakeLists.txt`, `CMakePresets.json` and the bootstrap sources (`version`, `main.cpp`, `smoke.h`, `cuda_check.h`, `smoke.cu`, the two first tests) described in these steps are already in the repo and have since been extended (CLI11, more libraries, todo_joe targets). Read the repo, not this guide. Notes kept from the guide: dependencies are found on the system first and downloaded only if missing; to also build for cloud GPUs, configure with `-DCMAKE_CUDA_ARCHITECTURES="120;89;75"`; use `debug` for development, `release` for every benchmark (never benchmark a Debug build), and `cpu` on a machine without CUDA; GPU tests carry the label `gpu`, so `ctest -LE gpu` runs only CPU tests and `ctest -L gpu` runs only GPU tests. The smoke kernel is throwaway scaffolding to prove the toolchain works; delete it once a hand-written `bandwidth_test` and first real kernel exist.

### 6.9 First build and first commit

```
cmake --preset debug          # must print ENGINE_CUDA = ON
cmake --build --preset debug
ctest --preset debug
./build/debug/bin/engine
```

On a machine without CUDA: `cmake --preset cpu && cmake --build --preset cpu && ctest --preset cpu` (prints `ENGINE_CUDA = OFF`).

## Step 7: CI and team conventions (together, ~45 min)

GitHub's free runners have no GPU, so CI checks three things that do not need one: CPU tests on Linux and macOS, that all CUDA code **compiles** for your GPU and cloud GPUs, and formatting. GPU tests run on your laptop before merging.

### 7.1 `.github/workflows/ci.yml`

The workflow in the repo is the source of truth (it has since gained `-LE todo_joe`, CLI11, a Python syntax job, job timeouts and apt retries). Notes kept from the guide: the CUDA container matches the 13.3 toolkit; the CUDA job only compiles and never runs `ctest`, because there is no GPU; after the first green run, mark the `cpu`, `cuda-compile`, and `format` checks as required for `main`.

### 7.2 Formatting before every commit

Run `scripts/format.sh` before committing (or set up VS Code "format on save" with clang-format). Both of you should use clang-format 18 or newer so formatting does not flip-flop between machines.

### 7.3 Pull request template

`.github/pull_request_template.md` (in the repo it lives at `.github/workflows/pull_request_template.md`):

```markdown
## What changed

## How it was tested
- [ ] `ctest --preset debug` passes on the GPU (laptop or over SSH)
- [ ] `compute-sanitizer` clean for new kernels

## Performance (if relevant)
| Metric | Before | After |
| --- | --- | --- |

## Screenshots (ncu / nsys, if relevant)
```

### 7.4 Issues, milestones, and branches

Create the milestones and labels once:

```
gh api repos/{owner}/{repo}/milestones -f title="M1 Foundations" -f due_on="2026-10-04T23:59:59Z"
gh api repos/{owner}/{repo}/milestones -f title="M2 Correct naive engine" -f due_on="2026-10-18T23:59:59Z"
gh api repos/{owner}/{repo}/milestones -f title="M3 Paged KV + batching" -f due_on="2026-11-01T23:59:59Z"
gh api repos/{owner}/{repo}/milestones -f title="M4 Fast" -f due_on="2026-11-08T23:59:59Z"
gh api repos/{owner}/{repo}/milestones -f title="M5 Shipped" -f due_on="2026-11-15T23:59:59Z"
gh label create kernels --color 5319e7
gh label create runtime --color 1d76db
gh label create perf --color d93f0b
gh label create later --color cccccc --description "Part 2, do not start before M5"
```

**Conventions:**

| Thing | Rule | Example |
| --- | --- | --- |
| Branch names | `<area>/<short-task>` | `kernel/rmsnorm-naive`, `runtime/block-manager` |
| Commit messages | Imperative, under ~72 characters, what and why | `Add vectorized RMSNorm (2.1x DRAM throughput)` |
| PR size | Under ~400 changed lines | Split big work into stacked PRs |
| Reviews | The other person reviews within 24 hours | Approve or leave specific comments |
| Merging | Squash merge, delete branch | Keeps `main` history readable |
| Performance claims | Every perf PR includes before/after numbers from the `release` preset | See PR template |

## Step 8: Day-1 smoke tests and troubleshooting

Run this list once when setup is finished, and again any time something feels off (after a Windows or driver update, for example). If every line passes, the environment is not the problem.

### 8.1 Smoke test checklist

| # | Where | Command | Expected |
| --- | --- | --- | --- |
| 1 | Laptop (WSL) | `nvidia-smi` | RTX 5060 Laptop GPU listed, driver version shown |
| 2 | Laptop (WSL) | `nvcc --version` | 12.8 or newer |
| 3 | Laptop (WSL) | `~/scratch/query` | Compute capability 12.0, ~8 GB |
| 4 | Laptop (WSL) | `ncu --section SpeedOfLight -k vadd -c 1 ~/scratch/vadd` | A DRAM Throughput %, no permission error |
| 5 | Laptop (WSL) | `python ~/scratch/hf_check.py` | `device: cuda (12, 0)` and a sensible sentence |
| 6 | Laptop (WSL) | `cmake --preset debug && cmake --build --preset debug && ctest --preset debug` | `ENGINE_CUDA = ON`, tests pass |
| 7 | Laptop (WSL) | `compute-sanitizer ./build/debug/bin/gpu_tests` | `ERROR SUMMARY: 0 errors` |
| 8 | Mac | `cmake --preset cpu && cmake --build --preset cpu && ctest --preset cpu` | `ENGINE_CUDA = OFF`, tests pass |
| 9 | Mac | `ssh gpu nvidia-smi` | Your GPU, from the Mac |
| 10 | Mac | `python hf_check.py` (path adjusted) | A sensible sentence on CPU |
| 11 | GitHub | Open a trivial PR (edit README) | All CI jobs green |

### 8.2 Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `nvidia-smi` not found or "no devices" in WSL | Old Windows driver, or WSL1 | Update the Windows NVIDIA driver; check `wsl -l -v` shows version 2; `wsl --update`; reboot |
| GPU broke after installing CUDA packages | A `cuda` or `cuda-drivers` meta-package installed a Linux driver | `sudo apt remove --purge` the driver packages it pulled in, keep only `cuda-toolkit-*`, then `wsl --shutdown` |
| `nvcc fatal: Unsupported gpu architecture 'compute_120'` | Toolkit older than 12.8 on the PATH | Check `which nvcc`; fix `PATH` in `~/.bashrc` to `/usr/local/cuda/bin` |
| `ERR_NVGPUCTRPERM` from `ncu` | GPU performance counters not allowed | Enable "allow access to all users" in the NVIDIA app / Control Panel developer settings, reboot, `wsl --shutdown` |
| PyTorch warns `sm_120 is not compatible` | Wheel built for an older CUDA | Reinstall with `--index-url https://download.pytorch.org/whl/cu128` (or newer) |
| CMake prints `ENGINE_CUDA = OFF` on the laptop | `nvcc` not on PATH when CMake ran | `source ~/.bashrc`, delete `build/debug`, reconfigure |
| Builds are very slow | Repo is under `/mnt/c` | Move it to `~/code` inside Linux |
| Out of memory while compiling | WSL memory limit too low | Raise `memory=` in `.wslconfig`, or build with `cmake --build --preset debug -j 4` |
| Partner's `ssh gpu` times out | Laptop asleep, WSL stopped, or firewall | Check Tailscale shows the laptop online; open an Ubuntu terminal (or the scheduled task); recheck both firewall rules from Step 5.3 |
| `ssh` connects to Windows instead of Ubuntu | Wrong port | Use port 2222 as configured, not 22 |
| clangd shows red errors in `.cu` files that compile fine | Clang's CUDA support lags nvcc | Make sure `.clangd` exists (it now points at `build/debug` directly); the errors are cosmetic if `cmake --build` succeeds |
| Numbers vary a lot between runs | Laptop throttling or on battery | Plug in, Best performance mode, log clocks with `nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv -lms 500` |

### 8.3 Where to go next

Once the checklist is all green, open the build plan and start Phase 1 of the learning plan: the PyTorch forward-pass rewrite and the CUDA exercises. Your first real code in this repo is the hand-written `bandwidth_test.cu` and `DeviceBuffer` for milestone M1.

### Sources

Checked on Sep 24, 2026:

- [NVIDIA CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) (driver rules, toolkit-only install, profiler requirements)
- [NVIDIA WSL-Ubuntu CUDA repository](https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/) (available toolkit versions)
- [Microsoft: WSL networking, mirrored mode](https://learn.microsoft.com/en-us/windows/wsl/networking) (requirements and Hyper-V firewall rule)
- [NVIDIA CUDA GPUs compute capability list](https://developer.nvidia.com/cuda-gpus)
- [Docker Hub nvidia/cuda tags](https://hub.docker.com/r/nvidia/cuda/tags) (CI container)
- [PyTorch install selector](https://pytorch.org/get-started/locally/)
