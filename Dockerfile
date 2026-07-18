FROM ghcr.io/coreweave/ml-containers/torch-extras:b1daf4e-base-cuda13.2.1-ubuntu24.04-torch2.12.0-vision0.27.0-audio2.11.0-abi1

ARG PYTHON_VERSION=3.13
ENV PATH="/opt/senpai-venv/bin:/root/.local/bin:${PATH}"

# Install system utilities
RUN apt-get update && \
    apt-get install -y netcat-openbsd gettext-base && rm -rf /var/lib/apt/lists/* && \
    curl -fsSL https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -o /usr/local/bin/yq && \
    chmod +x /usr/local/bin/yq

# Install kubectl
RUN curl -fsSL "https://dl.k8s.io/release/$(curl -fsSL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
      -o /usr/local/bin/kubectl && chmod +x /usr/local/bin/kubectl

# The CoreWeave CUDA 13.2 image currently ships Python 3.12. Install a
# uv-managed CPython 3.13 and make it the image-wide python/python3 runtime.
RUN python3 -m pip install --no-cache-dir --upgrade uv && \
    uv python install "$PYTHON_VERSION" && \
    uv venv --python "$PYTHON_VERSION" /opt/senpai-venv && \
    python --version

ENV SENPAI_PYTHON=/opt/senpai-venv/bin/python \
    UV_PROJECT_ENVIRONMENT=/opt/senpai-venv \
    UV_PYTHON=/opt/senpai-venv/bin/python \
    UV_PYTHON_DOWNLOADS=never

# Resolve and install project Python dependencies into the image.
COPY pyproject.toml /tmp/senpai/
RUN cd /tmp/senpai && \
    uv export --python 3.13 --no-dev --no-emit-project --format requirements.txt > requirements.txt && \
    uv pip install --python "$SENPAI_PYTHON" --upgrade -r requirements.txt && \
    python -c 'import openhands.sdk, sys, torch; assert sys.version_info[:2] == (3, 13); assert torch.__version__.startswith("2.13."); assert torch.version.cuda and torch.version.cuda.startswith("13.")' && \
    rm -rf /root/.cache/uv

# Install Claude Code + gh
RUN curl -fsSL https://claude.ai/install.sh | bash || true && \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg 2>/dev/null && \
    chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli-stable.list > /dev/null && \
    apt-get update && apt-get install -y gh && rm -rf /var/lib/apt/lists/*

# Use the standard NVIDIA Container Toolkit contract on supported x86_64 hosts.
# The extension targets cover current AWS NVIDIA families from T4 through
# Blackwell, including Ada targets that use Ampere-compatible PyTorch cubins.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=13.0" \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;10.3;12.0+PTX"

COPY scripts/senpai-gpu-smoke-test.py /usr/local/bin/senpai-gpu-smoke-test
RUN chmod +x /usr/local/bin/senpai-gpu-smoke-test && \
    python -c 'import torch; required={"sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"}; assert required <= set(torch.cuda.get_arch_list())' && \
    python -c 'from torch.utils.cpp_extension import _get_cuda_arch_flags; assert _get_cuda_arch_flags()'

WORKDIR /workspaces
