FROM ghcr.io/coreweave/ml-containers/torch-extras:c0f5966-base-cuda13.2.0-ubuntu24.04-torch2.11.0-vision0.26.0-audio2.11.0-abi1

ARG PYTHON_VERSION=3.13
ENV PATH="/opt/senpai-venv/bin:/root/.local/bin:${PATH}"

# Install Node.js 22 + yq
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs netcat-openbsd gettext-base && rm -rf /var/lib/apt/lists/* && \
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
    python -c 'import openhands.sdk, sys, torch; assert sys.version_info[:2] == (3, 13); assert torch.version.cuda and torch.version.cuda.startswith("13.")'

# Install Claude Code + gh
RUN curl -fsSL https://claude.ai/install.sh | bash || true && \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg 2>/dev/null && \
    chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli-stable.list > /dev/null && \
    apt-get update && apt-get install -y gh && rm -rf /var/lib/apt/lists/*

# Install weave-claude-plugin and patch inactivity timeout (10 min → 12 h).
# `weave-claude-plugin install` must run at runtime (needs GitHub access to
# clone the marketplace repo), so entrypoint scripts handle that step.
RUN npm install -g weave-claude-plugin@latest && \
    (sed -i "s/const INACTIVITY_TIMEOUT_MS = 10 \* 60 \* 1_000;/const INACTIVITY_TIMEOUT_MS = 12 * 60 * 60 * 1_000;/" \
      "$(npm root -g)/weave-claude-plugin/dist/daemon.js" || true)

RUN mkdir -p /root/.weave_claude_plugin/logs && \
    cat > /root/.weave_claude_plugin/settings.json <<'EOF'
{
  "log_file": "/root/.weave_claude_plugin/logs/daemon.log",
  "weave_project": null,
  "wandb_api_key": null,
  "debug": false,
  "version": "0.1.0",
  "daemon_socket": "/root/.weave_claude_plugin/daemon.sock"
}
EOF

WORKDIR /workspaces
