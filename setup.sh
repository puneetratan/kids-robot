#!/bin/bash
# setup.sh - one-shot dev environment setup for kids-robot
#
# - installs Homebrew (if missing)
# - installs uv (if missing)
# - installs Python 3.12 (via uv) and creates ./venv
# - installs all deps from requirements.txt
# - installs Ollama via brew (if missing), starts it bound to 0.0.0.0
#   so a Raspberry Pi on the LAN can reach it, and pulls the models used
#   by the pipeline
set -euo pipefail
cd "$(dirname "$0")"

export PATH="$HOME/.local/bin:$PATH"

# --- Homebrew -----------------------------------------------------------
# Needs an interactive sudo prompt on first install, so this script must
# be run from a real terminal (not piped/non-interactive).
if [[ "$OSTYPE" == "darwin"* ]]; then
    if ! command -v brew >/dev/null 2>&1; then
        echo ">>> Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        if [ -x /opt/homebrew/bin/brew ]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        elif [ -x /usr/local/bin/brew ]; then
            eval "$(/usr/local/bin/brew shellenv)"
        fi
    else
        echo ">>> Homebrew already installed."
    fi
fi

# --- uv ---------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo ">>> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# --- Python 3.12 + venv -------------------------------------------------
echo ">>> Installing Python 3.12..."
uv python install 3.12

echo ">>> Creating virtual environment (./venv)..."
uv venv --python 3.12 venv

echo ">>> Installing dependencies from requirements.txt..."
source venv/bin/activate
uv pip install -r requirements.txt
deactivate

# --- Ollama -------------------------------------------------------------
# Installed via Homebrew (not the ollama.com .app installer) so that
# `brew services` can manage it as a launchd service - it survives
# reboots/logins and doesn't need a GUI app running.
if [[ "$OSTYPE" == "darwin"* ]]; then
    if ! brew list ollama >/dev/null 2>&1; then
        echo ">>> Installing Ollama (brew formula \`ollama\`)..."
        brew install ollama
    else
        echo ">>> Ollama already installed."
    fi

    echo ">>> Starting Ollama (bound to 0.0.0.0:11434 so the Pi can reach it)..."
    launchctl setenv OLLAMA_HOST "0.0.0.0:11434"
    brew services start ollama
else
    if ! command -v ollama >/dev/null 2>&1; then
        echo ">>> Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo ">>> Ollama already installed."
    fi

    echo ">>> Starting Ollama (bound to 0.0.0.0:11434 so the Pi can reach it)..."
    (OLLAMA_HOST=0.0.0.0:11434 ollama serve > /tmp/ollama.log 2>&1 &) || true
fi
sleep 2

echo ""
echo ">>> Pulling Ollama models..."
ollama pull qwen2.5vl:7b
ollama pull llava:7b
ollama pull mxbai-embed-large
ollama pull hf.co/puneetsiet2005/robotai-v7-grpo

echo ">>> Setup complete."
echo "    Activate the venv with: source venv/bin/activate"
echo "    Ollama version: $(ollama --version 2>&1 | tail -1)"
