# kids-robot

A voice- and vision-driven robot for kids, running on a Raspberry Pi 5, with
the heavy models (vision-language, embeddings, and a fine-tuned LLM) served
from a separate machine (e.g. a Mac) over the network via Ollama's
OpenAI-compatible API.

"move forward" -> wheels turn, returns immediately
"what's the weather" -> answered while still moving
"what is this?" -> camera + vision-language model describes it
"stop" -> wheels stop

## Architecture

- **Raspberry Pi 5** — runs the robot itself: motor control, servos, camera,
  microphone, and the voice pipeline that routes each utterance to an action,
  a knowledge-base lookup, a live-data tool, or the vision model.
- **Ollama host** (this repo's `setup.sh` sets this up) — runs the models that
  are too slow for a Pi: the vision-language model, the embedding model, and
  the fine-tuned kids-robot LLM. Exposed to the Pi over LAN via Ollama's
  `/v1` OpenAI-compatible endpoint.

## Key files

| File | Purpose |
|---|---|
| `drive.py` | Keyboard driving / motor control (non-blocking GPIO). |
| `shiva.py` | Main voice pipeline: classifies each utterance into ACTION, KNOWLEDGE_BASE, LIVE_DATA, or VISION, and routes it. |
| `vision.py` | Camera capture + vision-language description (moves the neck while inferring, so the wait reads as "thinking"). |
| `voice_pipeline*.py` | Earlier/alternate voice pipeline variants (RAG, routed). |
| `motor_test.py`, `servo_test.py` | Hardware pin-mapping and sanity tests. |
| `classifier_*.py` | Utterance classifiers (keyword and LLM-based) used to route requests. |
| `rag_pipeline_*.py`, `build_rag_*.py` | Retrieval-augmented generation over the knowledge base (ChromaDB). |
| `gate_a0_safety.py` | Safety gate for content classified as needing a check (runs after routing, so plain ACTION commands skip it). |
| `run_ragas_eval.py`, `t2_nightly.sh` | Nightly evaluation of the pipeline against a frozen golden dataset (faithfulness, relevancy, recall, precision). |
| `Modelfile` | Ollama Modelfile giving the fine-tuned model its "fun kids science robot" persona. |
| `setup.sh` | One-shot environment setup (see below). |

## Setup

Run from a real terminal (Homebrew's installer needs an interactive `sudo`
prompt, so this won't work piped or non-interactively):

```bash
./setup.sh
```

This will:

1. Install **Homebrew** (macOS only), if missing.
2. Install **uv** (Python package/version manager), if missing.
3. Install **Python 3.12** via `uv` and create `./venv`.
4. Install all dependencies from `requirements.txt` into `./venv`.
5. Install and start **Ollama**, if missing.
6. Pull the models used by the pipeline:
   - `qwen2.5vl:7b` — vision-language model (see `vision.py`)
   - `llava:7b` — alternate vision-language model
   - `mxbai-embed-large` — embedding model for the RAG knowledge base
   - `hf.co/puneetsiet2005/robotai-v7-grpo` — fine-tuned kids-robot LLM

Afterwards, activate the venv with:

```bash
source venv/bin/activate
```

## Serving models to the Raspberry Pi

Ollama exposes an OpenAI-compatible API at `/v1`. `setup.sh` already starts
it bound to all interfaces (`0.0.0.0:11434`) instead of the default
`127.0.0.1`, so it's reachable from the Pi over LAN. On macOS this is done
via Homebrew (`brew install ollama` + `brew services start ollama`, with
`launchctl setenv OLLAMA_HOST "0.0.0.0:11434"` set beforehand) so it also
persists across reboots and logins.

From the Pi, point any OpenAI client at `http://<host-ip>:11434/v1`:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<host-ip>:11434/v1",
    api_key="ollama",  # unused, but required by the SDK
)

resp = client.chat.completions.create(
    model="hf.co/puneetsiet2005/robotai-v7-grpo",
    messages=[{"role": "user", "content": "hello"}],
)
```

Note: this exposes port 11434 to the whole LAN with no authentication —
fine on a trusted home network, but worth knowing.

## Nightly evaluation

`t2_nightly.sh` re-runs the frozen golden dataset through the current
pipeline every night (intended as a cron job on the Pi) and appends dated
scores to `t2_history.csv`, so pipeline regressions show up over time.
