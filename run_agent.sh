#!/bin/bash
cd ~/kids-robot
export OLLAMA_HOST=http://192.168.1.6:11434
export APLAY_DEVICE=pipewire
export AGENT_HOST=http://192.168.1.6:11434
export AGENT_MODEL=qwen2.5:7b
export TURN_SECONDS_90=0.6
export PIPER_BIN=$HOME/kids-robot/piper/piper
export PIPER_VOICE=$HOME/kids-robot/en_US-ryan-low.onnx
export TILT_DOWN=140
python3 shiva_agent.py --goal "$*"
