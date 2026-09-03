#!/bin/bash
cd ~/kids-robot
MAC="http://192.168.1.220:11434"  
export OLLAMA_HOST=$MAC
export APLAY_DEVICE=pipewire
export AGENT_HOST=$MAC
export AGENT_MODEL=qwen2.5:7b
export TURN_SECONDS_90=0.6
export PIPER_BIN=$HOME/kids-robot/piper/piper
export PIPER_VOICE=$HOME/kids-robot/en_US-ryan-low.onnx
export TILT_DOWN=140
FLAGS=""
if [ "$1" = "--speak-all" ]; then FLAGS="--speak-all"; shift; fi
python3 shiva_agent.py $FLAGS --goal "$*"
alias getframes='scp -r pi@192.168.1.7:~/kids-robot/agent_runs/frames ~/Downloads/'
while true; do
  rsync -a --info=progress2 pi@192.168.1.7:~/kids-robot/agent_runs/frames/ ~/Downloads/frames/
  sleep 5
done
