# okDriver Voice Assistant

Wake-word activated voice assistant prototype.

## Features

- Wake-word detection with fuzzy matching
- Speech-to-text pipeline
- LLM response generation with fallback
- Text-to-speech output
- Latency logging (STT, LLM, TTS, TOTAL)
- Demo mode with English + Hindi/Hinglish sample queries

## Files

- app.py — main application
- llm.py — LLM integration and fallback
- stt.py — STT helper module
- setup.ps1 — Windows setup script
- scripts/setup_vosk_model.ps1 — model setup script
- requirements.txt — dependencies

## Usage (for GitHub reviewers)

### 1) Setup

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

### 2) Run demo (recommended)

```powershell
.\.venv\Scripts\python.exe .\app.py --model .\vosk-model-en-us-0.22-lgraph --demo
```

### 3) Test microphone

```powershell
.\.venv\Scripts\python.exe .\app.py --model .\vosk-model-en-us-0.22-lgraph --test-mic
```

### 4) Live mode

```powershell
.\.venv\Scripts\python.exe .\app.py --model .\vosk-model-en-us-0.22-lgraph --debug
```

## Architecture Diagram

```text
Microphone
  ↓
Wake-word detection
  ↓
Audio capture (silence detection)
  ↓
Speech-to-text
  ↓
LLM response generation
  ↓
Text-to-speech
  ↓
Speaker output
  ↓
Latency logging
```

## Latency Benchmark

Command used:

```powershell
.\.venv\Scripts\python.exe .\app.py --model .\vosk-model-en-us-0.22-lgraph --demo
```

Sample results:

| Query | STT | LLM | TTS | TOTAL |
|---|---:|---:|---:|---:|
| how are you | 0.105s | 0.000s | 3.289s | 3.394s |
| what time is it | 0.101s | 0.000s | 0.980s | 1.082s |
| tell me a joke | 0.102s | 0.000s | 0.969s | 1.071s |
| kya haal chaal | 0.101s | 0.000s | 0.958s | 1.059s |
| aaj mausam kaisa hai | 0.101s | 0.000s | 1.040s | 1.141s |
| mujhe ek mazedaar baat bato | 0.101s | 0.000s | 1.036s | 1.137s |

Average total latency: ~1.30s
