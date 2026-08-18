# Federico — AI Voice Assistant

**Federico** is a local AI-powered voice assistant developed in Python, combining speech recognition, local LLM inference, text-to-speech and real-time audio visualization.

The project is designed to run primarily on Windows and interact with the user through natural spoken commands.

## Features

* 🎙️ Voice activation using the wake word **"Federico"**
* 🧠 Local AI inference with **Ollama**
* 🗣️ Speech recognition using **Faster-Whisper**
* 🔊 Text-to-Speech responses
* 📊 Real-time audio spectrum visualization
* ⏰ Voice-controlled alarms and timers
* 🌦️ Weather information
* 📰 News headlines
* 🎵 Spotify integration
* 🔇 Voice command to stop speech output
* 🎚️ Automatic audio input device detection
* 💬 Conversation history

## Tech Stack

* **Python**
* **Faster-Whisper**
* **Ollama**
* **NumPy**
* **SoundDevice**
* **Pydub**
* **Edge TTS**
* **PyQt5**
* **PyQtGraph**
* **Requests**

## Project Structure

```text
federico5/
├── federico5.py
├── spectrum_ui.py
├── alarma.wav
├── sonido1.wav
├── sonido2.wav
├── requirements.txt
├── .gitignore
└── README.md
```

### `federico5.py`

Main application.

Handles:

* microphone input
* wake-word detection
* speech recognition
* command processing
* Ollama communication
* text-to-speech
* alarms
* weather
* news
* assistant state management

### `spectrum_ui.py`

Real-time audio visualization interface developed with **PyQt5 + PyQtGraph**.

It displays:

* frequency spectrum
* waveform
* VU meter
* assistant state

## Requirements

You need:

* Python 3
* A microphone
* Internet connection for online services such as TTS, news and weather
* Ollama installed locally

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/eseuerreefe/federico5.git
cd federico5
```

### 2. Create a virtual environment

Recommended:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

## Ollama Setup

Federico uses **Ollama** to run the language model locally.

Install Ollama on your computer and make sure the `ollama` command is available.

The default model configured by the project is:

```text
llama3:8b
```

Download it with:

```bash
ollama pull llama3:8b
```

You can verify Ollama is working with:

```bash
ollama list
```

By default, Federico connects to:

```text
http://127.0.0.1:11434
```

## Running Federico

Start the assistant with:

```bash
python federico5.py
```

At startup, the application:

1. Checks the connection with Ollama.
2. Detects the available microphone/input device.
3. Opens the real-time spectrum interface.
4. Loads the Faster-Whisper speech recognition model.
5. Calibrates ambient noise.
6. Enters passive listening mode.

Then say:

```text
Federico
```

to activate the assistant.

## Example

```text
User: Federico

Federico: listening...

User: ¿Qué tiempo hace hoy?
```

The assistant transcribes the voice input, processes the request and responds using synthesized speech.

## Audio Spectrum

The project includes a custom real-time audio visualization interface.

It uses FFT-based audio analysis and displays:

* frequency spectrum
* peak hold
* waveform
* volume level
* assistant status

The spectrum interface can also be executed independently:

```bash
python spectrum_ui.py
```

## Configuration

Several parameters can be customized using environment variables.

### Ollama model

```text
OLLAMA_MODEL
```

Default:

```text
llama3:8b
```

### Ollama server

```text
OLLAMA_URL
```

Default:

```text
http://127.0.0.1:11434
```

### TTS voice

```text
EDGE_TTS_VOICE
```

Default:

```text
es-ES-ElviraNeural
```

### Weather location

The default location can be changed using:

```text
CITY_NAME
LAT
LON
```

## Speech Recognition

Federico uses **Faster-Whisper** for local speech-to-text processing.

The current configuration loads the Whisper `small` model on CPU using INT8 computation.

This allows speech recognition to run locally without sending microphone recordings to a remote transcription API.

## Architecture

```text
Microphone
    │
    ▼
Voice Activity Detection
    │
    ▼
Faster-Whisper
    │
    ▼
Wake Word / Command Detection
    │
    ▼
Intent ───────────────► Built-in Actions
    │                    ├── Alarm
    │                    ├── Weather
    │                    ├── News
    │                    └── Spotify
    │
    ▼
Ollama / Local LLM
    │
    ▼
Text-to-Speech
    │
    ▼
Audio Output
```

## Privacy

The core language model and speech recognition components are designed to run locally.

Some optional functionality, including weather, news and online TTS, communicates with external services.

## Purpose

This project was developed as a personal software engineering and AI project to explore the integration of:

* speech recognition
* local generative AI
* audio processing
* real-time interfaces
* natural-language interaction
* automation through voice commands

## Author

Developed by **eseuerreefe**.

---

If you find the project interesting, feel free to explore the source code and its implementation.
