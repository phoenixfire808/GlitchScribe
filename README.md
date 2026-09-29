# GlitchScribe

GlitchScribe is a Windows desktop speech-to-text and voice-dictation app. It records microphone audio locally, transcribes it with a model you configure, and can insert the final transcript into the focused text field. The project also includes optional dictation controls and editing helpers.

> This is a source-code release. It does not include model weights, a prebuilt installer, or a configured environment. Expect setup work, and review the privacy and safety notes before using it.

## Features

- Global hotkey dictation with a desktop status/preview interface.
- Local speech-recognition backends, with explicit model selection and no silent model fallback.
- Guarded text insertion that checks the target and focus before typing.
- Optional streaming preview and local editing tools.
- Windows-only integrations for audio capture, tray controls, and desktop automation.

## Requirements

- Windows 10 or later.
- Python 3.11 or later.
- A microphone and a compatible local ASR model/backend. Model files are intentionally not included.
- Some optional GPU-backed model configurations require a compatible NVIDIA GPU and matching CUDA/PyTorch packages.

## Setup

1. Clone or download this repository.
2. Create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   ```

3. Install dependencies using the instructions for your selected backend. `requirements.txt` currently describes the CUDA-enabled development setup and may need adjustment for your hardware.
4. Obtain model files from their upstream project and configure GlitchScribe to use them locally. Do not commit model weights or credentials.
5. Start the app with `START_LOCAL_VOICE.cmd` or run `python keepalive_launcher.py` from this folder.

The first launch may require microphone, hotkey, and model configuration. Verify the selected input device and test in a non-sensitive text field before using dictation in other applications.

## Privacy and safety

- Audio and transcripts can contain sensitive information. Review the app's settings and backend before use.
- Keep personal configuration, API keys, local model files, audio recordings, logs, and runtime state out of Git.
- Optional language-model integrations are separate from local transcription. Configure and enable them deliberately; do not assume every feature is offline.
- Text insertion affects the currently focused application. Test insertion safeguards before relying on them.

## Development

Run the automated tests from the project root:

```powershell
python -m pytest
```

Some UI and Windows integration tests require Windows desktop components. A passing unit suite does not replace testing microphone capture, model loading, or text insertion on your own system.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dependency notices. No project license is included yet, so reuse and redistribution rights are not granted by this repository's public visibility alone.
