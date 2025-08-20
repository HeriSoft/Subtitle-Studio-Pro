SubtitleStudio Pro
- Added HuggingFace auto-download support for Whisper model (large-v3) into ./models
- Settings include asr_device and hf_auto_download
- Auto-start hf model download in background on app start if enabled
- Keep prior v6 fixes: progress bars, ffmpeg local, downloader merge, logs, cancel buttons
Notes:
- Ensure ffmpeg.exe and ffprobe.exe are placed in project root.
- To install dependencies: pip install -r requirements.txt and pip install huggingface_hub
- For GPU: ensure torch + CUDA available.
