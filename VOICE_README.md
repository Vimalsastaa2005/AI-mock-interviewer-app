# AI Resume-Based Mock Interviewer - Voice Mode

## What was added

`app.py` supports microphone answers with local Whisper transcription as well
as typed answers, and speaks interview questions aloud using offline
text-to-speech. Edit the transcription before sending. The bot's messages remain
visible in the chat; use the audio player's **Play** control if your browser
blocks automatic playback. **Replay question** plays the latest question again.

`app_voice.py` additionally supports voice answers: user speech -> local Whisper
transcription -> editable transcript -> existing interview scoring, feedback,
and next-question logic. Typing remains available as a fallback.

## Setup and run

For spoken questions, typed answers, and microphone answers:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

For the optional voice-mode app with its own microphone controls:

```powershell
pip install -r requirements_voice.txt
streamlit run app_voice.py
```

`st.audio_input` requires a recent Streamlit release. Your browser will ask for
microphone permission; microphone recording works when the app is running on
localhost. Allow access to record an answer.

## Technologies and concepts

- **Speech-to-text:** faster-whisper transcribes recordings on the local CPU
  using the `int8` compute type.
- **Text-to-speech:** pyttsx3 speaks interview questions and sends temporary
  audio to the browser for playback.
- **Local inference and privacy:** Ollama and Whisper process interview data
  locally; no API keys or cloud speech services are used.
- **Session state:** Streamlit session state keeps the transcript and prevents a
  recording from being transcribed again after a rerun.
- **Fallback design:** edit the transcript before sending, or use the existing
  text input if recording, transcription, or speech playback is unavailable.
- **First microphone use:** the selected Whisper model is downloaded and loaded
  locally. Keep the page open while the loading message is displayed; later
  recordings reuse the cached model.

## Limitations

- Accents and background noise can affect transcription accuracy.
- The first run downloads the selected Whisper model; small models can make
  mistakes.
- Use the **tiny** model on low-RAM laptops.
- On Linux, pyttsx3 may require `espeak` to be installed. Windows and macOS
  generally work out of the box.
