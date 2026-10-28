# 💼 AI Mock Interviewer

An offline, privacy-first **AI Mock Interviewer** web application built with **Streamlit** and a **Local Large Language Model (LLM)** powered by **Ollama**.

Designed for job seekers and students preparing for technical, behavioral, or mixed interviews with real-time scoring, constructive feedback, model answers, and an executive hiring-readiness report.

---

## 📌 Project Overview & How It Works

The AI Mock Interviewer operates in a closed-loop interactive cycle entirely on your local machine—**no API keys, no paid subscriptions, no cloud latency, and 100% data privacy**.

### Architecture & Data Flow

```text
[ User Selects Role & Level ] 
               │
               ▼
   [ Streamlit UI Session ] 
               │
               ▼
 ┌─────────────────────────────┐
 │ 1. Question Generation      │ ◄── Injects previous questions to prevent repetition
 │    Ollama (qwen2.5:1.5b)   │
 └─────────────┬───────────────┘
               │
               ▼
    [ Candidate Submits Answer ]
               │
               ▼
 ┌─────────────────────────────┐
 │ 2. Answer Evaluation        │ ──► Returns strict JSON:
 │    Ollama (format="json")   │     - Score (0-10)
 └─────────────┬───────────────┘     - Strengths & Improvements
               │                     - Ideal model answer
               ▼
    [ Next Question / Loop ]
               │
               ▼
 ┌─────────────────────────────┐
 │ 3. Executive Final Report   │ ──► Aggregate metrics, full breakdown table,
 │    Ollama Synthesis         │     verdict & downloadable .txt report
 └─────────────────────────────┘
```

1. **User Input**: Candidate selects job role (Python, Java, Web, Data Analyst, HR, or Custom), difficulty (Fresher, Intermediate, Experienced), interview type, and number of questions (3–10).
2. **Context-Aware Prompting**: The system generates unique, difficulty-calibrated questions while injecting history to avoid duplicate questions.
3. **Structured JSON Evaluation**: The local LLM evaluates each response and outputs strict JSON containing a numerical score (0–10), strengths, areas for improvement, and a concise model answer.
4. **Final Assessment & Export**: Upon completing all questions, the app tallies scores, generates an executive hiring-readiness verdict via LLM synthesis, and provides a one-click `.txt` downloadable report.

---

## 🛠️ Technologies Used

| Technology | Purpose |
| :--- | :--- |
| **Python 3.10+** | Core programming language |
| **Streamlit** | Interactive web application framework and reactive UI |
| **Ollama** | Local LLM inference engine running offline models |
| **Qwen 2.5 (1.5B)** | Small instruction-tuned language model suitable for lower-memory laptops |

---

## 🚀 Setup & Execution Guide

### Prerequisites
1. **Python 3.10 or higher** installed.
2. **Ollama** installed from [ollama.com](https://ollama.com).

### Step 1: Install Ollama and Download the Model
Ollama must be installed, and the model must be downloaded separately. On Windows, open PowerShell and run:
```bash
ollama pull qwen2.5:1.5b
```
The Ollama app normally starts its local service automatically on Windows. If the command reports that Ollama is not running, open the Ollama app from the Start menu; alternatively, run `ollama serve` in a separate terminal.

### Step 2: Install Python Dependencies
In the project directory, run:
```bash
pip install -r requirements.txt
```
*(On Windows, you can also run: `py -m pip install -r requirements.txt`)*

### Step 3: Run the Application
Launch the Streamlit web app:
```bash
streamlit run app.py
```
*(Or: `py -m streamlit run app.py`)*

The application will open automatically in your browser at `http://localhost:8501`.

---

## 💡 Model choice

The app defaults to **Qwen 2.5 1.5B**, which is suitable for laptops with around 8GB RAM. If your laptop has more memory and you want to try **Llama 3.2 3B** instead:

1. Pull the model:
   ```bash
   ollama pull llama3.2:3b
   ```
2. Change the model configuration near the top of `app.py`:
   ```python
   MODEL = "llama3.2:3b"
   ```
3. Restart the Streamlit app.

---

## 🧠 Concepts Demonstrated (Viva Preparation)

1. **Local LLM Inference**:
   Running AI models entirely on consumer hardware using quantized weights via Ollama, eliminating cloud dependencies and privacy risks.
2. **Structured JSON Output**:
   Using `format="json"` in `ollama.chat()` and strict JSON prompt schemas. Evaluated with defensive parsing (`json.loads`), input validation/clamping (0–10), and an automatic single-retry fallback on parse exceptions.
3. **State Machine & Session State**:
   Managing complex multi-step UI flows (`answering` ➔ `feedback` ➔ `finished`) using `st.session_state` without losing state during page re-renders.
4. **Adaptive Prompt Engineering**:
   Dynamically passing question history into the system prompt to enforce question novelty and tailoring technical depth to the candidate's selected experience level (`Fresher`, `Intermediate`, `Experienced`).
5. **Defensive Error Handling**:
   Catching missing model errors or daemon disconnects gracefully with actionable terminal instructions (`ollama serve` and `ollama pull`).

---

## 🔮 Future Improvements

- **Speech-to-Text & Text-to-Speech**: Incorporating Whisper and Coqui TTS for a voice-interactive viva/interview experience.
- **Resume Parsing**: Allowing candidates to upload PDF resumes to tailor interview questions directly to their project experience.
- **Video & Emotion Analysis**: Using local computer vision models to evaluate eye contact, posture, and facial confidence.
- **Custom Rubrics**: Allowing recruiters to upload custom company scoring rubrics.
