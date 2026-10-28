# AI Mock Interviewer

An offline, privacy-focused resume-based interview chatbot built with Streamlit
and a local Ollama language model. Upload a resume and practice a conversational
interview based on your skills and projects, with follow-up questions, answer
scores, and a downloadable final report.

## How it works

```text
Resume upload (PDF/TXT)
        ↓
Text extraction and resume analysis (structured JSON)
        ↓
Conversational interview (resume-specific questions and follow-ups)
        ↓
Silent per-answer evaluation (score and feedback)
        ↓
Final report and downloadable TXT
```

1. Upload a text-based PDF or TXT resume. PDF text is extracted locally with
   `pypdf`; scanned or image-only PDFs are not supported.
2. The app extracts a compact profile from the resume locally so starting the
   interview does not wait for a slow model: candidate name, detected skills,
   projects, education, and an estimated experience level. You can optionally
   enter a target role.
3. The interviewer asks questions about the candidate's actual skills and
   projects, asks at most one topic-specific follow-up before the final main
   question, and includes a behavioral question. Question history is used to
   avoid repeating local questions.
4. Answers are evaluated on a 0–10 scale for relevance, context, actions,
   reasoning, and outcomes rather than length alone. The optional sidebar
   feedback setting shows answer-specific improvement advice, a reusable answer
   structure, and a template for the relevant question type. A local rubric
   keeps evaluation available if Ollama is slow or unavailable.
5. The final report includes a score and percentage, questions and short answers,
   strengths, areas to improve, resume improvement tips, and a hiring-readiness
   verdict. Download it as a TXT file.

## Technologies

| Technology | Purpose |
| --- | --- |
| Python 3.10+ | Application logic |
| Streamlit | Web interface, chat, and session state |
| Ollama | Local LLM inference |
| Qwen 2.5 1.5B | Default local model (`MODEL` in `app.py`) |
| pypdf | Extracting text from PDF resumes |

## Setup and run

### Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.com) installed

### 1. Download the model

In PowerShell or a terminal, run:

```powershell
ollama pull qwen2.5:1.5b
```

Ollama usually starts its local service automatically on Windows. If it is not
running, open the Ollama app or run `ollama serve` in a separate terminal.

### 2. Install Python dependencies

From the project directory:

```powershell
py -m pip install -r requirements.txt
```

### 3. Start the app

```powershell
py -m streamlit run app.py
```

Open the local address Streamlit prints (usually `http://localhost:8501`).

If Ollama reports that the model is missing, run
`ollama pull qwen2.5:1.5b`. To use Llama 3.2 3B instead, pull it with
`ollama pull llama3.2:3b` and change `MODEL` in `app.py` to
`"llama3.2:3b"`.

## Concepts demonstrated

- **Prompt engineering:** resume-grounded questions, difficulty matching,
  question history, and controlled conversational follow-ups.
- **Structured JSON output:** machine-readable Ollama answer evaluation and
  report generation, parsed and validated by Python.
- **Answer coaching:** feedback includes a concrete improvement, a reusable
  response structure, and an example template to adapt truthfully.
- **Local inference and privacy:** resume text and interview answers are
  processed by Ollama on the local machine; no cloud API or API key is used.
- **Streamlit session state:** preserving the resume, analysis, messages,
  question count, scores, and interview completion status across reruns.
- **Document parsing:** extracting text from PDF and TXT documents.
- **Conversational follow-ups:** reacting to answers and asking no more than one
  follow-up for each interview question.
- **Evaluation and reporting:** collecting scores and synthesizing strengths,
  improvement areas, resume tips, and readiness guidance.

## Limitations

- Scanned or image-only PDFs are not supported; upload a text-based PDF or TXT.
- The resume is limited to about 6,000 characters in analysis prompts.
- Small local models can misunderstand resumes, produce inconsistent scores,
  or generate imperfect questions and advice. Treat the report as practice
  guidance, not a hiring decision.
- Ollama and the selected model must be installed locally and have sufficient
  system resources.
- Interview startup uses local resume extraction and does not wait for Ollama.
  Answer scoring uses one Ollama request with an 8-second wait limit and falls
  back to a local rubric. The question count is entered manually with no fixed
  maximum.
  Check Ollama with `ollama serve` and test the model using
  `ollama run qwen2.5:1.5b "Reply with hello"`.

## Future improvements

- Add OCR for scanned resumes.
- Support editable scoring rubrics and interview styles.
- Add speech input and output for voice practice.
- Improve evaluation consistency with larger local models and configurable
  assessment criteria.
