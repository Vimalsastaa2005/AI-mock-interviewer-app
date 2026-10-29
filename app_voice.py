import hashlib
import io
import json
import os
import queue
import re
import tempfile
import threading

import ollama
import streamlit as st
from pypdf import PdfReader

# ==========================================
# CONFIGURATION: Local LLM Model via Ollama
# ==========================================
MODEL = "qwen2.5:1.5b"
OLLAMA_CLIENT = ollama.Client(timeout=20)
OLLAMA_TIMEOUT_SECONDS = 20


@st.cache_resource(
    show_spinner=(
        "Loading the Whisper speech model. The first run may take a few minutes "
        "while it downloads; please keep this page open."
    )
)
def load_whisper_model(model_size: str):
    """Load one local CPU Whisper model and reuse it across Streamlit reruns."""
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe_audio(audio_bytes: bytes, model_size: str) -> str:
    """Transcribe one recording locally and remove its temporary file."""
    audio_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
            audio_file.write(audio_bytes)
            audio_path = audio_file.name
        model = load_whisper_model(model_size)
        segments, _ = model.transcribe(audio_path, beam_size=5)
        return " ".join(segment.text.strip() for segment in segments).strip()
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


def clean_question_for_speech(question: str) -> str:
    """Remove basic Markdown formatting before sending a question to text-to-speech."""
    question = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", question)
    question = re.sub(r"[#*_`~>|]", "", question)
    return re.sub(r"\s+", " ", question).strip()


def create_question_audio(question: str) -> tuple[bytes | None, str | None]:
    """Create temporary offline speech audio and return its bytes for browser playback."""
    audio_path = None
    audio_bytes = None
    audio_error = None
    try:
        import pyttsx3

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
            audio_path = audio_file.name
        engine = pyttsx3.init()
        engine.save_to_file(clean_question_for_speech(question), audio_path)
        engine.runAndWait()
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        engine.stop()
    except Exception as error:
        audio_error = str(error)
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError as error:
                if audio_error is None:
                    audio_error = str(error)
    if audio_error:
        return None, audio_error
    return audio_bytes, None


def ollama_chat(timeout_seconds: float = OLLAMA_TIMEOUT_SECONDS, **request):
    """Run an Ollama request with a hard UI wait limit, even if its client stalls."""
    result_queue = queue.Queue(maxsize=1)

    def run_request() -> None:
        try:
            result_queue.put((True, OLLAMA_CLIENT.chat(**request)))
        except Exception as error:
            result_queue.put((False, error))

    threading.Thread(target=run_request, daemon=True).start()
    try:
        succeeded, result = result_queue.get(timeout=timeout_seconds)
    except queue.Empty as error:
        raise TimeoutError(f"Ollama did not respond within {timeout_seconds:g} seconds.") from error
    if not succeeded:
        raise result
    return result


def handle_ollama_error(e: Exception) -> None:
    """Show local Ollama troubleshooting steps when a model request fails."""
    st.error(
        f"⚠️ **Ollama Error:** {e}\n\n"
        "**Troubleshooting Checklist:**\n"
        "1. Ensure Ollama is running: execute `ollama serve` in your terminal.\n"
        f"2. Download the configured model: execute `ollama pull {MODEL}`.\n"
        "   To use Llama 3.2 instead, set MODEL to `llama3.2:3b` and run "
        "`ollama pull llama3.2:3b`."
    )


def reset_interview_state() -> None:
    """Clear interview progress while keeping the currently uploaded resume."""
    st.session_state.update(
        {
            "started": False,
            "interview_finished": False,
            "analysis": None,
            "chat_messages": [],
            "question_history": [],
            "history": [],
            "question_count": 0,
            "total_questions": 5,
            "current_question": "",
            "followup_used": False,
            "final_report": {},
            "role": "",
            "level": "",
            "fallback_notice": "",
            "ollama_unavailable": False,
            "last_audio_id": None,
            "transcript": "",
            "transcript_pending": False,
            "pending_voice_answer": "",
            "voice_error": "",
            "latest_tts_audio": None,
            "latest_tts_question": "",
            "tts_error": "",
            "tts_autoplay_pending": False,
        }
    )


def begin_interview(analysis: dict, target_role: str, total_questions: int) -> None:
    """Initialize a fresh chat and greet the candidate using the analyzed resume."""
    role = target_role.strip() or analysis["suggested_role"] or "a role suited to this resume"
    first_question = build_local_question(analysis, 1, total_questions)
    st.session_state.update(
        {
            "started": True,
            "interview_finished": False,
            "analysis": analysis,
            "chat_messages": [
                {
                    "role": "assistant",
                    "content": (
                        f"Hello {analysis['candidate_name'] or 'there'}! "
                        f"I’ve reviewed your resume for **{role}**. "
                        f"Let’s begin: {first_question}"
                    ),
                }
            ],
            "question_history": [first_question],
            "history": [],
            "question_count": 1,
            "total_questions": total_questions,
            "current_question": first_question,
            "followup_used": False,
            "final_report": {},
            "role": role,
            "level": analysis["experience_level"],
            "fallback_notice": "",
            "ollama_unavailable": False,
            "last_audio_id": None,
            "transcript": "",
            "transcript_pending": False,
            "pending_voice_answer": "",
            "voice_error": "",
            "latest_tts_audio": None,
            "latest_tts_question": "",
            "tts_error": "",
            "tts_autoplay_pending": False,
        }
    )


def use_local_fallback(message: str) -> None:
    """Keep the interview usable and clearly disclose when local heuristics replace Ollama."""
    st.session_state["ollama_unavailable"] = True
    st.session_state["fallback_notice"] = (
        f"{message} Ollama did not respond in time, so this part uses a simple local "
        "fallback. Check Ollama with `ollama serve`, then verify the model with "
        "`ollama pull qwen2.5:1.5b` and `ollama run qwen2.5:1.5b \"Reply with hello\"`."
    )


def build_local_analysis(resume_text: str) -> dict:
    """Extract a basic candidate profile from resume text if Ollama is too slow."""
    text = resume_text[:6000]
    lines = [line.strip(" •-\t") for line in text.splitlines() if line.strip()]
    candidate_name = next(
        (
            line[:80]
            for line in lines
            if len(line) <= 80
            and "@" not in line
            and not re.search(r"\b(resume|curriculum vitae|skills|education)\b", line, re.I)
        ),
        "",
    )
    skill_names = [
        "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "SQL",
        "HTML", "CSS", "React", "Node.js", "Flask", "Django", "Streamlit",
        "AWS", "Azure", "Docker", "Kubernetes", "Git", "Linux", "Excel",
        "Power BI", "Tableau", "Machine Learning", "Data Analysis",
    ]
    skills = [skill for skill in skill_names if re.search(rf"(?<!\w){re.escape(skill)}(?!\w)", text, re.I)]
    projects = []
    in_projects_section = False
    section_heading = re.compile(
        r"^(projects?|academic projects?|personal projects?|work experience|"
        r"experience|education|technical skills|skills|certifications?)\s*:?\s*$",
        re.I,
    )
    for line in lines:
        if section_heading.match(line):
            in_projects_section = bool(re.match(r"^(projects?|academic projects?|personal projects?)", line, re.I))
            continue
        explicit_project = re.match(r"^projects?(?: name| title)?\s*:\s*(.+)$", line, re.I)
        if explicit_project:
            title_and_summary = explicit_project.group(1).strip()
            name, separator, summary = title_and_summary.partition(" - ")
            projects.append(
                {
                    "name": name.strip()[:80],
                    "summary": (summary if separator else title_and_summary)[:240],
                }
            )
        elif in_projects_section and len(line.split()) <= 12:
            projects.append({"name": line[:80], "summary": line[:240]})
        elif in_projects_section and projects:
            projects[-1]["summary"] = (projects[-1]["summary"] + " " + line)[:240]
        if len(projects) >= 5:
            break
    education_line = next(
        (line for line in lines if re.search(r"\b(bachelor|master|degree|diploma|university|college)\b", line, re.I)),
        "",
    )
    years = [int(value) for value in re.findall(r"\b(\d{1,2})\+?\s+years?\b", text, re.I)]
    max_years = max(years, default=0)
    experience_level = (
        "Experienced" if max_years >= 5 or re.search(r"\bexperienced\b", text, re.I) else
        "Intermediate" if max_years >= 2 else
        "Fresher"
    )
    return {
        "candidate_name": candidate_name,
        "skills": skills,
        "projects": projects,
        "education": education_line,
        "experience_level": experience_level,
        "weak_areas": [],
        "suggested_role": "Software Developer" if skills else "a role suited to this resume",
        "leadership_experience": bool(
            re.search(r"\b(team leader|team lead|leadership|hackathon|coordinat\w*)\b", text, re.I)
        ),
    }


def build_local_question(
    analysis: dict,
    question_number: int,
    total_questions: int,
    previous_questions: list[str] | None = None,
) -> str:
    """Create a short resume-grounded question without an LLM response."""
    previous_questions = previous_questions or []
    projects = analysis["projects"]
    skills = analysis["skills"]
    if question_number == total_questions:
        candidates = [
            "Describe a team or project challenge you faced and the specific steps you took to resolve it."
        ]
    elif question_number == 1 and projects:
        project = projects[0]
        candidates = [
            f"What was the goal of your {project['name']} project, and what part did you personally deliver?"
        ]
    elif question_number == 1:
        candidates = [
            "Tell me about a project or hackathon where you contributed: what was the goal and what did you personally do?"
        ]
    elif analysis.get("leadership_experience"):
        candidates = [
            "During a hackathon, how did you coordinate the team and keep work on schedule?",
            "Tell me about a disagreement or coordination problem in a team and how you resolved it.",
            "How did you divide responsibilities and make sure the team delivered a working product?",
        ]
    elif skills:
        skill = skills[(question_number - 2) % len(skills)]
        candidates = [
            f"How did you use {skill} in your work, and what problem did it solve?",
            f"What was one technical decision you made while using {skill}, and why?",
            f"How did you test or verify the part you built with {skill}?",
            f"What would you improve in a solution you created with {skill}?",
        ]
    else:
        candidates = [
            "What is one skill from your resume that you have applied, and what did you accomplish with it?",
            "Which part of your experience best matches the target role, and what evidence can you share?",
        ]
    return next(
        (question for question in candidates if question not in previous_questions),
        candidates[0],
    )


def build_local_evaluation(question: str, answer: str, analysis: dict) -> dict:
    """Score the answer with a relevant rubric and give question-specific coaching."""
    normalized = answer.lower()
    words = re.findall(r"[a-zA-Z0-9+#.]+", normalized)
    if not words:
        return {
            "score": 0,
            "feedback": "No answer was provided, so there is not enough information to assess it.",
            "how_to_improve": "Try again with one direct answer, a specific action you took, and the outcome.",
            "answer_framework": "Direct answer → your action → result.",
            "sample_answer": "I [specific action] to address [the question's issue], which led to [result].",
        }

    answer_terms = set(words)
    action_terms = {
        "organized", "coordinated", "assigned", "delegated", "planned", "prioritized",
        "communicated", "resolved", "implemented", "built", "created", "developed",
        "designed", "tested", "debugged", "analyzed", "led", "managed", "tracked",
        "divided", "reassigned", "discussed", "proposed", "fixed", "worked",
    }
    reasoning_terms = {"because", "so", "therefore", "since", "decided", "choice", "reason"}
    outcome_terms = {
        "result", "completed", "delivered", "improved", "reduced", "increased",
        "saved", "achieved", "launched", "working", "learned", "finished",
    }
    has_action = bool(answer_terms & action_terms)
    has_reasoning = bool(answer_terms & reasoning_terms)
    has_outcome = bool(answer_terms & outcome_terms) or bool(re.search(r"\d+%?|\b\d+\b", normalized))
    has_context = len(words) >= 9 or bool(answer_terms & {"hackathon", "deadline", "conflict", "team", "project"})

    is_leadership = bool(
        re.search(r"\b(team|leader|leadership|coordination|coordinate|conflict|hackathon)\b", question, re.I)
        or (
            analysis.get("leadership_experience")
            and re.search(r"\b(team|leader|leadership|coordination|coordinate|conflict|hackathon)\b", normalized)
        )
    )
    is_project = bool(re.search(r"\b(project|built|developed|implemented)\b", question, re.I))
    question_terms = {
        term.lower()
        for term in re.findall(r"[a-zA-Z]+", question)
        if len(term) > 3
    }
    stop_words = {
        "what", "which", "could", "would", "your", "from", "about", "have",
        "that", "this", "with", "were", "when", "where", "tell", "explain",
        "specific", "personally", "during", "while", "work", "part", "into",
        "does", "did", "make", "made", "using", "used", "their", "there",
    }
    relevant_terms = question_terms - stop_words
    answered_topic = bool(relevant_terms & answer_terms)

    score = 2
    if answered_topic:
        score += 2
    if has_context:
        score += 1
    if has_action:
        score += 2
    if has_reasoning:
        score += 1
    if has_outcome:
        score += 2
    if len(words) >= 35 and (has_action or has_outcome):
        score += 1
    score = min(score, 10)

    if is_leadership:
        if not has_action:
            if answer_terms & {"issue", "issues", "problem", "problems", "conflict"}:
                feedback = "You identified a team coordination problem, but haven't said how you responded to it."
                improvement = "Explain one step you personally took to address the coordination problem and what happened next."
            elif "important" in answer_terms:
                feedback = "You explain why coordination matters, but not a specific action you took as team leader."
                improvement = "Give one concrete example of assigning tasks, resolving a disagreement, or tracking progress—only if you actually did it."
            else:
                feedback = "You named your team role, but haven't described a specific step you personally took."
                improvement = "Describe one concrete action you took to assign work, resolve a disagreement, or track the deadline."
        elif not has_outcome:
            feedback = "You mention an action, but the answer doesn't say whether it helped the team deliver or resolve the issue."
            improvement = "Add what changed after your action: the decision reached, work completed, deadline met, or lesson learned."
        else:
            feedback = "You connect a team action to an outcome; adding your reasoning would make the example clearer."
            improvement = "Briefly explain why you chose that approach and what you learned about leading a team."
        framework = "Team situation → your leadership action → why you chose it → team outcome."
        sample = (
            "During [specific team situation], I [action you actually took, such as "
            "agreeing priorities or coordinating tasks]. I chose this because "
            "[reason]. As a result, [what the team delivered or learned]."
        )
    elif is_project:
        if not has_action:
            feedback = "You refer to the project, but haven't identified your own contribution yet."
            improvement = "Name one feature or task you personally completed, the tool or skill you used if relevant, and its result."
        elif not has_outcome:
            feedback = "Your contribution is clearer, but the answer doesn't mention the project's result."
            improvement = "Add what worked, who benefited, or what you would change based on the outcome."
        else:
            feedback = "You describe a project contribution and outcome; a little more reasoning would make your choices easier to follow."
            improvement = "Explain why you chose that solution and how you checked it worked."
        framework = "Project goal → your contribution → tools or decisions → result."
        sample = (
            "The goal of [project] was [goal]. I personally [your contribution] using "
            "[tool or skill, if applicable]. I chose that approach because [reason], "
            "and the result was [actual outcome]."
        )
    else:
        if not answered_topic and not has_action:
            feedback = "The answer is too brief to show your reasoning or address the main point of the question."
            improvement = "Start with a direct answer, then give one specific example or action that supports it."
        elif not has_action:
            feedback = "You mention a relevant idea, but haven't explained what you did or how you applied it."
            improvement = "Describe the steps you took and why you chose them."
        elif not has_outcome:
            feedback = "You describe an action, but the answer stops before its result or lesson."
            improvement = "Finish with the outcome, evidence that it worked, or what you learned."
        else:
            feedback = "Your answer includes a relevant action and outcome; make the reasoning behind your choice explicit."
            improvement = "Add one sentence explaining why you chose that approach."
        framework = "Direct answer → specific action/example → reasoning → result or lesson."
        sample = (
            "For [specific situation from your experience], I [action you actually took]. "
            "I chose this approach because [reason]. The outcome was [actual result or lesson]."
        )

    return {
        "score": score,
        "feedback": feedback,
        "how_to_improve": improvement,
        "answer_framework": framework,
        "sample_answer": sample,
    }


def build_local_turn(
    analysis: dict,
    question_number: int,
    total_questions: int,
    followup_used: bool,
    previous_questions: list[str] | None = None,
    current_question: str = "",
    answer: str = "",
) -> dict:
    """Continue the interview with one follow-up or one locally generated question."""
    previous_questions = previous_questions or []
    if question_number >= total_questions:
        return {
            "reaction": "Thank you. That completes the interview.",
            "action": "finish",
            "question": "",
        }
    if not followup_used:
        answer_is_about_leadership = bool(
            analysis.get("leadership_experience")
            and re.search(r"\b(team|leader|leadership|coordination|coordinate|conflict|hackathon)\b", answer, re.I)
        )
        if re.search(r"\b(team|leader|leadership|coordination|coordinate|conflict|hackathon)\b", current_question, re.I) or answer_is_about_leadership:
            if re.search(r"\b(assign|delegat|organiz|coordinat|resolv|priorit|communicat|divid|managed|led)\w*\b", answer, re.I):
                followup = "What changed for the team after you took that step?"
            else:
                followup = "What specific action did you personally take to coordinate or support the team?"
        elif re.search(r"\b(project|built|developed|implemented)\b", current_question, re.I):
            followup = "Which specific part did you personally complete, and how did you check it worked?"
        else:
            followup = "Can you give one concrete example of how you applied that idea?"
        return {
            "reaction": "Thanks, that helps. One quick follow-up:",
            "action": "follow_up",
            "question": followup,
        }
    next_number = question_number + 1
    return {
        "reaction": "Thanks. Let's move to the next resume-based question.",
        "action": "next_question",
        "question": build_local_question(
            analysis, next_number, total_questions, previous_questions
        ),
    }


def build_local_report(analysis: dict, history: list[dict]) -> dict:
    """Summarize interview results with transparent, non-LLM report content."""
    average = sum(item["eval"]["score"] for item in history) / max(len(history), 1)
    strengths = ["You completed the practice interview."]
    if average >= 6:
        strengths.append("Your answers generally included useful detail.")
    areas = ["Add clear examples and explain your decisions in more detail."]
    if analysis["weak_areas"]:
        areas.extend(analysis["weak_areas"][:2])
    tips = ["Add measurable outcomes and specific contributions to project descriptions."]
    if analysis["skills"]:
        tips.append(f"Highlight evidence of your {', '.join(analysis['skills'][:3])} skills.")
    return {
        "strengths": strengths,
        "areas_to_improve": areas,
        "resume_improvement_tips": tips,
        "hiring_readiness_verdict": (
            "This practice score is only a rough local estimate. Use it as preparation "
            "guidance, not as a hiring recommendation."
        ),
    }


def extract_resume_text(uploaded_file) -> str:
    """Extract readable text from a TXT or text-based PDF resume."""
    file_bytes = uploaded_file.getvalue()
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    return file_bytes.decode("utf-8-sig", errors="replace").strip()


def analyze_resume(resume_text: str) -> dict | None:
    """Analyze resume text into validated, compact candidate-profile JSON."""
    prompt = (
        "Analyze this resume and return ONLY a valid JSON object with exactly "
        "these fields and types: "
        '{"candidate_name": string, "skills": [string], '
        '"projects": [{"name": string, "summary": string}], '
        '"education": string, "experience_level": '
        '"Fresher|Intermediate|Experienced", "weak_areas": [string], '
        '"suggested_role": string}. Use empty strings or lists when unknown. '
        "Choose experience level from evidence in the resume. Do not invent "
        "facts.\n\nResume text:\n"
        f"{resume_text[:6000]}"
    )
    last_error = None
    for attempt in range(2):
        try:
            current_prompt = prompt
            if attempt:
                current_prompt += (
                    "\nYour previous response was invalid. Return every required "
                    "field with the requested type, use only the three allowed "
                    "experience_level values, and include no extra fields."
                )
            response = ollama_chat(
                model=MODEL,
                messages=[{"role": "user", "content": current_prompt}],
                format="json",
                options={"num_predict": 96, "num_ctx": 2048, "temperature": 0.1},
            )
            data = json.loads(response["message"]["content"])
            valid_levels = {"Fresher", "Intermediate", "Experienced"}
            if (
                not isinstance(data, dict)
                or set(data)
                != {
                    "candidate_name",
                    "skills",
                    "projects",
                    "education",
                    "experience_level",
                    "weak_areas",
                    "suggested_role",
                }
                or not isinstance(data.get("candidate_name"), str)
                or not isinstance(data.get("skills"), list)
                or not all(isinstance(item, str) for item in data["skills"])
                or not isinstance(data.get("projects"), list)
                or not all(
                    isinstance(project, dict)
                    and isinstance(project.get("name"), str)
                    and isinstance(project.get("summary"), str)
                    for project in data["projects"]
                )
                or not isinstance(data.get("education"), str)
                or data.get("experience_level") not in valid_levels
                or not isinstance(data.get("weak_areas"), list)
                or not all(isinstance(item, str) for item in data["weak_areas"])
                or not isinstance(data.get("suggested_role"), str)
            ):
                raise ValueError("The resume analysis did not match the required fields.")
            return {
                "candidate_name": data["candidate_name"],
                "skills": data["skills"],
                "projects": data["projects"],
                "education": data["education"],
                "experience_level": data["experience_level"],
                "weak_areas": data["weak_areas"],
                "suggested_role": data["suggested_role"],
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            last_error = error
        except Exception as error:
            use_local_fallback(
                "Ollama could not analyze the resume within 25 seconds. The resume "
                "profile was extracted locally and may be less accurate."
            )
            return build_local_analysis(resume_text)
    if last_error is not None:
        if isinstance(last_error, (json.JSONDecodeError, KeyError, TypeError, ValueError)):
            use_local_fallback(
                "Ollama returned invalid resume-analysis JSON twice. A basic profile "
                "was extracted locally."
            )
            return build_local_analysis(resume_text)
    return None


def question_category(question_number: int, total_questions: int) -> str:
    """Choose categories so the interview includes projects, skills, and behavior."""
    if question_number == 1:
        return "a specific project from the candidate's resume"
    if question_number == total_questions:
        return "behavioral, based on a real situation or collaboration"
    return "a technical question about one of the candidate's listed skills"


def generate_question(analysis: dict, question_number: int, total_questions: int, previous_questions: list[str]) -> str:
    """Generate one resume-specific question without repeating interview history."""
    if st.session_state.get("ollama_unavailable"):
        return build_local_question(analysis, question_number, total_questions, previous_questions)
    category = question_category(question_number, total_questions)
    prompt = (
        "You are a professional interviewer. Ask exactly ONE short interview "
        "question grounded in the candidate profile JSON below. The question "
        f"must be {category}. Match the candidate's {analysis['experience_level']} "
        "experience level. Refer to a specific listed skill or project where "
        "appropriate. Do not add a greeting, number, explanation, or markdown. "
        "Never repeat or closely resemble a question in the history. Return only "
        f"the question.\nProfile JSON: {json.dumps(analysis, ensure_ascii=False)}"
        f"\nEarlier questions: {json.dumps(previous_questions, ensure_ascii=False)}"
    )
    try:
        response = ollama_chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 48, "num_ctx": 2048, "temperature": 0.2},
        )
        question = response["message"]["content"].strip().strip('"')
        return question or build_local_question(
            analysis, question_number, total_questions, previous_questions
        )
    except Exception as error:
        use_local_fallback(
            "Ollama could not prepare the opening question within 25 seconds. "
            "A resume-based question was created locally."
        )
        return build_local_question(analysis, question_number, total_questions, previous_questions)


def evaluate_answer(question: str, answer: str, analysis: dict) -> dict:
    """Score an answer and return actionable coaching using one fast model request."""
    if st.session_state.get("ollama_unavailable"):
        return build_local_evaluation(question, answer, analysis)
    prompt = (
        "You are a fair interview coach. Score the answer for relevance, correct "
        "reasoning, specific actions/examples, and results. Do not score mainly "
        "by answer length. Give 0 only for empty or completely irrelevant answers, "
        "1-3 for very weak or vague answers, 4-6 for partially correct answers "
        "missing important detail, 7-8 for correct, specific answers, and 9-10 "
        "for excellent answers with strong reasoning and evidence. Do not return "
        "0 or 2 by default. Return ONLY JSON with exactly these keys: "
        '{"score": integer 0-10, "feedback": "one sentence about this answer", '
        '"how_to_improve": "specific advice for this answer", '
        '"answer_framework": "short reusable structure", '
        '"sample_answer": "brief example using only known facts or clear placeholders"}. '
        "Never invent candidate achievements. The score must be an integer.\n"
        f"Candidate profile: {json.dumps(analysis, ensure_ascii=False)}\n"
        f"Question: {question}\nCandidate answer: {answer}"
    )
    try:
        response = ollama_chat(
            timeout_seconds=8,
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"num_predict": 180, "num_ctx": 1536, "temperature": 0.1},
        )
        data = json.loads(response["message"]["content"])
        required_fields = {
            "score", "feedback", "how_to_improve", "answer_framework", "sample_answer"
        }
        if (
            not isinstance(data, dict)
            or set(data) != required_fields
            or type(data.get("score")) is not int
            or not 0 <= data["score"] <= 10
            or any(not isinstance(data.get(field), str) for field in required_fields - {"score"})
        ):
            raise ValueError("The answer evaluation did not match the required fields.")
        rubric_score = build_local_evaluation(question, answer, analysis)["score"]
        data["score"] = round((data["score"] + rubric_score) / 2)
        return data
    except Exception:
        use_local_fallback(
            "Ollama did not return valid coaching within 8 seconds. The answer received a quick local rubric score."
        )
        return build_local_evaluation(question, answer, analysis)


def generate_interview_turn(
    analysis: dict,
    current_question: str,
    answer: str,
    question_number: int,
    total_questions: int,
    followup_used: bool,
    previous_questions: list[str],
) -> dict | None:
    """Create a brief reaction and choose a single follow-up or the next question."""
    if st.session_state.get("ollama_unavailable"):
        return build_local_turn(analysis, question_number, total_questions, followup_used)
    next_number = question_number + 1
    can_follow_up = not followup_used
    next_category = (
        question_category(next_number, total_questions)
        if next_number <= total_questions
        else "none; this is the final interview question"
    )
    prompt = (
        "Continue a conversational mock interview. Return ONLY JSON with exactly "
        'these fields: {"reaction": "one or two short sentences", '
        '"action": "follow_up|next_question|finish", "question": "question text"}. '
        "React briefly and constructively to the answer. Ask at most ONE short "
        "follow-up about the current question, and only if allowed. Otherwise "
        "move to exactly one next main question or finish. Do not ask multiple "
        "questions. If the candidate asks something unrelated, politely steer "
        "back to the interview. Never repeat the question history. For a next "
        "main question, "
        f"the required category is {next_category}. Match the "
        f"{analysis['experience_level']} level. If finishing, set question to an "
        "empty string.\n"
        f"Follow-up allowed: {can_follow_up}\n"
        f"Candidate profile JSON: {json.dumps(analysis, ensure_ascii=False)}\n"
        f"Current question: {current_question}\nCandidate answer: {answer}\n"
        f"Earlier questions: {json.dumps(previous_questions, ensure_ascii=False)}"
    )
    last_error = None
    for attempt in range(2):
        try:
            response = ollama_chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"num_predict": 100, "num_ctx": 2048, "temperature": 0.2},
            )
            data = json.loads(response["message"]["content"])
            if not isinstance(data, dict):
                raise ValueError("The interview turn response must be a JSON object.")
            if set(data) != {"reaction", "action", "question"}:
                raise ValueError("The interview turn response contained unexpected fields.")
            reaction = data.get("reaction")
            action = data.get("action")
            question = data.get("question")
            if (
                not isinstance(reaction, str)
                or not isinstance(question, str)
                or action not in {"follow_up", "next_question", "finish"}
                or (action == "follow_up" and not can_follow_up)
                or (action == "next_question" and next_number > total_questions)
                or (action == "finish" and next_number <= total_questions)
                or (action != "finish" and not question.strip())
            ):
                raise ValueError("The interview turn response contained invalid fields.")
            return {"reaction": reaction, "action": action, "question": question.strip()}
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            last_error = error
        except Exception as error:
            use_local_fallback(
                "Ollama could not continue the interview within 25 seconds. The next turn uses local prompts."
            )
            return build_local_turn(
                analysis, question_number, total_questions, followup_used
            )
    if last_error is not None:
        use_local_fallback(
            "Ollama returned invalid interview-turn JSON twice. The interview is continuing with local prompts."
        )
        return build_local_turn(analysis, question_number, total_questions, followup_used)


def generate_final_report(analysis: dict, history: list[dict], role: str) -> dict | None:
    """Summarize interview performance, resume tips, and hiring readiness."""
    if st.session_state.get("ollama_unavailable"):
        return build_local_report(analysis, history)
    transcript = [
        {
            "question": item["question"],
            "answer": item["answer"][:500],
            "score": item["eval"]["score"],
            "feedback": item["eval"]["feedback"],
        }
        for item in history
    ]
    prompt = (
        "Review this resume-based mock interview. Return ONLY strict JSON with "
        'exactly these fields: {"strengths": [string], '
        '"areas_to_improve": [string], "resume_improvement_tips": [string], '
        '"hiring_readiness_verdict": string}. Keep lists concise and give a '
        "short, realistic verdict. Base all advice on the profile and transcript; "
        "do not invent qualifications.\n"
        f"Target role: {role}\nProfile: {json.dumps(analysis, ensure_ascii=False)}\n"
        f"Transcript: {json.dumps(transcript, ensure_ascii=False)}"
    )
    last_error = None
    for attempt in range(2):
        try:
            response = ollama_chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"num_predict": 160, "num_ctx": 2048, "temperature": 0.2},
            )
            data = json.loads(response["message"]["content"])
            list_fields = ("strengths", "areas_to_improve", "resume_improvement_tips")
            if (
                not isinstance(data, dict)
                or set(data)
                != {
                    "strengths",
                    "areas_to_improve",
                    "resume_improvement_tips",
                    "hiring_readiness_verdict",
                }
                or any(
                    not isinstance(data.get(field), list)
                    or not all(isinstance(item, str) for item in data[field])
                    for field in list_fields
                )
                or not isinstance(data.get("hiring_readiness_verdict"), str)
            ):
                raise ValueError("The final report did not match the required fields.")
            return data
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            last_error = error
        except Exception as error:
            use_local_fallback(
                "Ollama could not compile the final report within 25 seconds. A basic local report is shown."
            )
            return build_local_report(analysis, history)
    if last_error is not None:
        use_local_fallback(
            "Ollama returned invalid final-report JSON twice. A basic local report is shown."
        )
        return build_local_report(analysis, history)


# ==========================================
# STREAMLIT CHAT INTERFACE & SESSION STATE
# ==========================================
st.set_page_config(
    page_title="AI Mock Interviewer",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    [data-testid="stDecoration"], [data-testid="stStatusWidget"],
    #MainMenu, footer, [data-testid="stAppDeployButton"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

for key, default in [
    ("started", False),
    ("interview_finished", False),
    ("resume_text", ""),
    ("resume_signature", ""),
    ("analysis", None),
    ("chat_messages", []),
    ("question_history", []),
    ("history", []),
    ("question_count", 0),
    ("total_questions", 5),
    ("current_question", ""),
    ("followup_used", False),
    ("final_report", {}),
    ("role", ""),
    ("level", ""),
    ("fallback_notice", ""),
    ("ollama_unavailable", False),
    ("last_audio_id", None),
    ("transcript", ""),
    ("transcript_pending", False),
    ("pending_voice_answer", ""),
    ("voice_error", ""),
    ("latest_tts_audio", None),
    ("latest_tts_question", ""),
    ("tts_error", ""),
    ("tts_autoplay_pending", False),
    ("voice_mode", True),
    ("speak_enabled", True),
    ("whisper_model_size", "base"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.sidebar.header("⚙️ Interview Settings")
target_role_input = st.sidebar.text_input("Target job role (optional)", placeholder="e.g., Python Developer")
num_questions = st.sidebar.number_input(
    "Number of Questions", min_value=2, value=5, step=1
)
show_feedback = st.sidebar.checkbox("Show feedback after each answer", value=True)
voice_mode = st.sidebar.checkbox("Voice mode", value=True, key="voice_mode")
speak_enabled = st.sidebar.checkbox(
    "Speak questions aloud", value=True, key="speak_enabled"
)
whisper_model_size = st.sidebar.selectbox(
    "Whisper model size",
    options=["tiny", "base"],
    index=1,
    key="whisper_model_size",
)
uploaded_resume = st.sidebar.file_uploader("Upload your resume", type=["pdf", "txt"])
if uploaded_resume is not None:
    uploaded_bytes = uploaded_resume.getvalue()
    upload_signature = hashlib.sha256(uploaded_bytes).hexdigest()
    if upload_signature != st.session_state.resume_signature:
        reset_interview_state()
        st.session_state.resume_signature = upload_signature
        try:
            st.session_state.resume_text = extract_resume_text(uploaded_resume)
        except Exception as error:
            st.session_state.resume_text = ""
            st.sidebar.error(f"Could not read this resume: {error}")
        if st.session_state.resume_text and len(st.session_state.resume_text) < 100:
            st.session_state.resume_text = ""
            st.sidebar.warning(
                "This resume contains too little readable text. Scanned or image-only "
                "PDFs are not supported; please upload a text-based PDF or TXT file."
            )
        elif st.session_state.resume_text:
            st.sidebar.success("Resume uploaded and text extracted.")
            profile = build_local_analysis(st.session_state.resume_text[:6000])
            begin_interview(profile, target_role_input, int(num_questions))

if st.sidebar.button("↩️ Reset", use_container_width=True):
    reset_interview_state()
    st.rerun()

st.title("💼 AI Mock Interviewer")
st.caption(f"Powered by Local LLM (`{MODEL}`) • 100% Private & Offline")

if st.session_state.analysis:
    with st.expander("Resume Analysis"):
        st.json(st.session_state.analysis)

if st.session_state.fallback_notice:
    st.warning(st.session_state.fallback_notice)

if not st.session_state.started:
    if st.session_state.resume_text:
        if st.sidebar.button("🚀 Start Interview", use_container_width=True, type="primary"):
            profile = build_local_analysis(st.session_state.resume_text[:6000])
            begin_interview(profile, target_role_input, int(num_questions))
            st.rerun()
        st.info("Resume ready. Click **Start Interview** to begin.")
    else:
        st.info("Upload a readable PDF or TXT resume in the sidebar to begin your interview.")
    st.stop()

progress = min(st.session_state.question_count, st.session_state.total_questions)
st.progress(
    progress / st.session_state.total_questions,
    text=f"Question {progress} of {st.session_state.total_questions}",
)

if speak_enabled and st.session_state.current_question:
    if st.session_state.latest_tts_question != st.session_state.current_question:
        audio_bytes, tts_error = create_question_audio(
            st.session_state.current_question
        )
        st.session_state.latest_tts_audio = audio_bytes
        st.session_state.latest_tts_question = st.session_state.current_question
        st.session_state.tts_error = tts_error or ""
        st.session_state.tts_autoplay_pending = audio_bytes is not None
elif not speak_enabled:
    st.session_state.tts_autoplay_pending = False

latest_assistant_index = max(
    (
        index
        for index, message in enumerate(st.session_state.chat_messages)
        if message["role"] == "assistant"
    ),
    default=-1,
)
for index, message in enumerate(st.session_state.chat_messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if index == latest_assistant_index and not st.session_state.interview_finished:
            if speak_enabled and st.session_state.latest_tts_audio:
                st.audio(
                    st.session_state.latest_tts_audio,
                    format="audio/wav",
                    autoplay=st.session_state.tts_autoplay_pending,
                )
                st.session_state.tts_autoplay_pending = False
                if st.button("Replay question", key="replay_latest_question"):
                    st.session_state.tts_autoplay_pending = True
                    st.rerun()

if not st.session_state.interview_finished and speak_enabled and st.session_state.tts_error:
    st.caption("Question audio is unavailable; the text is still shown.")

if not st.session_state.interview_finished:
    voice_answer = None
    if voice_mode:
        audio_recording = st.audio_input("Record your answer")
        if audio_recording is not None:
            audio_bytes = audio_recording.getvalue()
            audio_id = hashlib.sha256(audio_bytes).hexdigest()
            if audio_id != st.session_state.last_audio_id:
                st.session_state.last_audio_id = audio_id
                st.session_state.transcript = ""
                st.session_state.transcript_pending = False
                st.session_state.voice_error = ""
                try:
                    st.session_state.transcript = transcribe_audio(
                        audio_bytes, whisper_model_size
                    )
                    if not st.session_state.transcript:
                        st.session_state.voice_error = (
                            "I could not hear anything, please try again."
                        )
                    else:
                        st.session_state.transcript_pending = True
                except Exception:
                    st.session_state.voice_error = (
                        "Voice transcription is unavailable. Please try recording "
                        "again or type your answer below."
                    )
        else:
            st.caption(
                "Allow microphone access when your browser asks, then record an answer."
            )

        if st.session_state.voice_error:
            st.info(st.session_state.voice_error)
        elif st.session_state.transcript_pending:
            st.text_area(
                "Your answer (edit if needed)",
                key="transcript",
                height=140,
            )
            st.button(
                "Send answer",
                key="send_voice_answer",
                on_click=lambda: st.session_state.update(
                    {
                        "pending_voice_answer": st.session_state.transcript.strip(),
                        "transcript_pending": False,
                    }
                ),
            )

    user_answer = st.chat_input("Type your answer...")
    answer_to_send = st.session_state.pending_voice_answer.strip()
    answer_mode = "Voice" if answer_to_send else "Text"
    if user_answer and user_answer.strip():
        answer_to_send = user_answer.strip()

    if answer_to_send:
        answer = answer_to_send
        evaluated = evaluate_answer(st.session_state.current_question, answer, st.session_state.analysis)
        if evaluated is not None:
            st.session_state.history.append(
                {
                    "question": st.session_state.current_question,
                    "answer": answer,
                    "mode": answer_mode,
                    "eval": evaluated,
                }
            )
            st.session_state.chat_messages.append({"role": "user", "content": answer})
            st.session_state.voice_error = ""
            st.session_state.pending_voice_answer = ""
            turn = build_local_turn(
                st.session_state.analysis,
                st.session_state.question_count,
                st.session_state.total_questions,
                st.session_state.followup_used,
                st.session_state.question_history,
                st.session_state.current_question,
                answer,
            )
            assistant_text = turn["reaction"]
            if show_feedback or answer_mode == "Voice":
                assistant_text += (
                    f"\n\n**Score: {evaluated['score']}/10.** {evaluated['feedback']}"
                    f"\n\n**How to improve:** {evaluated['how_to_improve']}"
                    f"\n\n**Answer structure:** {evaluated['answer_framework']}"
                    f"\n\n**Example you can adapt:** {evaluated['sample_answer']}"
                )
            if turn["action"] == "follow_up":
                st.session_state.current_question = turn["question"]
                st.session_state.followup_used = True
                st.session_state.question_history.append(turn["question"])
                assistant_text += f"\n\n{turn['question']}"
            elif turn["action"] == "next_question":
                st.session_state.question_count += 1
                st.session_state.current_question = turn["question"]
                st.session_state.followup_used = False
                st.session_state.question_history.append(turn["question"])
                assistant_text += (
                    f"\n\n**Question {st.session_state.question_count} of "
                    f"{st.session_state.total_questions}:** {turn['question']}"
                )
            else:
                st.session_state.interview_finished = True
                st.session_state.final_report = generate_final_report(
                    st.session_state.analysis,
                    st.session_state.history,
                    st.session_state.role,
                ) or {}
            st.session_state.chat_messages.append({"role": "assistant", "content": assistant_text})
            st.rerun()

if st.session_state.interview_finished:
    st.header("🎯 Interview Final Report")
    answer_modes = {item.get("mode", "Text") for item in st.session_state.history}
    answer_mode_label = (
        "Voice/Text" if len(answer_modes) > 1 else next(iter(answer_modes), "Text")
    )
    st.markdown(f"**Answer mode:** {answer_mode_label}")
    total_score = sum(item["eval"]["score"] for item in st.session_state.history)
    max_score = len(st.session_state.history) * 10
    percentage = (total_score / max_score * 100) if max_score else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Score", f"{total_score} / {max_score}")
    col2.metric("Percentage", f"{percentage:.1f}%")
    col3.metric("Candidate Target", f"{st.session_state.role} ({st.session_state.level})")

    st.subheader("📋 Question Breakdown")
    st.table(
        [
            {
                "#": f"Q{index + 1}",
                "Question": item["question"],
                "Short Answer": item["answer"][:180],
                "Score": f"{item['eval']['score']}/10",
            }
            for index, item in enumerate(st.session_state.history)
        ]
    )

    report = st.session_state.final_report
    if report:
        st.subheader("Strengths")
        for item in report["strengths"]:
            st.markdown(f"- {item}")
        st.subheader("Areas to Improve")
        for item in report["areas_to_improve"]:
            st.markdown(f"- {item}")
        st.subheader("Resume Improvement Tips")
        for item in report["resume_improvement_tips"]:
            st.markdown(f"- {item}")
        st.subheader("Hiring-Readiness Verdict")
        st.write(report["hiring_readiness_verdict"])

    report_text = (
        "AI MOCK INTERVIEWER - FINAL REPORT\n"
        f"Role: {st.session_state.role} | Level: {st.session_state.level}\n"
        f"Answer mode: {answer_mode_label}\n"
        f"Overall Score: {total_score} / {max_score} ({percentage:.1f}%)\n\n"
        "QUESTION BREAKDOWN:\n"
    )
    for index, item in enumerate(st.session_state.history, 1):
        report_text += (
            f"\nQ{index}: {item['question']}\nAnswer: {item['answer']}\n"
            f"Score: {item['eval']['score']}/10\n"
            f"Feedback: {item['eval']['feedback']}\n"
            f"How to improve: {item['eval']['how_to_improve']}\n"
            f"Answer structure: {item['eval']['answer_framework']}\n"
            f"Example answer: {item['eval']['sample_answer']}\n"
        )
    if report:
        report_text += "\nSTRENGTHS:\n" + "\n".join(f"- {item}" for item in report["strengths"])
        report_text += "\n\nAREAS TO IMPROVE:\n" + "\n".join(
            f"- {item}" for item in report["areas_to_improve"]
        )
        report_text += "\n\nRESUME IMPROVEMENT TIPS:\n" + "\n".join(
            f"- {item}" for item in report["resume_improvement_tips"]
        )
        report_text += f"\n\nHIRING-READINESS VERDICT:\n{report['hiring_readiness_verdict']}\n"
    st.download_button(
        "📥 Download Report (.txt)",
        data=report_text,
        file_name="interview_report.txt",
        mime="text/plain",
        use_container_width=True,
    )
