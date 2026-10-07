"""AI Home Exercise Program — FastAPI + Gemini inference backend.

This is a clinical decision-support prototype. It uses synthetic/demo data and
requires licensed-clinician review before any output is acted upon.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from threading import Lock
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError, field_validator


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
load_dotenv(BASE_DIR / ".env")

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "minimal")
REQUEST_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "75000"))

_client_lock = Lock()
_gemini_client = None

app = FastAPI(
    title="AI Home Exercise Program API",
    version="1.0.0",
    description="AI-assisted HEP drafting with clinician review and safety gates.",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


GoalName = Literal["pain", "mobility", "strength", "balance", "activity"]
RegionName = Literal["knee", "shoulder", "back", "hip", "ankle", "neck"]


class PatientContext(BaseModel):
    age: int = Field(ge=0, le=120)
    sex: str = Field(min_length=1, max_length=40)
    primary_diagnosis: str = Field(min_length=2, max_length=160)
    episode: str = Field(min_length=2, max_length=100)
    affected_side: str = Field(min_length=2, max_length=40)
    mobility: str = Field(min_length=2, max_length=120)
    latest_reported_pain: int = Field(ge=0, le=10)
    precautions: str = Field(min_length=2, max_length=500)
    recent_signals: list[str] = Field(default_factory=list, max_length=5)


class PatientInput(BaseModel):
    region: RegionName
    phase: Literal["early", "mid", "return"]
    clinical_summary: str = Field(min_length=10, max_length=1600)
    pain: int = Field(ge=0, le=10)
    equipment: str = Field(min_length=2, max_length=200)
    duration_minutes: Literal[15, 20, 30]
    goals: list[GoalName] = Field(
        min_length=1,
        max_length=5,
    )
    red_flag: bool = False
    patient_context: PatientContext | None = None

    @field_validator("clinical_summary")
    @classmethod
    def strip_summary(cls, value: str) -> str:
        return value.strip()

class Exercise(BaseModel):
    code: str = Field(
        pattern=r"^[A-Z]{2,3}$",
        description="Two or three recognizable uppercase initials from the exercise name",
    )
    name: str = Field(description="Patient-friendly exercise name")
    instructions: str = Field(description="One concise form or safety cue")
    dose: str = Field(description="Sets, repetitions, hold time, or duration")
    frequency: str = Field(description="How often the patient performs it")
    estimated_minutes: int = Field(
        ge=1,
        le=12,
        description="Estimated minutes this exercise takes, including brief rests",
    )


class AIExercise(BaseModel):
    name: str = Field(min_length=3, max_length=60)
    instructions: str = Field(min_length=8, max_length=180)
    dose: str = Field(min_length=2, max_length=80)
    frequency: str = Field(min_length=2, max_length=40)
    goal_tags: list[GoalName] = Field(
        min_length=1,
        max_length=2,
        description="One or two selected goals directly addressed by this exercise",
    )
    target_region: RegionName = Field(
        description="Primary body region targeted by this exercise; must match the selected body region",
    )


class AIDraft(BaseModel):
    exercises: list[AIExercise] = Field(min_length=4, max_length=4)


class MonitoringItem(BaseModel):
    name: str
    schedule: str
    details: str


class CarePlan(BaseModel):
    program_title: str
    program_subtitle: str
    diagnosis_summary: str
    intensity: Literal["Low", "Low–moderate", "Moderate"]
    estimated_session_minutes: Literal[15, 20, 30]
    session_structure: str = Field(description="Concise time breakdown that adds up to the target session length")
    equipment_note: str = Field(description="Equipment used, or why no equipment is needed in this phase")
    review_interval_days: int = Field(ge=3, le=14)
    exercises: list[Exercise] = Field(min_length=3, max_length=5)
    monitoring: list[MonitoringItem] = Field(min_length=2, max_length=4)
    rationale: str
    safety_note: str
    clinician_review_required: bool


class CarePlanResponse(CarePlan):
    model: str
    generation_mode: Literal["gemini"] = "gemini"
    generation_ms: int = Field(ge=0)


RED_FLAG_PATTERNS: dict[str, str] = {
    r"\bchest pain\b": "chest pain",
    r"\bshortness of breath\b|\bdifficulty breathing\b": "breathing difficulty",
    r"\bfever\b": "fever",
    r"\bcalf tenderness\b|\bcalf pain\b": "calf pain/tenderness",
    r"\bnew (?:or worsening )?(?:\w+\s+){0,2}(?:weakness|numbness)\b": "new neurologic deficit",
    r"\bbowel or bladder\b|\bsaddle anesthesia\b": "possible cauda equina symptoms",
    r"\bsudden swelling\b|\brapid swelling\b": "sudden swelling",
}

REGION_RED_FLAG_PATTERNS: dict[str, dict[str, str]] = {
    "hip": {
        r"\b(?:cannot|can't|unable to) (?:walk|bear weight|put weight on (?:the|my) leg)\b": "unable to bear weight",
        r"\b(?:obvious )?(?:hip|leg|joint) deformity\b|\b(?:hip|leg|joint) (?:looks? )?(?:deformed|misshapen)\b": "possible deformity",
        r"\b(?:hot|red)(?:,? and)? swollen (?:hip|joint)\b|\b(?:hip|joint) (?:is )?(?:hot|red)(?:,? and)? swollen\b": "hot swollen joint",
    },
    "ankle": {
        r"\b(?:cannot|can't|unable to) (?:walk|bear weight|put weight on (?:the|my) (?:foot|ankle))\b": "unable to bear weight",
        r"\b(?:obvious )?(?:ankle|foot|joint) deformity\b|\b(?:ankle|foot|joint) (?:looks? )?(?:deformed|misshapen|at an odd angle)\b": "possible deformity",
        r"\b(?:hot|red)(?:,? and)? swollen (?:ankle|joint)\b|\b(?:ankle|joint) (?:is )?(?:hot|red)(?:,? and)? swollen\b": "hot swollen joint",
    },
    "neck": {
        r"\b(?:new|worsening|progressive) (?:problems? |difficulty )?(?:walking|with balance|with coordination)\b|\b(?:unsteady|unstable) (?:walking|gait)\b": "new walking or balance problem",
        r"\b(?:new|worsening|progressive) (?:hand )?(?:clumsiness|loss of coordination)\b": "new coordination problem",
        r"\b(?:new|worsening|progressive) (?:arm|leg|limb) (?:weakness|numbness)\b": "new limb neurologic deficit",
        r"\bloss of (?:bladder|bowel) control\b": "new bowel or bladder dysfunction",
    },
}

NEGATION_PATTERN = re.compile(r"\b(?:no|denies|denied|without|negative for)\b")
CONTRAST_PATTERN = re.compile(r"\b(?:but|however|except)\b")
PAIN_SCORE_PATTERN = re.compile(r"\b(10|[0-9])\s*/\s*10\b")
MINUTE_DOSE_PATTERN = re.compile(
    r"\b(\d{1,2})(?:\s*(?:to|[-–])\s*(\d{1,2}))?\s*(?:continuous\s*)?(?:minutes?|mins?)\b",
    re.IGNORECASE,
)

REGION_LABELS = {
    "knee": "Knee",
    "shoulder": "Shoulder",
    "back": "Lower back",
    "hip": "Hip",
    "ankle": "Ankle",
    "neck": "Neck",
}
PHASE_LABELS = {
    "early": "Early rehabilitation",
    "mid": "Strength & mobility",
    "return": "Return to activity",
}
PHASE_TITLE_LABELS = {
    "early": "Early Rehabilitation",
    "mid": "Strength & Mobility",
    "return": "Return-to-Activity",
}
TITLE_REGION_LABELS = {
    "knee": "Knee",
    "shoulder": "Shoulder",
    "back": "Lower Back",
    "hip": "Hip",
    "ankle": "Ankle",
    "neck": "Neck",
}
GOAL_DISPLAY_LABELS = {
    "pain": "Reduce pain",
    "mobility": "Restore mobility",
    "strength": "Build strength",
    "balance": "Improve balance",
    "activity": "Improve daily function",
}
EXERCISE_BUDGETS = {
    15: [2, 3, 3, 2],
    20: [3, 4, 4, 4],
    30: [5, 5, 5, 10],
}
GOAL_ALIGNMENT_PATTERNS: dict[str, str] = {
    "pain": r"\bpain relief\b|\breduc(?:e|ing) pain\b",
    "mobility": r"\bmobility\b|\brange of motion\b",
    "strength": r"\bstrength\b|\bstrengthening\b",
    "balance": r"\bimprov(?:e|es|ing) balance\b|\bbalance training\b|\bbalance exercise\b|\bsingle[- ]leg (?:balance|stance)\b|\btandem (?:balance|stance)\b",
    "activity": r"\bimprov(?:e|es|ing) daily function\b|\bdaily functional capacity\b|\bfunctional independence\b|\bactivities of daily living\b|\badls\b",
}
EXERCISE_REGION_PATTERNS: dict[str, str] = {
    "knee": r"\bknee (?:extension|flexion)\b|\b(?:straighten|bend)(?: your| the)? (?:left |right )?knee\b|\bquadriceps sets?\b|\bterminal knee extension\b",
    "shoulder": r"\bshoulder (?:flexion|extension|abduction|adduction|rotation)\b|\brotator cuff\b|\bscapular retraction\b",
    "back": r"\b(?:lumbar|lower back) (?:flexion|extension|rotation)\b|\bpelvic tilts?\b|\bcat[- ]cow\b",
    "hip": r"\bhip (?:flexion|extension|abduction|adduction|external rotation|internal rotation)\b|\bclamshells?\b",
    "ankle": r"\bankle (?:dorsiflexion|plantarflexion|inversion|eversion|pumps?)\b|\bcalf raises?\b",
    "neck": r"\b(?:neck|cervical) (?:flexion|extension|rotation|retraction)\b|\bchin tucks?\b",
}


def is_negated(summary: str, match_start: int) -> bool:
    """Return True when a red-flag term is negated in the same clause."""
    clause_start = max(
        summary.rfind(".", 0, match_start),
        summary.rfind(";", 0, match_start),
        summary.rfind("\n", 0, match_start),
    ) + 1
    context = summary[clause_start:match_start]
    negations = list(NEGATION_PATTERN.finditer(context))
    if not negations:
        return False
    words_after_negation = context[negations[-1].end() :]
    return not bool(CONTRAST_PATTERN.search(words_after_negation))


def detect_red_flags(patient: PatientInput) -> list[str]:
    if patient.red_flag:
        return ["red flag selected by clinician"]
    safety_context = [patient.clinical_summary]
    if patient.patient_context is not None:
        safety_context.extend(
            [patient.patient_context.precautions, *patient.patient_context.recent_signals]
        )
    summary = ". ".join(safety_context).lower()
    flags: list[str] = []
    patterns = {**RED_FLAG_PATTERNS, **REGION_RED_FLAG_PATTERNS.get(patient.region, {})}
    for pattern, label in patterns.items():
        matches = re.finditer(pattern, summary)
        if any(not is_negated(summary, match.start()) for match in matches):
            flags.append(label)
    return flags


def detect_input_conflicts(patient: PatientInput) -> list[str]:
    conflicts: list[str] = []
    scores = [int(match.group(1)) for match in PAIN_SCORE_PATTERN.finditer(patient.clinical_summary)]
    if scores and all(abs(score - patient.pain) >= 2 for score in scores):
        conflicts.append(
            f"The clinical summary reports pain {scores[0]}/10, but the pain slider is {patient.pain}/10."
        )
    if patient.phase == "return" and patient.pain >= 7:
        conflicts.append(
            "Return-to-activity phase with pain 7/10 or higher requires clinician review before generation."
        )
    return conflicts


def build_prompt(patient: PatientInput) -> str:
    goal_names = {
        "pain": "reduce pain",
        "mobility": "restore mobility",
        "strength": "build strength",
        "balance": "improve balance",
        "activity": "improve daily function",
    }
    goals = ", ".join(goal_names[item] for item in patient.goals) or "maintain safe activity"
    budgets = EXERCISE_BUDGETS[patient.duration_minutes]
    if patient.patient_context is None:
        patient_profile = "- Structured patient profile: not provided"
    else:
        context = patient.patient_context
        recent_signals = "; ".join(context.recent_signals) or "none provided"
        patient_profile = f"""- Age / sex: {context.age} / {context.sex}
- Primary diagnosis: {context.primary_diagnosis}
- Episode: {context.episode}
- Affected side: {context.affected_side}
- Mobility status: {context.mobility}
- Latest patient-reported pain: {context.latest_reported_pain}/10
- Precautions: {context.precautions}
- Recent signals: {recent_signals}"""

    return f"""
Draft only the exercise content for a conservative home exercise program that
a licensed physical therapist must review. This is not a diagnosis.

PATIENT INPUT (synthetic demonstration data)
{patient_profile}
- Body region: {REGION_LABELS[patient.region]}
- Recovery phase: {PHASE_LABELS[patient.phase]}
- Clinical summary: {patient.clinical_summary}
- Pain level selected by the clinician for this plan: {patient.pain}/10
- Available equipment: {patient.equipment}
- Target session length: {patient.duration_minutes} minutes
- Goals: {goals}

REQUIREMENTS
0. Recovery phase describes the patient's current rehabilitation stage and
   controls exercise difficulty. Goals describe the intended treatment outcomes.
   They are separate concepts: for example, an early-rehabilitation plan may
   improve daily function without implying readiness to return to sport or full activity.
1. Return exactly four low-risk exercises in this order of time budgets:
   {budgets[0]}, {budgets[1]}, {budgets[2]}, and {budgets[3]} minutes.
   Every exercise must primarily train the selected Body region. Set every
   target_region field to "{patient.region}" exactly. Do not include an exercise
   whose primary movement belongs to another joint, even if adjacent-joint
   training could sometimes be clinically useful.
2. Every dose must fit inside its assigned time budget. Use an exact duration,
   not a time range. The exercise name and instruction must describe the same
   movement.
3. Do not invent imaging findings, diagnoses, surgery dates, or clinician
   clearance. Do not imply tissue damage or healing unless the input says so.
   Respect every precaution in both the structured patient profile and the
   clinical summary. Use the clinician-selected pain level for this plan while
   treating the latest patient-reported pain as contextual history.
4. Use only the listed equipment. Do not force equipment when it is unnecessary.
5. The selected Goals list is exhaustive, not a suggestion. Every exercise
   must address only selected goals. Never introduce, name, or
   justify an unselected goal because of the recovery phase or clinical context.
   In particular, if improve balance is not selected, do not prescribe balance,
   single-leg stance, or tandem-stance exercises.
6. For every exercise, set goal_tags to one or two values copied exactly from
   the selected Goals list. Do not return any unselected goal tag.
7. Keep each instruction to one short sentence.
8. Return only the requested structured JSON. Do not use Markdown.
""".strip()


def get_gemini_client(api_key: str):
    global _gemini_client
    if _gemini_client is None:
        with _client_lock:
            if _gemini_client is None:
                _gemini_client = genai.Client(
                    api_key=api_key,
                    http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
                )
    return _gemini_client


def exercise_code(name: str) -> str:
    stop_words = {"a", "an", "and", "for", "of", "the", "with"}
    words = [word for word in re.findall(r"[A-Za-z]+", name) if word.lower() not in stop_words]
    if len(words) >= 2:
        return "".join(word[0] for word in words[:3]).upper()
    letters = re.sub(r"[^A-Za-z]", "", name).upper()
    return (letters[:2] or "EX").ljust(2, "X")


def equipment_note(patient: PatientInput, draft: AIDraft) -> str:
    if patient.equipment.lower().startswith("no equipment"):
        return "No special equipment is required for this draft."
    exercise_text = " ".join(
        f"{exercise.name} {exercise.instructions}" for exercise in draft.exercises
    ).lower()
    detected: list[str] = []
    for keyword, label in (("band", "a resistance band"), ("chair", "a chair"), ("towel", "a towel")):
        if keyword in patient.equipment.lower() and keyword in exercise_text:
            detected.append(label)
    if detected:
        if len(detected) == 1:
            used = detected[0]
        else:
            used = ", ".join(detected[:-1]) + f" and {detected[-1]}"
        return f"Uses {used} from the available equipment list."
    return "No special equipment is required for the selected exercises; listed equipment remains optional."


def normalize_draft_doses(patient: PatientInput, draft: AIDraft) -> AIDraft:
    """Fit minute-based model doses into deterministic UI time slots."""
    budgets = EXERCISE_BUDGETS[patient.duration_minutes]
    normalized_exercises: list[AIExercise] = []
    for index, exercise in enumerate(draft.exercises):
        budget = budgets[index]

        def fit_minutes(match: re.Match[str]) -> str:
            stated_maximum = int(match.group(2) or match.group(1))
            return f"{min(stated_maximum, budget)} minutes"

        normalized_exercises.append(
            exercise.model_copy(
                update={"dose": MINUTE_DOSE_PATTERN.sub(fit_minutes, exercise.dose)}
            )
        )
    return draft.model_copy(update={"exercises": normalized_exercises})


def validate_goal_alignment(patient: PatientInput, draft: AIDraft) -> None:
    selected = set(patient.goals)
    for index, exercise in enumerate(draft.exercises):
        unexpected = set(exercise.goal_tags) - selected
        if unexpected:
            labels = ", ".join(sorted(unexpected))
            raise ValueError(f"Exercise {index + 1} uses unselected goal tags: {labels}")

    draft_text = " ".join(
        f"{exercise.name} {exercise.instructions}" for exercise in draft.exercises
    ).lower()
    for goal, pattern in GOAL_ALIGNMENT_PATTERNS.items():
        if goal not in selected and re.search(pattern, draft_text):
            raise ValueError(f"Draft introduces the unselected goal: {goal}")


def validate_region_alignment(patient: PatientInput, draft: AIDraft) -> None:
    for index, exercise in enumerate(draft.exercises):
        if exercise.target_region != patient.region:
            raise ValueError(
                f"Exercise {index + 1} targets {exercise.target_region}, not {patient.region}"
            )
        exercise_text = f"{exercise.name} {exercise.instructions}".lower()
        for other_region, pattern in EXERCISE_REGION_PATTERNS.items():
            if other_region != patient.region and re.search(pattern, exercise_text):
                raise ValueError(
                    f"Exercise {index + 1} contains a primary {other_region} movement, not {patient.region}"
                )


def safety_note(patient: PatientInput) -> str:
    base = "Stop and contact the care team for sharp or rapidly worsening pain, new weakness, numbness, or tingling"
    if patient.region == "back":
        return base + ", or any new bowel or bladder changes."
    if patient.region == "knee":
        return base + ", rapid swelling, calf pain, fever, chest pain, or shortness of breath."
    if patient.region == "hip":
        return base + ", inability to bear weight, visible deformity, or a hot swollen joint with fever."
    if patient.region == "ankle":
        return base + ", inability to bear weight, visible deformity, a cold or numb foot, or a hot swollen joint with fever."
    if patient.region == "neck":
        return base + ", new walking or coordination problems, or any new bowel or bladder changes."
    return base + ", sudden swelling, chest pain, or shortness of breath."


def program_title(patient: PatientInput) -> str:
    return f"{TITLE_REGION_LABELS[patient.region]} {PHASE_TITLE_LABELS[patient.phase]} Program"


def program_subtitle(patient: PatientInput) -> str:
    goals = " · ".join(GOAL_DISPLAY_LABELS[goal] for goal in patient.goals)
    return f"Goals: {goals} · clinician review required"


def program_rationale(patient: PatientInput) -> str:
    goal_phrases = {
        "pain": "reduce discomfort",
        "mobility": "restore comfortable movement",
        "strength": "build controlled strength",
        "balance": "improve balance",
        "activity": "support daily function",
    }
    selected = [goal_phrases[goal] for goal in patient.goals]
    if len(selected) == 1:
        purpose = selected[0]
    else:
        purpose = ", ".join(selected[:-1]) + f", and {selected[-1]}"
    return (
        f"This {REGION_LABELS[patient.region].lower()} program uses conservative, "
        f"clinician-review exercises to {purpose} while respecting the selected "
        "recovery phase, pain level, and available equipment."
    )


def build_care_plan(patient: PatientInput, draft: AIDraft) -> CarePlan:
    budgets = EXERCISE_BUDGETS[patient.duration_minutes]
    exercises = [
        Exercise(
            code=exercise_code(exercise.name),
            name=exercise.name,
            instructions=exercise.instructions,
            dose=exercise.dose,
            frequency=exercise.frequency,
            estimated_minutes=budgets[index],
        )
        for index, exercise in enumerate(draft.exercises)
    ]
    intensity: Literal["Low", "Low–moderate", "Moderate"]
    if patient.pain >= 7 or patient.phase == "early":
        intensity = "Low"
    elif patient.pain >= 4:
        intensity = "Low–moderate"
    else:
        intensity = "Moderate" if patient.phase == "return" else "Low–moderate"
    exercise_minutes = sum(budgets)
    diagnosis = (
        patient.patient_context.primary_diagnosis
        if patient.patient_context is not None
        else f"{REGION_LABELS[patient.region]} symptoms"
    )
    return CarePlan(
        program_title=program_title(patient),
        program_subtitle=program_subtitle(patient),
        diagnosis_summary=(
            f"{diagnosis} · {PHASE_LABELS[patient.phase].lower()} · "
            f"pain {patient.pain}/10"
        ),
        intensity=intensity,
        estimated_session_minutes=patient.duration_minutes,
        session_structure=(
            f"3 min gentle warm-up · {exercise_minutes} min exercise sequence · 2 min recovery"
        ),
        equipment_note=equipment_note(patient, draft),
        review_interval_days=3 if patient.pain >= 7 else 7,
        exercises=exercises,
        monitoring=[
            MonitoringItem(
                name="Pain tracking",
                schedule="Each session",
                details="Record pain from 0–10 before and after the program.",
            ),
            MonitoringItem(
                name="Exercise adherence",
                schedule="Daily",
                details="Log completion and note movements that aggravated symptoms.",
            ),
        ],
        rationale=program_rationale(patient),
        safety_note=safety_note(patient),
        clinician_review_required=True,
    )


def generate_with_gemini(patient: PatientInput) -> CarePlan:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "paste_your_key_here":
        raise HTTPException(
            status_code=503,
            detail="Gemini API key is not configured. Add GEMINI_API_KEY to the local .env file.",
        )

    try:
        client = get_gemini_client(api_key)
        base_prompt = build_prompt(patient)
        correction = ""
        for quality_attempt in range(3):
            interaction = None
            for network_attempt in range(2):
                try:
                    interaction = client.interactions.create(
                        model=MODEL_NAME,
                        input=base_prompt + correction,
                        store=False,
                        generation_config={
                            "thinking_level": THINKING_LEVEL,
                            "max_output_tokens": 1050,
                        },
                        response_format=[
                            {
                                "type": "text",
                                "mime_type": "application/json",
                                "schema": AIDraft.model_json_schema(),
                            }
                        ],
                    )
                    break
                except Exception as exc:
                    transient_errors = {
                        "InternalServerError",
                        "ServerError",
                        "ServiceUnavailableError",
                    }
                    if network_attempt == 0 and type(exc).__name__ in transient_errors:
                        time.sleep(1)
                        continue
                    raise
            if interaction is None:
                raise RuntimeError("Gemini returned no interaction")
            try:
                draft = AIDraft.model_validate_json(interaction.output_text)
                draft = normalize_draft_doses(patient, draft)
                validate_goal_alignment(patient, draft)
                validate_region_alignment(patient, draft)
                return build_care_plan(patient, draft)
            except (ValidationError, ValueError) as validation_exc:
                if quality_attempt < 2:
                    issue = str(validation_exc).replace("\n", " ")[:400]
                    correction = (
                        "\n\nCORRECTION REQUIRED\n"
                        f"The previous draft failed validation: {issue}\n"
                        "Return a completely new draft with exactly four exercises using only "
                        "the selected Goals and the selected Body region. Do not add an unselected "
                        "goal or an exercise primarily targeting another joint. Set every "
                        f'target_region to "{patient.region}" and ensure every required JSON field '
                        "matches the schema."
                    )
                    continue
                raise
        raise RuntimeError("Gemini returned no valid draft")
    except HTTPException:
        raise
    except Exception as exc:
        error_name = type(exc).__name__
        if "Timeout" in error_name:
            message = f"Gemini did not respond within {REQUEST_TIMEOUT_MS // 1000} seconds. Please try again."
        elif error_name == "ValidationError":
            message = (
                "Gemini returned incomplete structured exercise data after automatic retry. "
                "Please generate once more."
            )
        elif error_name == "ValueError":
            message = (
                "Gemini kept introducing a treatment goal or body region that was not selected. "
                "The draft was blocked to preserve the clinician's inputs; please generate once more."
            )
        else:
            message = (
                f"Gemini request failed: {error_name}. "
                "Check the API key, model name, and free-tier quota."
            )
        raise HTTPException(
            status_code=502,
            detail=message,
        ) from exc


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str | bool]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    return {
        "status": "ok",
        "gemini_configured": bool(api_key and api_key != "paste_your_key_here"),
        "model": MODEL_NAME,
        "fast_mode": True,
    }


@app.post("/api/generate-plan", response_model=CarePlanResponse)
def generate_plan(patient: PatientInput) -> CarePlanResponse:
    flags = detect_red_flags(patient)
    if flags:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "clinical_red_flag",
                "message": "Plan generation paused. A licensed clinician must review the red flag first.",
                "flags": flags,
            },
        )

    conflicts = detect_input_conflicts(patient)
    if conflicts:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "input_conflict",
                "message": conflicts[0],
                "conflicts": conflicts,
            },
        )

    started_at = time.perf_counter()
    plan = generate_with_gemini(patient)
    generation_ms = int((time.perf_counter() - started_at) * 1000)
    return CarePlanResponse(
        **plan.model_dump(),
        model=MODEL_NAME,
        generation_mode="gemini",
        generation_ms=generation_ms,
    )
