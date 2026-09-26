"""
Multimodal AI Orchestrator for Study Vision Mode.

This is the layer the architecture diagram calls "Multimodal AI
Orchestrator": it sits between OCR output and the local LLM, turning
(extracted content + a task the user picked, e.g. "explain this circuit")
into one well-formed prompt. It does NOT call the LLM itself and does NOT
generate any answer text on its own -- there is nothing hardcoded or
templated as a fake response here, only prompt construction. The actual
reasoning happens in app/models/llm/llm_service.py, entirely on-device.
"""
from __future__ import annotations

from dataclasses import dataclass

# Each mode maps to a fixed instruction matching the brief's example
# questions ("Explain this circuit.", "Summarize this page.", "What is
# wrong with this code?", etc). "custom" is the escape hatch for anything
# else the user types.
MODES: dict[str, dict] = {
    "explain": {
        "label": "Explain this",
        "instruction": (
            "Explain what is shown in the extracted content clearly and accurately. "
            "Cover what it is, how its parts relate, and what it's for."
        ),
    },
    "summarize": {
        "label": "Summarize this page",
        "instruction": "Summarize the key points of the extracted content concisely, in a few sentences or a short list.",
    },
    "explain_simple": {
        "label": "Explain simply (2nd-year engineering level)",
        "instruction": (
            "Explain the extracted content the way you would to a second-year engineering "
            "student: clear, step-by-step, minimal jargon, and briefly define any technical "
            "term you do need to use."
        ),
    },
    "find_bug": {
        "label": "Find the bug / review the code",
        "instruction": (
            "Carefully review the extracted code for bugs, logic errors, and bad practices. "
            "List each issue found and how to fix it. If you genuinely find nothing wrong, say so explicitly."
        ),
    },
    "extract_equation": {
        "label": "Extract the equation",
        "instruction": (
            "Extract and clearly restate the mathematical equation(s) present in the extracted "
            "content, using standard notation. If there are multiple equations, list each one separately."
        ),
    },
    "custom": {
        "label": "Ask a custom question",
        "instruction": None,  # instruction comes from the user's own question
    },
}

# Purely a framing hint for the LLM -- "auto" omits the hint entirely and
# lets the model infer content type from the extracted text itself.
CONTENT_TYPES = [
    "auto",
    "circuit_diagram",
    "math_equation",
    "code",
    "graph_or_chart",
    "flowchart",
    "textbook_page",
    "handwritten_notes",
]

_CONTENT_TYPE_PHRASING = {
    "circuit_diagram": "a circuit diagram",
    "math_equation": "a mathematical equation or derivation",
    "code": "a piece of programming code",
    "graph_or_chart": "a graph or chart",
    "flowchart": "a flowchart",
    "textbook_page": "a textbook or lecture-notes page",
    "handwritten_notes": "handwritten notes",
}

SYSTEM_PREAMBLE = (
    "You are SnapSight AI's Study Vision assistant, running entirely on-device. "
    "You do not see the original image -- you only know what appears in EXTRACTED "
    "CONTENT below, which was produced by an OCR pipeline and may contain recognition "
    "errors or missing symbols. Reason only from that text; if it looks incomplete or "
    "garbled, say so rather than inventing details. Be direct and concrete."
)


@dataclass
class StudyVisionPrompt:
    mode: str
    content_type: str
    instruction: str
    prompt_text: str


def list_modes() -> list[dict]:
    return [{"id": k, "label": v["label"]} for k, v in MODES.items()]


def build_prompt(
    ocr_text: str,
    mode: str = "explain",
    custom_question: str = "",
    content_type: str = "auto",
) -> StudyVisionPrompt:
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}'. Valid modes: {', '.join(MODES)}")

    if mode == "custom":
        if not custom_question.strip():
            raise ValueError("mode='custom' requires a non-empty 'question' field.")
        instruction = custom_question.strip()
    else:
        instruction = MODES[mode]["instruction"]

    content_type_line = ""
    if content_type != "auto" and content_type in _CONTENT_TYPE_PHRASING:
        content_type_line = (
            f"\nThe user pointed their camera at what they believe is {_CONTENT_TYPE_PHRASING[content_type]}."
        )

    extracted = ocr_text.strip() or "(no text was recognized in the captured frame -- OCR returned nothing)"

    prompt_text = (
        f"{SYSTEM_PREAMBLE}{content_type_line}\n\n"
        f'EXTRACTED CONTENT:\n"""\n{extracted}\n"""\n\n'
        f"TASK: {instruction}\n\n"
        f"Answer:"
    )

    return StudyVisionPrompt(
        mode=mode,
        content_type=content_type,
        instruction=instruction,
        prompt_text=prompt_text,
    )
