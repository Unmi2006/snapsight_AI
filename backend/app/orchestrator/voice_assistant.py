"""
Prompt construction for the Voice Assistant's question path. Mirrors
app/orchestrator/study_vision.py's role exactly -- this module ONLY
builds a prompt string, it never generates or hardcodes an answer.
Reuses study_vision.build_prompt() when the user also captured/uploaded
an image (so a spoken question can reference on-screen content), and
falls back to a plain general-assistant preamble when there is no visual
context at all.
"""
from __future__ import annotations

from app.orchestrator.study_vision import build_prompt

GENERAL_SYSTEM_PREAMBLE = (
    "You are SnapSight AI's voice assistant, running entirely on-device with "
    "no internet connection. Answer the user's spoken question directly and "
    "concisely, in plain spoken-friendly sentences (no markdown, no bullet "
    "lists, no headings) since your reply will be read aloud by a "
    "text-to-speech engine."
)


def build_voice_prompt(question: str, ocr_text: str | None = None) -> str:
    question = (question or "").strip()
    if not question:
        raise ValueError(
            "No question was recognized from the audio (empty transcription) -- "
            "nothing to ask the LLM."
        )

    if ocr_text and ocr_text.strip():
        # Visual context is present -- reuse Study Vision's orchestrator so
        # the same OCR-grounding rules (don't invent details, flag garbled
        # text) apply here too, with the spoken question as a custom task.
        prompt = build_prompt(ocr_text, mode="custom", custom_question=question)
        return prompt.prompt_text

    return f"{GENERAL_SYSTEM_PREAMBLE}\n\nQUESTION: {question}\n\nAnswer:"
