"""Shared setup and transcript helpers for the standalone agent checks."""

from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def get_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set OPENAI_API_KEY in your environment or a local .env file before running tests."
        )
    return api_key


def show_and_save(filename: str, lines: list[str]) -> None:
    transcript = "\n".join(lines).rstrip() + "\n"
    print(transcript, end="")
    output_dir = PROJECT_ROOT / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    (output_dir / filename).write_text(transcript, encoding="utf-8")