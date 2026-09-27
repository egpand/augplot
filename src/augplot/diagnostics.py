"""Best-effort diagnostics for failed model generations."""

import json
import shutil
import tempfile
from pathlib import Path


def save_generation_failure(attempts: list[dict], parent: Path | None) -> Path | None:
    """Keep rejected source in a private directory beside visualization history."""
    if parent is None:
        return None
    directory = None
    try:
        parent.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="failure-", dir=parent))
        summary = []
        for attempt in attempts:
            number = attempt["attempt"]
            code = attempt["code"]
            source = None
            response = None
            if code is not None:
                source = f"attempt-{number}.py"
                (directory / source).write_text(code, encoding="utf-8")
            else:
                # Malformed responses have no parsed source to inspect.
                response = f"response-{number}.txt"
                (directory / response).write_text(attempt["response"][:60_000], encoding="utf-8")
            summary.append(
                {
                    "attempt": number,
                    "error": attempt["error"],
                    "violations": attempt["violations"],
                    "source": source,
                    "response": response,
                }
            )
        (directory / "errors.json").write_text(
            json.dumps({"attempts": summary}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return directory
    except OSError:
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)
        return None
