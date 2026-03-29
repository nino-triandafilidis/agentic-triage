"""ESI Handbook text loader for use as a system-level context prefix in triage prompts.

Not wired into the RAG pipeline — call load_esi_handbook_prefix() when building prompts
that should include the handbook as static context.
"""

import os
from pathlib import Path

# Default: cleaned handbook text shipped in data/corpus/.
# Override via ESI_HANDBOOK_PATH env var if using a different extraction.
DEFAULT_HANDBOOK_PATH = Path(
    os.environ.get(
        "ESI_HANDBOOK_PATH",
        str(Path(__file__).resolve().parent.parent.parent / "data" / "corpus" / "esi_handbook_clean.txt"),
    )
)


def load_esi_handbook_prefix(path: Path | None = None) -> str:
    """Load pre-extracted ESI Handbook text for use as a system-level context prefix.

    Run scripts/extract_esi_handbook_text.py first to generate the plaintext file.
    The file includes page markers (--- Page N ---) so the LLM can cite pages.

    Args:
        path: Path to esi_handbook_text.txt. If None, uses DEFAULT_HANDBOOK_PATH.

    Returns:
        Full handbook text as a string, ready to be injected into a prompt.

    Raises:
        FileNotFoundError: If the handbook text file does not exist.
    """
    p = path if path is not None else DEFAULT_HANDBOOK_PATH
    return p.read_text(encoding="utf-8")
