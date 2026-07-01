from pathlib import Path


def find_repo_root() -> Path:
    """
    Find parent directory of .git starting from utils.py file, with .git assumed to be at the repo root and
    that utils.py is contained within the repo root.
    :return: path to the root of the repository
    """
    start = Path(__file__).resolve()

    if start.is_file():
        start = start.parent

    for parent in (start, *start.parents):
        if (parent / ".git").exists():
            return parent

    raise RuntimeError("Could not find repository root")