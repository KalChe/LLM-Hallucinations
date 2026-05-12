import os
from pathlib import Path
from typing import Iterable, Optional


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(value: str | Path, base: Path = WORKSPACE_ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def get_env_path(var_name: str, default: str | Path, base: Path = WORKSPACE_ROOT) -> Path:
    raw = os.getenv(var_name, "").strip()
    selected = raw if raw else default
    return _resolve_path(selected, base=base)


PROJECT_ROOT = get_env_path("PROJECT_ROOT", WORKSPACE_ROOT)
DATA_ROOT = get_env_path("DATA_DIR", PROJECT_ROOT / "data", base=PROJECT_ROOT)
FIGS_DIR = get_env_path("FIGS_DIR", PROJECT_ROOT / "figs", base=PROJECT_ROOT)
HIDDEN_STATES_DIR = get_env_path(
    "HIDDEN_STATES_DIR", FIGS_DIR / "hidden_states", base=PROJECT_ROOT
)
RESULTS_DIR = get_env_path("RESULTS_DIR", PROJECT_ROOT / "code" / "json_results", base=PROJECT_ROOT)
TABLES_DIR = get_env_path("TABLES_DIR", PROJECT_ROOT / "code" / "tables", base=PROJECT_ROOT)


def data_root_candidates(explicit_root: Optional[Path] = None) -> list[Path]:
    if explicit_root is not None:
        root = _resolve_path(explicit_root, base=PROJECT_ROOT)
        return [root]

    candidates = [DATA_ROOT, PROJECT_ROOT, PROJECT_ROOT.parent]
    seen = set()
    unique = []
    for item in candidates:
        key = str(item)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def resolve_data_path(relative_path: str | Path, data_root: Optional[Path] = None) -> Path:
    rel = Path(relative_path)
    if rel.is_absolute():
        return rel

    for root in data_root_candidates(explicit_root=data_root):
        candidate = root / rel
        if candidate.exists():
            return candidate

    # Return the first candidate path even if missing so callers get a useful error path.
    return data_root_candidates(explicit_root=data_root)[0] / rel


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
