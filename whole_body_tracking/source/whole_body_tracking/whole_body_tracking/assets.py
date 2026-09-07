"""Resolve the asset root for source installs and the combined repository."""

from pathlib import Path


def _resolve_asset_dir() -> Path:
    this_file = Path(__file__).resolve()
    for parent in this_file.parents:
        candidates = (
            parent / "data" / "assets",
            parent / "GMR" / "assets",
        )
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
    return this_file.parent / "data" / "assets"


ASSET_DIR = str(_resolve_asset_dir())
