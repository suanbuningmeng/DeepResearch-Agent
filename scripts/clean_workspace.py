from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROTECTED_NAMES = {
    ".git",
    ".venv",
    "src",
    "tests",
    "scripts",
    "docs",
    "configs",
    "benchmarks",
    "README.md",
    "pyproject.toml",
    ".env.example",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safely clean local test caches and runtime artifacts.")
    parser.add_argument("--yes", action="store_true", help="Actually delete matched paths. Default is dry-run.")
    parser.add_argument("--include-outputs", action="store_true", help="Also clean outputs/ and outputs_*/.")
    parser.add_argument("--include-data", action="store_true", help="Also clean data/, *.sqlite, *.db, *.npz, and *.npy.")
    parser.add_argument("--include-eval", action="store_true", help="Also clean eval_outputs/.")
    parser.add_argument("--include-all", action="store_true", help="Equivalent to outputs + data + eval.")
    parser.add_argument("--root", default=str(PROJECT_ROOT), help=argparse.SUPPRESS)
    return parser


def collect_cleanup_targets(
    root: Path,
    include_outputs: bool = False,
    include_data: bool = False,
    include_eval: bool = False,
) -> list[Path]:
    root = root.resolve()
    targets: set[Path] = set()

    patterns = [
        ".pytest_cache",
        ".pytest_tmp",
        ".pytest_tmp*",
        "**/__pycache__",
        "**/*.pyc",
    ]
    if include_outputs:
        patterns.extend(["outputs", "outputs_*"])
    if include_data:
        patterns.extend(["data", "*.sqlite", "*.db", "*.npz", "*.npy"])
    if include_eval:
        patterns.append("eval_outputs")

    for pattern in patterns:
        for path in root.glob(pattern):
            resolved = path.resolve()
            if _is_safe_target(root, resolved):
                targets.add(resolved)

    return sorted(targets, key=lambda path: (len(path.parts), str(path)))


def clean_targets(targets: list[Path], dry_run: bool = True) -> list[tuple[Path, str]]:
    action = "Would delete" if dry_run else "Deleting"
    failures: list[tuple[Path, str]] = []
    if not targets:
        print("No cleanup targets found.")
        return failures

    for target in targets:
        print(f"{action}: {target}")
        if dry_run:
            continue
        try:
            if target.is_dir():
                shutil.rmtree(target, onerror=_handle_remove_readonly)
            elif target.exists():
                target.unlink()
        except OSError as exc:
            failures.append((target, str(exc)))
            print(f"Skipped: {target} ({exc})")
    return failures


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    include_outputs = args.include_outputs or args.include_all
    include_data = args.include_data or args.include_all
    include_eval = args.include_eval or args.include_all

    targets = collect_cleanup_targets(
        root=root,
        include_outputs=include_outputs,
        include_data=include_data,
        include_eval=include_eval,
    )
    failures = clean_targets(targets, dry_run=not args.yes)
    if not args.yes:
        print("Dry-run only. Re-run with --yes to delete these paths.")
    elif failures:
        print("Cleanup finished with skipped paths:")
        for path, error in failures:
            print(f"- {path}: {error}")
        print("If a path is locked on Windows, close terminals/editors using it and re-run the command.")
    return 0


def _is_safe_target(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    if path == root:
        return False
    if path.name in PROTECTED_NAMES:
        return False
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    if relative.parts and relative.parts[0] in {".git", ".venv"}:
        return False
    if relative.parts and relative.parts[0] in PROTECTED_NAMES:
        return path.name == "__pycache__" or path.suffix == ".pyc"
    return True


def _handle_remove_readonly(function, path: str, exc_info) -> None:
    """Best-effort Windows cleanup for readonly files before giving up."""
    del exc_info
    try:
        os.chmod(path, 0o700)
        function(path)
    except OSError:
        raise


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
