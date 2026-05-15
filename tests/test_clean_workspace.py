from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.clean_workspace import collect_cleanup_targets, main


def _make_fake_project(root: Path) -> None:
    for directory in ["src", "tests", "scripts", "docs", "configs", "benchmarks"]:
        (root / directory).mkdir(parents=True, exist_ok=True)
    for filename in ["README.md", "pyproject.toml", ".env.example"]:
        (root / filename).write_text("keep", encoding="utf-8")


def test_dry_run_does_not_delete_files(tmp_path: Path) -> None:
    _make_fake_project(tmp_path)
    cache = tmp_path / ".pytest_tmp_xxx"
    cache.mkdir()

    exit_code = main(["--root", str(tmp_path)])

    assert exit_code == 0
    assert cache.exists()


def test_yes_deletes_pytest_tmp(tmp_path: Path) -> None:
    _make_fake_project(tmp_path)
    cache = tmp_path / ".pytest_tmp_xxx"
    cache.mkdir()

    exit_code = main(["--root", str(tmp_path), "--yes"])

    assert exit_code == 0
    assert not cache.exists()


def test_protected_paths_are_not_deleted(tmp_path: Path) -> None:
    _make_fake_project(tmp_path)
    for protected in ["src", "tests", "scripts", "docs", "README.md"]:
        assert (tmp_path / protected).exists()

    main(["--root", str(tmp_path), "--include-all", "--yes"])

    for protected in ["src", "tests", "scripts", "docs", "README.md"]:
        assert (tmp_path / protected).exists()


def test_include_outputs_required_for_outputs(tmp_path: Path) -> None:
    _make_fake_project(tmp_path)
    outputs = tmp_path / "outputs"
    outputs.mkdir()

    main(["--root", str(tmp_path), "--yes"])
    assert outputs.exists()

    main(["--root", str(tmp_path), "--include-outputs", "--yes"])
    assert not outputs.exists()


def test_include_all_cleans_outputs_data_and_eval(tmp_path: Path) -> None:
    _make_fake_project(tmp_path)
    for directory in ["outputs", "outputs_test", "data", "eval_outputs"]:
        (tmp_path / directory).mkdir()
    for filename in ["memory.sqlite", "index.npz", "array.npy", "state.db"]:
        (tmp_path / filename).write_text("artifact", encoding="utf-8")

    main(["--root", str(tmp_path), "--include-all", "--yes"])

    for path in ["outputs", "outputs_test", "data", "eval_outputs", "memory.sqlite", "index.npz", "array.npy", "state.db"]:
        assert not (tmp_path / path).exists()


def test_collect_cleanup_targets_includes_pyc_and_pycache(tmp_path: Path) -> None:
    _make_fake_project(tmp_path)
    pycache = tmp_path / "src" / "__pycache__"
    pycache.mkdir()
    pyc = pycache / "module.pyc"
    pyc.write_text("cache", encoding="utf-8")

    targets = collect_cleanup_targets(tmp_path)

    assert pycache.resolve() in targets
    assert pyc.resolve() in targets


def test_never_cleans_venv_contents(tmp_path: Path) -> None:
    _make_fake_project(tmp_path)
    venv_cache = tmp_path / ".venv" / "Lib" / "site-packages" / "__pycache__"
    venv_cache.mkdir(parents=True)
    pyc = venv_cache / "module.pyc"
    pyc.write_text("cache", encoding="utf-8")

    targets = collect_cleanup_targets(tmp_path)

    assert venv_cache.resolve() not in targets
    assert pyc.resolve() not in targets


def test_permission_error_is_skipped_without_crashing(monkeypatch, tmp_path: Path) -> None:
    _make_fake_project(tmp_path)
    cache = tmp_path / ".pytest_tmp_locked"
    cache.mkdir()

    def raise_permission_error(*args, **kwargs):
        raise PermissionError("locked")

    monkeypatch.setattr("scripts.clean_workspace.shutil.rmtree", raise_permission_error)

    exit_code = main(["--root", str(tmp_path), "--yes"])

    assert exit_code == 0
    assert cache.exists()
