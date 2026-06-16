"""Unit tests for :mod:`bagel.ui.security` (path confinement + tokens).

Table-driven, ``tmp_path``-only, no network. Exercises the three escape vectors
:func:`resolve_within_root` must reject (absolute, ``..`` traversal, symlink
escape) plus valid nested paths, and the constant-time token comparison.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bagel.ui.security import (
    new_token,
    PathSecurityError,
    resolve_in_roots,
    resolve_within_root,
    Root,
    token_matches,
)


def test_resolve_valid_nested(tmp_path: Path) -> None:
    root = tmp_path / "bags"
    (root / "a" / "b").mkdir(parents=True)
    resolved = resolve_within_root(root, "a/b")
    assert resolved == (root / "a" / "b").resolve()


def test_resolve_empty_subpath_is_root(tmp_path: Path) -> None:
    root = tmp_path / "bags"
    root.mkdir()
    assert resolve_within_root(root, "") == root.resolve()


@pytest.mark.parametrize(
    "subpath",
    [
        "/etc/passwd",  # absolute
        "/abs/inside",  # absolute even if it would land inside
    ],
)
def test_resolve_absolute_rejected(tmp_path: Path, subpath: str) -> None:
    root = tmp_path / "bags"
    root.mkdir()
    with pytest.raises(PathSecurityError):
        resolve_within_root(root, subpath)


@pytest.mark.parametrize(
    "subpath",
    [
        "..",
        "../sibling",
        "../../etc",
        "a/../../escape",
    ],
)
def test_resolve_parent_traversal_rejected(tmp_path: Path, subpath: str) -> None:
    root = tmp_path / "bags"
    root.mkdir()
    with pytest.raises(PathSecurityError):
        resolve_within_root(root, subpath)


def test_resolve_symlink_escape_rejected(tmp_path: Path) -> None:
    root = tmp_path / "bags"
    root.mkdir()
    outside = tmp_path / "secret"
    outside.mkdir()
    (outside / "file.txt").write_text("nope")
    # A symlink *inside* the root that points outside it.
    link = root / "link"
    link.symlink_to(outside)
    with pytest.raises(PathSecurityError):
        resolve_within_root(root, "link/file.txt")


def test_resolve_symlink_within_root_ok(tmp_path: Path) -> None:
    root = tmp_path / "bags"
    (root / "real").mkdir(parents=True)
    (root / "real" / "f.txt").write_text("ok")
    link = root / "alias"
    link.symlink_to(root / "real")
    # Symlink staying inside the root is allowed.
    resolved = resolve_within_root(root, "alias/f.txt")
    assert resolved == (root / "real" / "f.txt").resolve()


def test_resolve_in_roots_dispatch(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    (a / "x").mkdir(parents=True)
    (b / "y").mkdir(parents=True)
    roots = [
        Root(id=0, label="bags", path=a),
        Root(id=1, label="output", path=b),
    ]
    assert resolve_in_roots(roots, 0, "x") == (a / "x").resolve()
    assert resolve_in_roots(roots, 1, "y") == (b / "y").resolve()
    with pytest.raises(PathSecurityError):
        resolve_in_roots(roots, 1, "../a/x")
    with pytest.raises(PathSecurityError):
        resolve_in_roots(roots, 99, "x")  # unknown root id


def test_token_matches() -> None:
    tok = new_token()
    assert isinstance(tok, str) and len(tok) > 20
    assert token_matches(tok, tok)
    assert not token_matches(tok, tok + "x")
    assert not token_matches(tok, None)
    assert not token_matches(tok, "")
