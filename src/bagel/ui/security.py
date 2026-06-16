"""Path confinement + token helpers for the ``bagel ui`` backend.

Pure logic, separated from I/O so it is unit-testable: the only filesystem
touch is :meth:`pathlib.Path.resolve` (needed to collapse ``..`` segments and
follow symlinks before the containment check). Every privileged filesystem
access in the backend (browse, inspect, convert in/out, dataset reads, the
config tempfile) must route through :func:`resolve_within_root` so an attacker
cannot escape the configured roots via absolute paths, ``..`` traversal, or
symlinks pointing outside.

Token comparison uses :func:`secrets.compare_digest` to avoid timing leaks.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from pathlib import Path


class PathSecurityError(Exception):
    """Raised when a requested subpath would escape its configured root."""


@dataclass(frozen=True)
class Root:
    """A configured, addressable filesystem root.

    Attributes:
        id: Stable integer index used by the FE to reference this root.
        label: Human-readable label (``"bags"`` / ``"output"``).
        path: The resolved absolute directory the root maps to.
    """

    id: int
    label: str
    path: Path

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view ``{id, label, path}``."""
        return {"id": self.id, "label": self.label, "path": str(self.path)}


def resolve_within_root(root: Path, subpath: str) -> Path:
    """Resolve ``subpath`` against ``root`` and assert it stays inside.

    The containment guarantee holds against three escape vectors:

    - **Absolute subpath** (``/etc/passwd``): rejected outright — an absolute
      component would otherwise discard ``root``.
    - **Parent traversal** (``../../etc``): the raw ``..`` segments are rejected
      before resolution, and the post-resolve containment check is the backstop.
    - **Symlink escape**: ``resolve()`` follows symlinks, so a link inside the
      root that points outside resolves to its real (outside) target and fails
      the containment check.

    Args:
        root: The configured root directory. Resolved for a stable base.
        subpath: A *relative* path under the root (``""`` selects the root).

    Returns:
        The resolved absolute path, guaranteed to be ``root`` or a descendant.

    Raises:
        PathSecurityError: If ``subpath`` is absolute, contains a ``..``
            segment, or the resolved target lies outside ``root``.
    """
    resolved_root = root.resolve()

    sub = Path(subpath)
    if sub.is_absolute():
        raise PathSecurityError(f"Absolute subpath not allowed: {subpath!r}")
    # Reject ``..`` anywhere in the *requested* path up-front (defence in depth);
    # the post-resolve containment check below is the authoritative backstop.
    if ".." in sub.parts:
        raise PathSecurityError(f"Parent traversal not allowed: {subpath!r}")

    candidate = (resolved_root / sub).resolve()
    if candidate != resolved_root and not candidate.is_relative_to(resolved_root):
        raise PathSecurityError(
            f"Path escapes root {resolved_root}: {subpath!r} -> {candidate}"
        )
    return candidate


def resolve_in_roots(roots: list[Root], root_id: int, subpath: str) -> Path:
    """Resolve ``subpath`` within the root identified by ``root_id``.

    Args:
        roots: Configured roots, indexed by their ``id``.
        root_id: The ``Root.id`` to resolve against.
        subpath: Relative path under that root.

    Returns:
        The confined absolute path.

    Raises:
        PathSecurityError: If ``root_id`` is unknown or ``subpath`` escapes.
    """
    for r in roots:
        if r.id == root_id:
            return resolve_within_root(r.path, subpath)
    raise PathSecurityError(f"Unknown root_id: {root_id}")


def new_token() -> str:
    """Return a fresh URL-safe session token (256 bits of entropy)."""
    return secrets.token_urlsafe(32)


def token_matches(expected: str, provided: str | None) -> bool:
    """Constant-time compare a presented token against the expected one.

    Args:
        expected: The server's session token.
        provided: The token supplied by the client (header or query), or
            ``None`` when absent.

    Returns:
        ``True`` iff a token was supplied and matches in constant time.
    """
    if not provided:
        return False
    return secrets.compare_digest(expected, provided)
