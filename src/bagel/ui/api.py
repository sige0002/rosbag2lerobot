"""JSON allow-list API for the ``bagel ui`` backend (plan.md D-7).

Thin wrappers over existing ``bagel`` verbs, kept deliberately narrow: the only
operations exposed are the scaffold→convert→quality reproduction loop. Each
method returns a plain ``dict`` (the server serializes it) and every response
carries a ``"command"`` string — the equivalent ``bagel ...`` CLI invocation —
so the CLI remains the source of truth and every UI action is copy-pasteable.

Read-only verbs (``inspect`` / ``scaffold`` / ``validate-config`` /
``validate-dataset`` / ``quality-report`` / ``preview``) are imported and run
**in-process** for latency. The single long-running verb (``convert``) is run as
a tracked subprocess via :mod:`bagel.ui.jobs`.

All filesystem access routes through :func:`bagel.ui.security.resolve_in_roots`
so requests cannot escape the roots configured at launch.
"""

from __future__ import annotations

import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any

import bagel
import yaml
from bagel import __version__ as _BAGEL_VERSION
from bagel.ui.jobs import JobRegistry
from bagel.ui.security import PathSecurityError, resolve_in_roots, Root

# Valid values for the global options patched by ``config_apply``.
_RESAMPLING_POLICIES = frozenset({"hold", "nearest", "drop"})
_STAMP_SOURCES = frozenset({"header", "receive"})


class ApiError(Exception):
    """A handled API failure carrying an HTTP status code.

    Attributes:
        status: The HTTP status to return (e.g. 400, 403, 404).
        detail: Human-readable explanation surfaced in the JSON body.
    """

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


# Bag markers — a directory is "a bag" when it holds any of these. Mirrors
# reader._resolve_bag_path without importing the private helper or raising.
def _is_bag_dir(path: Path) -> bool:
    """Return True when *path* looks like a rosbag2 bag directory."""
    if not path.is_dir():
        return False
    if (path / "metadata.yaml").exists():
        return True
    return any(path.glob("*.db3")) or any(path.glob("*.mcap"))


def _quote(value: str) -> str:
    """Shell-quote a value for the displayed ``command`` string."""
    return shlex.quote(value)


def _configs_dir() -> Path:
    """Return the repo-root ``configs/`` directory holding shipped configs.

    Located relative to the installed ``bagel`` package: ``src/bagel/__init__.py``
    -> ``bagel`` -> ``src`` -> repo root, then ``configs``. The directory may not
    exist (e.g. an installed wheel without the source tree); callers handle a
    missing directory gracefully.
    """
    return Path(bagel.__file__).resolve().parent.parent.parent / "configs"


class Api:
    """The backend API: roots, token, job registry, and verb wrappers.

    The HTTP layer (:mod:`bagel.ui.server`) owns auth/transport and dispatches
    parsed requests here. Methods raise :class:`ApiError` for handled failures;
    the server maps those to ``{"error","detail"}`` + the right status code.
    """

    def __init__(
        self,
        roots: list[Root],
        token: str,
        registry: JobRegistry,
    ) -> None:
        """Construct the API.

        Args:
            roots: Configured filesystem roots (``--bags-root`` / ``--output-root``).
            token: The session token required on ``/api/*``.
            registry: Convert-job registry for async conversions.
        """
        self.roots = roots
        self.token = token
        self.registry = registry
        self.bagel_version = _BAGEL_VERSION

    # -- helpers ---------------------------------------------------------

    def _resolve(self, root_id: int, subpath: str) -> Path:
        """Confine ``subpath`` to ``root_id``; raise :class:`ApiError` on escape."""
        try:
            return resolve_in_roots(self.roots, root_id, subpath)
        except PathSecurityError as exc:
            raise ApiError(403, str(exc)) from exc

    def _bagel_argv(self, *args: str) -> list[str]:
        """Return ``[python, -m, bagel.cli, *args]`` for a subprocess call.

        Using ``sys.executable -m bagel.cli`` keeps the subprocess pinned to the
        same interpreter/venv that is serving the UI (rather than relying on a
        ``bagel`` entry-point being on ``PATH``).
        """
        return [sys.executable, "-m", "bagel.cli", *args]

    # -- endpoints -------------------------------------------------------

    def config(self) -> dict[str, Any]:
        """``GET /api/config`` -- roots + bagel version for the FE to render."""
        return {
            "command": "bagel --version",
            "roots": [r.to_dict() for r in self.roots],
            "bagel_version": self.bagel_version,
        }

    # -- shipped-config template picker ----------------------------------

    def _list_configs(self) -> list[tuple[str, str]]:
        """Return ``(name, robot_type)`` for each shipped robot config.

        Scans :func:`_configs_dir` for ``*.yaml`` files, parses each with
        ``yaml.safe_load`` and keeps only those whose top level is a mapping
        (this drops all-comment templates like ``robot_template.yaml`` whose
        ``safe_load`` is ``None``). ``robot_type`` is the parsed ``robot_type``
        key if present, else ``""``. The list is sorted by name. A missing
        configs directory yields an empty list.
        """
        configs_dir = _configs_dir()
        if not configs_dir.is_dir():
            return []
        out: list[tuple[str, str]] = []
        for path in sorted(configs_dir.glob("*.yaml"), key=lambda p: p.name):
            try:
                parsed = yaml.safe_load(path.read_text())
            except (OSError, yaml.YAMLError):
                continue
            if not isinstance(parsed, dict):
                continue
            robot_type = parsed.get("robot_type")
            out.append((path.name, str(robot_type) if robot_type is not None else ""))
        return out

    def config_list(self) -> dict[str, Any]:
        """``GET /api/config-list`` -- shipped robot configs for the picker.

        Returns:
            ``{command, configs:[{name, robot_type}, ...]}`` sorted by name.
            Empty ``configs`` when no configs directory is present.
        """
        configs = [{"name": name, "robot_type": rt} for name, rt in self._list_configs()]
        return {"command": "ls configs", "configs": configs}

    def config_template(self, name: str) -> dict[str, Any]:
        """``GET /api/config-template?name=<name>`` -- raw text of a shipped config.

        Security: ``name`` must be a plain basename that appears in the
        :meth:`config_list` set — no ``/`` separators, no ``..``, no absolute
        paths — so reads cannot escape the configs directory.

        Args:
            name: The basename of a shipped config (e.g. ``"hsr.yaml"``).

        Returns:
            ``{command, name, yaml}`` where ``yaml`` is the raw file text.

        Raises:
            ApiError: 400 if ``name`` is empty/missing; 403 if ``name`` is not a
                plain listed basename.
        """
        if not name:
            raise ApiError(400, "Missing 'name'.")
        # Confine to the exact set of listed config basenames. This rejects any
        # path component ("/", "..", absolute) implicitly: such a value can never
        # equal a bare filename returned by _list_configs.
        allowed = {n for n, _ in self._list_configs()}
        if name not in allowed:
            raise ApiError(403, f"Not a shipped config: {name!r}")
        path = _configs_dir() / name
        try:
            text = path.read_text()
        except OSError as exc:
            raise ApiError(404, f"Config not readable: {name!r}") from exc
        return {"command": f"cat configs/{name}", "name": name, "yaml": text}

    def config_apply(self, body: dict[str, Any]) -> dict[str, Any]:
        """``POST /api/config-apply`` -- patch global options into a YAML string.

        The supplied ``yaml`` text is parsed with ``yaml.safe_load``, the
        present/non-null ``options`` are applied in Python, and the result is
        re-emitted with ``yaml.safe_dump``. Because this is a load/dump round
        trip, **YAML comments are dropped** from the returned text.

        Applied options (only when present and non-null):

        - top level: ``robot_type`` (str), ``task`` (str), ``fps`` (int).
        - under ``resampling`` (created if absent): ``resampling_policy`` ->
          ``default_policy``, ``tolerance_ms``, ``align_to_required`` (bool).
        - ``stamp_source`` (str): set on every item of ``observations`` and
          ``actions``.

        Args:
            body: ``{yaml: <text>, options: {...}}``.

        Returns:
            ``{command, yaml}`` with the patched YAML text.

        Raises:
            ApiError: 400 if the YAML is a non-mapping, or if
                ``resampling_policy`` / ``stamp_source`` carry invalid values.
        """
        raw = str(body.get("yaml", ""))
        options = body.get("options") or {}
        if not isinstance(options, dict):
            raise ApiError(400, "'options' must be an object.")

        if raw.strip():
            try:
                cfg = yaml.safe_load(raw)
            except yaml.YAMLError as exc:
                raise ApiError(400, f"Invalid YAML: {exc}") from exc
        else:
            cfg = {}
        if cfg is None:
            cfg = {}
        if not isinstance(cfg, dict):
            raise ApiError(400, "config is not a mapping")

        self._apply_options(cfg, options)

        patched = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
        return {"command": "# apply options", "yaml": patched}

    def _apply_options(self, cfg: dict[str, Any], options: dict[str, Any]) -> None:
        """Mutate ``cfg`` in place with the present/non-null ``options``.

        See :meth:`config_apply` for the option semantics. Validates
        ``resampling_policy`` and ``stamp_source`` before applying.
        """
        robot_type = options.get("robot_type")
        if robot_type is not None:
            cfg["robot_type"] = str(robot_type)
        task = options.get("task")
        if task is not None:
            cfg["task"] = str(task)
        fps = options.get("fps")
        if fps is not None:
            cfg["fps"] = int(fps)

        policy = options.get("resampling_policy")
        if policy is not None and policy not in _RESAMPLING_POLICIES:
            raise ApiError(400, f"Invalid resampling_policy: {policy!r}")
        tolerance_ms = options.get("tolerance_ms")
        align = options.get("align_to_required")
        if policy is not None or tolerance_ms is not None or align is not None:
            resampling = cfg.get("resampling")
            if not isinstance(resampling, dict):
                resampling = {}
                cfg["resampling"] = resampling
            if policy is not None:
                resampling["default_policy"] = str(policy)
            if tolerance_ms is not None:
                resampling["tolerance_ms"] = float(tolerance_ms)
            if align is not None:
                resampling["align_to_required"] = bool(align)

        stamp_source = options.get("stamp_source")
        if stamp_source is not None:
            if stamp_source not in _STAMP_SOURCES:
                raise ApiError(400, f"Invalid stamp_source: {stamp_source!r}")
            for section in ("observations", "actions"):
                items = cfg.get(section)
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            item["stamp_source"] = str(stamp_source)

    def browse(self, body: dict[str, Any]) -> dict[str, Any]:
        """``POST /api/browse`` -- list a directory under a root.

        Args:
            body: ``{root_id, subpath}``.

        Returns:
            ``{command, path, entries:[{name,is_dir,is_bag}], bags:[abs...]}``
            where ``bags`` lists the absolute paths of bag-shaped children.
        """
        root_id = int(body.get("root_id", 0))
        subpath = str(body.get("subpath", ""))
        target = self._resolve(root_id, subpath)
        if not target.is_dir():
            raise ApiError(404, f"Not a directory: {subpath!r}")

        entries: list[dict[str, Any]] = []
        bags: list[str] = []
        for child in sorted(target.iterdir(), key=lambda p: p.name):
            is_dir = child.is_dir()
            is_bag = is_dir and _is_bag_dir(child)
            entries.append({"name": child.name, "is_dir": is_dir, "is_bag": is_bag})
            if is_bag:
                bags.append(str(child))
        # The directory itself may be a single bag.
        if _is_bag_dir(target) and str(target) not in bags:
            bags.insert(0, str(target))

        return {
            "command": f"ls {_quote(str(target))}",
            "path": str(target),
            "entries": entries,
            "bags": bags,
        }

    def _bags_roots(self) -> list[Root]:
        """Return the roots that may hold bags (everything but the output root).

        Bag inputs (browse/inspect/scaffold/convert) must be confined to the
        bags roots only; the output root holds generated datasets and must not
        be accepted as a bag source.
        """
        return [r for r in self.roots if r.label != "output"]

    def _confine_bags(self, bags: list[str]) -> list[Path]:
        """Confine a list of bag paths to the bags roots.

        Each entry may be either an absolute path or a path relative to a
        bags-root (the frontend sends root-relative paths from ``browse``).
        Both forms are confined to the bags roots; anything outside every bags
        root — including a path under the output root — is rejected. Relative
        entries are resolved through :func:`resolve_in_roots`, so the
        path-traversal guard still applies.
        """
        if not bags:
            raise ApiError(400, "No bags specified.")
        confined: list[Path] = []
        for bag in bags:
            matched = self._confine_one_bag(bag)
            if matched is None:
                raise ApiError(403, f"Bag path outside configured roots: {bag!r}")
            confined.append(matched)
        return confined

    def _confine_one_bag(self, bag: str) -> Path | None:
        """Resolve a single bag (absolute or root-relative) within a bags root.

        Returns the confined path, or ``None`` if it falls outside every bags
        root.
        """
        bag_path = Path(bag)
        if bag_path.is_absolute():
            resolved = bag_path.resolve()
            for root in self._bags_roots():
                root_resolved = root.path.resolve()
                if resolved == root_resolved or resolved.is_relative_to(root_resolved):
                    rel = (
                        ""
                        if resolved == root_resolved
                        else str(resolved.relative_to(root_resolved))
                    )
                    return self._resolve(root.id, rel)
            return None
        # Root-relative path: try each bags root via the confined resolver.
        for root in self._bags_roots():
            try:
                candidate = resolve_in_roots(self.roots, root.id, bag)
            except PathSecurityError:
                continue
            if candidate.exists():
                return candidate
        return None

    def _bags_parent(self, bag_paths: list[Path]) -> Path:
        """Return the common parent dir to pass as ``--bags`` to the CLI.

        ``discover_bags`` accepts either a single bag dir or a parent of bags.
        When all selected bags share one parent we pass that parent; a single
        bag is passed directly.
        """
        if len(bag_paths) == 1:
            return bag_paths[0]
        parents = {p.parent for p in bag_paths}
        if len(parents) == 1:
            return next(iter(parents))
        # Mixed parents: fall back to the first bag's parent (best-effort);
        # the FE's single-group workflow keeps this from happening in practice.
        return bag_paths[0].parent

    def inspect(self, body: dict[str, Any]) -> dict[str, Any]:
        """``POST /api/inspect`` -- fps-stats report for the selected bags."""
        bags = [str(b) for b in body.get("bags", [])]
        bag_paths = self._confine_bags(bags)
        bags_arg = self._bags_parent(bag_paths)

        from bagel.cli import _build_stub_config, _run_fps_stats
        from bagel.reader import discover_bags

        try:
            discovered = discover_bags(bags_arg)
        except (FileNotFoundError, ValueError) as exc:
            raise ApiError(400, str(exc)) from exc

        report: dict[str, Any] = {"bags": []}
        for bp in discovered:
            stub = _build_stub_config(bp)
            entry: dict[str, Any] = {"bag": str(bp)}
            entry.update(_run_fps_stats(bp, stub, None, 200.0, 5))
            report["bags"].append(entry)

        command = f"bagel inspect --bags {_quote(str(bags_arg))} --fps-stats --json"
        return {"command": command, "report": report}

    def scaffold(self, body: dict[str, Any]) -> dict[str, Any]:
        """``POST /api/scaffold`` -- generate a starter config (stdout, no write).

        Delegates to :func:`bagel.cli.scaffold_config_yaml` (the shared body of
        the ``scaffold`` verb) with the same defaults, so the YAML is
        byte-identical to ``bagel scaffold`` printing to stdout.
        """
        bags = [str(b) for b in body.get("bags", [])]
        bag_paths = self._confine_bags(bags)
        bags_arg = self._bags_parent(bag_paths)
        robot_type = str(body.get("robot_type", "unknown_robot"))
        task = str(body.get("task", "TODO_describe_task"))
        fps = body.get("fps")
        fps_override = int(fps) if fps is not None else None

        from bagel.cli import scaffold_config_yaml
        from bagel.reader import discover_bags

        try:
            discovered = discover_bags(bags_arg)
        except (FileNotFoundError, ValueError) as exc:
            raise ApiError(400, str(exc)) from exc
        bag_path = discovered[0]

        # Same defaults as ``bagel scaffold`` so the YAML is byte-identical.
        yaml_text = scaffold_config_yaml(
            bag_path,
            robot_type=robot_type,
            task=task,
            fps=fps_override,
            min_count=1,
            samples=3,
        )

        parts = ["bagel scaffold", f"--bags {_quote(str(bags_arg))}"]
        parts.append(f"--robot-type {_quote(robot_type)}")
        parts.append(f"--task {_quote(task)}")
        if fps_override is not None:
            parts.append(f"--fps {fps_override}")
        command = " ".join(parts)
        return {"command": command, "yaml": yaml_text}

    def _write_config_tmp(self, config_yaml: str) -> Path:
        """Write ``config_yaml`` to a tempfile under the first output root.

        Keeping the tempfile *inside* a configured root means it too is subject
        to the confinement invariant. The caller is responsible for deleting it.
        """
        output_root = self._output_root()
        fd, name = tempfile.mkstemp(
            prefix="bagel_ui_cfg_", suffix=".yaml", dir=str(output_root.path)
        )
        path = Path(name)
        try:
            with open(fd, "w") as fh:
                fh.write(config_yaml)
        except OSError:
            path.unlink(missing_ok=True)
            raise
        return path

    def _output_root(self) -> Root:
        """Return the output :class:`Root` (the one labelled ``output``)."""
        for r in self.roots:
            if r.label == "output":
                return r
        raise ApiError(500, "No output root configured.")

    def validate_config(self, body: dict[str, Any]) -> dict[str, Any]:
        """``POST /api/validate-config`` -- validate FE-supplied YAML vs bags."""
        config_yaml = str(body.get("config_yaml", ""))
        bags = [str(b) for b in body.get("bags", [])]
        bag_paths = self._confine_bags(bags)
        bags_arg = self._bags_parent(bag_paths)

        from bagel.config import load_config
        from bagel.diagnostics import validate_config_against_bag
        from bagel.reader import BagReader, discover_bags

        tmp = self._write_config_tmp(config_yaml)
        try:
            try:
                cfg = load_config(tmp)
            except (ValueError, FileNotFoundError) as exc:
                raise ApiError(400, f"Invalid config: {exc}") from exc
            try:
                discovered = discover_bags(bags_arg)
            except (FileNotFoundError, ValueError) as exc:
                raise ApiError(400, str(exc)) from exc
            bag_path = discovered[0]
            with BagReader(bag_path, cfg) as reader:
                report = validate_config_against_bag(cfg, reader, samples=5)
            report.apply_verdict(strict=False)
            payload = {
                "config": "<in-memory>",
                "bag": str(bag_path),
                "results": report.to_dict(),
            }
        finally:
            tmp.unlink(missing_ok=True)

        command = (
            f"bagel validate-config --config <config.yaml> "
            f"--bags {_quote(str(bags_arg))} --json"
        )
        return {"command": command, "report": payload}

    def convert(self, body: dict[str, Any]) -> dict[str, Any]:
        """``POST /api/convert`` -- launch an async ``bagel convert`` subprocess.

        Writes the FE config to a tempfile under the output root, resolves the
        output dir within the output root, and starts the conversion. Returns a
        ``job_id`` immediately; the FE polls ``GET /api/convert/{job_id}``.
        """
        config_yaml = str(body.get("config_yaml", ""))
        bags = [str(b) for b in body.get("bags", [])]
        bag_paths = self._confine_bags(bags)
        bags_arg = self._bags_parent(bag_paths)
        output_rel = str(body.get("output", ""))
        if not output_rel:
            raise ApiError(400, "Missing 'output'.")
        output_root = self._output_root()
        output_dir = self._resolve(output_root.id, output_rel)

        fps = body.get("fps")
        workers = body.get("workers")
        video_codec = body.get("video_codec")

        # The config tempfile lives for the lifetime of the conversion; it is
        # under the output root (confined) and handed to the registry so it is
        # unlinked when the job reaches a terminal state (or on shutdown).
        tmp = self._write_config_tmp(config_yaml)

        # Count bags up-front so the FE can render done/total live.
        from bagel.reader import discover_bags

        try:
            total = len(discover_bags(bags_arg))
        except (FileNotFoundError, ValueError) as exc:
            tmp.unlink(missing_ok=True)
            raise ApiError(400, str(exc)) from exc

        cli_args = [
            "convert",
            "--config",
            str(tmp),
            "--bags",
            str(bags_arg),
            "--output",
            str(output_dir),
            "--json",
            "--quiet",
        ]
        display = [
            "bagel convert",
            "--config <config.yaml>",
            f"--bags {_quote(str(bags_arg))}",
            f"--output {_quote(str(output_dir))}",
        ]
        if fps is not None:
            cli_args += ["--fps", str(int(fps))]
            display.append(f"--fps {int(fps)}")
        if workers is not None:
            cli_args += ["--workers", str(int(workers))]
            display.append(f"--workers {int(workers)}")
        if video_codec:
            cli_args += ["--video-codec", str(video_codec)]
            display.append(f"--video-codec {_quote(str(video_codec))}")

        argv = self._bagel_argv(*cli_args)
        command = " ".join(display)
        job_id = self.registry.launch(
            argv,
            output_dir=output_dir,
            total=total,
            command=command,
            config_path=tmp,
        )
        return {"command": command, "job_id": job_id}

    def convert_status(self, job_id: str) -> dict[str, Any]:
        """``GET /api/convert/{job_id}`` -- poll a running/finished conversion."""
        job = self.registry.get(job_id)
        if job is None:
            raise ApiError(404, f"Unknown job_id: {job_id}")
        return job.status()

    def _confine_dataset(self, dataset_rel: str) -> Path:
        """Confine a dataset path (relative to the output root) and assert it exists."""
        output_root = self._output_root()
        dataset_dir = self._resolve(output_root.id, dataset_rel)
        if not dataset_dir.is_dir():
            raise ApiError(404, f"Dataset not found: {dataset_rel!r}")
        return dataset_dir

    def validate_dataset(self, body: dict[str, Any]) -> dict[str, Any]:
        """``POST /api/validate-dataset`` -- structural validation of a dataset."""
        dataset_rel = str(body.get("dataset", ""))
        dataset_dir = self._confine_dataset(dataset_rel)

        from bagel.validation import validate_dataset as _validate_dataset

        try:
            report = _validate_dataset(dataset_dir)
        except (OSError, ValueError) as exc:
            raise ApiError(400, f"validate-dataset failed: {exc}") from exc
        report.apply_verdict(strict=False)
        command = f"bagel validate-dataset --dataset {_quote(str(dataset_dir))} --json"
        return {"command": command, "report": report.to_dict()}

    def quality_report(self, body: dict[str, Any]) -> dict[str, Any]:
        """``POST /api/quality-report`` -- data-quality score for a dataset."""
        dataset_rel = str(body.get("dataset", ""))
        dataset_dir = self._confine_dataset(dataset_rel)

        from bagel.quality import compute_quality_report

        try:
            report = compute_quality_report(dataset_dir)
        except (OSError, ValueError, RuntimeError) as exc:
            raise ApiError(400, f"quality-report failed: {exc}") from exc
        command = f"bagel quality-report --dataset {_quote(str(dataset_dir))} --json"
        return {"command": command, "report": report.to_dict()}

    def preview(self, dataset_rel: str) -> str:
        """``GET /api/preview`` -- the self-contained HTML preview report.

        Returns the raw HTML string (the server sets ``text/html``).
        """
        dataset_dir = self._confine_dataset(dataset_rel)

        from bagel.preview import generate_preview

        try:
            return generate_preview(dataset_dir)
        except (OSError, ValueError, RuntimeError) as exc:
            raise ApiError(400, f"preview failed: {exc}") from exc
