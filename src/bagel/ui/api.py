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

from bagel import __version__ as _BAGEL_VERSION
from bagel.ui.jobs import JobRegistry
from bagel.ui.security import PathSecurityError, resolve_in_roots, Root


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
        """Confine a list of absolute bag paths to the bags roots.

        The FE sends absolute paths (from ``browse``). We re-confine each by
        computing its path relative to whichever bags-root contains it; anything
        outside every bags root — including a path under the output root — is
        rejected.
        """
        if not bags:
            raise ApiError(400, "No bags specified.")
        confined: list[Path] = []
        for bag in bags:
            bag_path = Path(bag)
            matched: Path | None = None
            for root in self._bags_roots():
                root_resolved = root.path.resolve()
                resolved = bag_path.resolve()
                if resolved == root_resolved or resolved.is_relative_to(root_resolved):
                    rel = (
                        ""
                        if resolved == root_resolved
                        else str(resolved.relative_to(root_resolved))
                    )
                    matched = self._resolve(root.id, rel)
                    break
            if matched is None:
                raise ApiError(403, f"Bag path outside configured roots: {bag!r}")
            confined.append(matched)
        return confined

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

        Reuses the ``scaffold`` verb's heuristics
        (:func:`bagel.cli._scaffold_from_topics` + :func:`config.config_to_yaml`)
        so the YAML is byte-identical to ``bagel scaffold`` printing to stdout.
        """
        bags = [str(b) for b in body.get("bags", [])]
        bag_paths = self._confine_bags(bags)
        bags_arg = self._bags_parent(bag_paths)
        robot_type = str(body.get("robot_type", "unknown_robot"))
        task = str(body.get("task", "TODO_describe_task"))
        fps = body.get("fps")
        fps_override = int(fps) if fps is not None else None

        from bagel.cli import (
            _build_stub_config,
            _IMAGE_MSG_TYPES,
            _INFRA_MSG_TYPES,
            _INFRA_TOPICS,
            _measure_topic_fps,
            _scaffold_from_topics,
        )
        from bagel.config import config_to_yaml, FeatureMapping
        from bagel.decoders import get_registered_types
        from bagel.diagnostics import detect_image_shape
        from bagel.reader import BagReader, discover_bags

        try:
            discovered = discover_bags(bags_arg)
        except (FileNotFoundError, ValueError) as exc:
            raise ApiError(400, str(exc)) from exc
        bag_path = discovered[0]

        registered = set(get_registered_types())
        min_count = 1
        samples = 3
        stub_cfg = _build_stub_config(bag_path)
        with BagReader(bag_path, stub_cfg) as reader:
            topics_info = reader.get_topics_info()
            measurable = [
                t
                for t, info in topics_info.items()
                if info.count >= min_count
                and info.msg_type not in _INFRA_MSG_TYPES
                and t not in _INFRA_TOPICS
            ]
            fps_by_topic = _measure_topic_fps(reader, measurable)
            image_shapes: dict[str, Any] = {}
            for topic, info in topics_info.items():
                if info.msg_type in _IMAGE_MSG_TYPES and info.count >= min_count:
                    fm = FeatureMapping(
                        key="probe",
                        topic=topic,
                        msg_type=info.msg_type,
                        dtype="image",
                    )
                    image_shapes[topic] = detect_image_shape(reader, fm, samples)

        cfg, obs_annotations, obs_candidates, act_candidates = _scaffold_from_topics(
            topics_info=topics_info,
            fps_by_topic=fps_by_topic,
            image_shapes=image_shapes,
            registered=registered,
            robot_type=robot_type,
            task=task,
            fps_override=fps_override,
            min_count=min_count,
            bag_name=bag_path.name,
        )
        header = obs_annotations.pop("__header__", [])
        yaml_text = config_to_yaml(
            cfg,
            header_lines=header,
            obs_annotations=obs_annotations,
            obs_candidates=obs_candidates,
            act_candidates=act_candidates,
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
