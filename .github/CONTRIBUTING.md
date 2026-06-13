# Contributing to bagel

Thanks for your interest in contributing! This document describes the
process for reporting issues, proposing changes, and submitting pull
requests.

By contributing to this project you agree that your contributions will be
licensed under the [Apache License 2.0](../LICENSE) (the project license).

## Table of Contents

- [Reporting bugs](#reporting-bugs)
- [Requesting features](#requesting-features)
- [Development setup](#development-setup)
- [Coding conventions](#coding-conventions)
- [Running tests](#running-tests)
- [Commit messages](#commit-messages)
- [Submitting a pull request](#submitting-a-pull-request)
- [Community expectations](#community-expectations)

## Reporting bugs

Please open an issue using the **Bug report** template. Include:

1. A minimal reproduction (config yaml + short rosbag or command).
2. The exact command you ran, including flags.
3. Full traceback or error output.
4. Environment info: OS, Python version (`python --version`), ffmpeg
   version (`ffmpeg -version`), GPU / NVENC availability if relevant
   (`ffmpeg -encoders | grep nvenc`).

## Requesting features

Open an issue using the **Feature request** template and describe:

1. The use case (which robot / workflow / dataset flavor).
2. What the current behavior is and why it's insufficient.
3. A proposed API or CLI shape (even rough).

Before implementing a large feature, please open a discussion issue first
so we can align on the scope and design.

## Development setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/sige0002/bagel.git
cd bagel
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

`ffmpeg` is a runtime dependency. On Ubuntu 24.04 the default package
already ships with NVENC encoders (`h264_nvenc`, `hevc_nvenc`,
`av1_nvenc`) for NVIDIA GPUs. See [`docs/performance.md`](../docs/performance.md)
for details.

**Do not** use `pip install` directly — this project uses `uv` for all
dependency management.

## Coding conventions

- **Python 3.11+** with type hints on every function signature
- **PEP 8** style, enforced by `ruff`
- Run `uv run ruff check --fix src/ tests/` before committing
- Prefer `pathlib.Path` over string paths
- Keep functions small and focused; new modules go under `src/bagel/`
- Comments explain **why**, not **what** — code and names carry the what

## Running tests

```bash
# Full suite
uv run pytest tests/ -q

# Exclude slow / NVENC / integration tests
uv run pytest tests/ -q -m "not slow and not nvenc and not integration"

# With coverage
uv run pytest tests/ --cov=bagel --cov-report=term-missing
```

Marker semantics (registered in `pyproject.toml`):

- `@pytest.mark.slow` — runs e2e conversion or tracemalloc memory checks
- `@pytest.mark.nvenc` — requires NVIDIA GPU + NVENC-enabled ffmpeg
- `@pytest.mark.integration` — uses real rosbag fixtures or network

New features must come with tests. Bug fixes should include a regression
test.

## Commit messages

Follow a lightweight [Conventional Commits](https://www.conventionalcommits.org/)
style:

```
<type>(<scope>): <short summary>

<optional body explaining context / motivation>
```

Common types used in this repo:

- `feat` — new user-visible capability
- `fix` — bug fix
- `perf` — performance improvement with no behavior change
- `refactor` — internal restructure, no behavior change
- `docs` — documentation only
- `test` — tests only
- `chore` — tooling, build, ignore, etc.

Example:

```
perf(resampler): vectorize resample loop with numpy.searchsorted

Replace per-frame bisect with a single np.searchsorted call per key.
2.1x-2.6x speedup on 10k-frame workloads, bit-level identical output.
```

**Do not** add `Co-Authored-By: Claude` or similar AI attribution trailers.

## Submitting a pull request

1. Fork the repo and create a branch from `main`. Name branches
   `feat/<topic>`, `fix/<topic>`, `perf/<topic>`, etc.
2. Make focused commits — one logical change per commit.
3. Ensure `uv run ruff check src/ tests/` and `uv run pytest tests/ -q`
   both pass locally.
4. Open a pull request targeting `main` using the PR template.
5. Reference any issues the PR closes (`Closes #123`).
6. Respond to review comments by pushing follow-up commits; squash on
   merge if requested by a maintainer.

## Community expectations

Be respectful, concise, and specific. Assume good faith. When in doubt
about a design choice, open an issue before coding rather than after.
