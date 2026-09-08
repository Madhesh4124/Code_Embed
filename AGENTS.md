# Agent Operating Guide — CodeEmbed

This document outlines the operational rules, environment requirements, and workflow conventions for AI agents working in this repository.

---

## 1. Environment & Package Management

* **Python Environment**: Always use the local virtual environment (`.venv`).
  * Activate on Windows PowerShell: `.venv\Scripts\Activate.ps1`
  * Activate on Git Bash / POSIX: `source .venv/Scripts/activate`
* **Package Manager**: Use **`uv`** as the primary package manager for dependency resolution, installation, and command execution.
  * Running scripts: `uv run python <script_path>`
  * Adding dependencies: `uv add <package>`
  * Installing from requirements: `uv pip install -r requirements.txt`
  * Syncing environment: `uv sync`

---

## 2. Documentation & Task Sources of Truth

Always consult and maintain the local documentation files before starting and after finishing tasks:

| Document | Purpose & Agent Action |
|---|---|
| [`Memory.md`](Memory.md) | **Active project state & tracker**: Check here first to see current progress, completed components, experiment logs, and immediate next actions. **Always update this file** when completing tasks, milestones, or experiments. |
| [`Phases.md`](Phases.md) | **Roadmap & Phase Specifications**: Defines Phases 0 through 8 in granular detail, including task breakdowns, deliverables, resource estimates, and strict exit criteria. |
| [`Rules.md`](Rules.md) | **Coding & Architecture Standards**: Non-negotiable rules for code style (Ruff, typing), architecture constraints (raw PyTorch `nn.Module`, Pre-LN, no HuggingFace model wrappers for core experiments), MLflow logging, and testing requirements (80%+ coverage). |
| [`Architecture.md`](Architecture.md) | **Technical Design**: Complete blueprint for Transformer architectures, embedding pipeline, data flow, configurations, and FAISS retrieval. |
| [`PRD.md`](PRD.md) | **Research & Product Objectives**: High-level context, target success criteria, and core research questions (RQ1–RQ5). |
| [`STUDY_GUIDE.md`](STUDY_GUIDE.md) | **Educational & Revision Companion**: Explains the "why", mathematical intuition, tensor shapes, and bug fixes across all phases. Keep updated for revision. |

---

## 3. Agent Execution Protocol

When starting or continuing a task in this project, follow this protocol:

1. **State Check**: Read [`Memory.md`](Memory.md) to determine the current active phase and outstanding next actions.
2. **Review Specifications**: Check [`Phases.md`](Phases.md) for the exact requirements and exit criteria of the task, and [`Rules.md`](Rules.md) for architectural and styling guardrails.
3. **Execution**: Implement changes adhering strictly to raw PyTorch conventions, type hints, and docstrings with explicit tensor shapes. Execute tests and scripts via `uv run`.
4. **Verification**: Run relevant unit tests (`uv run pytest tests/`) and ensure formatting and linting pass (`uv run ruff check .`).
5. **State Update**: Update [`Memory.md`](Memory.md) by checking off completed items, adding experiment results, and logging any architectural decisions made.

