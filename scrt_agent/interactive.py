"""Interactive session helpers for scMAA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hypothesis import AnalysisPlan, CandidateHypothesisMenu


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def format_candidate_menu_markdown(menu: CandidateHypothesisMenu) -> str:
    lines = [
        "# Candidate Hypotheses",
        "",
        f"Research focus: {menu.research_focus}",
        "",
    ]
    for idx, candidate in enumerate(menu.candidates, start=1):
        lines.extend(
            [
                f"## {idx}. {candidate.title}",
                "",
                f"Hypothesis: {candidate.hypothesis}",
                "",
                f"Rationale: {candidate.rationale}",
                "",
                f"Preferred analysis type: {candidate.preferred_analysis_type}",
                "",
                f"First test: {candidate.first_test}",
                "",
                "Cautions:",
            ]
        )
        lines.extend(f"- {item}" for item in candidate.cautions)
        lines.append("")
    lines.extend(
        [
            "## How To Review",
            "",
            "- Pick one candidate by number, or write your own hypothesis text.",
            "- Add freeform feedback if you want the agent to narrow, broaden, or redirect the question.",
            "- Then run the review step to produce an approved plan.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def format_analysis_plan_markdown(plan: AnalysisPlan) -> str:
    lines = [
        "# Approved Analysis Plan",
        "",
        f"Hypothesis: {plan.hypothesis}",
        f"Analysis type: {plan.analysis_type}",
        "",
        f"Priority question: {plan.priority_question}",
        "",
        f"Evidence goal: {plan.evidence_goal}",
        "",
        f"Decision rationale: {plan.decision_rationale}",
        "",
        "Validation checks:",
    ]
    lines.extend(f"- {item}" for item in plan.validation_checks)
    lines.extend(
        [
            "",
            "Remaining plan:",
        ]
    )
    lines.extend(f"{idx + 1}. {step}" for idx, step in enumerate(plan.analysis_plan))
    lines.extend(
        [
            "",
            f"Code description: {plan.code_description}",
            "",
            "Summary:",
            plan.summary,
            "",
            "First step code:",
            "```python",
            plan.first_step_code.strip(),
            "```",
        ]
    )
    return "\n".join(lines).strip() + "\n"

