"""Research state and evidence ledger utilities for scMAA."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field


class ResearchStepUpdate(BaseModel):
    """Structured summary of what a notebook step changed scientifically."""

    step_title: str = Field(description="Short label for the executed step.")
    claim: str = Field(description="Main scientific claim or conclusion from this step.")
    evidence_status: str = Field(
        description="One of: supports, weakens, reframes, inconclusive, setup_only."
    )
    supporting_evidence: list[str] = Field(
        description="Concrete observations from outputs that justify the claim."
    )
    caveats: list[str] = Field(
        description="Risks, confounders, or missing controls that limit interpretation."
    )
    next_priority_question: str = Field(
        description="Most important unresolved question to answer next."
    )
    recommended_direction: str = Field(
        description="Short recommendation for the next step direction."
    )


@dataclass
class ResearchLedger:
    """Lightweight state store tracking evidence and open questions."""

    dataset_strengths: list[str] = field(default_factory=list)
    dataset_warnings: list[str] = field(default_factory=list)
    guardrails: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    retired_paths: list[str] = field(default_factory=list)
    entries: list[ResearchStepUpdate] = field(default_factory=list)

    def add_entry(self, update: ResearchStepUpdate) -> None:
        self.entries.append(update)
        question = update.next_priority_question.strip()
        if question and question not in self.open_questions:
            self.open_questions.append(question)
        if update.evidence_status in {"supports", "weakens", "reframes"}:
            direction = update.recommended_direction.strip()
            if direction and direction in self.open_questions:
                self.open_questions.remove(direction)
        for caveat in update.caveats:
            text = caveat.strip()
            if text and text not in self.guardrails:
                self.guardrails.append(text)

    def to_prompt_text(self, max_entries: int = 6) -> str:
        lines: list[str] = []
        if self.dataset_strengths:
            lines.append("Dataset strengths:")
            lines.extend(f"- {item}" for item in self.dataset_strengths[:8])
        if self.dataset_warnings:
            lines.append("Dataset warnings:")
            lines.extend(f"- {item}" for item in self.dataset_warnings[:12])
        if self.guardrails:
            lines.append("Guardrails:")
            lines.extend(f"- {item}" for item in self.guardrails[:12])
        if self.open_questions:
            lines.append("Open questions:")
            lines.extend(f"- {item}" for item in self.open_questions[:10])
        if self.retired_paths:
            lines.append("Retired paths:")
            lines.extend(f"- {item}" for item in self.retired_paths[:10])
        if self.entries:
            lines.append("Evidence ledger:")
            for idx, entry in enumerate(self.entries[-max_entries:], start=1):
                lines.append(f"{idx}. {entry.step_title} [{entry.evidence_status}]")
                lines.append(f"   claim: {entry.claim}")
                if entry.supporting_evidence:
                    lines.extend(f"   evidence: {item}" for item in entry.supporting_evidence[:4])
                if entry.caveats:
                    lines.extend(f"   caveat: {item}" for item in entry.caveats[:3])
                lines.append(f"   next priority: {entry.next_priority_question}")
        return "\n".join(lines).strip() or "No research ledger entries yet."

    def to_markdown(self, max_entries: int = 6) -> str:
        text = self.to_prompt_text(max_entries=max_entries)
        return text if text else "No research ledger entries yet."

