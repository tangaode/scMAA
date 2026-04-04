"""Local literature ingestion and summarization for scMAA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterable

import instructor
import litellm
from pydantic import BaseModel, Field

from .utils import read_text, truncate_text


SUPPORTED_LITERATURE_SUFFIXES = {
    ".pdf",
    ".txt",
    ".md",
    ".markdown",
    ".rst",
}


@dataclass
class LiteratureDocument:
    """Normalized local literature document."""

    path: Path
    kind: str
    text: str

    @property
    def preview(self) -> str:
        return truncate_text(self.text.strip(), 3000)


class LiteratureHypothesisCandidate(BaseModel):
    """Dataset-aware literature-inspired hypothesis candidate."""

    title: str = Field(description="Short title for the candidate.")
    hypothesis: str = Field(description="Specific and testable hypothesis.")
    rationale: str = Field(description="Why this hypothesis is biologically motivated by the literature.")
    expected_evidence: str = Field(description="Concrete evidence that would support or weaken the hypothesis.")
    feasibility: str = Field(description="Why the hypothesis is executable with the current dataset and tools.")
    preferred_analysis_type: str = Field(description="One of: rna_only, tcr_only, joint.")
    required_fields: list[str] = Field(description="Key dataset fields or structures needed for the first test.")
    priority_score: int = Field(description="Priority score from 1 to 5, where 5 is best.")
    guardrail_notes: list[str] = Field(description="Guardrails or caveats that should constrain interpretation.")


class LiteratureHypothesisMenu(BaseModel):
    """Collection of candidate hypotheses distilled from local literature."""

    overview: str = Field(description="How the literature should bias the next round of planning.")
    candidates: list[LiteratureHypothesisCandidate] = Field(description="Ranked candidate hypotheses.")


def _read_pdf_with_pypdf(path: Path, max_pages: int = 80) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages[:max_pages]:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts).strip()


def _read_pdf_with_pdfplumber(path: Path, max_pages: int = 80) -> str:
    import pdfplumber

    parts: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages[:max_pages]:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue
    return "\n".join(parts).strip()


def read_literature_file(path: str | Path, max_chars: int = 60000) -> LiteratureDocument:
    """Read a supported local literature file into normalized text."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_LITERATURE_SUFFIXES:
        raise ValueError(f"Unsupported literature file type: {path}")

    text = ""
    if suffix == ".pdf":
        for reader in (_read_pdf_with_pypdf, _read_pdf_with_pdfplumber):
            try:
                text = reader(path)
            except Exception:
                continue
            if text.strip():
                break
    else:
        text = read_text(path, default="")

    cleaned = truncate_text(text.replace("\x00", " ").strip(), max_chars)
    if not cleaned:
        raise ValueError(f"Could not extract readable text from literature file: {path}")
    return LiteratureDocument(path=path.resolve(), kind=suffix.lstrip("."), text=cleaned)


def discover_literature_files(paths: Iterable[str | Path], max_files: int = 12) -> list[Path]:
    """Expand files/directories into a bounded list of supported literature files."""
    found: list[Path] = []
    seen: set[Path] = set()
    for item in paths:
        path = Path(item).resolve()
        candidates: list[Path]
        if path.is_dir():
            candidates = sorted(
                child for child in path.rglob("*") if child.is_file() and child.suffix.lower() in SUPPORTED_LITERATURE_SUFFIXES
            )
        elif path.is_file():
            candidates = [path]
        else:
            continue
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            found.append(candidate)
            if len(found) >= max_files:
                return found
    return found


class LiteratureSummarizer:
    """Summarize local literature for downstream planning prompts."""

    def __init__(self, model_name: str, logger: Any = None, log_prompts: bool = False) -> None:
        self.model_name = model_name
        self.logger = logger
        self.log_prompts = log_prompts
        self.client = instructor.from_litellm(litellm.completion)

    def _complete(self, prompt: str) -> str:
        if self.log_prompts and self.logger is not None:
            self.logger.log_prompt("user", prompt, "literature_summary")
        response = litellm.completion(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You summarize computational biology literature for a scRNA + scTCR research agent. "
                        "Focus on methods, biological hypotheses, result patterns, and reusable analysis ideas."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    def _complete_structured(self, prompt: str, response_model: type[BaseModel], prompt_name: str) -> BaseModel:
        if self.log_prompts and self.logger is not None:
            self.logger.log_prompt("user", prompt, prompt_name)
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You distill computational biology literature into concrete, executable hypotheses "
                        "for a scRNA + scTCR research agent. Favor ideas that are both biologically specific "
                        "and feasible with the available dataset."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_model=response_model,
        )

    def summarize_documents(self, documents: list[LiteratureDocument], context_summary: str = "") -> str:
        if not documents:
            return "No local literature was provided."

        per_doc_notes: list[str] = []
        for idx, document in enumerate(documents, start=1):
            prompt = (
                f"Summarize the local literature document below for scRNA + scTCR agent development.\n\n"
                f"Document {idx}: {document.path.name}\n"
                f"Document type: {document.kind}\n\n"
                "Extract these items:\n"
                "- core scientific question\n"
                "- analysis workflow or method pattern\n"
                "- useful biological priors for scRNA + scTCR\n"
                "- caveats or assumptions\n"
                "- ideas worth reusing in an agent\n\n"
                f"Optional research brief:\n{context_summary or 'No research brief provided.'}\n\n"
                f"Document text:\n{document.preview}"
            )
            note = self._complete(prompt).strip()
            if note:
                per_doc_notes.append(f"[{document.path.name}]\n{note}")

        combined_prompt = (
            "Combine the document summaries below into one planning-oriented literature brief for a scRNA + scTCR agent.\n\n"
            "Return a concise structured summary with:\n"
            "- methods or workflow ideas worth reusing\n"
            "- biological priors or recurrent result patterns\n"
            "- pitfalls to avoid\n"
            "- concrete hypothesis directions inspired by the literature\n\n"
            + "\n\n".join(per_doc_notes)
        )
        combined = self._complete(combined_prompt).strip()
        return combined or "\n\n".join(per_doc_notes)

    def propose_hypothesis_candidates(
        self,
        *,
        literature_summary: str,
        context_summary: str = "",
        rna_summary: str = "",
        tcr_summary: str = "",
        joint_summary: str = "",
        validation_summary: str = "",
        max_candidates: int = 4,
    ) -> LiteratureHypothesisMenu:
        """Generate literature-inspired, dataset-aware candidate hypotheses."""
        prompt = (
            "Using the literature summary and dataset notes below, generate ranked candidate hypotheses for the next "
            "scRNA + scTCR notebook analysis.\n\n"
            f"Return 3 to {max_candidates} candidates. Requirements:\n"
            "- Each candidate must be executable with the currently described dataset and Python-only notebook tools.\n"
            "- At least one candidate should explicitly reuse a mechanism, pathway, cell state, or biological pattern "
            "named in the literature when feasible.\n"
            "- Prefer candidates that can be tested with current RNA expression, tissue/sample metadata, and merged TCR fields.\n"
            "- Down-rank candidates that need unavailable annotations, external references, or heavy modeling.\n"
            "- Give higher priority to candidates that are more specific than generic 'expanded vs non-expanded' summaries.\n"
            "- The first test for each candidate should be small enough to run on a laptop.\n"
            "- Use `joint` when RNA/TCR integration is central.\n\n"
            f"RNA summary:\n{rna_summary or 'No RNA summary.'}\n\n"
            f"TCR summary:\n{tcr_summary or 'No TCR summary.'}\n\n"
            f"Joint summary:\n{joint_summary or 'No joint summary.'}\n\n"
            f"Validation summary:\n{validation_summary or 'No validation summary.'}\n\n"
            f"Research brief:\n{context_summary or 'No research brief provided.'}\n\n"
            f"Literature summary:\n{literature_summary or 'No local literature summary.'}"
        )
        return self._complete_structured(
            prompt,
            LiteratureHypothesisMenu,
            "literature_hypothesis_candidates",
        )

