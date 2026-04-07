"""Main orchestration for scMAA."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import re
from typing import Iterable

import litellm

from .deepresearch import DeepResearcher
from .execution import LegacyNotebookExecutor
from .figure_mode import FigureResult, build_publication_figure
from .hypothesis import AnalysisPlan, CandidateHypothesisMenu, HypothesisGenerator
from .literature import (
    LiteratureHypothesisMenu,
    LiteratureSummarizer,
    discover_literature_files,
    read_literature_file,
)
from .logger import AgentLogger
from .research import ResearchLedger
from .utils import (
    barcode_core,
    infer_sample_column,
    load_tcr_table,
    make_merge_key,
    normalize_barcode,
    normalize_tcr_columns,
    read_text,
    truncate_text,
)
from .validator import DatasetValidator


PACKAGE_CANDIDATES = (
    ("scanpy", "scanpy"),
    ("anndata", "anndata"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("sklearn", "scikit-learn"),
    ("scirpy", "scirpy"),
)

RNA_METADATA_HINTS = (
    "sample",
    "sample_id",
    "tissue",
    "orig.ident",
    "donor",
    "patient",
    "condition",
    "group",
    "batch",
    "timepoint",
    "cell_type",
    "annotation",
    "cluster",
    "leiden",
)


def _parse_status_text(status_text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in status_text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip().lower()] = value.strip()
    return parsed


def _publication_figure_section_from_status(status_path: Path) -> list[str]:
    if not status_path.exists():
        return []
    parsed = _parse_status_text(status_path.read_text(encoding="utf-8"))
    status = parsed.get("status", "unknown")
    lines = ["", "Publication figure", f"Status: {status}"]
    if status == "success":
        if parsed.get("png"):
            lines.append(f"PNG: {parsed['png']}")
        if parsed.get("pdf"):
            lines.append(f"PDF: {parsed['pdf']}")
        if parsed.get("summary"):
            lines.append(f"Summary: {parsed['summary']}")
    else:
        lines.append(f"Reason: {parsed.get('reason', 'unknown error')}")
    if parsed.get("note"):
        lines.append(f"Note: {parsed['note']}")
    return lines


def refresh_run_summary_from_artifacts(run_dir: str | Path) -> Path | None:
    run_dir = Path(run_dir)
    summary_path = run_dir / "run_summary.txt"
    figure_status_path = run_dir / "figure_status.txt"
    if not summary_path.exists() or not figure_status_path.exists():
        return None

    lines = summary_path.read_text(encoding="utf-8").splitlines()
    refreshed: list[str] = []
    in_publication_section = False
    for line in lines:
        if line == "Publication figure":
            in_publication_section = True
            continue
        if in_publication_section:
            continue
        refreshed.append(line)

    first_blank_idx = next((idx for idx, line in enumerate(refreshed) if line == ""), len(refreshed))
    figure_status_line = f"Figure status file: {figure_status_path}"
    figure_status_idx = next((idx for idx, line in enumerate(refreshed[:first_blank_idx]) if line.startswith("Figure status file:")), None)
    if figure_status_idx is not None:
        refreshed[figure_status_idx] = figure_status_line
    else:
        refreshed.insert(first_blank_idx, figure_status_line)

    refreshed.extend(_publication_figure_section_from_status(figure_status_path))
    summary_path.write_text("\n".join(refreshed).rstrip() + "\n", encoding="utf-8")
    return summary_path


def write_figure_status_file(
    run_dir: str | Path,
    *,
    figure_result: FigureResult | None = None,
    figure_error: str | None = None,
    note: str | None = None,
) -> Path:
    run_dir = Path(run_dir)
    status_path = run_dir / "figure_status.txt"
    if figure_result is not None:
        lines = [
            "status: success",
            f"png: {figure_result.png_path}",
            f"pdf: {figure_result.pdf_path}",
            f"summary: {figure_result.summary_path}",
        ]
    else:
        lines = [
            "status: failed",
            f"reason: {figure_error or 'unknown error'}",
        ]
    if note:
        lines.append(f"note: {note}")
    status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return status_path
def _top_counts(series, limit: int = 8) -> str:
    counts = series.value_counts(dropna=True).head(limit)
    if counts.empty:
        return "none"
    return ", ".join(f"{idx} ({int(val)})" for idx, val in counts.items())


def _detect_available_packages() -> str:
    installed: list[str] = []
    for module_name, display_name in PACKAGE_CANDIDATES:
        if importlib.util.find_spec(module_name) is not None:
            installed.append(display_name)
    return ", ".join(installed) or "pandas, numpy"


def _extract_requested_candidate_count(user_feedback: str, default: int = 5) -> int:
    text = user_feedback or ""
    patterns = (
        r"(\d+)\s*(?:candidate|candidates|hypotheses|hypothesis)\b",
        r"(?:generate|return|give|make)\s*(\d+)",
        r"(\d+)\s*个(?:候选假设|候选|假设)",
        r"生成\s*(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = int(match.group(1))
            return max(1, min(8, value))
    return default


def _feedback_requests_joint(user_feedback: str) -> bool:
    text = (user_feedback or "").lower()
    return "joint" in text or "联合" in text


def _normalize_candidate_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _candidate_key(candidate) -> str:
    return f"{_normalize_candidate_text(candidate.title)}||{_normalize_candidate_text(candidate.hypothesis)}"


def _candidate_theme_key(candidate) -> str:
    text = f"{getattr(candidate, 'title', '')} {getattr(candidate, 'hypothesis', '')}".lower()
    theme_map = {
        "treg": ("regulatory t", "treg"),
        "cytotoxic_t": ("cytotoxic t", "cd8", "exhausted cd8"),
        "effector_memory_t": ("effector/memory t", "effector memory", "memory t"),
        "mast_cell": ("mast cell",),
        "nk_cell": ("nk cell",),
        "trajectory": ("trajectory", "pseudotime", "lineage"),
        "clonotype": ("clonotype", "clone", "tcr repertoire", "clone type"),
        "checkpoint": ("checkpoint", "pd-1", "ctla-4", "immune checkpoint"),
        "exhaustion": ("exhaust", "pdcd1", "lag3", "havcr2", "tigit", "tox"),
        "trafficking": ("traffic", "migration", "homing", "cxcr", "ccr"),
        "stress": ("xbp1", "stress", "unfolded protein", "er stress"),
        "activation": ("activation", "effector", "cytotoxic", "ifng", "granzyme"),
    }
    for label, keywords in theme_map.items():
        if any(keyword in text for keyword in keywords):
            return label
    return re.sub(r"\s+", " ", text).strip()[:80]


def _candidate_has_mechanistic_direction(candidate) -> bool:
    text = " ".join(
        [
            getattr(candidate, "title", ""),
            getattr(candidate, "hypothesis", ""),
            getattr(candidate, "rationale", ""),
            getattr(candidate, "first_test", ""),
        ]
    ).lower()
    mechanism_tokens = (
        "xbp1",
        "pdcd1",
        "lag3",
        "havcr2",
        "tigit",
        "tox",
        "cxcr",
        "ccr",
        "pathway",
        "stress",
        "checkpoint",
        "exhaust",
        "traffick",
        "migration",
        "interferon",
        "metabolic",
        "signaling",
        "gene",
    )
    return any(token in text for token in mechanism_tokens)


def _merge_distinct_candidates(existing: list, new_items: list, limit: int) -> list:
    merged = list(existing)
    seen = {_candidate_key(item) for item in merged}
    for item in new_items:
        key = _candidate_key(item)
        if not key or key in seen:
            continue
        merged.append(item)
        seen.add(key)
        if len(merged) >= limit:
            break
    return merged


def _merge_theme_distinct_candidates(existing: list, new_items: list, limit: int) -> list:
    merged = list(existing)
    seen_keys = {_candidate_key(item) for item in merged}
    seen_themes = {_candidate_theme_key(item) for item in merged}
    for item in new_items:
        key = _candidate_key(item)
        theme = _candidate_theme_key(item)
        if not key or key in seen_keys or theme in seen_themes:
            continue
        merged.append(item)
        seen_keys.add(key)
        seen_themes.add(theme)
        if len(merged) >= limit:
            break
    return merged


class ScRTAgent:
    """Research-oriented agent for integrated scRNA + scTCR analysis."""

    def __init__(
        self,
        *,
        rna_h5ad_path: str,
        tcr_path: str,
        research_brief_path: str | None = None,
        context_path: str | None = None,
        literature_paths: Iterable[str] | None = None,
        analysis_name: str = "scrt_run",
        model_name: str = "gpt-4o",
        hypothesis_model: str | None = None,
        execution_model: str | None = None,
        vision_model: str = "gpt-4o",
        num_analyses: int = 3,
        max_iterations: int = 6,
        output_home: str = ".",
        prompt_dir: str | None = None,
        use_self_critique: bool = True,
        use_documentation: bool = True,
        use_VLM: bool = True,
        use_deepresearch: bool = False,
        generate_publication_figure: bool = False,
        publication_figure_name: str | None = None,
        log_prompts: bool = False,
        max_fix_attempts: int = 3,
    ) -> None:
        brief_path = research_brief_path or context_path
        if not brief_path:
            raise ValueError("research_brief_path or context_path must be provided.")
        self.rna_h5ad_path = str(Path(rna_h5ad_path).resolve())
        self.tcr_path = str(Path(tcr_path).resolve())
        self.research_brief_path = str(Path(brief_path).resolve())
        self.context_path = self.research_brief_path
        self.literature_paths = [str(Path(item).resolve()) for item in (literature_paths or [])]
        self.analysis_name = analysis_name
        self.model_name = model_name
        self.hypothesis_model = hypothesis_model or model_name
        self.execution_model = execution_model or model_name
        self.vision_model = vision_model
        self.num_analyses = max(1, int(num_analyses))
        self.max_iterations = max(1, int(max_iterations))
        self.use_self_critique = use_self_critique
        self.use_documentation = use_documentation
        self.use_VLM = use_VLM
        self.use_deepresearch = use_deepresearch
        self.generate_publication_figure = generate_publication_figure
        self.publication_figure_name = publication_figure_name
        self.log_prompts = log_prompts
        self.max_fix_attempts = max(1, int(max_fix_attempts))

        self.project_root = Path(__file__).resolve().parents[1]
        self.prompt_dir = Path(prompt_dir) if prompt_dir else self.project_root / "scrt_agent" / "prompts"
        self.output_home = Path(output_home).resolve()
        self.output_dir = self.output_home / analysis_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.output_dir / "logs"
        self.available_packages = _detect_available_packages()

        self._load_environment_files()
        self.logger = AgentLogger(analysis_name=analysis_name, log_dir=self.log_dir, log_prompts=log_prompts)
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")

        self.context_summary = truncate_text(read_text(self.research_brief_path, default=""), 12000)
        self.rna_summary = self._summarize_rna_data(self.rna_h5ad_path)
        self.tcr_summary = self._summarize_tcr_data(self.tcr_path)
        self.joint_summary = self._summarize_joint_data(self.rna_h5ad_path, self.tcr_path)
        self.dataset_validation = DatasetValidator().inspect_inputs(self.rna_h5ad_path, self.tcr_path)
        self.validation_summary = self.dataset_validation.to_prompt_text()
        self.standard_baseline_summary = self._summarize_standard_baseline(self.rna_h5ad_path, self.tcr_path)
        (self.output_dir / "standard_baseline_summary.txt").write_text(self.standard_baseline_summary, encoding="utf-8")
        self.baseline_planning_context = self._extract_baseline_planning_context(self.standard_baseline_summary)
        (self.output_dir / "baseline_planning_context.txt").write_text(self.baseline_planning_context, encoding="utf-8")
        self.baseline_tcell_labels = self._extract_baseline_tcell_labels(self.standard_baseline_summary)
        (self.output_dir / "baseline_tcell_labels.txt").write_text("\n".join(self.baseline_tcell_labels) + ("\n" if self.baseline_tcell_labels else ""), encoding="utf-8")
        self.literature_documents = self._load_literature_documents()
        self.literature_sources = "\n".join(str(doc.path) for doc in self.literature_documents) or "No local literature files."
        self.literature_summary = self._summarize_literature()
        self.literature_hypothesis_menu = self._generate_literature_hypothesis_menu()
        self.literature_hypothesis_candidates = self._format_literature_hypothesis_menu(self.literature_hypothesis_menu)

        coding_guidelines_template = self._read_prompt("coding_guidelines.txt")
        self.coding_guidelines = coding_guidelines_template.format(AVAILABLE_PACKAGES=self.available_packages)
        coding_system_template = self._read_prompt("coding_system_prompt.txt")
        self.coding_system_prompt = coding_system_template.format(AVAILABLE_PACKAGES=self.available_packages)

        self.deepresearch_background = ""
        if self.use_deepresearch:
            self.deepresearch_background = self._generate_deepresearch_background()

        self.hypothesis_generator = HypothesisGenerator(
            model_name=self.hypothesis_model,
            prompt_dir=self.prompt_dir,
            coding_guidelines=self.coding_guidelines,
            coding_system_prompt=self.coding_system_prompt,
            rna_summary=self.rna_summary,
            tcr_summary=self.tcr_summary,
            joint_summary=self.joint_summary,
            validation_summary=self.validation_summary,
            standard_baseline_summary=self.standard_baseline_summary,
            baseline_planning_context=self.baseline_planning_context,
            baseline_tcell_labels=self.baseline_tcell_labels,
            context_summary=self.context_summary,
            literature_summary=self.literature_summary,
            literature_candidates_summary=self.literature_hypothesis_candidates,
            logger=self.logger,
            use_self_critique=self.use_self_critique,
            use_documentation=self.use_documentation,
            max_iterations=self.max_iterations,
            deepresearch_background=self.deepresearch_background,
            log_prompts=self.log_prompts,
        )
        self.executor = LegacyNotebookExecutor(
            hypothesis_generator=self.hypothesis_generator,
            openai_api_key=self.openai_api_key,
            model_name=self.execution_model,
            vision_model=self.vision_model,
            prompt_dir=self.prompt_dir,
            coding_guidelines=self.coding_guidelines,
            coding_system_prompt=self.coding_system_prompt,
            rna_summary=self.rna_summary,
            tcr_summary=self.tcr_summary,
            joint_summary=self.joint_summary,
            validation_summary=self.validation_summary,
            standard_baseline_summary=self.standard_baseline_summary,
            context_summary=self.context_summary,
            logger=self.logger,
            rna_h5ad_path=self.rna_h5ad_path,
            tcr_path=self.tcr_path,
            output_dir=self.output_dir,
            analysis_name=self.analysis_name,
            max_iterations=self.max_iterations,
            max_fix_attempts=self.max_fix_attempts,
            use_VLM=self.use_VLM,
            use_documentation=self.use_documentation,
        )

    def prepare_candidate_hypotheses(self, user_feedback: str = "") -> CandidateHypothesisMenu:
        research_ledger = self._make_research_ledger()
        target_count = _extract_requested_candidate_count(user_feedback, default=5)
        hard_constraints = [
            f"Return exactly {target_count} candidate hypotheses.",
            "This is the scRNA + scTCR workflow.",
            "Keep the menu joint-led by default when RNA-TCR overlap is usable.",
            "Most candidates should be joint. Use rna_only only when the first evidence step does not need clonotype or repertoire support.",
            "Do not return a single-candidate menu unless the user explicitly asked for one.",
            "Every candidate must include a concrete first notebook step.",
        ]
        effective_feedback = user_feedback.strip()
        effective_feedback = (
            effective_feedback + "\n\nHard menu constraints:\n- " + "\n- ".join(hard_constraints)
            if effective_feedback
            else "Hard menu constraints:\n- " + "\n- ".join(hard_constraints)
        )
        research_state_summary = research_ledger.to_prompt_text()

        def _generate_menu(feedback_text: str) -> CandidateHypothesisMenu:
            return self.hypothesis_generator.generate_candidate_hypotheses(
                research_state_summary=research_state_summary,
                past_analyses="",
                user_feedback=feedback_text,
            )

        menu = _generate_menu(effective_feedback)
        joint_floor = max(3, target_count // 2 + 1)
        joint_light = sum(item.preferred_analysis_type == "joint" for item in menu.candidates) < joint_floor
        theme_keys = [_candidate_theme_key(item) for item in menu.candidates]
        unique_theme_count = len(set(theme_keys))
        mechanism_count = sum(_candidate_has_mechanistic_direction(item) for item in menu.candidates)
        needs_retry = (
            len(menu.candidates) != target_count
            or any(not item.first_test.strip() for item in menu.candidates)
            or (joint_light and not ("rna_only" in (user_feedback or "").lower()))
            or unique_theme_count < max(3, min(target_count, 4))
            or mechanism_count < min(2, target_count)
        )
        if needs_retry:
            retry_feedback = (
                effective_feedback
                + "\n\nThe previous menu did not satisfy the count or modality-balance requirements. "
                  "Retry and satisfy every hard constraint exactly. "
                  "Avoid near-duplicate themes and include at least two candidates with explicit mechanistic directions."
            )
            menu = _generate_menu(retry_feedback)

        merged_candidates = _merge_distinct_candidates([], menu.candidates, target_count)
        supplement_attempt = 0
        while len(merged_candidates) < target_count and supplement_attempt < 3:
            supplement_attempt += 1
            titles = [f"{idx + 1}. {item.title}" for idx, item in enumerate(merged_candidates)]
            summary_lines = [
                f"Current candidates already kept: {len(merged_candidates)}",
                "Keep all existing candidates unchanged.",
                f"Add {target_count - len(merged_candidates)} new, distinct candidates so that the final menu has exactly {target_count}.",
                "Do not repeat existing titles or hypotheses.",
                "Return a full candidate menu, not a partial patch.",
                "Preserve joint-led balance unless the user explicitly requested otherwise.",
            ]
            if titles:
                summary_lines.append("Existing titles:")
                summary_lines.extend(f"- {line}" for line in titles)
            top_up_feedback = effective_feedback + "\n\n" + "\n".join(summary_lines)
            top_up_menu = _generate_menu(top_up_feedback)
            merged_candidates = _merge_distinct_candidates(merged_candidates, top_up_menu.candidates, target_count)

        required_theme_floor = max(3, min(target_count, 4))
        theme_diversify_attempt = 0
        while len({_candidate_theme_key(item) for item in merged_candidates}) < required_theme_floor and theme_diversify_attempt < 3:
            theme_diversify_attempt += 1
            unique_by_theme: list = []
            seen_themes: set[str] = set()
            for item in merged_candidates:
                theme = _candidate_theme_key(item)
                if theme in seen_themes:
                    continue
                unique_by_theme.append(item)
                seen_themes.add(theme)
            banned_themes = sorted(seen_themes)
            diversify_feedback = (
                effective_feedback
                + "\n\nTheme diversification request:\n"
                + f"Keep these existing themes unchanged: {', '.join(banned_themes)}.\n"
                + f"Add {target_count - len(unique_by_theme)} new candidates with clearly different biological themes.\n"
                + "Do not return trajectory-only or clonotype-overlap-only duplicates.\n"
                + "Prefer mechanistic directions involving concrete genes, pathways, trafficking programs, stress programs, or exhaustion regulators."
            )
            diversify_menu = _generate_menu(diversify_feedback)
            merged_candidates = _merge_theme_distinct_candidates(unique_by_theme, diversify_menu.candidates, target_count)

        if len(merged_candidates) < target_count:
            final_fill_feedback = (
                effective_feedback
                + "\n\nFinal fill request:\n"
                + f"The menu currently has {len(merged_candidates)} candidates but needs exactly {target_count}.\n"
                + "Keep all existing candidates unchanged and add only the missing number of new candidates.\n"
                + "Prefer themes not already represented; if that is impossible, fill with the strongest remaining mechanistic candidates."
            )
            final_fill_menu = _generate_menu(final_fill_feedback)
            merged_candidates = _merge_distinct_candidates(merged_candidates, final_fill_menu.candidates, target_count)

        if len(merged_candidates) < target_count:
            self.logger.warning(
                "Candidate menu remained undersized after top-up attempts. "
                f"Returning {len(merged_candidates)} candidate(s); target was {target_count}."
            )
        else:
            merged_candidates = merged_candidates[:target_count]

        joint_count = sum(item.preferred_analysis_type == "joint" for item in merged_candidates)
        mechanism_count = sum(_candidate_has_mechanistic_direction(item) for item in merged_candidates)
        unique_theme_count = len({_candidate_theme_key(item) for item in merged_candidates})
        if joint_count < joint_floor and target_count > 1:
            self.logger.warning(
                "Candidate menu met count after top-up but joint coverage is still light. "
                f"joint={joint_count}, target_floor={joint_floor}."
            )
        if unique_theme_count < max(3, min(target_count, 4)):
            self.logger.warning(
                "Candidate menu still contains repeated themes after top-up. "
                f"unique_themes={unique_theme_count}, candidates={len(merged_candidates)}."
            )
        if mechanism_count < min(2, target_count):
            self.logger.warning(
                "Candidate menu still lacks enough mechanistic candidates after top-up. "
                f"mechanistic={mechanism_count}, candidates={len(merged_candidates)}."
            )

        research_focus = menu.research_focus if getattr(menu, "research_focus", "").strip() else "Ranked integrated scRNA + scTCR candidate menu."
        return CandidateHypothesisMenu(research_focus=research_focus, candidates=merged_candidates)

    def revise_hypothesis(self, *, hypothesis: str, user_feedback: str) -> str:
        research_ledger = self._make_research_ledger()
        revision = self.hypothesis_generator.revise_hypothesis_with_feedback(
            hypothesis=hypothesis,
            user_feedback=user_feedback,
            research_state_summary=research_ledger.to_prompt_text(),
            past_analyses="",
        )
        self.logger.log_response(
            (
                f"Revised hypothesis: {revision.revised_hypothesis}\n"
                f"Revision rationale: {revision.revision_rationale}\n"
                f"Preferred analysis type: {revision.preferred_analysis_type}\n"
                f"Retained constraints:\n" + "\n".join(f"- {item}" for item in revision.retained_constraints)
            ),
            "revised_hypothesis",
        )
        return revision.revised_hypothesis

    def build_plan_from_hypothesis(self, hypothesis: str, user_strategy_feedback: str = ""):
        research_ledger = self._make_research_ledger()
        return self.hypothesis_generator.generate_analysis_from_hypothesis(
            hypothesis,
            past_analyses="",
            research_state_summary=research_ledger.to_prompt_text(),
            seed_context="User-approved hypothesis from interactive review.",
            user_strategy_feedback=user_strategy_feedback,
        )

    def revise_plan(self, current_plan: AnalysisPlan, user_strategy_feedback: str):
        research_ledger = self._make_research_ledger()
        return self.hypothesis_generator.revise_analysis_plan(
            current_plan,
            past_analyses="",
            research_state_summary=research_ledger.to_prompt_text(),
            user_strategy_feedback=user_strategy_feedback,
        )

    def _write_executed_hypotheses_artifact(self, executed_hypotheses: list[str]) -> Path:
        executed_hypotheses_path = self.output_dir / "executed_hypotheses.txt"
        executed_lines = [f"Analysis {idx + 1}: {text}" for idx, text in enumerate(executed_hypotheses)]
        executed_hypotheses_path.write_text(
            "\n".join(executed_lines) + ("\n" if executed_lines else ""),
            encoding="utf-8",
        )
        return executed_hypotheses_path

    def _finalize_run_outputs(
        self,
        *,
        past_analyses: str,
        notebook_paths: list[Path],
        ledger_summaries: list[str],
        executed_hypotheses: list[str],
        seeded: list[str],
        figure_result: FigureResult | None,
        figure_error: str | None,
    ) -> Path:
        executed_hypotheses_path = self._write_executed_hypotheses_artifact(executed_hypotheses)
        executed_lines = [f"Analysis {idx + 1}: {text}" for idx, text in enumerate(executed_hypotheses)]

        seeded_hypotheses_path: Path | None = None
        if seeded:
            seeded_hypotheses_path = self.output_dir / "seeded_hypotheses.txt"
            seeded_lines = [f"Analysis {idx + 1}: {text}" for idx, text in enumerate(seeded)]
            seeded_hypotheses_path.write_text("\n".join(seeded_lines) + "\n", encoding="utf-8")

        figure_status_path: Path | None = None
        if self.generate_publication_figure:
            figure_status_path = write_figure_status_file(
                self.output_dir,
                figure_result=figure_result,
                figure_error=figure_error,
            )

        summary_path = self.output_dir / "run_summary.txt"
        summary_lines = [
            f"Analysis name: {self.analysis_name}",
            f"RNA input: {self.rna_h5ad_path}",
            f"TCR input: {self.tcr_path}",
            f"Research brief input: {self.research_brief_path}",
            f"Model (planning): {self.hypothesis_model}",
            f"Model (execution support): {self.execution_model}",
            f"Vision model: {self.vision_model}",
            f"Detected packages: {self.available_packages}",
            f"Executed hypotheses file: {executed_hypotheses_path}",
        ]
        if seeded_hypotheses_path is not None:
            summary_lines.append(f"Seeded hypotheses file: {seeded_hypotheses_path}")
        if figure_status_path is not None:
            summary_lines.append(f"Figure status file: {figure_status_path}")
        summary_lines.extend(
            [
                "",
                "Executed hypotheses",
                "\n".join(executed_lines) or "None",
                "",
                "RNA summary",
                self.rna_summary,
                "",
                "TCR summary",
                self.tcr_summary,
                "",
                "Joint summary",
                self.joint_summary,
                "",
                "Validation summary",
                self.validation_summary,
                "",
                "Literature sources",
                self.literature_sources,
                "",
                "Literature summary",
                self.literature_summary,
                "",
                "Literature-derived hypothesis candidates",
                self.literature_hypothesis_candidates,
                "",
                "Past analyses",
                past_analyses or "No analyses were completed.",
                "",
                "Research ledger summaries",
                "\n\n".join(ledger_summaries) or "No research ledger entries.",
                "",
                "Generated notebooks",
                "\n".join(str(path) for path in notebook_paths) or "None",
            ]
        )
        summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
        if figure_status_path is not None:
            refresh_run_summary_from_artifacts(self.output_dir)
        self.logger.info(f"Run complete. Summary written to {summary_path}")
        return summary_path

    def run_approved_plan(self, approved_plan: AnalysisPlan | dict) -> Path:
        plan = approved_plan if isinstance(approved_plan, AnalysisPlan) else AnalysisPlan.model_validate(approved_plan)
        notebook_paths: list[Path] = []
        ledger_summaries: list[str] = []
        figure_result: FigureResult | None = None
        figure_error: str | None = None
        research_ledger = self._make_research_ledger()

        self.logger.info(
            f"Starting scMAA run from approved plan. "
            f"RNA={self.rna_h5ad_path}, TCR={self.tcr_path}, research_brief={self.research_brief_path}"
        )

        past_analyses, research_ledger = self.executor.execute_idea(
            analysis=plan,
            past_analyses="",
            research_ledger=research_ledger,
            analysis_idx=0,
            seeded=True,
        )
        notebook_paths.append(self.output_dir / f"{self.analysis_name}_analysis_1.ipynb")
        ledger_summaries.append(research_ledger.to_prompt_text())

        self._write_executed_hypotheses_artifact([plan.hypothesis])
        if self.generate_publication_figure:
            try:
                figure_result = self._build_publication_figure()
            except Exception as exc:
                figure_error = str(exc)
                self.logger.warning(f"Publication figure generation failed: {exc}")

        return self._finalize_run_outputs(
            past_analyses=past_analyses,
            notebook_paths=notebook_paths,
            ledger_summaries=ledger_summaries,
            executed_hypotheses=[plan.hypothesis],
            seeded=[plan.hypothesis],
            figure_result=figure_result,
            figure_error=figure_error,
        )

    def _build_publication_figure(self) -> FigureResult:
        figure_dir = self.output_dir / "figure"
        figure_name = self.publication_figure_name or f"{self.analysis_name}_publication_figure"
        result = build_publication_figure(
            rna_h5ad_path=self.rna_h5ad_path,
            tcr_path=self.tcr_path,
            output_dir=figure_dir,
            figure_name=figure_name,
            hypothesis_model=self.hypothesis_model,
            baseline_summary_text=self.standard_baseline_summary,
            prompt_dir=self.prompt_dir,
        )
        self.logger.info(f"Publication figure generated at {result.png_path}")
        return result

    def _make_research_ledger(self) -> ResearchLedger:
        ledger = ResearchLedger(
            dataset_strengths=list(self.dataset_validation.strengths),
            dataset_warnings=list(self.dataset_validation.warnings),
            guardrails=list(self.dataset_validation.guardrails),
        )
        if self.dataset_validation.warnings:
            ledger.open_questions.extend(
                [
                    "Which conclusions remain valid after accounting for dataset-level risks?",
                    "What is the strongest next analysis that can survive the listed guardrails?",
                ]
            )
        return ledger

    def _load_literature_documents(self):
        if not self.literature_paths:
            return []
        documents = []
        for path in discover_literature_files(self.literature_paths):
            try:
                documents.append(read_literature_file(path))
            except Exception as exc:
                self.logger.warning(f"Skipping literature file {path}: {exc}")
        return documents

    def _summarize_literature(self) -> str:
        if not self.literature_documents:
            return "No local literature was provided."
        try:
            summarizer = LiteratureSummarizer(
                model_name=self.hypothesis_model,
                logger=self.logger,
                log_prompts=self.log_prompts,
            )
            summary = summarizer.summarize_documents(
                self.literature_documents,
                context_summary=self.context_summary,
            )
            self.logger.log_response(summary, "literature_summary")
            return truncate_text(summary, 20000)
        except Exception as exc:
            self.logger.warning(f"Literature summarization failed, falling back to raw previews: {exc}")
            fallback = []
            for document in self.literature_documents:
                fallback.append(f"[{document.path.name}]\n{document.preview}")
            return truncate_text("\n\n".join(fallback), 20000)

    def _generate_literature_hypothesis_menu(self) -> LiteratureHypothesisMenu | None:
        if not self.literature_documents:
            return None
        try:
            summarizer = LiteratureSummarizer(
                model_name=self.hypothesis_model,
                logger=self.logger,
                log_prompts=self.log_prompts,
            )
            menu = summarizer.propose_hypothesis_candidates(
                literature_summary=self.literature_summary,
                context_summary=self.context_summary,
                rna_summary=self.rna_summary,
                tcr_summary=self.tcr_summary,
                joint_summary=self.joint_summary,
                validation_summary=self.validation_summary,
            )
            self.logger.log_response(self._format_literature_hypothesis_menu(menu), "literature_hypothesis_candidates")
            return menu
        except Exception as exc:
            self.logger.warning(f"Literature hypothesis generation failed: {exc}")
            return None

    def _format_literature_hypothesis_menu(self, menu: LiteratureHypothesisMenu | None) -> str:
        if menu is None or not menu.candidates:
            return "No literature-derived hypothesis candidates."
        lines = [
            f"Overview: {menu.overview}",
            "",
            "Candidates:",
        ]
        for idx, candidate in enumerate(menu.candidates, start=1):
            lines.extend(
                [
                    f"{idx}. {candidate.title}",
                    f"   hypothesis: {candidate.hypothesis}",
                    f"   rationale: {candidate.rationale}",
                    f"   expected evidence: {candidate.expected_evidence}",
                    f"   feasibility: {candidate.feasibility}",
                    f"   preferred analysis type: {candidate.preferred_analysis_type}",
                    f"   priority score: {candidate.priority_score}",
                    f"   required fields: {', '.join(candidate.required_fields) or 'none listed'}",
                    f"   guardrails: {'; '.join(candidate.guardrail_notes) or 'none listed'}",
                ]
            )
        return "\n".join(lines)

    def _read_prompt(self, name: str) -> str:
        return (self.prompt_dir / name).read_text(encoding="utf-8")

    def _load_environment_files(self) -> None:
        try:
            from dotenv import load_dotenv
        except Exception:
            return

        candidate_dirs = [
            Path.cwd(),
            self.project_root,
            self.project_root.parent,
            Path(self.rna_h5ad_path).resolve().parent,
        ]
        seen: set[Path] = set()
        for directory in candidate_dirs:
            if directory in seen:
                continue
            seen.add(directory)
            for name in (".env", "OPENAI.env", "deepseek.env"):
                env_path = directory / name
                if env_path.exists():
                    load_dotenv(env_path, override=False)

    def _summarize_rna_data(self, rna_h5ad_path: str) -> str:
        import anndata as ad

        adata = ad.read_h5ad(rna_h5ad_path, backed="r")
        try:
            obs_columns = [str(col) for col in adata.obs.columns]
            var_columns = [str(col) for col in adata.var.columns]
            obsm_keys = list(adata.obsm.keys())
            layers_keys = list(adata.layers.keys())
            uns_keys = list(adata.uns.keys())

            metadata_lines: list[str] = []
            lower_to_original = {col.lower(): col for col in obs_columns}
            for hint in RNA_METADATA_HINTS:
                column = lower_to_original.get(hint)
                if not column:
                    continue
                series = adata.obs[column]
                metadata_lines.append(
                    f"- {column}: {series.nunique(dropna=True)} unique values; top levels: {_top_counts(series)}"
                )

            summary_lines = [
                f"RNA file: {rna_h5ad_path}",
                f"RNA matrix shape: {adata.n_obs} cells x {adata.n_vars} genes",
                f"obs columns ({len(obs_columns)}): {', '.join(obs_columns[:30]) or 'none'}",
                f"var columns ({len(var_columns)}): {', '.join(var_columns[:20]) or 'none'}",
                f"obsm keys: {', '.join(obsm_keys[:20]) or 'none'}",
                f"layers: {', '.join(layers_keys[:20]) or 'none'}",
                f"uns keys: {', '.join(uns_keys[:20]) or 'none'}",
                f"raw present: {adata.raw is not None}",
            ]
            if metadata_lines:
                summary_lines.append("Candidate RNA metadata columns:")
                summary_lines.extend(metadata_lines)
            return "\n".join(summary_lines)
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()

    def _summarize_tcr_data(self, tcr_path: str) -> str:
        df = normalize_tcr_columns(load_tcr_table(tcr_path))
        columns = [str(col) for col in df.columns]
        lines = [
            f"TCR file: {tcr_path}",
            f"TCR rows: {len(df)}",
            f"TCR columns ({len(columns)}): {', '.join(columns[:30]) or 'none'}",
        ]

        if "barcode" in df.columns:
            barcodes = df["barcode"].astype(str)
            lines.append(f"Unique TCR barcodes: {barcodes.nunique()}")
        if "sample_id" in df.columns:
            lines.append(
                f"Sample IDs: {df['sample_id'].nunique(dropna=True)} unique; top levels: {_top_counts(df['sample_id'])}"
            )
        if "clonotype_id" in df.columns:
            clonotypes = df["clonotype_id"].dropna().astype(str)
            if not clonotypes.empty:
                lines.append(f"Unique clonotypes: {clonotypes.nunique()}")
                lines.append(f"Top clonotypes: {_top_counts(clonotypes)}")
            else:
                lines.append("Clonotype IDs are present but mostly empty.")
        if "chain" in df.columns:
            lines.append(f"TCR chain distribution: {_top_counts(df['chain'].astype(str))}")
        if "v_gene" in df.columns:
            lines.append(f"Top V genes: {_top_counts(df['v_gene'].astype(str))}")
        if "j_gene" in df.columns:
            lines.append(f"Top J genes: {_top_counts(df['j_gene'].astype(str))}")
        if "productive" in df.columns:
            productive_rate = float(df["productive"].fillna(False).astype(bool).mean())
            lines.append(f"Productive fraction: {productive_rate:.3f}")
        return "\n".join(lines)

    def _summarize_joint_data(self, rna_h5ad_path: str, tcr_path: str) -> str:
        import anndata as ad

        adata = ad.read_h5ad(rna_h5ad_path, backed="r")
        try:
            lower_to_original = {str(col).lower(): str(col) for col in adata.obs.columns}
            barcode_column = lower_to_original.get("barcode")
            sample_column = infer_sample_column(adata.obs.columns)
            rna_barcodes = (
                adata.obs[barcode_column].astype(str).tolist()
                if barcode_column is not None
                else [str(idx) for idx in adata.obs_names]
            )
            rna_samples = (
                adata.obs[sample_column].tolist()
                if sample_column is not None
                else [None] * len(rna_barcodes)
            )

            df = normalize_tcr_columns(load_tcr_table(tcr_path))
            exact_overlap = 0
            core_overlap = 0
            sample_exact_overlap = 0
            sample_core_overlap = 0
            tcr_unique_barcodes = 0
            tcr_unique_core = 0
            tcr_clonotypes = 0
            sample_lines: list[str] = []

            if "barcode" in df.columns:
                tcr_barcodes = set(df["barcode"].dropna().astype(str))
                tcr_core = {barcode_core(value) for value in tcr_barcodes}
                exact_overlap = sum(1 for barcode in rna_barcodes if normalize_barcode(barcode) in tcr_barcodes)
                core_overlap = sum(1 for barcode in rna_barcodes if barcode_core(barcode) in tcr_core)
                tcr_unique_barcodes = len(tcr_barcodes)
                tcr_unique_core = len(tcr_core)
                tcr_sample_column = infer_sample_column(df.columns)
                if sample_column and tcr_sample_column:
                    tcr_exact_keys = {
                        make_merge_key(barcode, sample, use_core=False)
                        for barcode, sample in zip(df["barcode"], df[tcr_sample_column])
                        if make_merge_key(barcode, sample, use_core=False)
                    }
                    tcr_core_keys = {
                        make_merge_key(barcode, sample, use_core=True)
                        for barcode, sample in zip(df["barcode"], df[tcr_sample_column])
                        if make_merge_key(barcode, sample, use_core=True)
                    }
                    sample_exact_overlap = sum(
                        1
                        for barcode, sample in zip(rna_barcodes, rna_samples)
                        if make_merge_key(barcode, sample, use_core=False) in tcr_exact_keys
                    )
                    sample_core_overlap = sum(
                        1
                        for barcode, sample in zip(rna_barcodes, rna_samples)
                        if make_merge_key(barcode, sample, use_core=True) in tcr_core_keys
                    )
            if "clonotype_id" in df.columns:
                tcr_clonotypes = df["clonotype_id"].dropna().astype(str).nunique()
            if sample_column:
                sample_series = adata.obs[sample_column]
                sample_lines.append(
                    f"RNA sample column selected for stratified analyses: {sample_column} "
                    f"({sample_series.nunique(dropna=True)} groups)."
                )
            if "tissue" in adata.obs.columns:
                sample_lines.append(
                    f"RNA tissue groups: {adata.obs['tissue'].nunique(dropna=True)}; "
                    f"top levels: {_top_counts(adata.obs['tissue'])}"
                )
            if "sample_id" in df.columns:
                sample_lines.append(
                    f"TCR sample_id groups: {df['sample_id'].nunique(dropna=True)}; "
                    f"top levels: {_top_counts(df['sample_id'])}"
                )
            if "tissue" in df.columns:
                sample_lines.append(
                    f"TCR tissue groups: {df['tissue'].nunique(dropna=True)}; "
                    f"top levels: {_top_counts(df['tissue'])}"
                )

            overlap_modes = {
                "sample_exact": sample_exact_overlap,
                "sample_barcode_core": sample_core_overlap,
                "exact": exact_overlap,
                "barcode_core": core_overlap,
            }
            chosen_mode = max(overlap_modes, key=overlap_modes.get)
            best_overlap = overlap_modes[chosen_mode]
            lines = [
                f"Exact barcode overlap between RNA obs_names and TCR barcodes: {exact_overlap}",
                f"Core barcode overlap after stripping suffixes: {core_overlap}",
                f"Sample-aware exact overlap: {sample_exact_overlap}",
                f"Sample-aware core overlap: {sample_core_overlap}",
                f"Preferred merge mode: {chosen_mode}",
                f"RNA cells: {adata.n_obs}",
                f"TCR unique barcodes: {tcr_unique_barcodes}",
                f"TCR unique barcode cores: {tcr_unique_core}",
                f"TCR unique clonotypes: {tcr_clonotypes}",
                f"Approximate RNA coverage by TCR ({chosen_mode}): "
                f"{(best_overlap / max(adata.n_obs, 1)):.3f}",
            ]
            lines.extend(sample_lines)
            lines.append(
                "Merged notebook fields that will exist on adata_rna.obs after setup: "
                "barcode, barcode_core, clonotype_id, chain, cdr3, v_gene, j_gene, "
                "productive_any, tcr_chain_count, tcr_reads, has_tcr, clone_size, expanded_clone."
            )
            return "\n".join(lines)
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()

    def _summarize_standard_baseline(self, rna_h5ad_path: str, tcr_path: str) -> str:
        import anndata as ad
        import pandas as pd
        from .notebook_tools import (
            assign_clone_type_labels,
            is_t_cell_like_label,
            paired_tcr_subset,
            recluster_and_annotate_t_cells,
        )

        adata = ad.read_h5ad(rna_h5ad_path, backed="r")
        try:
            obs = adata.obs.copy()
            if "barcode" not in obs.columns:
                obs["barcode"] = [str(idx) for idx in adata.obs_names]
            obs["barcode_exact"] = obs["barcode"].astype(str).map(normalize_barcode)
            obs["barcode_core"] = obs["barcode"].astype(str).map(barcode_core)

            group_col = next(
                (
                    column
                    for column in ("cell_type", "cluster_cell_type", "annotation", "leiden")
                    if column in obs.columns
                ),
                None,
            )
            tissue_col = next(
                (
                    column
                    for column in ("tissue", "sample_key", "sample_id")
                    if column in obs.columns
                ),
                None,
            )
            sample_col = infer_sample_column(obs.columns)

            lines = [
                "Mandatory baseline analysis before hypothesis generation:",
                "- scRNA UMAP with current annotations.",
                "- Global cell-type proportion bar plot across tissues/groups.",
                "- T-cell-only reclustering/annotation UMAP.",
                "- T-cell subtype proportion bar plot across tissues/groups.",
                "- T-cell clonotype/cloneType UMAP.",
                "- cloneType composition bar plot across T-cell subtypes.",
            ]
            if group_col is None or tissue_col is None:
                lines.append("Baseline warning: RNA metadata does not expose both annotation and tissue columns clearly.")
                return "\n".join(lines)

            lines.extend(
                [
                    f"Baseline annotation column: {group_col}",
                    f"Baseline tissue column: {tissue_col}",
                    f"RNA has UMAP coordinates: {'X_umap' in adata.obsm}",
                    "Top global cell types: " + _top_counts(obs[group_col].astype(str), limit=10),
                    "Top tissue labels: " + _top_counts(obs[tissue_col].astype(str), limit=10),
                ]
            )

            celltype_by_tissue = pd.crosstab(obs[tissue_col].astype(str), obs[group_col].astype(str))
            if not celltype_by_tissue.empty:
                dominant = []
                for tissue_name, row in celltype_by_tissue.iterrows():
                    top_label = row.sort_values(ascending=False).index[0]
                    dominant.append(f"{tissue_name}: {top_label}")
                lines.append("Dominant cell type per tissue: " + "; ".join(dominant[:8]))

            t_mask = obs[group_col].astype(str).map(is_t_cell_like_label)
            t_obs = obs.loc[t_mask].copy()
            if t_obs.empty:
                lines.append("No T-cell-like annotations detected from the current RNA labels.")
                return "\n".join(lines)

            lines.append(
                "Coarse T-cell gate from current annotations: "
                + _top_counts(t_obs[group_col].astype(str), limit=12)
            )

            tcr_df = normalize_tcr_columns(load_tcr_table(tcr_path))
            if "barcode" not in tcr_df.columns:
                lines.append("Baseline warning: TCR table is missing a barcode column.")
                return "\n".join(lines)

            tcr_df = tcr_df.copy()
            tcr_df["barcode_exact"] = tcr_df["barcode"].astype(str).map(normalize_barcode)
            tcr_df["barcode_core"] = tcr_df["barcode"].astype(str).map(barcode_core)
            tcr_sample_col = infer_sample_column(tcr_df.columns)

            merge_key = "barcode_core"
            if sample_col and tcr_sample_col:
                obs["sample_merge_key"] = [
                    make_merge_key(barcode, sample, use_core=True)
                    for barcode, sample in zip(obs["barcode"], obs[sample_col])
                ]
                tcr_df["sample_merge_key"] = [
                    make_merge_key(barcode, sample, use_core=True)
                    for barcode, sample in zip(tcr_df["barcode"], tcr_df[tcr_sample_col])
                ]
                if pd.Series(obs["sample_merge_key"]).isin(set(tcr_df["sample_merge_key"])).sum() >= pd.Series(obs["barcode_core"]).isin(set(tcr_df["barcode_core"])).sum():
                    merge_key = "sample_merge_key"

            grouped = tcr_df.groupby(merge_key, dropna=False).agg(clonotype_id=("clonotype_id", "first"))
            obs = obs.join(grouped, on=merge_key)
            obs["has_tcr"] = obs["clonotype_id"].notna()
            clone_sizes = obs.loc[obs["has_tcr"], "clonotype_id"].astype(str).value_counts()
            obs["clone_size"] = obs["clonotype_id"].astype(str).map(clone_sizes).fillna(0).astype(int)
            obs["expanded_clone"] = obs["clone_size"] >= 3

            paired_t = obs.loc[t_mask & obs["has_tcr"].fillna(False).astype(bool)].copy()
            if paired_t.empty:
                lines.append("No paired T-cell subset with TCR information was found after baseline merge.")
                return "\n".join(lines)

            full_adata = ad.read_h5ad(rna_h5ad_path)
            try:
                if "cell_type" not in full_adata.obs.columns:
                    full_adata.obs["cell_type"] = obs[group_col].reindex(full_adata.obs_names).astype(str)
                for column in ("clonotype_id", "has_tcr", "clone_size", "expanded_clone"):
                    full_adata.obs[column] = obs[column].reindex(full_adata.obs_names).values
                assign_clone_type_labels(full_adata)
                paired_adata = paired_tcr_subset(full_adata)
                t_adata, t_marker_df, t_annotation_df = recluster_and_annotate_t_cells(
                    paired_adata,
                    group_col="cell_type",
                    model_name=self.hypothesis_model,
                    logger=self.logger,
                )
                t_group_col = "tcell_cluster_cell_type" if "tcell_cluster_cell_type" in t_adata.obs.columns else "tcell_leiden"

                lines.append(
                    "LLM-defined T-cell subclusters: "
                    + _top_counts(t_adata.obs[t_group_col].astype(str), limit=12)
                )
                if not t_annotation_df.empty:
                    annotation_lines = []
                    for row in t_annotation_df.head(8).itertuples():
                        markers = [item for item in str(row.supporting_markers).split("|") if item][:5]
                        marker_text = ",".join(markers) if markers else "no markers"
                        annotation_lines.append(f"{row.cluster_id}->{row.cell_type} ({row.confidence}; {marker_text})")
                    lines.append("LLM T-cell cluster calls: " + " | ".join(annotation_lines))

                subtype_summary = (
                    t_adata.obs.groupby(t_group_col, observed=False)
                    .agg(
                        paired_cells=("cloneType", "size"),
                        expanded_fraction=("expanded_clone", "mean"),
                        median_clone_size=("clone_size", "median"),
                    )
                    .sort_values(["expanded_fraction", "paired_cells"], ascending=[False, False])
                )
                top_focus = subtype_summary.head(5)
                focus_lines = [
                    f"{idx}: expanded_fraction={row.expanded_fraction:.3f}, paired_cells={int(row.paired_cells)}, median_clone_size={float(row.median_clone_size):.1f}"
                    for idx, row in top_focus.iterrows()
                ]
                lines.append("T-cell subclusters prioritized by clonal expansion: " + " | ".join(focus_lines))

                tissue_skew = pd.crosstab(t_adata.obs[t_group_col].astype(str), t_adata.obs[tissue_col].astype(str))
                skew_lines = []
                for subtype, row in tissue_skew.iterrows():
                    total = int(row.sum())
                    if total == 0:
                        continue
                    dominant_tissue = row.sort_values(ascending=False).index[0]
                    dominant_fraction = float(row.max() / total)
                    skew_lines.append((dominant_fraction, subtype, dominant_tissue, total))
                skew_lines.sort(reverse=True)
                if skew_lines:
                    lines.append(
                        "T-cell subclusters with strongest tissue skew: "
                        + " | ".join(
                            f"{subtype} -> {dominant_tissue} ({fraction:.2f}, n={total})"
                            for fraction, subtype, dominant_tissue, total in skew_lines[:5]
                        )
                    )

                clone_type_mix = pd.crosstab(t_adata.obs[t_group_col].astype(str), t_adata.obs["cloneType"].astype(str))
                if not clone_type_mix.empty:
                    hyper_col = "Hyperexpanded (100 < X <= 500)" if "Hyperexpanded (100 < X <= 500)" in clone_type_mix.columns else clone_type_mix.columns[-1]
                    ranked = clone_type_mix.div(clone_type_mix.sum(axis=1).replace(0, 1), axis=0).sort_values(hyper_col, ascending=False)
                    lines.append(
                        "T-cell subclusters enriched for large/hyperexpanded clones: "
                        + " | ".join(
                            f"{idx}: {ranked.loc[idx, hyper_col]:.2f}"
                            for idx in ranked.head(5).index
                        )
                    )
            finally:
                if getattr(full_adata, "file", None) is not None:
                    full_adata.file.close()

            lines.extend(
                [
                    "Use this baseline to choose the next innovative hypothesis.",
                    "Prioritize T-cell subsets that are newly emerged, relatively under-studied in this disease, tissue-skewed, or enriched for large/hyperexpanded clonotypes.",
                    "After the baseline, the next plan should test how those subsets differ across tissues, how their clone types distribute across tissues, which genes are specifically elevated in tumor/metastasis, and which candidate functional genes could explain the state.",
                ]
            )
            deterministic_summary = "\n".join(lines)
            return (
                deterministic_summary
                + "\n\nBaseline interpretation by the planning model:\n"
                + self._interpret_standard_baseline_summary(deterministic_summary)
            )
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()

    def _interpret_standard_baseline_summary(self, baseline_summary: str) -> str:
        prompt = self._read_prompt("baseline_interpretation.txt").format(
            baseline_summary=truncate_text(baseline_summary, 12000),
            context_summary=getattr(self, "context_summary", "") or "No research brief content was provided.",
            literature_summary=getattr(self, "literature_summary", "") or "No local literature was provided.",
            validation_summary=getattr(self, "validation_summary", "") or "No validation summary available.",
        )
        try:
            response = litellm.completion(
                model=self.hypothesis_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You interpret fixed baseline scRNA+scTCR findings and identify the most promising "
                            "T-cell subsets and mechanistic directions for deeper follow-up."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            content = (response.choices[0].message.content or "").strip()
            if content:
                self.logger.log_response(content, "standard_baseline_interpretation")
                return content
        except Exception as exc:
            self.logger.warning(f"Baseline interpretation failed; using deterministic-only summary: {exc}")
        return (
            "- No model interpretation was available.\n"
            "- Use the deterministic baseline observations directly when selecting the next hypothesis.\n"
            "- Prioritize T-cell subsets with tissue skew, clonal expansion, unusual cloneType composition, or tumor-enriched candidate genes."
        )

    def _extract_baseline_planning_context(self, baseline_summary: str) -> str:
        marker = "Baseline interpretation by the planning model:"
        text = (baseline_summary or "").strip()
        if marker in text:
            extracted = text.split(marker, 1)[1].strip()
            if extracted:
                return extracted
        return text or "No baseline planning context available."

    def _extract_baseline_tcell_labels(self, baseline_summary: str) -> list[str]:
        text = baseline_summary or ""
        match = re.search(r"LLM-defined T-cell subclusters:\s*(.+)", text)
        if not match:
            return []
        line = match.group(1)
        labels: list[str] = []
        for item in re.finditer(r"([^,]+?)\s*\(\d+\)", line):
            label = item.group(1).strip()
            if label and label not in labels:
                labels.append(label)
        return labels

    def _generate_deepresearch_background(self) -> str:
        if not self.openai_api_key:
            self.logger.warning("Deep Research requested but OPENAI_API_KEY is not available.")
            return ""
        prompt = self._read_prompt("deepresearch.txt").format(
            rna_summary=self.rna_summary,
            tcr_summary=self.tcr_summary,
            joint_summary=self.joint_summary,
            context_summary=self.context_summary or "No research brief content was provided.",
        )
        try:
            researcher = DeepResearcher(self.openai_api_key)
            background = researcher.research(prompt)
            self.logger.log_response(background, "deepresearch_background")
            return truncate_text(background, 20000)
        except Exception as exc:
            self.logger.warning(f"Deep Research background generation failed: {exc}")
            return ""

    def run(self, seeded_hypotheses: Iterable[str] | None = None) -> Path:
        seeded = [item.strip() for item in (seeded_hypotheses or []) if item and item.strip()]
        total_analyses = max(self.num_analyses, len(seeded))
        past_analyses = ""
        notebook_paths: list[Path] = []
        ledger_summaries: list[str] = []
        executed_hypotheses: list[str] = []
        figure_result: FigureResult | None = None
        figure_error: str | None = None

        self.logger.info(
            f"Starting scMAA run with {total_analyses} analyses. "
            f"RNA={self.rna_h5ad_path}, TCR={self.tcr_path}, research_brief={self.research_brief_path}"
        )

        for analysis_idx in range(total_analyses):
            research_ledger = self._make_research_ledger()
            seeded_hypothesis = seeded[analysis_idx] if analysis_idx < len(seeded) else None
            analysis = self.hypothesis_generator.generate_idea(
                past_analyses=past_analyses,
                research_state_summary=research_ledger.to_prompt_text(),
                analysis_idx=analysis_idx,
                seeded_hypothesis=seeded_hypothesis,
            )
            executed_hypotheses.append(analysis.hypothesis)
            past_analyses, research_ledger = self.executor.execute_idea(
                analysis=analysis,
                past_analyses=past_analyses,
                research_ledger=research_ledger,
                analysis_idx=analysis_idx,
                seeded=seeded_hypothesis is not None,
            )
            notebook_paths.append(self.output_dir / f"{self.analysis_name}_analysis_{analysis_idx + 1}.ipynb")
            ledger_summaries.append(research_ledger.to_prompt_text())

        self._write_executed_hypotheses_artifact(executed_hypotheses)
        if self.generate_publication_figure:
            try:
                figure_result = self._build_publication_figure()
            except Exception as exc:
                figure_error = str(exc)
                self.logger.warning(f"Publication figure generation failed: {exc}")

        return self._finalize_run_outputs(
            past_analyses=past_analyses,
            notebook_paths=notebook_paths,
            ledger_summaries=ledger_summaries,
            executed_hypotheses=executed_hypotheses,
            seeded=seeded,
            figure_result=figure_result,
            figure_error=figure_error,
        )

