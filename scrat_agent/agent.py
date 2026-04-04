"""Main orchestration for scRNA + scATAC analysis."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Iterable

from scrt_agent.agent import refresh_run_summary_from_artifacts, write_figure_status_file

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
from .utils import infer_sample_column, read_text, truncate_text
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
    ("mudata", "mudata"),
    ("muon", "muon"),
)

RNA_METADATA_HINTS = (
    "sample",
    "sample_id",
    "sample_key",
    "tissue",
    "cell_type",
    "annotation",
    "cluster",
    "leiden",
)

ATAC_METADATA_HINTS = (
    "sample",
    "sample_id",
    "sample_key",
    "tissue",
    "cell_type",
    "annotation",
    "cluster",
    "leiden",
)


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


class ScRATAgent:
    """Research-oriented agent for integrated scRNA + scATAC analysis."""

    def __init__(
        self,
        *,
        rna_h5ad_path: str,
        atac_h5ad_path: str,
        research_brief_path: str | None = None,
        context_path: str | None = None,
        literature_paths: Iterable[str] | None = None,
        analysis_name: str = "scrat_run",
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
        self.atac_h5ad_path = str(Path(atac_h5ad_path).resolve())
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
        self.prompt_dir = Path(prompt_dir) if prompt_dir else self.project_root / "scrat_agent" / "prompts"
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
        self.atac_summary = self._summarize_atac_data(self.atac_h5ad_path)
        self.joint_summary = self._summarize_joint_data(self.rna_h5ad_path, self.atac_h5ad_path)
        self.dataset_validation = DatasetValidator().inspect_inputs(self.rna_h5ad_path, self.atac_h5ad_path)
        self.validation_summary = self.dataset_validation.to_prompt_text()
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
            tcr_summary=self.atac_summary,
            joint_summary=self.joint_summary,
            validation_summary=self.validation_summary,
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
            atac_summary=self.atac_summary,
            joint_summary=self.joint_summary,
            validation_summary=self.validation_summary,
            context_summary=self.context_summary,
            logger=self.logger,
            rna_h5ad_path=self.rna_h5ad_path,
            atac_h5ad_path=self.atac_h5ad_path,
            output_dir=self.output_dir,
            analysis_name=self.analysis_name,
            max_iterations=self.max_iterations,
            max_fix_attempts=self.max_fix_attempts,
            use_VLM=self.use_VLM,
            use_documentation=self.use_documentation,
        )

    def prepare_candidate_hypotheses(self, user_feedback: str = "") -> CandidateHypothesisMenu:
        research_ledger = self._make_research_ledger()
        return self.hypothesis_generator.generate_candidate_hypotheses(
            research_state_summary=research_ledger.to_prompt_text(),
            past_analyses="",
            user_feedback=user_feedback,
        )

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

    def _write_executed_hypotheses_artifact(self, executed_hypotheses: list[str]) -> Path:
        executed_hypotheses_path = self.output_dir / "executed_hypotheses.txt"
        executed_lines = [f"Analysis {idx + 1}: {text}" for idx, text in enumerate(executed_hypotheses)]
        executed_hypotheses_path.write_text("\n".join(executed_lines) + ("\n" if executed_lines else ""), encoding="utf-8")
        return executed_hypotheses_path

    def _build_publication_figure(self) -> FigureResult:
        figure_dir = self.output_dir / "figure"
        figure_name = self.publication_figure_name or f"{self.analysis_name}_publication_figure"
        result = build_publication_figure(
            rna_h5ad_path=self.rna_h5ad_path,
            atac_h5ad_path=self.atac_h5ad_path,
            output_dir=figure_dir,
            figure_name=figure_name,
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
                    "Which RNA-ATAC findings can survive the listed guardrails and paired-modality limits?",
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
                tcr_summary=self.atac_summary,
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
        lines = [f"Overview: {menu.overview}", "", "Candidates:"]
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
            Path(self.atac_h5ad_path).resolve().parent,
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
            metadata_lines: list[str] = []
            lower_to_original = {col.lower(): col for col in obs_columns}
            for hint in RNA_METADATA_HINTS:
                column = lower_to_original.get(hint)
                if not column:
                    continue
                metadata_lines.append(
                    f"- {column}: {adata.obs[column].nunique(dropna=True)} unique values; top levels: {_top_counts(adata.obs[column])}"
                )
            lines = [
                f"RNA file: {rna_h5ad_path}",
                f"RNA matrix shape: {adata.n_obs} cells x {adata.n_vars} genes",
                f"obs columns ({len(obs_columns)}): {', '.join(obs_columns[:30]) or 'none'}",
                f"obsm keys: {', '.join(list(adata.obsm.keys())[:20]) or 'none'}",
                f"layers: {', '.join(list(adata.layers.keys())[:20]) or 'none'}",
                f"uns keys: {', '.join(list(adata.uns.keys())[:20]) or 'none'}",
                f"raw present: {adata.raw is not None}",
            ]
            if metadata_lines:
                lines.append("Candidate RNA metadata columns:")
                lines.extend(metadata_lines)
            return "\n".join(lines)
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()

    def _summarize_atac_data(self, atac_h5ad_path: str) -> str:
        import anndata as ad

        adata = ad.read_h5ad(atac_h5ad_path, backed="r")
        try:
            obs_columns = [str(col) for col in adata.obs.columns]
            metadata_lines: list[str] = []
            lower_to_original = {col.lower(): col for col in obs_columns}
            for hint in ATAC_METADATA_HINTS:
                column = lower_to_original.get(hint)
                if not column:
                    continue
                metadata_lines.append(
                    f"- {column}: {adata.obs[column].nunique(dropna=True)} unique values; top levels: {_top_counts(adata.obs[column])}"
                )
            feature_name_columns = [str(col) for col in adata.var.columns if str(col).lower() in {"gene_name", "gene", "symbol"}]
            lines = [
                f"ATAC file: {atac_h5ad_path}",
                f"ATAC matrix shape: {adata.n_obs} cells x {adata.n_vars} features",
                f"obs columns ({len(obs_columns)}): {', '.join(obs_columns[:30]) or 'none'}",
                f"var columns ({len(adata.var.columns)}): {', '.join(map(str, list(adata.var.columns)[:20])) or 'none'}",
                f"obsm keys: {', '.join(list(adata.obsm.keys())[:20]) or 'none'}",
                f"layers: {', '.join(list(adata.layers.keys())[:20]) or 'none'}",
                f"uns keys: {', '.join(list(adata.uns.keys())[:20]) or 'none'}",
                f"gene-like feature annotation columns: {', '.join(feature_name_columns) or 'none'}",
                f"raw present: {adata.raw is not None}",
            ]
            if metadata_lines:
                lines.append("Candidate ATAC metadata columns:")
                lines.extend(metadata_lines)
            return "\n".join(lines)
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()

    def _summarize_joint_data(self, rna_h5ad_path: str, atac_h5ad_path: str) -> str:
        import anndata as ad

        rna = ad.read_h5ad(rna_h5ad_path, backed="r")
        atac = ad.read_h5ad(atac_h5ad_path, backed="r")
        try:
            shared_obs = sorted(set(map(str, rna.obs_names)).intersection(map(str, atac.obs_names)))
            shared_gene_names = sorted(set(map(str, rna.var_names)).intersection(map(str, atac.var_names)))
            sample_lines: list[str] = []
            rna_sample_col = infer_sample_column(rna.obs.columns)
            atac_sample_col = infer_sample_column(atac.obs.columns)
            if rna_sample_col:
                sample_lines.append(
                    f"RNA sample column selected for stratified analyses: {rna_sample_col} ({rna.obs[rna_sample_col].nunique(dropna=True)} groups)."
                )
            if atac_sample_col:
                sample_lines.append(
                    f"ATAC sample column selected for stratified analyses: {atac_sample_col} ({atac.obs[atac_sample_col].nunique(dropna=True)} groups)."
                )
            if "tissue" in rna.obs.columns:
                sample_lines.append(f"RNA tissue groups: {rna.obs['tissue'].nunique(dropna=True)}; top levels: {_top_counts(rna.obs['tissue'])}")
            if "tissue" in atac.obs.columns:
                sample_lines.append(f"ATAC tissue groups: {atac.obs['tissue'].nunique(dropna=True)}; top levels: {_top_counts(atac.obs['tissue'])}")
            lines = [
                f"Shared obs names between RNA and ATAC objects: {len(shared_obs)}",
                f"Shared feature names by var_names: {len(shared_gene_names)}",
                f"RNA cells: {rna.n_obs}",
                f"ATAC cells: {atac.n_obs}",
                f"Approximate paired-cell fraction relative to RNA cells: {(len(shared_obs) / max(rna.n_obs, 1)):.3f}",
                f"Approximate paired-cell fraction relative to ATAC cells: {(len(shared_obs) / max(atac.n_obs, 1)):.3f}",
            ]
            lines.extend(sample_lines)
            lines.append(
                "Notebook objects that will exist after setup: adata_rna, adata_atac, shared_obs_names, shared_gene_features, "
                "and helper functions for paired subsets, RNA expression, ATAC gene activity or accessibility signals, and RNA-ATAC link summaries."
            )
            return "\n".join(lines)
        finally:
            if getattr(rna, "file", None) is not None:
                rna.file.close()
            if getattr(atac, "file", None) is not None:
                atac.file.close()

    def _generate_deepresearch_background(self) -> str:
        if not self.openai_api_key:
            self.logger.warning("Deep Research requested but OPENAI_API_KEY is not available.")
            return ""
        prompt = self._read_prompt("deepresearch.txt").format(
            rna_summary=self.rna_summary,
            tcr_summary=self.atac_summary,
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
            f"ATAC input: {self.atac_h5ad_path}",
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
                "ATAC summary",
                self.atac_summary,
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

    def run(self, seeded_hypotheses: Iterable[str] | None = None) -> Path:
        seeded = [item.strip() for item in (seeded_hypotheses or []) if item and item.strip()]
        total_analyses = max(self.num_analyses, len(seeded))
        past_analyses = ""
        notebook_paths: list[Path] = []
        ledger_summaries: list[str] = []
        executed_hypotheses: list[str] = []
        figure_result: FigureResult | None = None
        figure_error: str | None = None
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
