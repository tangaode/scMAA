"""Main orchestration for scRT-agent."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Iterable

from .deepresearch import DeepResearcher
from .execution import LegacyNotebookExecutor
from .figure_mode import FigureResult, build_publication_figure
from .hypothesis import CandidateHypothesisMenu, HypothesisGenerator
from .literature import (
    LiteratureHypothesisMenu,
    LiteratureSummarizer,
    discover_literature_files,
    read_literature_file,
)
from .logger import AgentLogger
from .research import ResearchLedger
from .utils import load_tcr_table, normalize_tcr_columns, read_text, truncate_text
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


def _normalize_barcode(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text.split("-")[0]


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

    def prepare_candidate_hypotheses(self) -> CandidateHypothesisMenu:
        research_ledger = self._make_research_ledger()
        menu = self.hypothesis_generator.generate_candidate_hypotheses(
            research_state_summary=research_ledger.to_prompt_text(),
            past_analyses="",
        )
        return menu

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

    def build_plan_from_hypothesis(self, hypothesis: str):
        research_ledger = self._make_research_ledger()
        return self.hypothesis_generator.generate_analysis_from_hypothesis(
            hypothesis,
            past_analyses="",
            research_state_summary=research_ledger.to_prompt_text(),
            seed_context="User-approved hypothesis from interactive review.",
        )

    def _build_publication_figure(self) -> FigureResult:
        figure_dir = self.output_dir / "figure"
        figure_name = self.publication_figure_name or f"{self.analysis_name}_publication_figure"
        result = build_publication_figure(
            rna_h5ad_path=self.rna_h5ad_path,
            tcr_path=self.tcr_path,
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
            obs_names = [str(idx) for idx in adata.obs_names]
            obs_core = {_normalize_barcode(idx) for idx in obs_names}
            lower_to_original = {str(col).lower(): str(col) for col in adata.obs.columns}
            sample_column = None
            for hint in ("sample", "sample_id", "orig.ident", "donor", "patient"):
                if hint in lower_to_original:
                    sample_column = lower_to_original[hint]
                    break

            df = normalize_tcr_columns(load_tcr_table(tcr_path))
            exact_overlap = 0
            core_overlap = 0
            tcr_unique_barcodes = 0
            tcr_unique_core = 0
            tcr_clonotypes = 0
            sample_lines: list[str] = []

            if "barcode" in df.columns:
                tcr_barcodes = set(df["barcode"].dropna().astype(str))
                tcr_core = {_normalize_barcode(value) for value in tcr_barcodes}
                exact_overlap = sum(1 for barcode in obs_names if barcode in tcr_barcodes)
                core_overlap = sum(1 for barcode in obs_core if barcode and barcode in tcr_core)
                tcr_unique_barcodes = len(tcr_barcodes)
                tcr_unique_core = len(tcr_core)
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

            chosen_mode = "exact" if exact_overlap >= core_overlap else "barcode_core"
            lines = [
                f"Exact barcode overlap between RNA obs_names and TCR barcodes: {exact_overlap}",
                f"Core barcode overlap after stripping suffixes: {core_overlap}",
                f"Preferred merge mode: {chosen_mode}",
                f"RNA cells: {adata.n_obs}",
                f"TCR unique barcodes: {tcr_unique_barcodes}",
                f"TCR unique barcode cores: {tcr_unique_core}",
                f"TCR unique clonotypes: {tcr_clonotypes}",
                f"Approximate RNA coverage by TCR ({chosen_mode}): "
                f"{(max(exact_overlap, core_overlap) / max(adata.n_obs, 1)):.3f}",
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
        figure_result: FigureResult | None = None

        self.logger.info(
            f"Starting scRT-agent run with {total_analyses} analyses. "
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
            past_analyses, research_ledger = self.executor.execute_idea(
                analysis=analysis,
                past_analyses=past_analyses,
                research_ledger=research_ledger,
                analysis_idx=analysis_idx,
                seeded=seeded_hypothesis is not None,
            )
            notebook_paths.append(self.output_dir / f"{self.analysis_name}_analysis_{analysis_idx + 1}.ipynb")
            ledger_summaries.append(research_ledger.to_prompt_text())

        if self.generate_publication_figure:
            try:
                figure_result = self._build_publication_figure()
            except Exception as exc:
                self.logger.warning(f"Publication figure generation failed: {exc}")

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
        if figure_result is not None:
            summary_lines.extend(
                [
                    "",
                    "Publication figure",
                    f"PNG: {figure_result.png_path}",
                    f"PDF: {figure_result.pdf_path}",
                    f"Summary: {figure_result.summary_path}",
                ]
            )
        summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
        self.logger.info(f"Run complete. Summary written to {summary_path}")
        return summary_path
