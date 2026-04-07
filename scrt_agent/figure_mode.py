"""Publication-style figure builder for scRNA + scTCR analysis."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import traceback

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import scanpy as sc
import seaborn as sns

try:
    import instructor
    import litellm

    litellm.drop_params = True
except Exception:  # pragma: no cover
    instructor = None
    litellm = None

from scrt_agent.figure_common import (
    ensure_display_column,
    extract_hypothesis_genes,
    extract_result_genes,
    panel_label,
    plot_categorical_embedding,
    read_run_result_context,
    save_figure_bundle,
)

from .notebook_tools import (
    assign_clone_type_labels,
    clone_expansion_table,
    clone_type_distribution_table,
    expression_frame,
    paired_tcr_subset,
    proportion_table,
    recluster_and_annotate_t_cells,
    resolve_gene_names,
    tissue_stratified_expansion_de,
    tumor_like_subset,
)
from .utils import load_tcr_table, normalize_tcr_columns


@dataclass
class FigureResult:
    png_path: Path
    pdf_path: Path
    summary_path: Path


class HypothesisFigureResponse(BaseModel):
    analysis_focus: str = Field(description="Main biological focus chosen from the baseline results.")
    rationale: str = Field(description="Why this focus should be visualized next, grounded in baseline results and user feedback.")
    code: str = Field(description="Executable Python code that creates a publication-quality hypothesis figure and assigns it to variable `fig`.")


T_CELL_HINTS = ("t cell", "t cells", "cd8", "cd4", "treg", "regulatory t", "cytotoxic t")
CD8_T_CELL_HINTS = ("cd8", "cytotoxic")
PSEUDOTIME_HINTS = ("pseudotime", "trajectory", "dpt", "pseudotemporal", "trajectory analysis")
EXHAUSTION_MARKER_CANDIDATES = ["PDCD1", "LAG3", "HAVCR2", "TIGIT", "CTLA4", "TOX", "CXCL13"]
BAD_FOCUS_LABEL_HINTS = ("doublet", "contaminant", "ambiguous", "unknown", "artifact", "low quality")
DEFAULT_FOCUS_GENES = [
    "TCF7",
    "IL7R",
    "LTB",
    "PDCD1",
    "TIGIT",
    "HAVCR2",
    "LAG3",
    "CXCL13",
    "CCL5",
    "NKG7",
    "GZMB",
    "XBP1",
]
PLACEHOLDER_PATTERNS = (
    "no umap",
    "no data",
    "not available",
    "unavailable",
    "placeholder",
    "no marker data",
    "no contrast data",
    "no diversity data",
    "no paired clonotype data",
    "no clone",
    "empty panel",
)
INVALID_CODE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"=\s*\.\.\.", "Do not assign placeholder ellipsis values such as `t_adata = ...`."),
    (r"if\s+__name__\s*==\s*[\"']__main__[\"']\s*:", "Do not use a `__main__` guard; the code is executed directly."),
    (r"def\s+main\s*\(", "Do not wrap the plotting workflow inside `main()`; execute it at top level."),
    (r"Load your AnnData object here", "Do not include placeholder comments about loading data."),
)


def _read_figure_prompt(prompt_dir: str | Path | None, name: str) -> str:
    base = Path(prompt_dir) if prompt_dir else (Path(__file__).resolve().parent / "prompts")
    return (base / name).read_text(encoding="utf-8")


def _load_figure_env_files(*paths: str | Path) -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    project_root = Path(__file__).resolve().parents[1]
    candidate_dirs = [Path.cwd(), project_root, project_root.parent]
    for item in paths:
        if item:
            candidate_dirs.append(Path(item).resolve().parent)
    seen: set[Path] = set()
    for directory in candidate_dirs:
        if directory in seen:
            continue
        seen.add(directory)
        for name in (".env", "OPENAI.env", "deepseek.env"):
            env_path = directory / name
            if env_path.exists():
                load_dotenv(env_path, override=False)


def _safe_exec_code(code: str, namespace: dict) -> tuple[plt.Figure | None, str | None]:
    try:
        exec(code, namespace, namespace)
    except Exception:
        return None, traceback.format_exc()
    fig = namespace.get("fig")
    if fig is None:
        try:
            fig = plt.gcf()
        except Exception:
            fig = None
    if fig is None:
        return None, "Generated code did not create a matplotlib figure named `fig`."
    return fig, None


def _strip_code_fences(text: str) -> str:
    cleaned = (text or "").strip()
    lines = []
    for line in cleaned.splitlines():
        if line.strip().startswith("```"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _dynamic_hypothesis_context(
    *,
    t_adata: ad.AnnData,
    t_group_col: str,
    tissue_col: str,
) -> str:
    if t_adata.n_obs == 0 or t_group_col not in t_adata.obs.columns:
        return "No focused T-cell reclustering context is available."
    lines: list[str] = []
    counts = t_adata.obs[t_group_col].astype(str).value_counts()
    lines.append("LLM-defined T-cell subclusters: " + ", ".join(f"{idx} ({int(val)})" for idx, val in counts.head(12).items()))
    tissue_mix = pd.crosstab(t_adata.obs[t_group_col].astype(str), t_adata.obs[tissue_col].astype(str))
    if not tissue_mix.empty:
        summary_bits = []
        for subtype, row in tissue_mix.iterrows():
            top_tissue = row.sort_values(ascending=False).index[0]
            top_fraction = float(row.max() / max(row.sum(), 1))
            summary_bits.append(f"{subtype}->{top_tissue} ({top_fraction:.2f})")
        lines.append("Tissue skew by T-cell subcluster: " + " | ".join(summary_bits[:8]))
    if "cloneType" in t_adata.obs.columns:
        clone_mix = pd.crosstab(t_adata.obs[t_group_col].astype(str), t_adata.obs["cloneType"].astype(str))
        if not clone_mix.empty:
            ranked = clone_mix.div(clone_mix.sum(axis=1).replace(0, 1), axis=0)
            hyper_col = "Hyperexpanded (100 < X <= 500)" if "Hyperexpanded (100 < X <= 500)" in ranked.columns else ranked.columns[-1]
            top_rows = ranked.sort_values(hyper_col, ascending=False).head(8)
            lines.append(
                "Large/hyperexpanded enrichment by T-cell subcluster: "
                + " | ".join(f"{idx}:{top_rows.loc[idx, hyper_col]:.2f}" for idx in top_rows.index)
            )
    return "\n".join(lines)


def _normalize_basis_alias(basis: str | None) -> str:
    if not basis:
        return "X_umap"
    normalized = str(basis).strip().lower()
    if normalized in {"umap", "x_umap"}:
        return "X_umap"
    if normalized in {"pca", "x_pca"}:
        return "X_pca"
    return str(basis)


def _is_probable_colorbar_axis(ax) -> bool:
    bbox = ax.get_position()
    return bbox.width < 0.05 or bbox.height < 0.05


def _axis_free_text(ax) -> list[str]:
    texts: list[str] = []
    for text in ax.texts:
        value = str(text.get_text() or "").strip()
        if not value:
            continue
        if re.fullmatch(r"[A-Fa-f]", value):
            continue
        texts.append(value)
    return texts


def _validate_generated_figure(fig: plt.Figure | None) -> str | None:
    if fig is None:
        return "The generated code did not produce a matplotlib figure."
    main_axes = [ax for ax in fig.axes if not _is_probable_colorbar_axis(ax)]
    if not main_axes:
        return "The generated figure does not contain any main plotting axes."
    issues: list[str] = []
    contentful_axes = 0
    for idx, ax in enumerate(main_axes, start=1):
        title = str(ax.get_title() or "").strip()
        free_text = _axis_free_text(ax)
        candidate_text = [title, *free_text]
        placeholder = next(
            (
                item
                for item in candidate_text
                if any(pattern in item.lower() for pattern in PLACEHOLDER_PATTERNS)
            ),
            None,
        )
        if placeholder:
            issues.append(f"Axis {idx} contains placeholder text: {placeholder!r}.")
            continue
        if not ax.has_data():
            issues.append(
                f"Axis {idx} has no plotted data. Remove the axis entirely or replace it with a real plot."
            )
            continue
        contentful_axes += 1
    if contentful_axes < 3:
        issues.append(
            f"The figure has only {contentful_axes} contentful axes. Create at least 3 biologically informative panels."
        )
    if issues:
        return "\n".join(issues)
    return None


def _validate_generated_code_text(code: str) -> str | None:
    for pattern, message in INVALID_CODE_PATTERNS:
        if re.search(pattern, code or "", flags=re.IGNORECASE):
            return message
    return None


def _generate_model_hypothesis_figure(
    *,
    adata_rna: ad.AnnData,
    paired_adata: ad.AnnData,
    t_adata: ad.AnnData,
    t_group_col: str,
    t_marker_df: pd.DataFrame,
    t_annotation_df: pd.DataFrame,
    tcr_df: pd.DataFrame,
    tissue_col: str,
    output_dir: str | Path,
    figure_name: str,
    result_context: dict[str, str],
    hypothesis_model: str = "gpt-4o",
    baseline_summary_text: str = "",
    prompt_dir: str | Path | None = None,
) -> tuple[Path, Path, Path, Path, str, str]:
    if instructor is None or litellm is None:
        raise RuntimeError("Dynamic hypothesis figure generation requires instructor and litellm.")

    output_dir = Path(output_dir)
    hyp_png = output_dir / f"{figure_name}_hypothesis.png"
    hyp_pdf = output_dir / f"{figure_name}_hypothesis.pdf"
    code_path = output_dir / f"{figure_name}_hypothesis_generated_code.py"
    plan_path = output_dir / f"{figure_name}_hypothesis_generated_plan.txt"
    prompt_template = _read_figure_prompt(prompt_dir, "hypothesis_figure_code.txt")

    marker_preview = ""
    if not t_annotation_df.empty:
        marker_preview = t_annotation_df.head(12).to_csv(index=False)
    marker_summary = ""
    if not t_marker_df.empty:
        top_marker_lines = []
        for cluster in sorted(t_marker_df["cluster"].astype(str).unique(), key=lambda x: (len(x), x)):
            subset = t_marker_df.loc[
                (t_marker_df["cluster"].astype(str) == cluster)
                & (~t_marker_df["is_linc_like"].astype(bool))
            ].head(12)
            genes = subset["names"].astype(str).tolist()
            top_marker_lines.append(f"Cluster {cluster}: {', '.join(genes)}")
        marker_summary = "\n".join(top_marker_lines[:12])

    prompt = prompt_template.format(
        executed_hypothesis=result_context.get("executed_hypothesis", "") or "none",
        approved_priority_question=result_context.get("approved_priority_question", "") or "none",
        approved_plan_steps=result_context.get("approved_plan_steps", "") or "none",
        approved_strategy_feedback=result_context.get("approved_strategy_feedback", "") or "none",
        user_feedback=result_context.get("user_feedback", "") or "none",
        final_interpretation=result_context.get("final_interpretation", "") or "none",
        baseline_summary=(baseline_summary_text or result_context.get("standard_baseline_summary", "") or "none"),
        tcell_context=_dynamic_hypothesis_context(t_adata=t_adata, t_group_col=t_group_col, tissue_col=tissue_col),
        tcell_annotation_table=marker_preview or "none",
        tcell_marker_summary=marker_summary or "none",
        t_group_col=t_group_col,
        tissue_col=tissue_col,
        obs_columns=", ".join(map(str, adata_rna.obs.columns)),
        tcell_obs_columns=", ".join(map(str, t_adata.obs.columns)),
        tcell_obsm_keys=", ".join(map(str, t_adata.obsm.keys())) or "none",
        paired_obs_columns=", ".join(map(str, paired_adata.obs.columns)),
        tissue_levels=", ".join(sorted(adata_rna.obs[tissue_col].astype(str).dropna().unique().tolist())) if tissue_col in adata_rna.obs.columns else "unknown",
        figure_png_path=str(hyp_png),
        figure_pdf_path=str(hyp_pdf),
    )
    client = instructor.from_litellm(litellm.completion)
    response = client.chat.completions.create(
        model=hypothesis_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a computational immunology figure author. "
                    "Given baseline scRNA+scTCR results, approved analysis plan, and user feedback, "
                    "decide the next hypothesis-driven analysis and write executable Python figure code."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_model=HypothesisFigureResponse,
    )

    initial_code = _strip_code_fences(response.code or "")
    code_path.write_text(initial_code + "\n", encoding="utf-8")
    plan_path.write_text(
        "\n".join(
            [
                f"analysis_focus: {response.analysis_focus}",
                "",
                "rationale:",
                response.rationale,
                "",
                "model: " + hypothesis_model,
            ]
        ),
        encoding="utf-8",
    )

    def _compat_plot_categorical_embedding(*args, **kwargs):
        adata = kwargs.pop("adata", None)
        ax = kwargs.pop("ax", None)
        obs_col = kwargs.pop("obs_col", None)
        color = kwargs.pop("color", None) or kwargs.pop("category", None)
        basis = _normalize_basis_alias(kwargs.pop("basis", None))
        subset_categories = kwargs.pop("subset_categories", None)
        subset_category = kwargs.pop("subset_category", None)
        title = kwargs.pop("title", "")
        remaining = list(args)
        if adata is None and remaining and hasattr(remaining[0], "obs") and hasattr(remaining[0], "obsm"):
            adata = remaining.pop(0)
        if ax is None and remaining and not (hasattr(remaining[0], "obs") and hasattr(remaining[0], "obsm")):
            ax = remaining.pop(0)
        if adata is None and remaining and hasattr(remaining[0], "obs") and hasattr(remaining[0], "obsm"):
            adata = remaining.pop(0)
        if ax is None and remaining:
            ax = remaining.pop(0)
        if subset_category is not None and subset_categories is None:
            subset_categories = [subset_category]
        if adata is None:
            raise ValueError("adata is required for plot_categorical_embedding.")
        chosen_color = color or obs_col or t_group_col
        working = adata
        if subset_categories and chosen_color in adata.obs.columns:
            categories = {str(item) for item in subset_categories}
            mask = adata.obs[chosen_color].astype(str).isin(categories)
            if mask.any():
                working = adata[mask].copy()
        target_ax = ax if ax is not None else plt.gca()
        return plot_categorical_embedding(target_ax, working, color=chosen_color, title=title, basis=basis, **kwargs)

    def _compat_clone_type_distribution_table(*args, **kwargs):
        adata = kwargs.pop("adata", None)
        group_col = kwargs.pop("group_col", None)
        clone_type_col = kwargs.pop("clone_type_col", "cloneType")
        groupby_col = kwargs.pop("groupby_col", None)
        subset_col = kwargs.pop("subset_col", None)
        subset_value = kwargs.pop("subset_value", None)
        if adata is None and args:
            adata = args[0]
        if adata is None:
            raise ValueError("adata is required for clone_type_distribution_table.")
        working = adata
        if subset_col and subset_col in working.obs.columns and subset_value is not None:
            working = working[working.obs[subset_col].astype(str) == str(subset_value)].copy()
        chosen_group = group_col or groupby_col or t_group_col
        columns = [clone_type_col]
        for extra in (tissue_col, "clone_size", chosen_group, t_group_col, "cell_type"):
            if extra in working.obs.columns and extra not in columns:
                columns.append(extra)
        return working.obs[columns].copy()

    def _compat_expression_frame(*args, **kwargs):
        adata = kwargs.pop("adata", None)
        genes = kwargs.pop("genes", None)
        obs_columns = kwargs.pop("obs_columns", None)
        group_col = kwargs.pop("group_col", None) or kwargs.pop("color", None)
        if adata is None and args:
            adata = args[0]
            args = args[1:]
        if genes is None and args:
            genes = args[0]
        if adata is None or genes is None:
            raise ValueError("adata and genes are required for expression_frame.")
        chosen_obs = list(obs_columns or [])
        if group_col and group_col not in chosen_obs:
            chosen_obs.append(group_col)
        return expression_frame(adata, genes, obs_columns=chosen_obs or None)

    def _compat_proportion_table(*args, **kwargs):
        adata = kwargs.pop("adata", None)
        group_col = kwargs.pop("group_col", None)
        tissue_name = kwargs.pop("tissue_col", None)
        normalize = kwargs.pop("normalize", "index")
        if adata is None and args:
            adata = args[0]
            args = args[1:]
        if group_col is None and args:
            group_col = args[0]
            args = args[1:]
        if tissue_name is None and args:
            tissue_name = args[0]
        if adata is None:
            raise ValueError("adata is required for proportion_table.")
        chosen_group = group_col or t_group_col
        chosen_tissue = tissue_name or tissue_col
        requested_long = kwargs.get("long_form", False)
        if isinstance(normalize, bool):
            requested_long = True
            normalize = "index" if normalize else "none"
        table = proportion_table(adata, group_col=chosen_group, tissue_col=chosen_tissue, normalize=normalize if normalize in {"index", "columns"} else "index")
        if chosen_group == chosen_tissue:
            long_df = (
                adata.obs[[chosen_tissue]]
                .dropna()
                .assign(proportion=1.0)
                .reset_index(drop=True)
            )
            return long_df
        if requested_long:
            long_df = table.reset_index().melt(id_vars=table.index.name or chosen_tissue, var_name=chosen_group, value_name="proportion")
            if table.index.name != chosen_tissue:
                long_df = long_df.rename(columns={table.index.name or "index": chosen_tissue})
            return long_df
        return table

    def _compat_tissue_stratified_expansion_de(*args, **kwargs):
        try:
            result = tissue_stratified_expansion_de(*args, **kwargs)
        except Exception:
            return pd.DataFrame(), pd.DataFrame()
        if result is None or (hasattr(result, "empty") and result.empty):
            return pd.DataFrame(), pd.DataFrame()
        if not isinstance(result, pd.DataFrame):
            return pd.DataFrame(), pd.DataFrame()
        top_genes = result["names"].astype(str).head(12).tolist() if "names" in result.columns else []
        source_adata = args[0] if args else kwargs.get("adata")
        expr = pd.DataFrame()
        if source_adata is not None and top_genes:
            try:
                obs_cols = [tissue_col] if tissue_col in source_adata.obs.columns else None
                expr = expression_frame(source_adata, top_genes, obs_columns=obs_cols)
                if obs_cols:
                    expr = expr.groupby(tissue_col, observed=False).mean(numeric_only=True).T
            except Exception:
                expr = pd.DataFrame()
        return expr, result

    def _repair_generated_code(bad_code: str, error_text: str) -> str:
        repair_prompt = (
            "The previous hypothesis-figure code failed. Rewrite it into a simpler, more robust version and return only executable Python.\n\n"
            f"Execution error:\n{error_text}\n\n"
            f"Current subtype column: {t_group_col}\n"
            f"Current tissue column: {tissue_col}\n"
            f"Available T-cell obs columns: {', '.join(map(str, t_adata.obs.columns))}\n"
            f"Available T-cell embedding keys: {', '.join(map(str, t_adata.obsm.keys())) or 'none'}\n\n"
            "Use the existing namespace only. Important helper signatures:\n"
            "- plot_categorical_embedding(ax, adata, *, color, title, basis='X_umap', label_points=True, show_legend=True, legend_title=None)\n"
            "- proportion_table(...) in this environment returns either a long-form or simple table suitable for plotting; prefer inspecting columns before plotting.\n"
            "- clone_type_distribution_table(...) in this environment returns a row-level DataFrame with columns like cloneType, tissue, clone_size, and cell_type when available.\n"
            "- expression_frame(adata, genes, obs_columns=None) requires a concrete list of genes.\n"
            "- tissue_stratified_expansion_de(...) in this environment may return `(expr_matrix, de_table)`; handle both objects defensively.\n"
            "- panel_label(ax, label)\n\n"
            "Rules:\n"
            "- If a prior approach is brittle, rewrite from scratch instead of patching line-by-line.\n"
            "- Do not assume the first panel must be UMAP. Let the biological story determine the plot types.\n"
            "- You may use any sensible multi-panel layout, but every panel you create must contain a real biological plot.\n"
            "- If you only have 3 or 4 strong panels, create 3 or 4 panels. Do not leave unused subplot slots blank.\n"
            "- If one planned analysis is unsupported, drop that panel and rebuild the figure with fewer panels instead of keeping an empty slot.\n"
            "- Execute the plotting code at top level. Do not define `main()` and do not use `if __name__ == '__main__'`.\n"
            "- Do not assign placeholder values like `t_adata = ...`.\n"
            "- Always check that required columns exist before plotting.\n"
            "- Always inspect available subtype labels from `t_adata.obs[t_group_col]` before subsetting; never invent subtype names.\n"
            "- Always inspect available embeddings from `t_adata.obsm.keys()`; use `X_umap` or `X_pca`, not guessed aliases.\n"
            "- Never assume a helper returns a matrix shaped exactly as needed; transform it explicitly.\n"
            "- Placeholder text such as 'No UMAP' or empty axes is a failure, not a success.\n"
            "- Always create variable `fig` and save it to `HYP_FIG_PNG` and `HYP_FIG_PDF`.\n\n"
            f"Broken code:\n{bad_code}"
        )
        response = litellm.completion(
            model=hypothesis_model,
            messages=[
                {"role": "system", "content": "You fix Python figure-generation code. Return only executable Python."},
                {"role": "user", "content": repair_prompt},
            ],
        )
        return _strip_code_fences(response.choices[0].message.content or "")

    base_namespace = {
        "__builtins__": __builtins__,
        "adata_rna": adata_rna,
        "paired_adata": paired_adata,
        "t_adata": t_adata,
        "t_group_col": t_group_col,
        "t_marker_df": t_marker_df,
        "t_annotation_df": t_annotation_df,
        "tcr_df": tcr_df,
        "result_context": result_context,
        "standard_baseline_summary": baseline_summary_text or result_context.get("standard_baseline_summary", ""),
        "HYP_FIG_PNG": str(hyp_png),
        "HYP_FIG_PDF": str(hyp_pdf),
        "Path": Path,
        "np": np,
        "pd": pd,
        "sc": sc,
        "sns": sns,
        "plt": plt,
        "GridSpec": GridSpec,
        "assign_clone_type_labels": assign_clone_type_labels,
        "clone_type_distribution_table": _compat_clone_type_distribution_table,
        "expression_frame": _compat_expression_frame,
        "paired_tcr_subset": paired_tcr_subset,
        "proportion_table": _compat_proportion_table,
        "resolve_gene_names": resolve_gene_names,
        "tissue_stratified_expansion_de": _compat_tissue_stratified_expansion_de,
        "tumor_like_subset": tumor_like_subset,
        "panel_label": panel_label,
        "plot_categorical_embedding": _compat_plot_categorical_embedding,
    }
    current_code = initial_code
    fig = None
    exec_error = None
    max_attempts = 8
    for attempt in range(1, max_attempts + 1):
        plt.close("all")
        namespace = dict(base_namespace)
        code_error = _validate_generated_code_text(current_code)
        if code_error:
            exec_error = f"Code validation failed:\n{code_error}"
            fig = None
        else:
            fig, exec_error = _safe_exec_code(current_code, namespace)
        if not exec_error:
            validation_error = _validate_generated_figure(fig)
            if not validation_error:
                break
            exec_error = f"Figure validation failed:\n{validation_error}"
            if fig is not None:
                plt.close(fig)
        if attempt >= max_attempts:
            raise RuntimeError(exec_error)
        repaired_code = _repair_generated_code(current_code, exec_error)
        current_code = repaired_code or current_code
        code_path.write_text(current_code + "\n", encoding="utf-8")
    if exec_error:
        raise RuntimeError(exec_error)
    if not hyp_png.exists() or not hyp_pdf.exists():
        fig.savefig(hyp_png, dpi=300, bbox_inches="tight")
        fig.savefig(hyp_pdf, bbox_inches="tight")
    hyp_summary = output_dir / f"{figure_name}_hypothesis_summary.txt"
    hyp_summary.write_text(
        "\n".join(
            [
                f"Executed hypothesis: {result_context.get('executed_hypothesis', '') or 'none'}",
                f"Approved priority question: {result_context.get('approved_priority_question', '') or 'none'}",
                f"Approved strategy feedback: {result_context.get('approved_strategy_feedback', '') or 'none'}",
                f"User feedback: {result_context.get('user_feedback', '') or 'none'}",
                f"Analysis focus chosen by model: {response.analysis_focus}",
                "Rationale:",
                response.rationale,
                f"Generated code: {code_path}",
                "This hypothesis figure was generated by model-authored analysis code, grounded in baseline results, approved plan, and user feedback.",
            ]
        ),
        encoding="utf-8",
    )
    return hyp_png, hyp_pdf, hyp_summary, code_path, response.analysis_focus, response.rationale


def _detect_plan_focus(result_context: dict[str, str], tokens: tuple[str, ...]) -> bool:
    joined = "\n".join(
        [
            result_context.get("approved_priority_question", ""),
            result_context.get("approved_plan_steps", ""),
        ]
    ).lower()
    return any(token in joined for token in tokens)


def _tcr_diversity_by_sample_tissue(adata: ad.AnnData, tissue_col: str) -> pd.DataFrame:
    if adata.n_obs == 0 or "clonotype_id" not in adata.obs.columns:
        return pd.DataFrame()
    sample_col = (
        "sample_id"
        if "sample_id" in adata.obs.columns
        else ("sample_key" if "sample_key" in adata.obs.columns else None)
    )
    if sample_col is None:
        return pd.DataFrame()
    frame = adata.obs[[sample_col, tissue_col, "clonotype_id"]].dropna().copy()
    if frame.empty:
        return pd.DataFrame()
    frame["clonotype_id"] = frame["clonotype_id"].astype(str)
    frame = frame.loc[frame["clonotype_id"] != "nan"]
    if frame.empty:
        return pd.DataFrame()
    records: list[dict[str, object]] = []
    for (sample_value, tissue_value), sample_frame in frame.groupby([sample_col, tissue_col], observed=False):
        clone_counts = sample_frame["clonotype_id"].value_counts()
        if clone_counts.sum() <= 0:
            continue
        probs = clone_counts / clone_counts.sum()
        shannon = float(-(probs * np.log(probs + 1e-12)).sum())
        records.append(
            {
                "sample": str(sample_value),
                tissue_col: str(tissue_value),
                "clonotype_diversity": shannon,
                "paired_cells": int(len(sample_frame)),
            }
        )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records)


def _plot_tcr_diversity_by_tissue(ax, data: pd.DataFrame, tissue_col: str) -> None:
    if data.empty:
        ax.text(0.5, 0.5, "No diversity data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("TCR diversity by tissue")
        ax.axis("off")
        return
    sns.boxplot(data=data, x=tissue_col, y="clonotype_diversity", ax=ax, color="#9ecae1", fliersize=1.5)
    sns.stripplot(
        data=data,
        x=tissue_col,
        y="clonotype_diversity",
        ax=ax,
        color="#08519c",
        alpha=0.45,
        size=4,
    )
    ax.set_title("TCR diversity by tissue", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("Shannon diversity")
    ax.tick_params(axis="x", rotation=35)


def _plot_group_fraction(ax, adata: ad.AnnData, tissue_col: str, value_col: str, title: str) -> None:
    if adata.n_obs == 0 or value_col not in adata.obs.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.axis("off")
        return
    frame = adata.obs[[tissue_col, value_col]].copy()
    frame[value_col] = frame[value_col].fillna(False).astype(bool)
    summary = (
        frame.groupby(tissue_col, observed=False)[value_col]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    if summary.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.axis("off")
        return
    sns.barplot(data=summary, x=tissue_col, y=value_col, ax=ax, color="#6baed6")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("Fraction")
    ax.tick_params(axis="x", rotation=35)


def _approved_plan_step_lines(result_context: dict[str, str]) -> list[str]:
    return [line.strip() for line in result_context.get("approved_plan_steps", "").splitlines() if line.strip()]


def _plan_requests_focus_subset(result_context: dict[str, str]) -> bool:
    return _detect_plan_focus(
        result_context,
        ("cd8", "t cell", "t-cell", "cytotoxic", "subset", "identify", "annotation"),
    )


def _plan_requests_diversity_panel(result_context: dict[str, str]) -> bool:
    return _detect_plan_focus(
        result_context,
        ("diversity", "shannon", "clonotype diversity", "tcr diversity"),
    )


def _plan_requests_de_panel(result_context: dict[str, str]) -> bool:
    return _detect_plan_focus(
        result_context,
        ("differential expression", "exhaust", "pdcd1", "pd-1", "lag3", "havcr2", "tim-3", "marker"),
    )


def _plan_requests_expression_panel(result_context: dict[str, str]) -> bool:
    return _detect_plan_focus(
        result_context,
        (
            "differential expression",
            "heatmap",
            "candidate gene",
            "driver gene",
            "functional gene",
            "gene expression",
            "deg",
            "marker",
        ),
    )


def _plan_requests_clonotype_panel(result_context: dict[str, str]) -> bool:
    return _detect_plan_focus(
        result_context,
        (
            "clonotype",
            "clone",
            "tcr",
            "repertoire",
            "v gene",
            "j gene",
            "sharing",
        ),
    )


def _context_blob(result_context: dict[str, str]) -> str:
    return "\n".join(
        [
            result_context.get("executed_hypothesis", ""),
            result_context.get("approved_priority_question", ""),
            result_context.get("approved_plan_steps", ""),
            result_context.get("approved_strategy_feedback", ""),
            result_context.get("user_feedback", ""),
            result_context.get("final_interpretation", ""),
            result_context.get("notebook_text", ""),
        ]
    )


def _normalized_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _select_focus_label(result_context: dict[str, str], t_subset: ad.AnnData, t_group_col: str) -> str | None:
    if t_subset.n_obs == 0 or t_group_col not in t_subset.obs.columns:
        return None
    labels = [str(value) for value in t_subset.obs[t_group_col].dropna().astype(str).unique().tolist()]
    if not labels:
        return None
    context = _context_blob(result_context)
    context_tokens = set(_normalized_tokens(context))
    best_label = None
    best_score = -1.0
    for label in labels:
        label_tokens = [token for token in _normalized_tokens(label) if token not in {"t", "cell", "cells"}]
        score = 0.0
        normalized_label = " ".join(_normalized_tokens(label))
        if normalized_label and normalized_label in " ".join(_normalized_tokens(context)):
            score += 5.0
        score += sum(1.0 for token in label_tokens if token in context_tokens)
        if any(hint in label.lower() for hint in BAD_FOCUS_LABEL_HINTS):
            score -= 10.0
        if "expanded_clone" in t_subset.obs.columns:
            try:
                expanded_fraction = (
                    t_subset.obs.assign(_group=t_subset.obs[t_group_col].astype(str))
                    .groupby("_group", observed=False)["expanded_clone"]
                    .mean()
                    .get(label, 0.0)
                )
                score += float(expanded_fraction)
            except Exception:
                pass
        score += float((t_subset.obs[t_group_col].astype(str) == label).sum()) / max(t_subset.n_obs, 1) * 0.25
        if score > best_score:
            best_label = label
            best_score = score
    return best_label or labels[0]


def _select_focus_subset(
    result_context: dict[str, str],
    *,
    adata: ad.AnnData,
    t_subset: ad.AnnData,
    t_group_col: str,
    group_col: str,
    tissue_col: str,
) -> tuple[ad.AnnData, str]:
    focus_label = _select_focus_label(result_context, t_subset, t_group_col)
    if focus_label:
        subset = t_subset[t_subset.obs[t_group_col].astype(str) == focus_label].copy()
        if subset.n_obs >= 25:
            return subset, focus_label
    subset = (
        _focus_cd8_subset(adata, group_col, tissue_col)
        if _detect_plan_focus(result_context, ("cd8", "cd8+", "cytotoxic"))
        else _focus_tcell_subset(adata, group_col, tissue_col)
    )
    label = "CD8+ T-cell subset" if _detect_plan_focus(result_context, ("cd8", "cd8+", "cytotoxic")) else "Focused T-cell subset"
    return subset, label


def _focus_fraction_summary(t_subset: ad.AnnData, focus_label: str, t_group_col: str, tissue_col: str) -> pd.DataFrame:
    if t_subset.n_obs == 0 or t_group_col not in t_subset.obs.columns or tissue_col not in t_subset.obs.columns:
        return pd.DataFrame()
    frame = t_subset.obs[[t_group_col, tissue_col]].dropna().copy()
    frame["is_focus"] = frame[t_group_col].astype(str) == str(focus_label)
    summary = (
        frame.groupby(tissue_col, observed=False)["is_focus"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    return summary


def _plot_focus_fraction(ax, summary: pd.DataFrame, tissue_col: str, title: str) -> None:
    if summary.empty:
        ax.axis("off")
        return
    sns.barplot(data=summary, x=tissue_col, y="is_focus", ax=ax, color="#6baed6")
    sns.stripplot(data=summary, x=tissue_col, y="is_focus", ax=ax, color="#084594", size=5, alpha=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("Fraction within T cells")
    ax.tick_params(axis="x", rotation=35)


def _clone_type_mix_by_tissue(adata: ad.AnnData, tissue_col: str) -> pd.DataFrame:
    if adata.n_obs == 0 or "cloneType" not in adata.obs.columns or tissue_col not in adata.obs.columns:
        return pd.DataFrame()
    frame = adata.obs[[tissue_col, "cloneType"]].dropna().copy()
    if frame.empty:
        return pd.DataFrame()
    table = pd.crosstab(frame[tissue_col].astype(str), frame["cloneType"].astype(str))
    denom = table.sum(axis=1).replace(0, 1)
    return table.div(denom, axis=0)


def _mean_expression_by_tissue(adata: ad.AnnData, genes: list[str], tissue_col: str) -> pd.DataFrame:
    if adata.n_obs == 0 or tissue_col not in adata.obs.columns or not genes:
        return pd.DataFrame()
    try:
        expr = expression_frame(adata, genes, obs_columns=[tissue_col]).groupby(tissue_col, observed=False).mean()
    except Exception:
        return pd.DataFrame()
    return expr.T if not expr.empty else pd.DataFrame()


def _select_contrast_tissues(result_context: dict[str, str], available_tissues: list[str]) -> tuple[str, str] | None:
    tissues = [str(item) for item in available_tissues if str(item)]
    if len(tissues) < 2:
        return None
    context = _context_blob(result_context).lower()
    aliases = {
        "primary_focus": ("primary_focus", "primary focus", "primary tumor", "primary"),
        "metastasis": ("metastasis", "metastatic"),
        "lymph_node": ("lymph_node", "lymph node"),
        "pbmc": ("pbmc", "peripheral blood"),
    }
    scored: list[tuple[float, str]] = []
    for tissue in tissues:
        patterns = aliases.get(tissue.lower(), (tissue.lower(), tissue.lower().replace("_", " ")))
        score = float(sum(context.count(pattern) for pattern in patterns))
        scored.append((score, tissue))
    scored.sort(key=lambda item: (-item[0], item[1]))
    mentioned = [tissue for score, tissue in scored if score > 0]
    if len(mentioned) >= 2:
        return mentioned[0], mentioned[1]
    preferred_pairs = [
        ("primary_focus", "metastasis"),
        ("PBMC", "lymph_node"),
        ("pbmc", "lymph_node"),
    ]
    available_lookup = {tissue.lower(): tissue for tissue in tissues}
    for left, right in preferred_pairs:
        left_value = available_lookup.get(left.lower())
        right_value = available_lookup.get(right.lower())
        if left_value and right_value:
            return left_value, right_value
    return tissues[0], tissues[1]


def _contrast_expression_table(
    adata: ad.AnnData,
    genes: list[str],
    tissue_col: str,
    contrast_tissues: tuple[str, str] | None,
) -> pd.DataFrame:
    if not contrast_tissues:
        return pd.DataFrame()
    left_tissue, right_tissue = contrast_tissues
    if adata.n_obs == 0 or tissue_col not in adata.obs.columns or not genes:
        return pd.DataFrame()
    subset = adata[adata.obs[tissue_col].astype(str).isin([left_tissue, right_tissue])].copy()
    if subset.n_obs == 0:
        return pd.DataFrame()
    expr = _mean_expression_by_tissue(subset, genes, tissue_col)
    if expr.empty or left_tissue not in expr.columns or right_tissue not in expr.columns:
        return pd.DataFrame()
    return (expr[right_tissue] - expr[left_tissue]).to_frame(name=f"{right_tissue} - {left_tissue}")


def _focus_cd8_subset(adata: ad.AnnData, group_col: str, tissue_col: str) -> ad.AnnData:
    subset = _focus_tcell_subset(adata, group_col, tissue_col)
    labels = subset.obs[group_col].astype(str).str.lower()
    mask = labels.map(lambda text: any(token in text for token in CD8_T_CELL_HINTS))
    cd8_subset = subset[mask].copy()
    if cd8_subset.n_obs < 100:
        return subset
    return cd8_subset


def _marker_genes_from_plan(result_context: dict[str, str], adata: ad.AnnData, top_n: int = 6) -> list[str]:
    genes: list[str] = []
    if _detect_plan_focus(result_context, ("exhaust", "pdcd1", "pd-1", "lag3", "havcr2", "tim-3", "tigit", "ctla4", "tox")):
        requested = resolve_gene_names(adata, EXHAUSTION_MARKER_CANDIDATES)
        genes.extend(list(dict.fromkeys(requested.values())))
    for gene in _focus_genes(result_context, adata, top_n=max(top_n, 8)):
        if gene not in genes:
            genes.append(gene)
    return genes[:top_n]


def _table_has_signal(table: pd.DataFrame, *, min_rows: int = 2, min_cols: int = 1, min_range: float = 0.15) -> bool:
    if table.empty or table.shape[0] < min_rows or table.shape[1] < min_cols:
        return False
    numeric = table.select_dtypes(include=[np.number])
    if numeric.empty:
        return False
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).any():
        return False
    return float(np.nanmax(values) - np.nanmin(values)) >= min_range


def _plot_marker_expression_by_tissue(
    ax,
    adata: ad.AnnData,
    genes: list[str],
    tissue_col: str,
    title: str,
) -> None:
    if adata.n_obs == 0 or not genes:
        ax.text(0.5, 0.5, "No marker data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.axis("off")
        return
    try:
        expr = expression_frame(adata, genes, obs_columns=[tissue_col]).groupby(tissue_col, observed=False).mean()
    except Exception:
        expr = pd.DataFrame()
    if expr.empty:
        ax.text(0.5, 0.5, "No marker data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.axis("off")
        return
    _plot_heatmap(ax, expr.T, title, cbar_label="mean RNA", cmap="magma")


def _plot_plan_de_heatmap(
    ax,
    adata: ad.AnnData,
    genes: list[str],
    tissue_col: str,
    title: str,
) -> None:
    if adata.n_obs == 0 or tissue_col not in adata.obs.columns or not genes:
        ax.text(0.5, 0.5, "No contrast data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.axis("off")
        return
    tissues = adata.obs[tissue_col].dropna().astype(str)
    available = set(tissues.unique().tolist())
    primary = "primary_focus" if "primary_focus" in available else None
    metastasis = "metastasis" if "metastasis" in available else None
    if primary is None or metastasis is None:
        _plot_marker_expression_by_tissue(ax, adata, genes, tissue_col, title)
        return
    subset = adata[adata.obs[tissue_col].astype(str).isin([primary, metastasis])].copy()
    if subset.n_obs == 0:
        _plot_marker_expression_by_tissue(ax, adata, genes, tissue_col, title)
        return
    try:
        expr = expression_frame(subset, genes, obs_columns=[tissue_col]).groupby(tissue_col, observed=False).mean()
    except Exception:
        expr = pd.DataFrame()
    if expr.empty or primary not in expr.index or metastasis not in expr.index:
        _plot_marker_expression_by_tissue(ax, adata, genes, tissue_col, title)
        return
    contrast = (expr.loc[metastasis] - expr.loc[primary]).to_frame(name=f"{metastasis} - {primary}")
    _plot_heatmap(ax, contrast, title, cbar_label="mean RNA delta", cmap="vlag")


def _normalize_barcode(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text.split("-")[0]


def _normalize_sample(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def _make_merge_key(barcode: object, sample: object = None) -> str:
    barcode_value = _normalize_barcode(barcode)
    sample_value = _normalize_sample(sample)
    if not barcode_value:
        return ""
    return f"{sample_value}::{barcode_value}" if sample_value else barcode_value


def _prepare_joint_adata(rna_h5ad_path: str | Path, tcr_path: str | Path) -> ad.AnnData:
    adata = sc.read_h5ad(rna_h5ad_path)
    if "barcode" not in adata.obs.columns:
        adata.obs["barcode"] = adata.obs_names.astype(str)
    adata.obs["barcode_core"] = adata.obs["barcode"].map(_normalize_barcode)
    sample_col = "sample_key" if "sample_key" in adata.obs.columns else ("sample_id" if "sample_id" in adata.obs.columns else None)
    adata.obs["merge_key"] = [
        _make_merge_key(barcode, adata.obs.iloc[idx][sample_col] if sample_col else None)
        for idx, barcode in enumerate(adata.obs["barcode"])
    ]

    tcr_df = normalize_tcr_columns(load_tcr_table(tcr_path)).copy()
    if "barcode" not in tcr_df.columns:
        raise ValueError("TCR table must contain a barcode column.")
    tcr_df["barcode_core"] = tcr_df["barcode"].map(_normalize_barcode)
    tcr_sample_col = "sample_key" if "sample_key" in tcr_df.columns else ("sample_id" if "sample_id" in tcr_df.columns else None)
    tcr_df["merge_key"] = [_make_merge_key(row["barcode"], row[tcr_sample_col] if tcr_sample_col else None) for _, row in tcr_df.iterrows()]

    grouped = tcr_df.groupby("merge_key", dropna=False).agg(
        clonotype_id=("clonotype_id", "first"),
        v_gene=("v_gene", lambda s: "|".join(sorted({str(v) for v in s if pd.notna(v) and str(v).strip()}))),
        j_gene=("j_gene", lambda s: "|".join(sorted({str(v) for v in s if pd.notna(v) and str(v).strip()}))),
        productive_any=("productive", "max"),
    )
    adata.obs = adata.obs.join(grouped, on="merge_key")
    adata.obs["has_tcr"] = adata.obs["clonotype_id"].notna()
    clone_sizes = adata.obs.loc[adata.obs["has_tcr"], "clonotype_id"].value_counts()
    adata.obs["clone_size"] = adata.obs["clonotype_id"].map(clone_sizes).fillna(0).astype(int)
    adata.obs["expanded_clone"] = adata.obs["clone_size"] >= 3
    adata.obs["expanded_label"] = adata.obs["expanded_clone"].map({True: "expanded", False: "non-expanded"})
    return adata


def _plot_heatmap(ax, table: pd.DataFrame, title: str, cbar_label: str = "mean signal", cmap: str = "mako") -> None:
    if table.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.axis("off")
        return
    sns.heatmap(table, cmap=cmap, linewidths=0.3, linecolor="white", ax=ax, cbar_kws={"label": cbar_label})
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("")


def _plot_stacked_bar(
    ax,
    table: pd.DataFrame,
    title: str,
    *,
    show_legend: bool = True,
    legend_title: str | None = None,
) -> None:
    if table.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.axis("off")
        return
    table.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", width=0.85, legend=show_legend)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("Fraction")
    ax.tick_params(axis="x", rotation=35)
    if show_legend:
        legend = ax.get_legend()
        if legend is not None:
            legend.set_bbox_to_anchor((1.02, 1.0))
            legend._loc = 2
            legend.set_frame_on(False)
            legend.set_title(legend_title or "")
            for text in legend.get_texts():
                text.set_fontsize(7)
            if legend.get_title() is not None:
                legend.get_title().set_fontsize(8)


def _standard_composition_table(adata: ad.AnnData, group_col: str, tissue_col: str, top_n_groups: int = 8) -> pd.DataFrame:
    frame = adata.obs[[group_col, tissue_col]].dropna().copy()
    if frame.empty:
        return pd.DataFrame()
    top_groups = frame[group_col].astype(str).value_counts().head(top_n_groups).index.tolist()
    frame = frame.loc[frame[group_col].astype(str).isin(top_groups)]
    table = pd.crosstab(frame[tissue_col].astype(str), frame[group_col].astype(str))
    denom = table.sum(axis=1).replace(0, 1)
    return table.div(denom, axis=0)


def _clone_size_summary(paired: ad.AnnData, tissue_col: str) -> pd.DataFrame:
    if paired.n_obs == 0 or "clone_size" not in paired.obs.columns:
        return pd.DataFrame()
    frame = paired.obs[[tissue_col, "clone_size"]].copy()
    frame["clone_size_log10"] = np.log10(frame["clone_size"].clip(lower=1))
    return frame


def _plot_clone_size_distribution(ax, paired: ad.AnnData, tissue_col: str) -> None:
    frame = _clone_size_summary(paired, tissue_col)
    if frame.empty:
        ax.text(0.5, 0.5, "No paired clonotype data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Clone size distribution by tissue")
        ax.axis("off")
        return
    sns.boxplot(data=frame, x=tissue_col, y="clone_size_log10", ax=ax, color="#9ecae1", fliersize=1.5)
    ax.set_title("Clone size distribution by tissue", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("log10(clone size)")
    ax.tick_params(axis="x", rotation=35)


def _tissue_clonotype_sharing(paired: ad.AnnData, tissue_col: str) -> pd.DataFrame:
    if paired.n_obs == 0 or "clonotype_id" not in paired.obs.columns:
        return pd.DataFrame()
    frame = paired.obs[[tissue_col, "clonotype_id"]].dropna().copy()
    if frame.empty:
        return pd.DataFrame()
    tissue_to_clones = {
        tissue: set(group["clonotype_id"].astype(str))
        for tissue, group in frame.groupby(tissue_col, observed=False)
    }
    tissues = list(tissue_to_clones.keys())
    matrix = pd.DataFrame(index=tissues, columns=tissues, dtype=float)
    for left in tissues:
        for right in tissues:
            union = tissue_to_clones[left] | tissue_to_clones[right]
            matrix.loc[left, right] = len(tissue_to_clones[left] & tissue_to_clones[right]) / max(len(union), 1)
    return matrix


def _v_gene_usage_heatmap(adata: ad.AnnData, tissue_col: str, top_n: int = 10) -> pd.DataFrame:
    if "v_gene" not in adata.obs.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for _, row in adata.obs[[tissue_col, "v_gene"]].dropna().iterrows():
        tissue = str(row[tissue_col])
        for gene in str(row["v_gene"]).split("|"):
            gene = gene.strip()
            if gene:
                rows.append({"tissue": tissue, "v_gene": gene})
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    top_genes = frame["v_gene"].value_counts().head(top_n).index.tolist()
    frame = frame.loc[frame["v_gene"].isin(top_genes)]
    table = pd.crosstab(frame["v_gene"], frame["tissue"])
    denom = table.sum(axis=0).replace(0, 1)
    return table.div(denom, axis=1)


def _expanded_fraction_table(adata: ad.AnnData, row_col: str, column_col: str) -> pd.DataFrame:
    if adata.n_obs == 0 or "expanded_clone" not in adata.obs.columns:
        return pd.DataFrame()
    frame = adata.obs[[row_col, column_col, "expanded_clone"]].copy()
    frame["expanded_clone"] = (
        frame["expanded_clone"]
        .astype(str)
        .str.lower()
        .isin({"true", "1", "expanded"})
    )
    return frame.groupby([row_col, column_col], observed=False)["expanded_clone"].mean().unstack(fill_value=0.0)


def _cluster_marker_heatmap(adata: ad.AnnData, group_col: str, marker_csv_path: str | Path | None, top_n_groups: int = 8) -> pd.DataFrame:
    marker_genes: list[str] = []
    top_groups = adata.obs[group_col].astype(str).value_counts().head(top_n_groups).index.tolist()
    if marker_csv_path and Path(marker_csv_path).exists():
        marker_df = pd.read_csv(marker_csv_path)
        if {"cluster", "names"}.issubset(marker_df.columns):
            cluster_to_group = (
                adata.obs.assign(_group=adata.obs[group_col].astype(str), _cluster=adata.obs["leiden"].astype(str))
                .groupby("_cluster", observed=False)["_group"]
                .agg(lambda s: s.value_counts().index[0] if not s.empty else "Unknown")
                .to_dict()
            )
            if "used_for_annotation" in marker_df.columns:
                marker_df = marker_df.loc[marker_df["used_for_annotation"].astype(bool)]
            if "is_linc_like" in marker_df.columns:
                marker_df = marker_df.loc[~marker_df["is_linc_like"].astype(bool)]
            marker_df["mapped_group"] = marker_df["cluster"].astype(str).map(cluster_to_group)
            marker_df = marker_df.loc[marker_df["mapped_group"].isin(top_groups)].copy()
            sort_cols = ["mapped_group"]
            ascending = [True]
            if "rank" in marker_df.columns:
                sort_cols.append("rank")
                ascending.append(True)
            elif "scores" in marker_df.columns:
                sort_cols.append("scores")
                ascending.append(False)
            marker_df = marker_df.sort_values(sort_cols, ascending=ascending)
            for group in top_groups:
                chosen: list[str] = []
                for gene in marker_df.loc[marker_df["mapped_group"] == group, "names"].astype(str):
                    if gene in adata.var_names and gene not in chosen:
                        chosen.append(gene)
                    if len(chosen) >= 2:
                        break
                marker_genes.extend(chosen)
    if not marker_genes and "rank_genes_groups" in adata.uns:
        try:
            for cluster in pd.Series(adata.obs["leiden"].astype(str)).value_counts().index.tolist():
                frame = sc.get.rank_genes_groups_df(adata, group=cluster).head(2)
                marker_genes.extend([gene for gene in frame["names"].astype(str).tolist() if gene in adata.var_names])
        except Exception:
            marker_genes = []
    marker_genes = list(dict.fromkeys(marker_genes))
    if not marker_genes:
        return pd.DataFrame()
    expr = expression_frame(adata, marker_genes, obs_columns=[group_col]).groupby(group_col, observed=False).mean()
    keep_cols = [col for col in top_groups if col in expr.index]
    expr = expr.loc[keep_cols]
    return expr.T


def _plan_requests_pseudotime(result_context: dict[str, str]) -> bool:
    joined = "\n".join(
        [
            result_context.get("executed_hypothesis", ""),
            result_context.get("approved_plan_text", ""),
            result_context.get("approved_strategy_feedback", ""),
            result_context.get("approved_plan_steps", ""),
        ]
    ).lower()
    return any(token in joined for token in PSEUDOTIME_HINTS)


def _focus_tcell_subset(adata: ad.AnnData, group_col: str, tissue_col: str) -> ad.AnnData:
    working = paired_tcr_subset(adata)
    try:
        working = tumor_like_subset(working, tissue_col=tissue_col)
    except Exception:
        pass
    labels = working.obs[group_col].astype(str)
    mask = labels.str.lower().map(lambda text: any(token in text for token in T_CELL_HINTS))
    subset = working[mask].copy()
    if subset.n_obs < 200:
        subset = working.copy()
    return subset


def _compute_pseudotime(subset: ad.AnnData) -> ad.AnnData | None:
    if subset.n_obs < 100:
        return None
    working = subset.copy()
    if "X_pca" not in working.obsm or working.obsm["X_pca"].shape[1] < 5:
        sc.pp.pca(working, n_comps=min(30, max(5, working.n_vars - 1)))
    sc.pp.neighbors(working, use_rep="X_pca", n_neighbors=min(20, max(8, working.n_obs // 80)))
    sc.tl.diffmap(working)
    root_idx = 0
    if "expanded_clone" in working.obs.columns:
        non_expanded = np.where(~working.obs["expanded_clone"].fillna(False).astype(bool).to_numpy())[0]
        if len(non_expanded):
            root_idx = int(non_expanded[0])
    working.uns["iroot"] = root_idx
    sc.tl.dpt(working)
    if "X_umap" not in working.obsm:
        sc.tl.umap(working)
    return working


def _pseudotime_bin_table(subset: ad.AnnData, genes: list[str], tissue_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "dpt_pseudotime" not in subset.obs.columns or not genes:
        return pd.DataFrame(), pd.DataFrame()
    work = subset.copy()
    n_bins = min(5, max(2, work.n_obs // 150))
    bin_codes = pd.qcut(
        work.obs["dpt_pseudotime"].rank(method="first"),
        q=n_bins,
        labels=False,
        duplicates="drop",
    )
    unique_codes = sorted(pd.Series(bin_codes).dropna().astype(int).unique().tolist())
    if not unique_codes:
        return pd.DataFrame(), pd.DataFrame()
    labels = []
    for idx, code in enumerate(unique_codes):
        if idx == 0:
            labels.append("Early")
        elif idx == len(unique_codes) - 1:
            labels.append("Late")
        else:
            labels.append(f"Mid-{idx}")
    code_to_label = {code: label for code, label in zip(unique_codes, labels)}
    work.obs["pseudotime_bin"] = pd.Series(bin_codes, index=work.obs.index).map(code_to_label).astype(str)
    expr = expression_frame(work, genes, obs_columns=["pseudotime_bin"]).groupby("pseudotime_bin", observed=False).mean()
    if "expanded_clone" in work.obs.columns:
        clone_by_bin = (
            work.obs.groupby(["pseudotime_bin", tissue_col], observed=False)["expanded_clone"]
            .mean()
            .unstack(fill_value=0.0)
        )
    else:
        clone_by_bin = pd.DataFrame()
    ordered_labels = [label for label in labels if label in expr.index]
    expr = expr.loc[ordered_labels]
    if not clone_by_bin.empty:
        clone_by_bin = clone_by_bin.reindex(ordered_labels)
    return expr.T, clone_by_bin


def _focused_de_heatmap(subset: ad.AnnData, tissue_col: str) -> pd.DataFrame:
    try:
        de_frame = tissue_stratified_expansion_de(
            subset,
            tissue_col=tissue_col,
            expansion_col="expanded_clone",
            paired_only=False,
            min_cells_per_group=15,
            top_n=5,
        )
    except Exception:
        return pd.DataFrame()
    return de_frame.pivot_table(index="names", columns="tissue", values="logfoldchanges", aggfunc="mean").fillna(0.0)


def _focus_genes(result_context: dict[str, str], adata: ad.AnnData, top_n: int = 8) -> list[str]:
    result_genes = extract_result_genes(
        available_genes=adata.var_names,
        texts=[
            result_context.get("approved_strategy_feedback", ""),
            result_context.get("notebook_text", ""),
            result_context.get("final_interpretation", ""),
        ],
        top_n=top_n,
    )
    merged = [gene for gene in DEFAULT_FOCUS_GENES if gene in adata.var_names]
    for gene in result_genes:
        if gene not in merged:
            merged.append(gene)
    if len(merged) < top_n:
        genes = extract_hypothesis_genes(result_context.get("executed_hypothesis", ""), adata.var_names, top_n=top_n)
        for gene in genes:
            if gene not in merged:
                merged.append(gene)
    return merged[:top_n]


def _summary_lines(
    *,
    figure_name: str,
    rna_h5ad_path: str | Path,
    tcr_path: str | Path,
    paired: ad.AnnData,
    hypothesis_text: str,
    focus_genes: list[str],
    pseudotime_used: bool,
) -> list[str]:
    return [
        f"Figure name: {figure_name}",
        f"RNA input: {Path(rna_h5ad_path).resolve()}",
        f"TCR input: {Path(tcr_path).resolve()}",
        f"Executed hypothesis: {hypothesis_text or 'none'}",
        f"Paired cells: {paired.n_obs}",
        f"Focus genes: {', '.join(focus_genes) or 'none'}",
        f"Pseudotime requested or used: {'yes' if pseudotime_used else 'no'}",
    ]


def build_publication_figure(
    *,
    rna_h5ad_path: str | Path,
    tcr_path: str | Path,
    output_dir: str | Path,
    figure_name: str = "scrt_publication_figure",
    hypothesis_model: str = "gpt-4o",
    baseline_summary_text: str = "",
    prompt_dir: str | Path | None = None,
) -> FigureResult:
    _load_figure_env_files(rna_h5ad_path, tcr_path, output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    sc.settings.set_figure_params(dpi=120, facecolor="white", frameon=False)

    adata = _prepare_joint_adata(rna_h5ad_path, tcr_path)
    group_col = ensure_display_column(adata, "cell_type", ("cluster_cell_type", "annotation", "leiden"))
    tissue_col = "tissue" if "tissue" in adata.obs.columns else ("sample_key" if "sample_key" in adata.obs.columns else group_col)
    result_context = read_run_result_context(output_dir)
    hypothesis_text = result_context["executed_hypothesis"]
    approved_priority = result_context.get("approved_priority_question", "")

    paired = paired_tcr_subset(adata)
    assign_clone_type_labels(paired)
    global_prop = proportion_table(adata, group_col=group_col, tissue_col=tissue_col, normalize="index")
    annotation_model = os.environ.get("SCRT_TCELL_ANNOTATION_MODEL", "gpt-4o")
    try:
        t_subset, t_marker_df, t_annotation_df = recluster_and_annotate_t_cells(
            paired,
            group_col=group_col,
            model_name=annotation_model,
        )
    except Exception:
        t_subset = paired.copy()
        t_marker_df = pd.DataFrame()
        t_annotation_df = pd.DataFrame()
    t_group_col = "tcell_cluster_cell_type" if "tcell_cluster_cell_type" in t_subset.obs.columns else group_col
    if t_subset.n_obs:
        assign_clone_type_labels(t_subset)
        if "cell_type" not in t_subset.obs.columns or t_subset.obs["cell_type"].astype(str).str.contains("T-cell cluster").all():
            t_subset.obs["cell_type"] = t_subset.obs[t_group_col].astype(str)
    t_prop = proportion_table(t_subset, group_col=t_group_col, tissue_col=tissue_col, normalize="index") if t_subset.n_obs else pd.DataFrame()
    clone_mix = clone_type_distribution_table(t_subset, group_col=t_group_col, clone_type_col="cloneType") if t_subset.n_obs else pd.DataFrame()

    summary_fig = plt.figure(figsize=(24, 18))
    gs = GridSpec(3, 2, figure=summary_fig, hspace=0.34, wspace=0.68)

    ax_a = summary_fig.add_subplot(gs[0, 0])
    plot_categorical_embedding(
        ax_a,
        adata,
        color=group_col,
        title="scRNA UMAP with annotated cell types",
        label_points=False,
        show_legend=True,
        legend_title="Cell type",
    )
    panel_label(ax_a, "a")

    ax_b = summary_fig.add_subplot(gs[0, 1])
    _plot_stacked_bar(ax_b, global_prop, "Cell composition across tissues", show_legend=True, legend_title="Cell type")
    panel_label(ax_b, "b")

    ax_c = summary_fig.add_subplot(gs[1, 0])
    plot_categorical_embedding(
        ax_c,
        t_subset if t_subset.n_obs else paired,
        color=t_group_col if t_subset.n_obs else group_col,
        title="Reclustered T-cell UMAP with LLM-defined subtype annotations",
        label_points=False,
        show_legend=True,
        legend_title="T-cell subtype",
    )
    panel_label(ax_c, "c")

    ax_d = summary_fig.add_subplot(gs[1, 1])
    _plot_stacked_bar(ax_d, t_prop, "T-cell subtype composition across tissues", show_legend=True, legend_title="T-cell subtype")
    panel_label(ax_d, "d")

    ax_e = summary_fig.add_subplot(gs[2, 0])
    plot_categorical_embedding(
        ax_e,
        t_subset if t_subset.n_obs else paired,
        color="cloneType" if t_subset.n_obs and "cloneType" in t_subset.obs.columns else group_col,
        title="T-cell cloneType UMAP",
        label_points=False,
        show_legend=True,
        legend_title="cloneType",
    )
    panel_label(ax_e, "e")

    ax_f = summary_fig.add_subplot(gs[2, 1])
    _plot_stacked_bar(ax_f, clone_mix, "cloneType composition across T-cell subtypes", show_legend=True, legend_title="cloneType")
    panel_label(ax_f, "f")

    summary_fig.suptitle("Standard scRNA + scTCR baseline figure", fontsize=18, y=0.995)
    summary_fig.tight_layout(rect=[0, 0, 1, 0.985])
    png_path, pdf_path, summary_path = save_figure_bundle(summary_fig, output_dir, figure_name)

    diversity_by_tissue = _tcr_diversity_by_sample_tissue(paired, tissue_col=tissue_col)
    normalize_tcr_columns(load_tcr_table(tcr_path)).copy()

    focus_subset, focus_label = _select_focus_subset(
        result_context,
        adata=adata,
        t_subset=t_subset,
        t_group_col=t_group_col,
        group_col=group_col,
        tissue_col=tissue_col,
    )
    plan_requests_pseudotime = _plan_requests_pseudotime(result_context)
    plan_requests_expression = _plan_requests_expression_panel(result_context)
    plan_requests_clonotype = _plan_requests_clonotype_panel(result_context)
    pseudotime_subset = _compute_pseudotime(focus_subset) if plan_requests_pseudotime else None
    focus_genes = _focus_genes(result_context, focus_subset if focus_subset.n_obs else adata)
    plan_marker_genes = _marker_genes_from_plan(result_context, focus_subset if focus_subset.n_obs else adata)
    focus_fraction = _focus_fraction_summary(t_subset, focus_label, t_group_col, tissue_col) if t_subset.n_obs else pd.DataFrame()
    focus_clone_mix = _clone_type_mix_by_tissue(focus_subset, tissue_col)
    contrast_tissues = _select_contrast_tissues(
        result_context,
        sorted(focus_subset.obs[tissue_col].astype(str).dropna().unique().tolist()) if focus_subset.n_obs and tissue_col in focus_subset.obs.columns else [],
    )
    marker_expression = _mean_expression_by_tissue(focus_subset, plan_marker_genes, tissue_col)
    contrast_expression = _contrast_expression_table(focus_subset, plan_marker_genes, tissue_col, contrast_tissues)
    trend_heatmap = pd.DataFrame()
    clone_by_bin = pd.DataFrame()
    if pseudotime_subset is not None and focus_genes:
        trend_heatmap, clone_by_bin = _pseudotime_bin_table(pseudotime_subset, focus_genes[:6], tissue_col)
    focus_de = _focused_de_heatmap(focus_subset, tissue_col=tissue_col) if focus_subset.n_obs else pd.DataFrame()
    focus_clone = _expanded_fraction_table(
        focus_subset,
        tissue_col,
        "cloneType" if focus_subset.n_obs and "cloneType" in focus_subset.obs.columns else group_col,
    ) if focus_subset.n_obs else pd.DataFrame()
    focus_clone_sharing = _tissue_clonotype_sharing(focus_subset, tissue_col=tissue_col) if focus_subset.n_obs else pd.DataFrame()
    focus_v_gene_usage = _v_gene_usage_heatmap(focus_subset, tissue_col=tissue_col, top_n=8) if focus_subset.n_obs else pd.DataFrame()

    panel_specs: list[tuple[str, callable]] = []
    focus_basis = "X_umap" if focus_subset.n_obs and "X_umap" in focus_subset.obsm else ("X_pca" if focus_subset.n_obs and "X_pca" in focus_subset.obsm else "")
    if focus_basis:
        panel_specs.append(
            (
                "focus_embedding",
                lambda fig, ax: plot_categorical_embedding(
                    ax,
                    focus_subset,
                    color=tissue_col if tissue_col in focus_subset.obs.columns else t_group_col,
                    title=f"{focus_label} embedding by tissue context",
                    basis=focus_basis,
                    label_points=False,
                    show_legend=True,
                    legend_title="Tissue" if tissue_col in focus_subset.obs.columns else t_group_col,
                ),
            )
        )
    if not focus_fraction.empty:
        panel_specs.append(
            (
                "focus_fraction",
                lambda fig, ax: _plot_focus_fraction(ax, focus_fraction, tissue_col, f"{focus_label} abundance across tissues"),
            )
        )
    if not focus_clone_mix.empty and plan_requests_clonotype:
        panel_specs.append(
            (
                "clone_mix",
                lambda fig, ax: _plot_stacked_bar(ax, focus_clone_mix, f"{focus_label} cloneType composition across tissues", show_legend=True, legend_title="cloneType"),
            )
        )
    if not _clone_size_summary(focus_subset, tissue_col).empty and plan_requests_clonotype:
        panel_specs.append(
            (
                "clone_size",
                lambda fig, ax: _plot_clone_size_distribution(ax, focus_subset, tissue_col),
            )
        )
    if plan_requests_expression and _table_has_signal(marker_expression, min_rows=2, min_cols=2, min_range=0.2):
        panel_specs.append(
            (
                "marker_expression",
                lambda fig, ax: _plot_heatmap(ax, marker_expression, f"{focus_label} marker expression by tissue", cbar_label="mean RNA", cmap="magma"),
            )
        )
    if plan_requests_expression and _table_has_signal(contrast_expression, min_rows=2, min_cols=1, min_range=0.2):
        contrast_title = f"{focus_label} marker contrast"
        if contrast_tissues:
            contrast_title = f"{focus_label} marker contrast: {contrast_tissues[1]} vs {contrast_tissues[0]}"
        panel_specs.append(
            (
                "marker_contrast",
                lambda fig, ax: _plot_heatmap(ax, contrast_expression, contrast_title, cbar_label="mean RNA delta", cmap="vlag"),
            )
        )
    elif plan_requests_expression and _table_has_signal(focus_de, min_rows=2, min_cols=2, min_range=0.25):
        panel_specs.append(
            (
                "focused_de",
                lambda fig, ax: _plot_heatmap(ax, focus_de, f"{focus_label} differential expression support", cbar_label="logFC", cmap="vlag"),
            )
        )
    if plan_requests_pseudotime and pseudotime_subset is not None and "dpt_pseudotime" in pseudotime_subset.obs.columns:
        pseudo_basis = "X_umap" if "X_umap" in pseudotime_subset.obsm else ("X_pca" if "X_pca" in pseudotime_subset.obsm else "")
        if pseudo_basis:
            def _render_pseudotime(fig, ax):
                coords = pseudotime_subset.obsm[pseudo_basis][:, :2]
                values = pseudotime_subset.obs["dpt_pseudotime"].astype(float).to_numpy()
                scatter = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap="viridis", s=14, alpha=0.85, linewidths=0)
                cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("dpt_pseudotime")
                ax.set_title(f"{focus_label} {'UMAP' if pseudo_basis == 'X_umap' else 'PCA'} pseudotime ordering", fontsize=11)
                ax.set_xlabel("UMAP1" if pseudo_basis == "X_umap" else "PC1")
                ax.set_ylabel("UMAP2" if pseudo_basis == "X_umap" else "PC2")
            panel_specs.append(("pseudotime_embedding", _render_pseudotime))
        if not clone_by_bin.empty:
            panel_specs.append(
                (
                    "pseudotime_clone_support",
                    lambda fig, ax: _plot_heatmap(ax, clone_by_bin.T, "Expanded clone fraction across pseudotime bins", cbar_label="expanded fraction", cmap="crest"),
                )
            )
        elif not trend_heatmap.empty:
            panel_specs.append(
                (
                    "pseudotime_marker_support",
                    lambda fig, ax: _plot_heatmap(ax, trend_heatmap, "Marker dynamics along pseudotime", cbar_label="mean RNA", cmap="rocket"),
                )
            )
    if not focus_clone.empty and plan_requests_clonotype:
        panel_specs.append(
            (
                "expanded_clone_support",
                lambda fig, ax: _plot_heatmap(ax, focus_clone, f"{focus_label} expanded clone support", cbar_label="expanded fraction", cmap="crest"),
            )
        )
    if not diversity_by_tissue.empty and (_plan_requests_diversity_panel(result_context) or plan_requests_clonotype):
        panel_specs.append(
            (
                "diversity",
                lambda fig, ax: _plot_tcr_diversity_by_tissue(ax, diversity_by_tissue, tissue_col),
            )
        )
    if plan_requests_clonotype and _table_has_signal(focus_clone_sharing, min_rows=2, min_cols=2, min_range=0.05):
        panel_specs.append(
            (
                "clone_sharing",
                lambda fig, ax: _plot_heatmap(ax, focus_clone_sharing, f"{focus_label} clonotype sharing across tissues", cbar_label="Jaccard overlap", cmap="crest"),
            )
        )
    if plan_requests_clonotype and _table_has_signal(focus_v_gene_usage, min_rows=3, min_cols=2, min_range=0.05):
        panel_specs.append(
            (
                "v_gene_usage",
                lambda fig, ax: _plot_heatmap(ax, focus_v_gene_usage, f"{focus_label} V gene usage across tissues", cbar_label="column fraction", cmap="mako"),
            )
        )
    if not panel_specs:
        panel_specs.append(
            (
                "tcell_overview",
                lambda fig, ax: plot_categorical_embedding(
                    ax,
                    t_subset if t_subset.n_obs else paired,
                    color=t_group_col if t_subset.n_obs else group_col,
                    title="Focused T-cell subtype overview",
                    label_points=False,
                    show_legend=True,
                    legend_title="T-cell subtype",
                ),
            )
        )

    hypothesis_pages: list[tuple[Path, Path]] = []
    max_panels_per_page = 6
    panel_chunks = [panel_specs[idx: idx + max_panels_per_page] for idx in range(0, len(panel_specs), max_panels_per_page)]
    hyp_png = None
    hyp_pdf = None
    hyp_summary = output_dir / f"{figure_name}_hypothesis_summary.txt"
    for page_index, chunk in enumerate(panel_chunks, start=1):
        n_panels = len(chunk)
        ncols = 2 if n_panels > 1 else 1
        nrows = int(np.ceil(n_panels / ncols))
        hyp_fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, max(7, 6 * nrows)))
        panel_axes = np.atleast_1d(axes).ravel().tolist()
        for idx, (ax, (_, renderer)) in enumerate(zip(panel_axes, chunk)):
            renderer(hyp_fig, ax)
            panel_label(ax, chr(ord("a") + idx))
        for ax in panel_axes[n_panels:]:
            ax.remove()
        hyp_fig.suptitle("Hypothesis-driven scRNA + scTCR figure", fontsize=18, y=0.995)
        hyp_fig.tight_layout(rect=[0, 0, 1, 0.985])
        page_name = f"{figure_name}_hypothesis" if page_index == 1 else f"{figure_name}_hypothesis_page{page_index}"
        page_png, page_pdf, _ = save_figure_bundle(hyp_fig, output_dir, page_name)
        hypothesis_pages.append((page_png, page_pdf))
        if page_index == 1:
            hyp_png, hyp_pdf = page_png, page_pdf
    for stale_path in (
        output_dir / f"{figure_name}_hypothesis_generated_code.py",
        output_dir / f"{figure_name}_hypothesis_generated_plan.txt",
    ):
        if stale_path.exists():
            stale_path.unlink()
    hypothesis_generation_note = (
        f"Result-driven hypothesis figure built directly from executed results for focus subset: {focus_label}. "
        f"Panels generated: {len(panel_specs)} across {len(panel_chunks)} page(s)."
    )
    hyp_summary.write_text(
        "\n".join(
            [
                f"Executed hypothesis: {hypothesis_text or 'none'}",
                f"Approved priority question: {approved_priority or 'none'}",
                f"Approved strategy feedback: {result_context.get('approved_strategy_feedback', '') or 'none'}",
                f"User feedback: {result_context.get('user_feedback', '') or 'none'}",
                f"Focus subset: {focus_label}",
                f"Contrast tissues: {contrast_tissues[0]} vs {contrast_tissues[1]}" if contrast_tissues else "Contrast tissues: none",
                f"Panels generated: {len(panel_specs)}",
                f"Hypothesis figure pages: {len(panel_chunks)}",
                "Panel order: " + ", ".join(name for name, _ in panel_specs),
                "This hypothesis figure was generated directly from approved plan context and executed analysis results; no secondary model-authored figure code was used.",
                *[
                    f"Page {page_idx}: {page_png}"
                    for page_idx, (page_png, _) in enumerate(hypothesis_pages, start=1)
                ],
            ]
        ),
        encoding="utf-8",
    )

    summary_path.write_text(
        "\n".join(
            _summary_lines(
                figure_name=figure_name,
                rna_h5ad_path=rna_h5ad_path,
                tcr_path=tcr_path,
                paired=paired,
                hypothesis_text=hypothesis_text,
                focus_genes=focus_genes,
                pseudotime_used=pseudotime_subset is not None,
            )
            + [
                "Standard figure layout: baseline six-panel figure (global UMAP, global composition, T-cell reclustering UMAP, T-cell composition, cloneType UMAP, cloneType composition).",
                hypothesis_generation_note,
                f"Summary figure PNG: {png_path}",
                f"Summary figure PDF: {pdf_path}",
                f"Hypothesis figure PNG: {hyp_png}",
                f"Hypothesis figure PDF: {hyp_pdf}",
            ]
        ),
        encoding="utf-8",
    )
    return FigureResult(png_path=png_path, pdf_path=pdf_path, summary_path=summary_path)
