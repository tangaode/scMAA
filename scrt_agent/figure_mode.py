"""Publication-style figure builder for scRNA + scTCR analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from scrt_agent.figure_common import (
    ensure_display_column,
    extract_hypothesis_genes,
    extract_result_genes,
    panel_label,
    plot_categorical_embedding,
    plot_text_panel,
    read_run_result_context,
    save_figure_bundle,
)

from .notebook_tools import (
    clone_expansion_table,
    expression_frame,
    paired_tcr_subset,
    tissue_stratified_expansion_de,
    tumor_like_subset,
)
from .utils import load_tcr_table, normalize_tcr_columns


@dataclass
class FigureResult:
    png_path: Path
    pdf_path: Path
    summary_path: Path


T_CELL_HINTS = ("t cell", "t cells", "cd8", "cd4", "treg", "regulatory t", "cytotoxic t")
PSEUDOTIME_HINTS = ("pseudotime", "trajectory", "dpt", "pseudotemporal", "trajectory analysis")
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


def _detect_plan_focus(result_context: dict[str, str], tokens: tuple[str, ...]) -> bool:
    joined = "\n".join(
        [
            result_context.get("executed_hypothesis", ""),
            result_context.get("approved_plan_text", ""),
            result_context.get("approved_priority_question", ""),
            result_context.get("approved_plan_steps", ""),
            result_context.get("approved_strategy_feedback", ""),
            result_context.get("run_summary", ""),
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


def _plot_stacked_bar(ax, table: pd.DataFrame, title: str) -> None:
    if table.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.axis("off")
        return
    table.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", width=0.85, legend=False)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("Fraction")
    ax.tick_params(axis="x", rotation=35)


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
    work.obs["pseudotime_bin"] = pd.qcut(
        work.obs["dpt_pseudotime"].rank(method="first"),
        q=min(5, max(2, work.n_obs // 150)),
        labels=False,
        duplicates="drop",
    ).astype(str)
    expr = expression_frame(work, genes, obs_columns=["pseudotime_bin"]).groupby("pseudotime_bin", observed=False).mean()
    if "expanded_clone" in work.obs.columns:
        clone_by_bin = (
            work.obs.groupby(["pseudotime_bin", tissue_col], observed=False)["expanded_clone"]
            .mean()
            .unstack(fill_value=0.0)
        )
    else:
        clone_by_bin = pd.DataFrame()
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
) -> FigureResult:
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
    marker_csv_path = Path(rna_h5ad_path).resolve().with_name("cluster_markers.csv")

    paired = paired_tcr_subset(adata)
    composition = _standard_composition_table(adata, group_col, tissue_col)
    marker_heatmap = _cluster_marker_heatmap(adata, group_col, marker_csv_path)
    expansion_heatmap = _expanded_fraction_table(paired, group_col, tissue_col) if paired.n_obs else pd.DataFrame()
    clone_sharing = _tissue_clonotype_sharing(paired, tissue_col=tissue_col)
    v_gene_heatmap = _v_gene_usage_heatmap(paired, tissue_col=tissue_col)
    diversity_by_tissue = _tcr_diversity_by_sample_tissue(paired, tissue_col=tissue_col)

    summary_fig = plt.figure(figsize=(18, 24))
    gs = GridSpec(4, 2, figure=summary_fig, hspace=0.42, wspace=0.24)

    ax_a = summary_fig.add_subplot(gs[0, 0])
    plot_categorical_embedding(ax_a, adata, color=group_col, title="scRNA UMAP with annotated cell types")
    panel_label(ax_a, "a")

    ax_b = summary_fig.add_subplot(gs[0, 1])
    _plot_stacked_bar(ax_b, composition, "Cell composition across tissues")
    panel_label(ax_b, "b")

    ax_c = summary_fig.add_subplot(gs[1, 0])
    plot_categorical_embedding(ax_c, paired if paired.n_obs else adata, color="expanded_label", title="Paired cells: clonal expansion overlay")
    panel_label(ax_c, "c")

    ax_d = summary_fig.add_subplot(gs[1, 1])
    _plot_heatmap(ax_d, expansion_heatmap, "Expanded clone fraction by cell type and tissue", cbar_label="expanded fraction")
    panel_label(ax_d, "d")

    ax_e = summary_fig.add_subplot(gs[2, 0])
    _plot_clone_size_distribution(ax_e, paired, tissue_col)
    panel_label(ax_e, "e")

    ax_f = summary_fig.add_subplot(gs[2, 1])
    _plot_heatmap(ax_f, clone_sharing, "Clonotype sharing across tissues", cbar_label="Jaccard overlap", cmap="crest")
    panel_label(ax_f, "f")

    ax_g = summary_fig.add_subplot(gs[3, 0])
    _plot_heatmap(ax_g, marker_heatmap, "Top scRNA marker heatmap", cbar_label="mean RNA")
    panel_label(ax_g, "g")

    ax_h = summary_fig.add_subplot(gs[3, 1])
    _plot_heatmap(ax_h, v_gene_heatmap, "Top V gene usage across tissues", cbar_label="column fraction")
    panel_label(ax_h, "h")

    summary_fig.suptitle("Standard scRNA + scTCR summary figure", fontsize=18, y=0.995)
    summary_fig.tight_layout(rect=[0, 0, 1, 0.985])
    png_path, pdf_path, summary_path = save_figure_bundle(summary_fig, output_dir, figure_name)

    focus_subset = _focus_tcell_subset(adata, group_col, tissue_col)
    plan_requests_pseudotime = _plan_requests_pseudotime(result_context)
    pseudotime_subset = _compute_pseudotime(focus_subset) if plan_requests_pseudotime else None
    focus_genes = _focus_genes(result_context, focus_subset if focus_subset.n_obs else adata)
    trend_heatmap = pd.DataFrame()
    clone_by_bin = pd.DataFrame()
    if pseudotime_subset is not None and focus_genes:
        trend_heatmap, clone_by_bin = _pseudotime_bin_table(pseudotime_subset, focus_genes[:6], tissue_col)
    focus_de = _focused_de_heatmap(focus_subset, tissue_col=tissue_col) if focus_subset.n_obs else pd.DataFrame()
    focus_clone = _expanded_fraction_table(focus_subset, tissue_col, group_col) if focus_subset.n_obs else pd.DataFrame()

    hypothesis_text_block = "\n\n".join(
        [
            f"Executed hypothesis:\n{hypothesis_text or 'No executed hypothesis found.'}",
            f"Approved priority question:\n{approved_priority or 'No approved priority question found.'}",
            (
                "Approved plan steps:\n"
                + (result_context.get("approved_plan_steps", "No approved plan steps found.") or "No approved plan steps found.")
            ),
        ]
    )

    hyp_fig = plt.figure(figsize=(18, 22))
    hyp_gs = GridSpec(3, 2, figure=hyp_fig, hspace=0.36, wspace=0.24)

    ax_h1 = hyp_fig.add_subplot(hyp_gs[0, 0])
    plot_text_panel(ax_h1, "Plan-aligned hypothesis context", hypothesis_text_block)
    panel_label(ax_h1, "a")

    ax_h2 = hyp_fig.add_subplot(hyp_gs[0, 1])
    if _detect_plan_focus(result_context, ("diversity", "metastasis", "immune escape")) and not diversity_by_tissue.empty:
        _plot_tcr_diversity_by_tissue(ax_h2, diversity_by_tissue, tissue_col)
    elif focus_genes:
        sc.pl.umap(
            focus_subset if focus_subset.n_obs else adata,
            color=focus_genes[0],
            ax=ax_h2,
            show=False,
            frameon=False,
            title=f"{focus_genes[0]} expression in focused subset",
        )
    else:
        plot_categorical_embedding(ax_h2, focus_subset if focus_subset.n_obs else adata, color=group_col, title="Focused subset")
    panel_label(ax_h2, "b")

    ax_h3 = hyp_fig.add_subplot(hyp_gs[1, 0])
    if _detect_plan_focus(result_context, ("diversity", "metastasis", "immune escape")) and paired.n_obs:
        _plot_clone_size_distribution(ax_h3, paired, tissue_col)
    else:
        _plot_heatmap(ax_h3, focus_clone, "Expanded clone support in focused subset", cbar_label="expanded fraction")
    panel_label(ax_h3, "c")

    ax_h4 = hyp_fig.add_subplot(hyp_gs[1, 1])
    if not focus_de.empty:
        _plot_heatmap(ax_h4, focus_de, "Focused differential-expression support", cbar_label="log fold change", cmap="vlag")
    elif not trend_heatmap.empty:
        _plot_heatmap(ax_h4, trend_heatmap, "Marker dynamics along pseudotime", cbar_label="mean RNA", cmap="rocket")
    panel_label(ax_h4, "d")

    ax_h5 = hyp_fig.add_subplot(hyp_gs[2, 0])
    if pseudotime_subset is not None and "dpt_pseudotime" in pseudotime_subset.obs.columns:
        basis = "X_umap" if "X_umap" in pseudotime_subset.obsm else "X_pca"
        coords = pseudotime_subset.obsm[basis][:, :2]
        values = pseudotime_subset.obs["dpt_pseudotime"].astype(float).to_numpy()
        scatter = ax_h5.scatter(
            coords[:, 0],
            coords[:, 1],
            c=values,
            cmap="viridis",
            s=14,
            alpha=0.85,
            linewidths=0,
        )
        cbar = hyp_fig.colorbar(scatter, ax=ax_h5, fraction=0.046, pad=0.04)
        cbar.set_label("dpt_pseudotime")
        ax_h5.set_title(
            f"Focused T-cell { 'UMAP' if basis == 'X_umap' else 'PCA'} pseudotime ordering",
            fontsize=11,
        )
        ax_h5.set_xlabel("UMAP1" if basis == "X_umap" else "PC1")
        ax_h5.set_ylabel("UMAP2" if basis == "X_umap" else "PC2")
    else:
        clone_sharing_focus = _tissue_clonotype_sharing(focus_subset, tissue_col=tissue_col)
        _plot_heatmap(ax_h5, clone_sharing_focus, "Focused clonotype sharing", cbar_label="Jaccard overlap", cmap="crest")
    panel_label(ax_h5, "e")

    ax_h6 = hyp_fig.add_subplot(hyp_gs[2, 1])
    if not clone_by_bin.empty:
        _plot_heatmap(ax_h6, clone_by_bin.T, "Expanded clone fraction across pseudotime bins", cbar_label="expanded fraction", cmap="crest")
    elif not trend_heatmap.empty:
        _plot_heatmap(ax_h6, trend_heatmap, "Marker dynamics along pseudotime", cbar_label="mean RNA", cmap="rocket")
    else:
        focus_expr = pd.DataFrame()
        if focus_genes:
            focus_expr = expression_frame(focus_subset if focus_subset.n_obs else adata, focus_genes[:6], obs_columns=[tissue_col]).groupby(tissue_col, observed=False).mean()
        _plot_heatmap(ax_h6, focus_expr.T if not focus_expr.empty else pd.DataFrame(), "Focused marker expression by tissue", cbar_label="mean RNA")
    panel_label(ax_h6, "f")

    hyp_fig.suptitle("Hypothesis-driven scRNA + scTCR figure", fontsize=18, y=0.995)
    hyp_fig.tight_layout(rect=[0, 0, 1, 0.985])
    hyp_png, hyp_pdf, hyp_summary = save_figure_bundle(hyp_fig, output_dir, f"{figure_name}_hypothesis")

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
                f"Summary figure PNG: {png_path}",
                f"Summary figure PDF: {pdf_path}",
                f"Hypothesis figure PNG: {hyp_png}",
                f"Hypothesis figure PDF: {hyp_pdf}",
            ]
        ),
        encoding="utf-8",
    )
    hyp_summary.write_text(
        "\n".join(
            [
                f"Executed hypothesis: {hypothesis_text or 'none'}",
                f"Approved priority question: {approved_priority or 'none'}",
                f"Approved strategy feedback: {result_context.get('approved_strategy_feedback', '') or 'none'}",
                f"Focus genes: {', '.join(focus_genes) or 'none'}",
                f"Pseudotime used: {'yes' if pseudotime_subset is not None else 'no'}",
                "This figure is built from the approved plan, the executed hypothesis, and focused RNA+TCR evidence panels.",
            ]
        ),
        encoding="utf-8",
    )
    return FigureResult(png_path=png_path, pdf_path=pdf_path, summary_path=summary_path)
