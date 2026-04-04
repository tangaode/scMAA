"""Publication-style figure builder for scRNA + scTCR analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    rank_marker_matrix,
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


def _plot_heatmap(ax, table: pd.DataFrame, title: str, cbar_label: str = "mean signal") -> None:
    if table.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.axis("off")
        return
    sns.heatmap(table, cmap="mako", linewidths=0.3, linecolor="white", ax=ax, cbar_kws={"label": cbar_label})
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("")


def _v_gene_usage_heatmap(adata, tissue_col: str, top_n: int = 10) -> pd.DataFrame:
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
    result_genes = extract_result_genes(
        available_genes=adata.var_names,
        texts=[
            hypothesis_text,
            result_context["notebook_text"],
            result_context["final_interpretation"],
        ],
        top_n=10,
    )

    paired = paired_tcr_subset(adata)
    rna_marker_matrix = rank_marker_matrix(adata, groupby=group_col, top_n_per_group=2)
    expansion_heatmap = (
        paired.obs.groupby([group_col, tissue_col], observed=False)["expanded_clone"].mean().unstack(fill_value=0.0)
        if paired.n_obs
        else pd.DataFrame()
    )
    v_gene_heatmap = _v_gene_usage_heatmap(paired, tissue_col=tissue_col)

    summary_fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(3, 2, figure=summary_fig, hspace=0.35, wspace=0.22)

    ax_a = summary_fig.add_subplot(gs[0, 0])
    plot_categorical_embedding(ax_a, adata, color=group_col, title="scRNA UMAP with annotated cell types")
    panel_label(ax_a, "a")

    ax_b = summary_fig.add_subplot(gs[0, 1])
    plot_categorical_embedding(ax_b, paired if paired.n_obs else adata, color="expanded_label", title="Paired cells: clonal expansion overlay")
    panel_label(ax_b, "b")

    ax_c = summary_fig.add_subplot(gs[1, 0])
    plot_categorical_embedding(ax_c, paired if paired.n_obs else adata, color=tissue_col, title="Paired cells: tissue or sample context")
    panel_label(ax_c, "c")

    ax_d = summary_fig.add_subplot(gs[1, 1])
    _plot_heatmap(ax_d, rna_marker_matrix.iloc[:16] if not rna_marker_matrix.empty else pd.DataFrame(), "Top scRNA marker heatmap")
    panel_label(ax_d, "d")

    ax_e = summary_fig.add_subplot(gs[2, 0])
    _plot_heatmap(ax_e, expansion_heatmap, "Expanded clone fraction by cell type and tissue", cbar_label="expanded fraction")
    panel_label(ax_e, "e")

    ax_f = summary_fig.add_subplot(gs[2, 1])
    _plot_heatmap(ax_f, v_gene_heatmap, "Top V gene usage across tissues", cbar_label="column fraction")
    panel_label(ax_f, "f")

    summary_fig.suptitle("Standard scRNA + scTCR summary figure", fontsize=18, y=0.99)
    summary_fig.tight_layout(rect=[0, 0, 1, 0.97])
    png_path, pdf_path, summary_path = save_figure_bundle(summary_fig, output_dir, figure_name)

    hypothesis_genes = result_genes[:6] or extract_hypothesis_genes(hypothesis_text, adata.var_names, top_n=6)
    if not hypothesis_genes:
        defaults = ["CCL5", "NKG7", "GZMB", "PDCD1", "TIGIT", "CXCL13", "XBP1"]
        hypothesis_genes = [gene for gene in defaults if gene in adata.var_names][:4]
    hyp_group = tissue_col if tissue_col in adata.obs.columns and adata.obs[tissue_col].nunique() > 1 else "expanded_label"
    hyp_expr = pd.DataFrame()
    if hypothesis_genes:
        hyp_expr = expression_frame(paired if paired.n_obs else adata, hypothesis_genes, obs_columns=[hyp_group]).groupby(hyp_group, observed=False).mean()

    hyp_de = pd.DataFrame()
    if "expanded_clone" in paired.obs.columns and paired.obs["expanded_clone"].nunique() > 1:
        try:
            hyp_de = tissue_stratified_expansion_de(tumor_like_subset(adata), top_n=4)
        except Exception:
            hyp_de = pd.DataFrame()

    hyp_fig = plt.figure(figsize=(16, 12))
    hyp_gs = GridSpec(2, 2, figure=hyp_fig, hspace=0.3, wspace=0.22)
    ax_h1 = hyp_fig.add_subplot(hyp_gs[0, 0])
    plot_text_panel(ax_h1, "Executed hypothesis", hypothesis_text)
    panel_label(ax_h1, "a")

    ax_h2 = hyp_fig.add_subplot(hyp_gs[0, 1])
    if hypothesis_genes:
        sc.pl.umap(adata, color=hypothesis_genes[0], ax=ax_h2, show=False, frameon=False, title=f"{hypothesis_genes[0]} expression")
    else:
        plot_categorical_embedding(ax_h2, adata, color=group_col, title="UMAP")
    panel_label(ax_h2, "b")

    ax_h3 = hyp_fig.add_subplot(hyp_gs[1, 0])
    _plot_heatmap(ax_h3, hyp_expr.T if not hyp_expr.empty else pd.DataFrame(), "Hypothesis gene expression", cbar_label="mean RNA")
    panel_label(ax_h3, "c")

    ax_h4 = hyp_fig.add_subplot(hyp_gs[1, 1])
    if not hyp_de.empty:
        de_table = hyp_de.pivot_table(index="names", columns="tissue", values="logfoldchanges", aggfunc="mean").fillna(0.0)
        _plot_heatmap(ax_h4, de_table, "Hypothesis-related DE support", cbar_label="log fold change")
    else:
        clone_table = clone_expansion_table(adata, groupby=hyp_group if hyp_group in adata.obs.columns else tissue_col)
        clone_display = clone_table.set_index(hyp_group if hyp_group in clone_table.columns else clone_table.columns[0])[["expanded_fraction"]]
        _plot_heatmap(ax_h4, clone_display.T if not clone_display.empty else pd.DataFrame(), "Clone expansion support", cbar_label="expanded fraction")
    panel_label(ax_h4, "d")

    hyp_fig.suptitle("Hypothesis-driven scRNA + scTCR figure", fontsize=18, y=0.99)
    hyp_fig.tight_layout(rect=[0, 0, 1, 0.97])
    hyp_png, hyp_pdf, hyp_summary = save_figure_bundle(hyp_fig, output_dir, f"{figure_name}_hypothesis")

    summary_lines = [
        f"Figure name: {figure_name}",
        f"RNA input: {Path(rna_h5ad_path).resolve()}",
        f"TCR input: {Path(tcr_path).resolve()}",
        f"Executed hypothesis: {hypothesis_text or 'none'}",
        f"Summary figure PNG: {png_path}",
        f"Summary figure PDF: {pdf_path}",
        f"Hypothesis figure PNG: {hyp_png}",
        f"Hypothesis figure PDF: {hyp_pdf}",
        f"Paired cells: {paired.n_obs}",
        f"Result-context genes: {', '.join(result_genes[:12]) or 'none'}",
        f"Hypothesis genes: {', '.join(hypothesis_genes) or 'none'}",
        "",
        "Top RNA marker genes:",
        ", ".join(rna_marker_matrix.index.tolist()[:16]) if not rna_marker_matrix.empty else "none",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    hyp_summary.write_text(
        "\n".join(
            [
                f"Executed hypothesis: {hypothesis_text or 'none'}",
                f"Result-context genes: {', '.join(result_genes[:12]) or 'none'}",
                f"Hypothesis genes: {', '.join(hypothesis_genes) or 'none'}",
                f"Grouping variable: {hyp_group}",
                "This figure is intended to test the executed hypothesis using RNA expression and clonal support panels.",
            ]
        ),
        encoding="utf-8",
    )
    return FigureResult(png_path=png_path, pdf_path=pdf_path, summary_path=summary_path)
