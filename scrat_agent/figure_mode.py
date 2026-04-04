"""Publication-style figure builder for scRNA + scATAC analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import scanpy as sc
import seaborn as sns

from scrt_agent.figure_common import (
    build_joint_umap_from_gene_signals,
    ensure_display_column,
    extract_hypothesis_genes,
    extract_result_genes,
    group_overlap_heatmap,
    panel_label,
    plot_categorical_embedding,
    plot_text_panel,
    rank_marker_matrix,
    read_run_result_context,
    save_figure_bundle,
)

from .notebook_tools import (
    atac_signal_frame,
    expression_frame,
    resolve_atac_signal_names,
    resolve_gene_names,
    summarize_rna_atac_link,
)


@dataclass
class FigureResult:
    png_path: Path
    pdf_path: Path
    summary_path: Path


def _candidate_joint_genes(adata_rna, adata_atac, hypothesis_text: str) -> list[str]:
    atac_gene_space = [str(item) for item in adata_atac.uns.get("gene_activity_var_names", [])]
    if not atac_gene_space:
        atac_gene_space = [str(item) for item in adata_atac.var_names]
    shared = list(pd.Index(adata_rna.var_names).intersection(atac_gene_space))
    rna_markers = rank_marker_matrix(adata_rna, groupby="cell_type", top_n_per_group=2)
    marker_genes = rna_markers.index.tolist() if not rna_markers.empty else []
    hypothesis_genes = extract_hypothesis_genes(hypothesis_text, shared, top_n=6)
    defaults = ["COL1A1", "RUNX2", "CXCL13", "CCL5", "NKG7", "PDCD1", "KRT17", "ACTA2", "FN1"]
    genes = []
    for gene in hypothesis_genes + marker_genes + defaults + shared[:40]:
        if gene in shared and gene not in genes:
            genes.append(gene)
    return genes[:40]


def _gene_activity_group_matrix(adata_atac, genes: list[str], groupby: str) -> pd.DataFrame:
    if not genes:
        return pd.DataFrame()
    frame = atac_signal_frame(adata_atac, genes=genes, obs_columns=[groupby])
    value_cols = [col for col in frame.columns if col != groupby]
    if not value_cols:
        return pd.DataFrame()
    grouped = frame.groupby(groupby, observed=False)[value_cols].mean()
    return grouped[value_cols].T


def _plot_heatmap(ax, table: pd.DataFrame, title: str, cbar_label: str = "mean signal") -> None:
    if table.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        return
    sns.heatmap(table, cmap="mako", linewidths=0.3, linecolor="white", ax=ax, cbar_kws={"label": cbar_label})
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("")


def build_publication_figure(
    *,
    rna_h5ad_path: str | Path,
    atac_h5ad_path: str | Path,
    output_dir: str | Path,
    figure_name: str = "scrat_publication_figure",
) -> FigureResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    sc.settings.set_figure_params(dpi=120, facecolor="white", frameon=False)

    adata_rna = sc.read_h5ad(rna_h5ad_path)
    adata_atac = sc.read_h5ad(atac_h5ad_path)
    rna_group = ensure_display_column(adata_rna, "cell_type", ("cluster_cell_type", "annotation", "leiden"))
    atac_group = ensure_display_column(adata_atac, "cell_type", ("cluster_cell_type", "annotation", "leiden"))
    result_context = read_run_result_context(output_dir)
    hypothesis_text = result_context["executed_hypothesis"]
    result_genes = extract_result_genes(
        available_genes=list(adata_rna.var_names) + [str(item) for item in adata_atac.uns.get("gene_activity_var_names", [])],
        texts=[
            hypothesis_text,
            result_context["notebook_text"],
            result_context["final_interpretation"],
        ],
        top_n=10,
    )

    candidate_genes = _candidate_joint_genes(adata_rna, adata_atac, hypothesis_text)
    for gene in result_genes:
        if gene not in candidate_genes:
            candidate_genes.insert(0, gene)
    candidate_genes = list(dict.fromkeys(candidate_genes))[:40]
    joint_umap = build_joint_umap_from_gene_signals(
        adata_rna,
        adata_atac,
        expression_getter=expression_frame,
        other_signal_getter=atac_signal_frame,
        candidate_genes=candidate_genes,
        max_genes=30,
    )

    rna_marker_matrix = rank_marker_matrix(adata_rna, groupby=rna_group, top_n_per_group=2)
    if not rna_marker_matrix.empty and len(candidate_genes) < 12:
        for gene in rna_marker_matrix.index.tolist():
            if gene not in candidate_genes:
                candidate_genes.append(gene)

    atac_activity_matrix = _gene_activity_group_matrix(adata_atac, candidate_genes[:16], atac_group)
    overlap = group_overlap_heatmap(
        adata_rna.obs.loc[adata_rna.obs_names.intersection(adata_atac.obs_names), rna_group],
        adata_atac.obs.loc[adata_rna.obs_names.intersection(adata_atac.obs_names), atac_group],
    )

    summary_fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(3, 2, figure=summary_fig, hspace=0.35, wspace=0.22)

    ax_a = summary_fig.add_subplot(gs[0, 0])
    plot_categorical_embedding(ax_a, adata_rna, color=rna_group, title="scRNA UMAP with annotated cell types")
    panel_label(ax_a, "a")

    ax_b = summary_fig.add_subplot(gs[0, 1])
    plot_categorical_embedding(ax_b, adata_atac, color=atac_group, title="scATAC UMAP with annotated cell types")
    panel_label(ax_b, "b")

    ax_c = summary_fig.add_subplot(gs[1, 0])
    if joint_umap is not None:
        plot_categorical_embedding(ax_c, joint_umap, color=rna_group, title="Integrated RNA-ATAC UMAP")
    else:
        ax_c.text(0.5, 0.5, "Integrated UMAP unavailable", ha="center", va="center", transform=ax_c.transAxes)
        ax_c.set_title("Integrated RNA-ATAC UMAP")
        ax_c.axis("off")
    panel_label(ax_c, "c")

    ax_d = summary_fig.add_subplot(gs[1, 1])
    _plot_heatmap(ax_d, rna_marker_matrix.iloc[:16] if not rna_marker_matrix.empty else pd.DataFrame(), "Top scRNA marker heatmap")
    panel_label(ax_d, "d")

    ax_e = summary_fig.add_subplot(gs[2, 0])
    _plot_heatmap(ax_e, atac_activity_matrix.iloc[:16] if not atac_activity_matrix.empty else pd.DataFrame(), "ATAC gene-activity heatmap")
    panel_label(ax_e, "e")

    ax_f = summary_fig.add_subplot(gs[2, 1])
    _plot_heatmap(ax_f, overlap, "RNA cell type vs ATAC cell type overlap", cbar_label="row fraction")
    panel_label(ax_f, "f")

    summary_fig.suptitle("Standard scRNA + scATAC summary figure", fontsize=18, y=0.99)
    summary_fig.tight_layout(rect=[0, 0, 1, 0.97])
    png_path, pdf_path, summary_path = save_figure_bundle(summary_fig, output_dir, figure_name)

    hypothesis_genes = result_genes[:6] or extract_hypothesis_genes(
        hypothesis_text,
        list(adata_rna.var_names) + [str(item) for item in adata_atac.uns.get("gene_activity_var_names", [])],
        top_n=6,
    )
    if not hypothesis_genes:
        hypothesis_genes = candidate_genes[:4]
    hypothesis_genes = [gene for gene in hypothesis_genes if gene in candidate_genes][:4] or candidate_genes[:4]
    sample_group = "sample_key" if "sample_key" in adata_rna.obs.columns and adata_rna.obs["sample_key"].nunique() > 1 else rna_group
    hypothesis_rna = pd.DataFrame()
    hypothesis_atac = pd.DataFrame()
    if hypothesis_genes:
        hypothesis_rna = expression_frame(adata_rna, hypothesis_genes, obs_columns=[sample_group]).groupby(sample_group, observed=False).mean()
        hypothesis_atac = atac_signal_frame(adata_atac, genes=hypothesis_genes, obs_columns=[sample_group]).groupby(sample_group, observed=False).mean()
    link_table = pd.DataFrame()
    if hypothesis_genes:
        try:
            link_table = summarize_rna_atac_link(adata_rna, adata_atac, rna_groupby=sample_group, atac_groupby=sample_group, genes=hypothesis_genes)
        except Exception:
            link_table = pd.DataFrame()

    hyp_fig = plt.figure(figsize=(16, 12))
    hyp_gs = GridSpec(2, 2, figure=hyp_fig, hspace=0.3, wspace=0.22)
    ax_h1 = hyp_fig.add_subplot(hyp_gs[0, 0])
    plot_text_panel(ax_h1, "Executed hypothesis", hypothesis_text)
    panel_label(ax_h1, "a")

    ax_h2 = hyp_fig.add_subplot(hyp_gs[0, 1])
    if joint_umap is not None and hypothesis_genes:
        plot_gene = hypothesis_genes[0]
        sc.pl.umap(joint_umap, color=plot_gene if plot_gene in joint_umap.var_names else rna_group, ax=ax_h2, show=False, frameon=False, title=f"Integrated view: {plot_gene}")
    elif joint_umap is not None:
        plot_categorical_embedding(ax_h2, joint_umap, color=rna_group, title="Integrated RNA-ATAC UMAP")
    else:
        ax_h2.text(0.5, 0.5, "Integrated UMAP unavailable", ha="center", va="center", transform=ax_h2.transAxes)
        ax_h2.axis("off")
    panel_label(ax_h2, "b")

    ax_h3 = hyp_fig.add_subplot(hyp_gs[1, 0])
    _plot_heatmap(ax_h3, hypothesis_rna.T if not hypothesis_rna.empty else pd.DataFrame(), "Hypothesis gene RNA expression", cbar_label="mean RNA")
    panel_label(ax_h3, "c")

    ax_h4 = hyp_fig.add_subplot(hyp_gs[1, 1])
    if not link_table.empty:
        link_heatmap = link_table.pivot_table(index="gene", columns="group", values="atac_mean", aggfunc="mean")
        _plot_heatmap(ax_h4, link_heatmap, "Hypothesis gene ATAC support", cbar_label="mean ATAC")
    else:
        _plot_heatmap(ax_h4, hypothesis_atac.T if not hypothesis_atac.empty else pd.DataFrame(), "Hypothesis gene ATAC support", cbar_label="mean ATAC")
    panel_label(ax_h4, "d")

    hyp_fig.suptitle("Hypothesis-driven scRNA + scATAC figure", fontsize=18, y=0.99)
    hyp_fig.tight_layout(rect=[0, 0, 1, 0.97])
    hyp_png, hyp_pdf, hyp_summary = save_figure_bundle(hyp_fig, output_dir, f"{figure_name}_hypothesis")

    summary_lines = [
        f"Figure name: {figure_name}",
        f"RNA input: {Path(rna_h5ad_path).resolve()}",
        f"ATAC input: {Path(atac_h5ad_path).resolve()}",
        f"Executed hypothesis: {hypothesis_text or 'none'}",
        f"Summary figure PNG: {png_path}",
        f"Summary figure PDF: {pdf_path}",
        f"Hypothesis figure PNG: {hyp_png}",
        f"Hypothesis figure PDF: {hyp_pdf}",
        f"Shared cells: {len(set(map(str, adata_rna.obs_names)).intersection(map(str, adata_atac.obs_names)))}",
        f"Candidate joint genes: {', '.join(candidate_genes[:16]) or 'none'}",
        f"Result-context genes: {', '.join(result_genes[:12]) or 'none'}",
        f"Hypothesis genes: {', '.join(hypothesis_genes) or 'none'}",
        "",
        "RNA marker heatmap genes:",
        ", ".join(rna_marker_matrix.index.tolist()[:16]) if not rna_marker_matrix.empty else "none",
        "",
        "ATAC activity heatmap genes:",
        ", ".join(atac_activity_matrix.index.tolist()[:16]) if not atac_activity_matrix.empty else "none",
        "",
        "RNA-ATAC link table:",
        link_table.to_string(index=False) if not link_table.empty else "No RNA-ATAC link table available.",
        "",
        f"Hypothesis figure summary placeholder: {hyp_summary}",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    hyp_summary.write_text(
        "\n".join(
            [
                f"Executed hypothesis: {hypothesis_text or 'none'}",
                f"Result-context genes: {', '.join(result_genes[:12]) or 'none'}",
                f"Hypothesis genes: {', '.join(hypothesis_genes) or 'none'}",
                f"Grouping variable: {sample_group}",
                "This figure is intended to test the executed hypothesis using integrated UMAP plus RNA and ATAC support panels.",
            ]
        ),
        encoding="utf-8",
    )
    return FigureResult(png_path=png_path, pdf_path=pdf_path, summary_path=summary_path)
