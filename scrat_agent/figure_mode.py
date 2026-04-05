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


def _plot_fraction_bar(ax, series: pd.Series, title: str, color: str = "#9ecae1") -> None:
    if series.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.axis("off")
        return
    plotting = series.sort_values(ascending=True)
    plotting.plot(kind="barh", ax=ax, color=color, width=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Fraction")
    ax.set_ylabel("")


def _plot_numeric_embedding(ax, adata, values: pd.Series, title: str, basis: str = "X_umap") -> None:
    if basis not in adata.obsm:
        ax.text(0.5, 0.5, f"No {basis}", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.axis("off")
        return
    coords = adata.obsm[basis]
    frame = pd.DataFrame(coords[:, :2], columns=["x", "y"], index=adata.obs_names)
    frame["value"] = values.reindex(frame.index).astype(float).values
    scatter = ax.scatter(frame["x"], frame["y"], c=frame["value"], s=8, cmap="magma", rasterized=True)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)


def _peak_marker_matrix(adata_atac, groupby: str, top_n_per_group: int = 2) -> pd.DataFrame:
    if "rank_genes_groups" not in adata_atac.uns:
        return pd.DataFrame()
    groups = adata_atac.obs[groupby].astype(str).value_counts().index.tolist()
    rows: list[pd.DataFrame] = []
    for group in groups:
        try:
            frame = sc.get.rank_genes_groups_df(adata_atac, group=str(group)).head(top_n_per_group)
        except Exception:
            continue
        if frame.empty:
            continue
        rows.append(frame.assign(group=str(group)))
    if not rows:
        return pd.DataFrame()
    peak_df = pd.concat(rows, ignore_index=True)
    peak_df = peak_df.dropna(subset=["names"])
    return peak_df.pivot_table(index="names", columns="group", values="logfoldchanges", aggfunc="mean").fillna(0.0)


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
    approved_priority_question = result_context["approved_priority_question"]

    result_genes = extract_result_genes(
        available_genes=list(adata_rna.var_names) + [str(item) for item in adata_atac.uns.get("gene_activity_var_names", [])],
        texts=[
            hypothesis_text,
            approved_priority_question,
            result_context["approved_plan_steps"],
            result_context["notebook_text"],
            result_context["final_interpretation"],
        ],
        top_n=12,
    )

    candidate_genes = _candidate_joint_genes(adata_rna, adata_atac, hypothesis_text + "\n" + approved_priority_question)
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
    atac_activity_matrix = _gene_activity_group_matrix(adata_atac, candidate_genes[:16], atac_group)
    overlap = group_overlap_heatmap(
        adata_rna.obs.loc[adata_rna.obs_names.intersection(adata_atac.obs_names), rna_group],
        adata_atac.obs.loc[adata_rna.obs_names.intersection(adata_atac.obs_names), atac_group],
    )
    peak_marker_matrix = _peak_marker_matrix(adata_atac, atac_group)
    link_table = pd.DataFrame()
    try:
        link_table = summarize_rna_atac_link(adata_rna, adata_atac, rna_groupby=rna_group, atac_groupby=atac_group, genes=candidate_genes[:16])
    except Exception:
        link_table = pd.DataFrame()
    link_heatmap = (
        link_table.pivot_table(index="gene", columns="group", values="atac_mean", aggfunc="mean").fillna(0.0)
        if not link_table.empty
        else pd.DataFrame()
    )

    rna_composition = adata_rna.obs[rna_group].astype(str).value_counts(normalize=True).head(10)

    summary_fig = plt.figure(figsize=(20, 20), constrained_layout=True)
    gs = GridSpec(4, 2, figure=summary_fig, hspace=0.35, wspace=0.22)

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
    _plot_fraction_bar(ax_d, rna_composition, "scRNA cell-type composition")
    panel_label(ax_d, "d")

    ax_e = summary_fig.add_subplot(gs[2, 0])
    _plot_heatmap(ax_e, rna_marker_matrix.iloc[:16] if not rna_marker_matrix.empty else pd.DataFrame(), "Top scRNA marker heatmap")
    panel_label(ax_e, "e")

    ax_f = summary_fig.add_subplot(gs[2, 1])
    _plot_heatmap(ax_f, atac_activity_matrix.iloc[:16] if not atac_activity_matrix.empty else pd.DataFrame(), "ATAC gene-activity heatmap")
    panel_label(ax_f, "f")

    ax_g = summary_fig.add_subplot(gs[3, 0])
    _plot_heatmap(ax_g, overlap, "RNA cell type vs ATAC cell type overlap", cbar_label="row fraction")
    panel_label(ax_g, "g")

    ax_h = summary_fig.add_subplot(gs[3, 1])
    focus_link_heatmap = link_heatmap.iloc[:16] if not link_heatmap.empty else peak_marker_matrix.iloc[:16] if not peak_marker_matrix.empty else pd.DataFrame()
    focus_title = "Top RNA-ATAC support heatmap" if not link_heatmap.empty else "Top ATAC marker feature heatmap"
    focus_cbar = "mean ATAC" if not link_heatmap.empty else "log fold change"
    _plot_heatmap(ax_h, focus_link_heatmap, focus_title, cbar_label=focus_cbar)
    panel_label(ax_h, "h")

    summary_fig.suptitle("Standard scRNA + scATAC summary figure", fontsize=18)
    png_path, pdf_path, summary_path = save_figure_bundle(summary_fig, output_dir, figure_name)

    hypothesis_genes = result_genes[:6] or extract_hypothesis_genes(
        hypothesis_text + "\n" + approved_priority_question,
        list(adata_rna.var_names) + [str(item) for item in adata_atac.uns.get("gene_activity_var_names", [])],
        top_n=6,
    )
    if not hypothesis_genes:
        hypothesis_genes = candidate_genes[:4]
    hypothesis_genes = [gene for gene in hypothesis_genes if gene in candidate_genes][:6] or candidate_genes[:6]
    sample_group = "sample_key" if "sample_key" in adata_rna.obs.columns and adata_rna.obs["sample_key"].nunique() > 1 else rna_group
    hypothesis_rna = pd.DataFrame()
    hypothesis_atac = pd.DataFrame()
    if hypothesis_genes:
        hypothesis_rna = expression_frame(adata_rna, hypothesis_genes, obs_columns=[sample_group]).groupby(sample_group, observed=False).mean()
        hypothesis_atac = atac_signal_frame(adata_atac, genes=hypothesis_genes, obs_columns=[sample_group]).groupby(sample_group, observed=False).mean()
    hypothesis_link = pd.DataFrame()
    if hypothesis_genes:
        try:
            hypothesis_link = summarize_rna_atac_link(adata_rna, adata_atac, rna_groupby=sample_group, atac_groupby=sample_group, genes=hypothesis_genes)
        except Exception:
            hypothesis_link = pd.DataFrame()

    hyp_fig = plt.figure(figsize=(18, 16), constrained_layout=True)
    hyp_gs = GridSpec(3, 2, figure=hyp_fig, hspace=0.32, wspace=0.22)

    ax_h1 = hyp_fig.add_subplot(hyp_gs[0, 0])
    plot_text_panel(
        ax_h1,
        "Executed hypothesis",
        "\n\n".join(
            part
            for part in [
                hypothesis_text,
                f"Priority question: {approved_priority_question}" if approved_priority_question else "",
                result_context["approved_plan_steps"],
            ]
            if part
        ),
    )
    panel_label(ax_h1, "a")

    ax_h2 = hyp_fig.add_subplot(hyp_gs[0, 1])
    if joint_umap is not None and hypothesis_genes:
        plot_gene = hypothesis_genes[0]
        if plot_gene in joint_umap.var_names:
            sc.pl.umap(joint_umap, color=plot_gene, ax=ax_h2, show=False, frameon=False, title=f"Integrated RNA signal: {plot_gene}")
        else:
            plot_categorical_embedding(ax_h2, joint_umap, color=rna_group, title="Integrated RNA-ATAC UMAP")
    else:
        plot_categorical_embedding(ax_h2, adata_rna, color=rna_group, title="scRNA UMAP")
    panel_label(ax_h2, "b")

    ax_h3 = hyp_fig.add_subplot(hyp_gs[1, 0])
    if hypothesis_genes:
        signal_map = resolve_atac_signal_names(adata_atac, [hypothesis_genes[0]])
        if signal_map:
            signal_gene = next(iter(signal_map.values()))
            signal_frame = atac_signal_frame(adata_atac, genes=[signal_gene])
            _plot_numeric_embedding(ax_h3, adata_atac, signal_frame.iloc[:, 0], f"scATAC support: {signal_gene}")
        else:
            plot_categorical_embedding(ax_h3, adata_atac, color=atac_group, title="scATAC UMAP")
    else:
        plot_categorical_embedding(ax_h3, adata_atac, color=atac_group, title="scATAC UMAP")
    panel_label(ax_h3, "c")

    ax_h4 = hyp_fig.add_subplot(hyp_gs[1, 1])
    _plot_heatmap(ax_h4, hypothesis_rna.T if not hypothesis_rna.empty else pd.DataFrame(), "Hypothesis gene RNA expression", cbar_label="mean RNA")
    panel_label(ax_h4, "d")

    ax_h5 = hyp_fig.add_subplot(hyp_gs[2, 0])
    _plot_heatmap(ax_h5, hypothesis_atac.T if not hypothesis_atac.empty else pd.DataFrame(), "Hypothesis gene ATAC support", cbar_label="mean ATAC")
    panel_label(ax_h5, "e")

    ax_h6 = hyp_fig.add_subplot(hyp_gs[2, 1])
    link_matrix = (
        hypothesis_link.pivot_table(index="gene", columns="group", values="atac_mean", aggfunc="mean").fillna(0.0)
        if not hypothesis_link.empty
        else pd.DataFrame()
    )
    _plot_heatmap(ax_h6, link_matrix if not link_matrix.empty else pd.DataFrame(), "RNA-ATAC link support", cbar_label="mean ATAC")
    panel_label(ax_h6, "f")

    hyp_fig.suptitle("Hypothesis-driven scRNA + scATAC figure", fontsize=18)
    hyp_png, hyp_pdf, hyp_summary = save_figure_bundle(hyp_fig, output_dir, f"{figure_name}_hypothesis")

    summary_lines = [
        f"Figure name: {figure_name}",
        f"RNA input: {Path(rna_h5ad_path).resolve()}",
        f"ATAC input: {Path(atac_h5ad_path).resolve()}",
        f"Executed hypothesis: {hypothesis_text or 'none'}",
        f"Approved priority question: {approved_priority_question or 'none'}",
        f"Approved plan steps: {result_context['approved_plan_steps'] or 'none'}",
        f"Summary figure PNG: {png_path}",
        f"Summary figure PDF: {pdf_path}",
        f"Hypothesis figure PNG: {hyp_png}",
        f"Hypothesis figure PDF: {hyp_pdf}",
        f"Shared cells: {len(set(map(str, adata_rna.obs_names)).intersection(map(str, adata_atac.obs_names)))}",
        f"Candidate joint genes: {', '.join(candidate_genes[:16]) or 'none'}",
        f"Result-context genes: {', '.join(result_genes[:12]) or 'none'}",
        f"Hypothesis genes: {', '.join(hypothesis_genes) or 'none'}",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    hyp_summary.write_text(
        "\n".join(
            [
                f"Executed hypothesis: {hypothesis_text or 'none'}",
                f"Approved priority question: {approved_priority_question or 'none'}",
                f"Approved plan steps: {result_context['approved_plan_steps'] or 'none'}",
                f"Result-context genes: {', '.join(result_genes[:12]) or 'none'}",
                f"Hypothesis genes: {', '.join(hypothesis_genes) or 'none'}",
                f"Grouping variable: {sample_group}",
                "This figure emphasizes integrated RNA-ATAC structure, RNA/ATAC support, and hypothesis-linked gene panels.",
            ]
        ),
        encoding="utf-8",
    )
    return FigureResult(png_path=png_path, pdf_path=pdf_path, summary_path=summary_path)
