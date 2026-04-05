"""Publication-style figure builder for scRNA + spatial analysis."""

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
    ensure_display_column,
    extract_hypothesis_genes,
    extract_result_genes,
    panel_label,
    plot_categorical_embedding,
    plot_text_panel,
    rank_marker_matrix,
    read_run_result_context,
    save_figure_bundle,
    top_numeric_obs_columns,
)

from .notebook_tools import (
    expression_frame,
    get_spatial_mapping_method,
    resolve_gene_names,
    run_reference_mapping,
    safe_rank_genes_groups,
    spatial_coordinate_frame,
    spatial_domain_de,
)


@dataclass
class FigureResult:
    png_path: Path
    pdf_path: Path
    summary_path: Path


def _draw_spatial_categories(ax, adata_spatial, color: str, title: str) -> None:
    if "spatial" not in adata_spatial.obsm:
        plot_categorical_embedding(ax, adata_spatial, color=color, title=title)
        return
    coords = spatial_coordinate_frame(adata_spatial)
    coords[color] = adata_spatial.obs[color].astype(str).values
    categories = coords[color].astype(str).value_counts().index.tolist()
    palette = dict(zip(categories, sns.color_palette("tab20", max(len(categories), 3))))
    for category in categories:
        subset = coords.loc[coords[color].astype(str) == category]
        ax.scatter(
            subset["spatial_x"],
            subset["spatial_y"],
            s=8,
            alpha=0.85,
            color=palette[category],
            label=category,
            rasterized=True,
        )
    if len(categories) <= 14:
        for category, subset in coords.groupby(color, observed=False):
            row = subset[["spatial_x", "spatial_y"]].median()
            ax.text(row["spatial_x"], row["spatial_y"], str(category), fontsize=8, weight="bold")
    ax.invert_yaxis()
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("spatial_x")
    ax.set_ylabel("spatial_y")


def _draw_spatial_numeric(ax, adata_spatial, values: pd.Series, title: str) -> None:
    coords = spatial_coordinate_frame(adata_spatial)
    coords["value"] = values.reindex(coords.index).astype(float).values
    scatter = ax.scatter(coords["spatial_x"], coords["spatial_y"], c=coords["value"], s=8, cmap="magma", rasterized=True)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("spatial_x")
    ax.set_ylabel("spatial_y")
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)


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


def _plot_fraction_bar(ax, series: pd.Series, title: str, color: str = "#6baed6") -> None:
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


def build_publication_figure(
    *,
    rna_h5ad_path: str | Path,
    spatial_h5ad_path: str | Path,
    output_dir: str | Path,
    figure_name: str = "scrst_publication_figure",
    mapping_method: str = "auto",
) -> FigureResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    sc.settings.set_figure_params(dpi=120, facecolor="white", frameon=False)

    adata_rna = sc.read_h5ad(rna_h5ad_path)
    adata_spatial = sc.read_h5ad(spatial_h5ad_path)
    rna_group = ensure_display_column(adata_rna, "cell_type", ("cluster_cell_type", "annotation", "leiden"))
    domain_col = ensure_display_column(adata_spatial, "spatial_domain", ("cluster_cell_type", "annotation", "leiden"))
    result_context = read_run_result_context(output_dir)
    hypothesis_text = result_context["executed_hypothesis"]
    approved_priority_question = result_context["approved_priority_question"]

    result_genes = extract_result_genes(
        available_genes=adata_spatial.var_names,
        texts=[
            hypothesis_text,
            approved_priority_question,
            result_context["approved_plan_steps"],
            result_context["notebook_text"],
            result_context["final_interpretation"],
        ],
        top_n=12,
    )

    effective_mapping_method = get_spatial_mapping_method(mapping_method)
    marker_scores = run_reference_mapping(adata_rna, adata_spatial, cell_type_col=rna_group, top_n=15, method=effective_mapping_method)
    score_cols = marker_scores["score_name"].tolist() if not marker_scores.empty else top_numeric_obs_columns(adata_spatial, prefix="sig_", top_n=8)

    rna_marker_matrix = rank_marker_matrix(adata_rna, groupby=rna_group, top_n_per_group=2)
    try:
        domain_de = spatial_domain_de(adata_spatial, domain_col=domain_col, top_n=4)
    except Exception:
        try:
            ranked = safe_rank_genes_groups(adata_spatial, groupby=domain_col, min_cells_per_group=10, key_added="spatial_domain_markers")
            domain_de = pd.concat(
                [
                    sc.get.rank_genes_groups_df(ranked, group=str(group), key="spatial_domain_markers").head(4).assign(domain=str(group))
                    for group in ranked.obs[domain_col].cat.categories
                ],
                ignore_index=True,
            )
        except Exception:
            domain_de = pd.DataFrame()

    domain_marker_matrix = (
        domain_de.pivot_table(index="names", columns="domain", values="logfoldchanges", aggfunc="mean").fillna(0.0)
        if not domain_de.empty
        else pd.DataFrame()
    )

    mapping_heatmap = pd.DataFrame()
    dominant_mapping = pd.Series(dtype="object")
    if score_cols:
        mapping_heatmap = adata_spatial.obs.groupby(domain_col, observed=False)[score_cols].mean(numeric_only=True).T
        mapping_heatmap.index = [name.replace("sig_", "").replace("c2l_", "") for name in mapping_heatmap.index]
        dominant_mapping = adata_spatial.obs[score_cols].idxmax(axis=1).astype(str).str.replace("sig_", "", regex=False).str.replace("c2l_", "", regex=False)
        adata_spatial.obs["_dominant_mapping"] = dominant_mapping.values

    rna_composition = adata_rna.obs[rna_group].astype(str).value_counts(normalize=True).head(10)

    summary_fig = plt.figure(figsize=(20, 20), constrained_layout=True)
    gs = GridSpec(4, 2, figure=summary_fig, hspace=0.35, wspace=0.22)

    ax_a = summary_fig.add_subplot(gs[0, 0])
    plot_categorical_embedding(ax_a, adata_rna, color=rna_group, title="scRNA UMAP with annotated cell types")
    panel_label(ax_a, "a")

    ax_b = summary_fig.add_subplot(gs[0, 1])
    _plot_fraction_bar(ax_b, rna_composition, "scRNA cell-type composition")
    panel_label(ax_b, "b")

    ax_c = summary_fig.add_subplot(gs[1, 0])
    _draw_spatial_categories(ax_c, adata_spatial, domain_col, "Spatial domains")
    panel_label(ax_c, "c")

    ax_d = summary_fig.add_subplot(gs[1, 1])
    _plot_heatmap(
        ax_d,
        mapping_heatmap.iloc[:16] if not mapping_heatmap.empty else pd.DataFrame(),
        "Reference-to-spatial mapping heatmap",
        cbar_label="mean score",
    )
    panel_label(ax_d, "d")

    ax_e = summary_fig.add_subplot(gs[2, 0])
    _plot_heatmap(ax_e, rna_marker_matrix.iloc[:16] if not rna_marker_matrix.empty else pd.DataFrame(), "Top scRNA marker heatmap")
    panel_label(ax_e, "e")

    ax_f = summary_fig.add_subplot(gs[2, 1])
    _plot_heatmap(
        ax_f,
        domain_marker_matrix.iloc[:16] if not domain_marker_matrix.empty else pd.DataFrame(),
        "Top spatial domain marker heatmap",
        cbar_label="log fold change",
    )
    panel_label(ax_f, "f")

    ax_g = summary_fig.add_subplot(gs[3, 0])
    if not dominant_mapping.empty:
        _draw_spatial_categories(ax_g, adata_spatial, "_dominant_mapping", "Dominant mapped cell type across spots")
    elif score_cols:
        _draw_spatial_numeric(ax_g, adata_spatial, adata_spatial.obs[score_cols[0]], f"Spatial map: {score_cols[0].replace('sig_', '').replace('c2l_', '')}")
    else:
        ax_g.text(0.5, 0.5, "No mapping score available", ha="center", va="center", transform=ax_g.transAxes)
        ax_g.axis("off")
    panel_label(ax_g, "g")

    ax_h = summary_fig.add_subplot(gs[3, 1])
    resolved = resolve_gene_names(adata_spatial, result_genes[:4] or ["COL1A1", "CXCL13", "KRT19", "NKG7"])
    if resolved:
        gene = next(iter(resolved.values()))
        gene_frame = expression_frame(adata_spatial, [gene])
        _draw_spatial_numeric(ax_h, adata_spatial, gene_frame[gene], f"Spatial expression: {gene}")
    else:
        ax_h.text(0.5, 0.5, "No focus gene available", ha="center", va="center", transform=ax_h.transAxes)
        ax_h.axis("off")
    panel_label(ax_h, "h")

    summary_fig.suptitle("Standard scRNA + spatial summary figure", fontsize=18)
    png_path, pdf_path, summary_path = save_figure_bundle(summary_fig, output_dir, figure_name)

    hypothesis_genes = result_genes[:6] or extract_hypothesis_genes(hypothesis_text + "\n" + approved_priority_question, adata_spatial.var_names, top_n=6)
    if not hypothesis_genes:
        defaults = ["COL1A1", "CXCL13", "NKG7", "KRT19", "XBP1", "ACTA2"]
        hypothesis_genes = [gene for gene in defaults if gene in adata_spatial.var_names][:4]
    hyp_spatial_matrix = pd.DataFrame()
    if hypothesis_genes:
        hyp_spatial_matrix = expression_frame(adata_spatial, hypothesis_genes, obs_columns=[domain_col]).groupby(domain_col, observed=False).mean()

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
    if hypothesis_genes:
        gene = hypothesis_genes[0]
        gene_frame = expression_frame(adata_spatial, [gene])
        _draw_spatial_numeric(ax_h2, adata_spatial, gene_frame[gene], f"Spatial expression: {gene}")
    else:
        _draw_spatial_categories(ax_h2, adata_spatial, domain_col, "Spatial domains")
    panel_label(ax_h2, "b")

    ax_h3 = hyp_fig.add_subplot(hyp_gs[1, 0])
    if len(hypothesis_genes) > 1:
        gene = hypothesis_genes[1]
        gene_frame = expression_frame(adata_spatial, [gene])
        _draw_spatial_numeric(ax_h3, adata_spatial, gene_frame[gene], f"Spatial expression: {gene}")
    elif not dominant_mapping.empty:
        _draw_spatial_categories(ax_h3, adata_spatial, "_dominant_mapping", "Dominant mapped cell type across spots")
    else:
        ax_h3.text(0.5, 0.5, "No second focus panel", ha="center", va="center", transform=ax_h3.transAxes)
        ax_h3.axis("off")
    panel_label(ax_h3, "c")

    ax_h4 = hyp_fig.add_subplot(hyp_gs[1, 1])
    _plot_heatmap(
        ax_h4,
        hyp_spatial_matrix.T if not hyp_spatial_matrix.empty else pd.DataFrame(),
        "Hypothesis gene expression by spatial domain",
        cbar_label="mean expression",
    )
    panel_label(ax_h4, "d")

    ax_h5 = hyp_fig.add_subplot(hyp_gs[2, 0])
    focused_mapping = mapping_heatmap
    if not focused_mapping.empty and len(focused_mapping) > 12:
        keep_rows = focused_mapping.mean(axis=1).sort_values(ascending=False).head(12).index
        focused_mapping = focused_mapping.loc[keep_rows]
    _plot_heatmap(ax_h5, focused_mapping if not focused_mapping.empty else pd.DataFrame(), "Mapping support by spatial domain", cbar_label="mean score")
    panel_label(ax_h5, "e")

    ax_h6 = hyp_fig.add_subplot(hyp_gs[2, 1])
    _plot_heatmap(
        ax_h6,
        domain_marker_matrix.iloc[:12] if not domain_marker_matrix.empty else pd.DataFrame(),
        "Spatial domain marker heatmap",
        cbar_label="log fold change",
    )
    panel_label(ax_h6, "f")

    hyp_fig.suptitle("Hypothesis-driven scRNA + spatial figure", fontsize=18)
    hyp_png, hyp_pdf, hyp_summary = save_figure_bundle(hyp_fig, output_dir, f"{figure_name}_hypothesis")

    summary_lines = [
        f"Figure name: {figure_name}",
        f"RNA input: {Path(rna_h5ad_path).resolve()}",
        f"Spatial input: {Path(spatial_h5ad_path).resolve()}",
        f"Executed hypothesis: {hypothesis_text or 'none'}",
        f"Approved priority question: {approved_priority_question or 'none'}",
        f"Approved plan steps: {result_context['approved_plan_steps'] or 'none'}",
        f"Spatial mapping method: {effective_mapping_method}",
        f"Summary figure PNG: {png_path}",
        f"Summary figure PDF: {pdf_path}",
        f"Hypothesis figure PNG: {hyp_png}",
        f"Hypothesis figure PDF: {hyp_pdf}",
        f"Score columns: {', '.join(score_cols) or 'none'}",
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
                f"Grouping variable: {domain_col}",
                "This figure emphasizes spatial domains, mapping support, and spatial expression aligned with the executed hypothesis and approved plan.",
            ]
        ),
        encoding="utf-8",
    )
    return FigureResult(png_path=png_path, pdf_path=pdf_path, summary_path=summary_path)
