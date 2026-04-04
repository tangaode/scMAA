"""Shared figure helpers for multi-omics summary and hypothesis-driven plots."""

from __future__ import annotations

import json
from pathlib import Path
import re
import textwrap

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns


DEFAULT_PALETTE = sns.color_palette("tab20", 20)
NOISY_GENE_TOKENS = {
    "PDF",
    "PNG",
    "JPG",
    "JPEG",
    "TSV",
    "CSV",
    "TXT",
    "JSON",
    "H5AD",
    "IPYNB",
    "FIGURE",
    "SUMMARY",
    "ANALYSIS",
    "RUN",
    "PATH",
    "NOTEBOOK",
}


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )


def ensure_display_column(adata, preferred: str = "cell_type", fallbacks: tuple[str, ...] = ("cluster_cell_type", "annotation", "leiden")) -> str:
    if preferred in adata.obs.columns:
        return preferred
    for column in fallbacks:
        if column in adata.obs.columns:
            adata.obs[preferred] = adata.obs[column].astype(str)
            return preferred
    adata.obs[preferred] = "Unknown"
    return preferred


def read_executed_hypothesis(figure_output_dir: str | Path) -> str:
    run_dir = Path(figure_output_dir).resolve().parent
    hypothesis_path = run_dir / "executed_hypotheses.txt"
    if not hypothesis_path.exists():
        return ""
    text = hypothesis_path.read_text(encoding="utf-8").strip()
    if not text:
        return ""
    first_line = text.splitlines()[0]
    if ":" in first_line:
        return first_line.split(":", 1)[1].strip()
    return first_line.strip()


def _collect_notebook_text(notebook_path: Path) -> str:
    try:
        payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    parts: list[str] = []
    for cell in payload.get("cells", []):
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown" and source.strip():
            parts.append(source)
        if cell.get("cell_type") == "code":
            for output in cell.get("outputs", []):
                output_type = output.get("output_type")
                if output_type == "stream":
                    text = str(output.get("text", ""))
                    if text.strip():
                        parts.append(text)
                elif output_type == "execute_result":
                    text = str(output.get("data", {}).get("text/plain", ""))
                    if text.strip():
                        parts.append(text)
                elif output_type == "display_data":
                    text = str(output.get("data", {}).get("text/plain", ""))
                    if text.strip():
                        parts.append(text)
                elif output_type == "error":
                    text = f"{output.get('ename', '')}: {output.get('evalue', '')}"
                    if text.strip():
                        parts.append(text)
    return "\n".join(parts)


def read_run_result_context(figure_output_dir: str | Path) -> dict[str, str]:
    run_dir = Path(figure_output_dir).resolve().parent
    context = {
        "executed_hypothesis": read_executed_hypothesis(figure_output_dir),
        "run_summary": "",
        "notebook_text": "",
        "final_interpretation": "",
    }
    run_summary_path = run_dir / "run_summary.txt"
    if run_summary_path.exists():
        try:
            context["run_summary"] = run_summary_path.read_text(encoding="utf-8")
        except Exception:
            context["run_summary"] = ""
        if "Final interpretation:" in context["run_summary"]:
            context["final_interpretation"] = context["run_summary"].split("Final interpretation:", 1)[1][:2500].strip()
    notebooks = sorted(run_dir.glob("*.ipynb"))
    if notebooks:
        context["notebook_text"] = _collect_notebook_text(notebooks[-1])
    return context


def extract_hypothesis_genes(hypothesis_text: str, available_genes: list[str] | pd.Index, top_n: int = 6) -> list[str]:
    if not hypothesis_text:
        return []
    lookup = {str(gene).upper(): str(gene) for gene in available_genes}
    matches: list[str] = []
    for token in re.findall(r"[A-Za-z0-9\-]{3,}", hypothesis_text):
        normalized = token.upper().replace("-", "")
        direct = lookup.get(token.upper()) or lookup.get(normalized)
        if direct and direct not in matches:
            matches.append(direct)
    return matches[:top_n]


def extract_result_genes(
    *,
    available_genes: list[str] | pd.Index,
    top_n: int = 8,
    texts: list[str] | tuple[str, ...],
) -> list[str]:
    lookup = {str(gene).upper(): str(gene) for gene in available_genes}
    counts: dict[str, int] = {}
    for text in texts:
        if not text:
            continue
        cleaned = re.sub(r"[A-Za-z]:\\\\[^\s]+", " ", text)
        cleaned = re.sub(r"\b\S+\.(pdf|png|jpg|jpeg|tsv|csv|txt|json|h5ad|ipynb)\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"https?://\S+", " ", cleaned)
        for token in re.findall(r"[A-Za-z0-9\-_]{3,}", cleaned):
            normalized = token.upper().replace("-", "")
            if normalized in NOISY_GENE_TOKENS:
                continue
            matched = lookup.get(token.upper()) or lookup.get(normalized)
            if matched:
                counts[matched] = counts.get(matched, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [gene for gene, _ in ranked[:top_n]]


def plot_text_panel(ax, title: str, text: str) -> None:
    ax.axis("off")
    wrapped = textwrap.fill(text or "No hypothesis text available.", width=60)
    ax.set_title(title, fontsize=11, loc="left")
    ax.text(0.0, 0.98, wrapped, va="top", ha="left", fontsize=10, transform=ax.transAxes)


def plot_categorical_embedding(
    ax,
    adata,
    *,
    color: str,
    title: str,
    basis: str = "X_umap",
    label_points: bool = True,
) -> None:
    if basis not in adata.obsm:
        ax.text(0.5, 0.5, f"No {basis}", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.axis("off")
        return
    coords = adata.obsm[basis]
    frame = pd.DataFrame(coords[:, :2], columns=["x", "y"], index=adata.obs_names)
    if color in adata.obs.columns:
        labels = adata.obs[color].astype(str).fillna("Unknown")
    else:
        labels = pd.Series(["Unknown"] * adata.n_obs, index=adata.obs_names)
    frame[color] = labels.values
    categories = frame[color].astype(str).value_counts().index.tolist()
    palette = dict(zip(categories, sns.color_palette("tab20", max(len(categories), 3))))
    for category in categories:
        subset = frame.loc[frame[color].astype(str) == category]
        ax.scatter(subset["x"], subset["y"], s=7, alpha=0.8, color=palette[category], label=category, rasterized=True)
    if label_points and len(categories) <= 18:
        centers = frame.groupby(color, observed=False)[["x", "y"]].median()
        for category, row in centers.iterrows():
            ax.text(row["x"], row["y"], str(category), fontsize=8, weight="bold")
    elif len(categories) <= 18:
        ax.legend(loc="best", fontsize=7, frameon=False)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")


def rank_marker_matrix(
    adata,
    *,
    groupby: str,
    top_n_per_group: int = 3,
    key_candidates: tuple[str, ...] = ("rank_genes_groups",),
    use_raw: bool = True,
) -> pd.DataFrame:
    working = adata.copy()
    if groupby not in working.obs.columns:
        raise KeyError(f"'{groupby}' not present in adata.obs")
    working.obs[groupby] = working.obs[groupby].astype(str).astype("category")
    key_name = next((key for key in key_candidates if key in working.uns), None)
    if key_name is None:
        sc.tl.rank_genes_groups(working, groupby=groupby, method="wilcoxon", key_added="_figure_markers", use_raw=use_raw)
        key_name = "_figure_markers"
    marker_frames: list[pd.DataFrame] = []
    for group in working.obs[groupby].cat.categories:
        try:
            frame = sc.get.rank_genes_groups_df(working, group=str(group), key=key_name).head(top_n_per_group)
        except Exception:
            continue
        if frame.empty:
            continue
        frame = frame.assign(group=str(group))
        marker_frames.append(frame)
    if not marker_frames:
        return pd.DataFrame()
    marker_table = pd.concat(marker_frames, ignore_index=True)
    genes = [gene for gene in marker_table["names"].astype(str).tolist() if gene in working.var_names]
    genes = list(dict.fromkeys(genes))
    if not genes:
        return pd.DataFrame()
    matrix = working[:, genes].to_df()
    matrix[groupby] = working.obs[groupby].values
    grouped = matrix.groupby(groupby, observed=False).mean(numeric_only=True)
    return grouped[genes].T


def build_joint_umap_from_gene_signals(
    adata_rna,
    adata_other,
    *,
    expression_getter,
    other_signal_getter,
    candidate_genes: list[str],
    obs_source: str = "rna",
    max_genes: int = 40,
) -> ad.AnnData | None:
    shared = sorted(set(map(str, adata_rna.obs_names)).intersection(map(str, adata_other.obs_names)))
    if len(shared) < 50:
        return None
    rna_sub = adata_rna[shared].copy()
    other_sub = adata_other[shared].copy()
    selected = list(dict.fromkeys(candidate_genes))[:max_genes]
    if len(selected) < 3:
        return None
    try:
        rna_frame = expression_getter(rna_sub, selected)
        other_frame = other_signal_getter(other_sub, selected)
    except Exception:
        return None
    common = rna_frame.columns.intersection(other_frame.columns)
    if len(common) < 3:
        return None
    rna_values = rna_frame[common].astype(float)
    other_values = other_frame[common].astype(float)
    rna_z = (rna_values - rna_values.mean(axis=0)) / rna_values.std(axis=0).replace(0, 1)
    other_z = (other_values - other_values.mean(axis=0)) / other_values.std(axis=0).replace(0, 1)
    merged = (rna_z.fillna(0.0) + other_z.fillna(0.0)) / 2.0
    obs = rna_sub.obs.copy() if obs_source == "rna" else other_sub.obs.copy()
    joint = ad.AnnData(merged.values, obs=obs, var=pd.DataFrame(index=list(common)))
    sc.pp.pca(joint, n_comps=min(30, max(2, merged.shape[1] - 1)))
    sc.pp.neighbors(joint, n_neighbors=15)
    sc.tl.umap(joint)
    return joint


def group_overlap_heatmap(
    left_labels: pd.Series,
    right_labels: pd.Series,
    *,
    normalize: str = "index",
) -> pd.DataFrame:
    table = pd.crosstab(left_labels.astype(str), right_labels.astype(str))
    if normalize == "index":
        denom = table.sum(axis=1).replace(0, 1)
        return table.div(denom, axis=0)
    if normalize == "columns":
        denom = table.sum(axis=0).replace(0, 1)
        return table.div(denom, axis=1)
    return table


def top_numeric_obs_columns(adata, *, prefix: str | None = None, top_n: int = 4) -> list[str]:
    numeric_cols = []
    for column in adata.obs.columns:
        if prefix and not str(column).startswith(prefix):
            continue
        if pd.api.types.is_numeric_dtype(adata.obs[column]):
            numeric_cols.append(str(column))
    return numeric_cols[:top_n]


def save_figure_bundle(fig, output_dir: str | Path, figure_name: str) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{figure_name}.png"
    pdf_path = output_dir / f"{figure_name}.pdf"
    summary_path = output_dir / f"{figure_name}_summary.txt"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path, summary_path
