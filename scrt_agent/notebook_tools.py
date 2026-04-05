"""Reusable notebook helper functions for scMAA."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns


TUMOR_HINT_TOKENS = ("tumor", "metast", "primary", "focus", "lesion", "cancer", "carcinoma", "malignan")
NON_TUMOR_HINT_TOKENS = ("pbmc", "blood", "normal", "healthy", "control", "adjacent", "benign", "spleen")
COMMON_GENE_ALIASES = {
    "PD1": "PDCD1",
    "PD-1": "PDCD1",
    "PDCD1": "PDCD1",
    "TIGIT": "TIGIT",
    "TIM3": "HAVCR2",
    "TIM-3": "HAVCR2",
    "LAG3": "LAG3",
    "CTLA4": "CTLA4",
    "XBP-1": "XBP1",
    "XBP1": "XBP1",
}
CURATED_T_CELL_PROGRAMS = {
    "Cytotoxic effector pathway": {
        "CCL4",
        "CCL5",
        "CTSW",
        "FCGR3A",
        "GNLY",
        "GZMB",
        "GZMH",
        "IFNG",
        "KLRD1",
        "NKG7",
        "PRF1",
    },
    "T-cell activation pathway": {
        "CD27",
        "CD69",
        "CD8A",
        "CD8B",
        "IL7R",
        "LTB",
        "MALAT1",
        "SAT1",
        "TRAC",
    },
    "Checkpoint/exhaustion pathway": {
        "CTLA4",
        "ENTPD1",
        "HAVCR2",
        "LAG3",
        "PDCD1",
        "TIGIT",
        "TOX",
    },
    "Interferon response pathway": {
        "IFI6",
        "IFI44L",
        "IFIT1",
        "IFIT2",
        "IFIT3",
        "IFITM1",
        "IFITM2",
        "IFITM3",
        "IRF7",
        "ISG15",
        "MX1",
        "OAS1",
        "STAT1",
        "XBP1",
    },
    "Chemokine trafficking pathway": {
        "CCL3",
        "CCL4",
        "CCL5",
        "CCR7",
        "CXCL13",
        "CXCR3",
        "IL32",
        "LTB",
        "S1PR1",
    },
    "Antigen presentation pathway": {
        "B2M",
        "HLA-A",
        "HLA-B",
        "HLA-C",
        "HLA-DRA",
        "HLA-DRB1",
        "TAP1",
        "TAPBP",
    },
}


def ensure_obs_column(adata, column: str, fill_value: str = "Unknown", as_category: bool = True) -> None:
    """Ensure an obs column exists and is optionally categorical."""
    if column not in adata.obs.columns:
        adata.obs[column] = fill_value
    else:
        adata.obs[column] = adata.obs[column].astype("object").where(adata.obs[column].notna(), fill_value)
    if as_category:
        adata.obs[column] = adata.obs[column].astype("category")


def ensure_obs_columns(adata, columns: Iterable[str], fill_value: str = "Unknown", as_category: bool = True) -> None:
    for column in columns:
        ensure_obs_column(adata, column, fill_value=fill_value, as_category=as_category)


def paired_tcr_subset(adata, copy: bool = True):
    """Return the paired scRNA + scTCR subset."""
    if "has_tcr" not in adata.obs.columns:
        raise KeyError("'has_tcr' is not present in adata.obs.")
    mask = adata.obs["has_tcr"].fillna(False).astype(bool)
    return adata[mask].copy() if copy else adata[mask]


def infer_tumor_like_tissues(adata, tissue_col: str = "tissue") -> list[str]:
    """Infer tumor-like tissue labels from observed metadata values."""
    if tissue_col not in adata.obs.columns:
        raise KeyError(f"'{tissue_col}' is not present in adata.obs.")
    labels = (
        adata.obs[tissue_col]
        .dropna()
        .astype(str)
        .map(str.strip)
    )
    inferred: list[str] = []
    for label in labels.value_counts(dropna=False).index.tolist():
        lowered = label.lower()
        if any(token in lowered for token in NON_TUMOR_HINT_TOKENS):
            continue
        if any(token in lowered for token in TUMOR_HINT_TOKENS):
            inferred.append(label)
    return inferred


def infer_primary_metastasis_tissues(adata, tissue_col: str = "tissue") -> tuple[str, str]:
    """Infer a primary/metastasis tissue pair from observed metadata."""
    if tissue_col not in adata.obs.columns:
        raise KeyError(f"'{tissue_col}' is not present in adata.obs.")
    labels = [str(value).strip() for value in adata.obs[tissue_col].dropna().astype(str).unique().tolist()]
    primary = [label for label in labels if "primary" in label.lower()]
    metastasis = [label for label in labels if "metast" in label.lower()]
    if not primary or not metastasis:
        raise ValueError(
            f"Could not infer both primary and metastasis labels from '{tissue_col}'. "
            f"Observed labels: {labels}"
        )
    return primary[0], metastasis[0]


def tumor_like_subset(adata, tissue_col: str = "tissue", copy: bool = True):
    """Return a tumor-like subset using heuristic tissue label inference."""
    tissues = infer_tumor_like_tissues(adata, tissue_col=tissue_col)
    if not tissues:
        raise ValueError(
            f"No tumor-like labels inferred from '{tissue_col}'. "
            "Inspect the available tissue values before making tumor-specific claims."
        )
    mask = adata.obs[tissue_col].astype(str).isin(tissues)
    subset = adata[mask].copy() if copy else adata[mask]
    print(f"Tumor-like tissues inferred from {tissue_col}: {', '.join(tissues)}")
    print(f"Tumor-like subset cells: {subset.n_obs}")
    return subset


def resolve_gene_names(adata_or_genes, genes: Iterable[str] | None = None) -> dict[str, str]:
    """Resolve requested markers to canonical dataset gene names.

    Supports both ``resolve_gene_names(adata, genes)`` and a permissive fallback
    ``resolve_gene_names(genes)``. The fallback returns an identity mapping so
    downstream code can continue when the model omits the dataset argument.
    """
    if genes is None:
        if hasattr(adata_or_genes, "var_names") and hasattr(adata_or_genes, "obs"):
            return adata_or_genes
        gene_iterable = adata_or_genes
        try:
            return {str(gene).strip(): str(gene).strip() for gene in gene_iterable if str(gene).strip()}
        except TypeError:
            requested = str(adata_or_genes).strip()
            return {requested: requested} if requested else {}
    adata = adata_or_genes
    var_lookup = {str(name).upper(): str(name) for name in adata.var_names}
    resolved: dict[str, str] = {}
    for gene in genes:
        requested = str(gene).strip()
        if not requested:
            continue
        candidates = [requested]
        alias = COMMON_GENE_ALIASES.get(requested.upper())
        if alias and alias not in candidates:
            candidates.append(alias)
        if requested.upper() in COMMON_GENE_ALIASES:
            canonical = COMMON_GENE_ALIASES[requested.upper()]
            if canonical not in candidates:
                candidates.append(canonical)
        matched = None
        for candidate in candidates:
            matched = var_lookup.get(candidate.upper())
            if matched:
                break
        if matched:
            resolved[requested] = matched
    return resolved


def expression_frame(adata, genes: Iterable[str], obs_columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Extract a dense expression DataFrame using resolved marker names."""
    resolved = resolve_gene_names(adata, genes)
    actual_genes = list(dict.fromkeys(resolved.values()))
    if not actual_genes:
        raise ValueError("None of the requested genes were found in adata.var_names after alias resolution.")
    matrix = adata[:, actual_genes].X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    frame = pd.DataFrame(matrix, index=adata.obs_names, columns=actual_genes)
    if obs_columns:
        for column in obs_columns:
            if column in adata.obs.columns:
                frame[column] = adata.obs[column].values
    print("Resolved genes:", ", ".join(f"{src}->{dst}" for src, dst in resolved.items()))
    return frame


def clone_expansion_table(adata, groupby: str = "tissue", paired_only: bool = True) -> pd.DataFrame:
    """Summarize expanded-clone counts by grouping column."""
    if paired_only:
        adata = paired_tcr_subset(adata, copy=True)
    if "expanded_clone" not in adata.obs.columns:
        raise KeyError("'expanded_clone' is not present in adata.obs.")
    ensure_obs_column(adata, groupby, fill_value="Unknown", as_category=True)
    summary = (
        adata.obs.groupby(groupby, observed=False)["expanded_clone"]
        .agg(total_paired="size", expanded_cells="sum")
        .reset_index()
    )
    summary["expanded_fraction"] = summary["expanded_cells"] / summary["total_paired"].clip(lower=1)
    return summary.sort_values(["expanded_fraction", "expanded_cells"], ascending=[False, False]).reset_index(drop=True)


def print_clone_expansion_table(adata, groupby: str = "tissue", paired_only: bool = True) -> pd.DataFrame:
    """Print and return the clone expansion summary."""
    summary = clone_expansion_table(adata, groupby=groupby, paired_only=paired_only)
    print(summary.to_string(index=False))
    return summary


def safe_rank_genes_groups(
    adata,
    *,
    groupby: str,
    groups: list[str] | None = None,
    reference: str | None = None,
    method: str = "wilcoxon",
    layer: str | None = None,
    use_raw: bool | None = None,
    min_cells_per_group: int = 20,
    key_added: str | None = None,
):
    """Run rank_genes_groups after enforcing categorical group labels and minimum group sizes."""
    ensure_obs_column(adata, groupby, fill_value="Unknown", as_category=True)
    counts = adata.obs[groupby].value_counts(dropna=False)
    eligible = counts[counts >= min_cells_per_group].index.astype(str).tolist()
    if len(eligible) < 2:
        raise ValueError(
            f"Not enough groups with >= {min_cells_per_group} cells for rank_genes_groups on '{groupby}'."
        )

    working = adata[adata.obs[groupby].astype(str).isin(eligible)].copy()
    working.obs[groupby] = working.obs[groupby].astype(str).astype("category")
    selected_groups = groups
    if selected_groups is not None:
        selected_groups = [item for item in groups if item in set(eligible)]
        if not selected_groups:
            raise ValueError(f"Requested groups are not eligible for '{groupby}'.")

    sc.tl.rank_genes_groups(
        working,
        groupby=groupby,
        groups=selected_groups,
        reference=reference,
        method=method,
        layer=layer,
        use_raw=use_raw,
        key_added=key_added,
    )
    return working


def tissue_stratified_expansion_de(
    adata,
    *,
    tissue_col: str = "tissue",
    expansion_col: str = "expanded_clone",
    paired_only: bool = True,
    min_cells_per_group: int = 20,
    top_n: int = 10,
    method: str = "wilcoxon",
    **kwargs,
) -> pd.DataFrame:
    """Run expanded-vs-non-expanded DE within each tissue using safe categorical handling."""
    tissue_col = kwargs.pop("tissue", tissue_col)
    expansion_col = kwargs.pop("expanded", expansion_col)
    expansion_col = kwargs.pop("group_col", expansion_col)
    expansion_col = kwargs.pop("expansion", expansion_col)
    kwargs.pop("sample_aware", None)
    if kwargs:
        print(f"Ignoring unsupported kwargs in tissue_stratified_expansion_de: {sorted(kwargs)}")
    if paired_only:
        adata = paired_tcr_subset(adata, copy=True)
    ensure_obs_columns(adata, [tissue_col], fill_value="Unknown", as_category=True)
    if expansion_col not in adata.obs.columns:
        raise KeyError(f"'{expansion_col}' is not present in adata.obs.")

    adata.obs[expansion_col] = adata.obs[expansion_col].fillna(False).astype(bool).map(
        {True: "expanded", False: "non_expanded"}
    )
    adata.obs[expansion_col] = adata.obs[expansion_col].astype("category")

    frames: list[pd.DataFrame] = []
    for tissue in adata.obs[tissue_col].astype(str).value_counts(dropna=False).index.tolist():
        tissue_data = adata[adata.obs[tissue_col].astype(str) == tissue].copy()
        counts = tissue_data.obs[expansion_col].value_counts()
        expanded_n = int(counts.get("expanded", 0))
        non_expanded_n = int(counts.get("non_expanded", 0))
        print(
            f"Tissue={tissue}: expanded={expanded_n}, non_expanded={non_expanded_n}, total={tissue_data.n_obs}"
        )
        if expanded_n < min_cells_per_group or non_expanded_n < min_cells_per_group:
            print(
                f"Skipping tissue={tissue} because one group has fewer than {min_cells_per_group} cells."
            )
            continue

        ranked = safe_rank_genes_groups(
            tissue_data,
            groupby=expansion_col,
            groups=["expanded"],
            reference="non_expanded",
            method=method,
            min_cells_per_group=min_cells_per_group,
            key_added=f"{tissue}_expanded_vs_non_expanded",
        )
        frame = sc.get.rank_genes_groups_df(
            ranked,
            group="expanded",
            key=f"{tissue}_expanded_vs_non_expanded",
        ).head(top_n)
        frame.insert(0, "tissue", tissue)
        frame.insert(1, "expanded_n", expanded_n)
        frame.insert(2, "non_expanded_n", non_expanded_n)
        frames.append(frame)

    if not frames:
        raise ValueError("No tissues had enough cells for stratified differential expression.")
    result = pd.concat(frames, ignore_index=True)
    print(result[["tissue", "names", "scores", "logfoldchanges", "pvals_adj"]].to_string(index=False))
    return result


def expanded_clone_tissue_de(
    adata,
    *,
    tissue_col: str = "tissue",
    expansion_col: str = "expanded_clone",
    tissues: Iterable[str] | None = None,
    case_tissue: str | None = None,
    reference_tissue: str | None = None,
    paired_only: bool = True,
    min_cells_per_group: int = 20,
    method: str = "wilcoxon",
    top_n: int = 30,
):
    """Run DE between tissues within expanded clonotypes and return the filtered subset plus results."""
    if paired_only:
        adata = paired_tcr_subset(adata, copy=True)
    ensure_obs_column(adata, tissue_col, fill_value="Unknown", as_category=True)
    if expansion_col not in adata.obs.columns:
        raise KeyError(f"'{expansion_col}' is not present in adata.obs.")

    subset = adata[adata.obs[expansion_col].fillna(False).astype(bool)].copy()
    observed_tissues = subset.obs[tissue_col].dropna().astype(str).unique().tolist()
    if tissues is None:
        tissues = infer_primary_metastasis_tissues(subset, tissue_col=tissue_col)
    selected_tissues = [str(tissue) for tissue in tissues if str(tissue) in set(observed_tissues)]
    if len(selected_tissues) < 2:
        raise ValueError(
            f"Need at least two tissues for contrast. Requested={list(tissues)}, observed={observed_tissues}"
        )
    subset = subset[subset.obs[tissue_col].astype(str).isin(selected_tissues)].copy()

    if reference_tissue is None or case_tissue is None:
        inferred_reference, inferred_case = infer_primary_metastasis_tissues(subset, tissue_col=tissue_col)
        reference_tissue = reference_tissue or inferred_reference
        case_tissue = case_tissue or inferred_case

    counts = subset.obs[tissue_col].astype(str).value_counts()
    reference_n = int(counts.get(reference_tissue, 0))
    case_n = int(counts.get(case_tissue, 0))
    print(
        f"Expanded-clone tissue contrast: reference={reference_tissue} (n={reference_n}), "
        f"case={case_tissue} (n={case_n})"
    )
    if reference_n < min_cells_per_group or case_n < min_cells_per_group:
        raise ValueError(
            f"Not enough expanded-clone cells for tissue contrast: "
            f"{reference_tissue}={reference_n}, {case_tissue}={case_n}, "
            f"required>={min_cells_per_group}"
        )

    key = f"{case_tissue}_vs_{reference_tissue}_expanded_clone_tissue_de"
    ranked = safe_rank_genes_groups(
        subset,
        groupby=tissue_col,
        groups=[case_tissue],
        reference=reference_tissue,
        method=method,
        min_cells_per_group=min_cells_per_group,
        key_added=key,
    )
    frame = sc.get.rank_genes_groups_df(ranked, group=case_tissue, key=key).head(top_n).copy()
    frame.insert(0, "case_tissue", case_tissue)
    frame.insert(1, "reference_tissue", reference_tissue)
    print(frame[["names", "scores", "logfoldchanges", "pvals_adj"]].to_string(index=False))
    return ranked, frame


def plot_de_barplot(
    de_results: pd.DataFrame,
    *,
    n: int = 10,
    gene_col: str = "names",
    lfc_col: str = "logfoldchanges",
    title: str | None = None,
):
    """Plot a bidirectional barplot of the strongest DE genes."""
    required = {gene_col, lfc_col}
    if not required.issubset(de_results.columns):
        raise KeyError(f"de_results must contain columns: {sorted(required)}")
    working = de_results[[gene_col, lfc_col]].dropna().copy()
    if working.empty:
        raise ValueError("de_results has no rows available for plotting.")

    top_case = working.nlargest(n, lfc_col).copy()
    top_case["direction"] = "case_up"
    top_reference = working.nsmallest(n, lfc_col).copy()
    top_reference["direction"] = "reference_up"
    plot_df = pd.concat([top_reference.iloc[::-1], top_case], ignore_index=True)

    fig_height = max(4.0, 0.45 * len(plot_df))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    sns.barplot(
        data=plot_df,
        x=lfc_col,
        y=gene_col,
        hue="direction",
        dodge=False,
        palette={"case_up": "#c43c35", "reference_up": "#2c7fb8"},
        ax=ax,
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("log fold change")
    ax.set_ylabel("Gene")
    ax.set_title(title or "Top differentially expressed genes")
    ax.legend(title="")
    fig.tight_layout()
    return ax


def plot_de_heatmap(
    adata,
    genes: Iterable[str],
    *,
    tissue_col: str = "tissue",
    tissues: Iterable[str] | None = None,
    expansion_col: str = "expanded_clone",
    paired_only: bool = True,
    standard_scale: bool = True,
    title: str | None = None,
):
    """Plot a tissue-level heatmap for selected DE genes within expanded clonotypes."""
    if paired_only:
        adata = paired_tcr_subset(adata, copy=True)
    ensure_obs_column(adata, tissue_col, fill_value="Unknown", as_category=True)
    if expansion_col not in adata.obs.columns:
        raise KeyError(f"'{expansion_col}' is not present in adata.obs.")

    subset = adata[adata.obs[expansion_col].fillna(False).astype(bool)].copy()
    if tissues is None:
        tissues = infer_primary_metastasis_tissues(subset, tissue_col=tissue_col)
    selected_tissues = [str(tissue) for tissue in tissues]
    subset = subset[subset.obs[tissue_col].astype(str).isin(selected_tissues)].copy()
    frame = expression_frame(subset, genes, obs_columns=[tissue_col])
    mean_expr = frame.groupby(tissue_col, observed=False).mean(numeric_only=True).T
    mean_expr = mean_expr.loc[:, [col for col in selected_tissues if col in mean_expr.columns]]
    if mean_expr.empty:
        raise ValueError("No gene expression values were available for the requested heatmap.")

    if standard_scale:
        centered = mean_expr.sub(mean_expr.mean(axis=1), axis=0)
        scaled = centered.div(mean_expr.std(axis=1).replace(0, 1), axis=0)
        mean_expr = scaled.fillna(0.0)

    fig_width = max(5.0, 2.4 * max(1, mean_expr.shape[1]))
    fig_height = max(4.0, 0.4 * max(1, mean_expr.shape[0]))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(mean_expr, cmap="vlag", center=0, linewidths=0.5, cbar_kws={"label": "scaled mean expression"}, ax=ax)
    ax.set_xlabel(tissue_col)
    ax.set_ylabel("Gene")
    ax.set_title(title or "Heatmap of top DE genes across tissues")
    fig.tight_layout()
    return mean_expr, ax


def summarize_de_pathways(
    de_results: pd.DataFrame,
    *,
    gene_col: str = "names",
    lfc_col: str = "logfoldchanges",
    case_label: str = "case",
    reference_label: str = "reference",
    n_case: int = 15,
    n_reference: int = 15,
) -> dict[str, pd.DataFrame]:
    """Print a lightweight pathway interpretation using curated T-cell programs."""
    required = {gene_col, lfc_col}
    if not required.issubset(de_results.columns):
        raise KeyError(f"de_results must contain columns: {sorted(required)}")

    working = de_results[[gene_col, lfc_col]].dropna().copy()
    if working.empty:
        raise ValueError("de_results has no rows available for pathway interpretation.")

    case_genes = working.nlargest(n_case, lfc_col)[gene_col].astype(str).tolist()
    reference_genes = working.nsmallest(n_reference, lfc_col)[gene_col].astype(str).tolist()

    def _pathway_frame(genes: list[str]) -> pd.DataFrame:
        gene_set = {gene.upper() for gene in genes}
        rows: list[dict[str, object]] = []
        for pathway, members in CURATED_T_CELL_PROGRAMS.items():
            overlap = sorted(gene_set & members)
            if overlap:
                rows.append(
                    {
                        "pathway": pathway,
                        "overlap_n": len(overlap),
                        "overlap_genes": ", ".join(overlap),
                    }
                )
        if not rows:
            return pd.DataFrame(columns=["pathway", "overlap_n", "overlap_genes"])
        frame = pd.DataFrame(rows)
        return frame.sort_values(["overlap_n", "pathway"], ascending=[False, True]).reset_index(drop=True)

    case_frame = _pathway_frame(case_genes)
    reference_frame = _pathway_frame(reference_genes)

    print(f"Curated pathway enrichment for {case_label}-up genes:")
    if case_frame.empty:
        print("No curated pathway overlap detected.")
    else:
        print(case_frame.to_string(index=False))

    print(f"\nCurated pathway enrichment for {reference_label}-up genes:")
    if reference_frame.empty:
        print("No curated pathway overlap detected.")
    else:
        print(reference_frame.to_string(index=False))

    return {
        f"{case_label}_up": case_frame,
        f"{reference_label}_up": reference_frame,
    }


def plot_tissue_embedding(
    adata,
    *,
    tissue_col: str = "tissue",
    expansion_col: str = "expanded_clone",
    tissues: Iterable[str] | None = None,
    paired_only: bool = True,
    basis: str = "auto",
    title: str | None = None,
):
    """Plot a tissue-annotated embedding for expanded clonotypes."""
    if paired_only:
        adata = paired_tcr_subset(adata, copy=True)
    ensure_obs_column(adata, tissue_col, fill_value="Unknown", as_category=True)
    if expansion_col not in adata.obs.columns:
        raise KeyError(f"'{expansion_col}' is not present in adata.obs.")

    subset = adata[adata.obs[expansion_col].fillna(False).astype(bool)].copy()
    if tissues is None:
        tissues = infer_primary_metastasis_tissues(subset, tissue_col=tissue_col)
    selected_tissues = [str(tissue) for tissue in tissues]
    subset = subset[subset.obs[tissue_col].astype(str).isin(selected_tissues)].copy()
    if subset.n_obs == 0:
        raise ValueError("No cells remained after filtering expanded clonotypes by the requested tissues.")

    chosen_basis = basis.lower()
    if chosen_basis == "auto":
        chosen_basis = "umap" if "X_umap" in subset.obsm else "pca"

    if chosen_basis == "umap":
        if "X_umap" not in subset.obsm:
            raise ValueError("UMAP coordinates are not available in adata.obsm['X_umap'].")
        coords = np.asarray(subset.obsm["X_umap"])[:, :2]
        axis_labels = ("UMAP1", "UMAP2")
    elif chosen_basis == "pca":
        if "X_pca" not in subset.obsm:
            sc.tl.pca(subset, svd_solver="arpack")
        coords = np.asarray(subset.obsm["X_pca"])[:, :2]
        axis_labels = ("PC1", "PC2")
    else:
        raise ValueError("basis must be one of: auto, umap, pca")

    plot_df = pd.DataFrame(
        {
            axis_labels[0]: coords[:, 0],
            axis_labels[1]: coords[:, 1],
            tissue_col: subset.obs[tissue_col].astype(str).values,
        }
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x=axis_labels[0],
        y=axis_labels[1],
        hue=tissue_col,
        s=14,
        linewidth=0,
        alpha=0.85,
        ax=ax,
    )
    ax.set_title(title or f"{chosen_basis.upper()} of expanded clonotypes by {tissue_col}")
    fig.tight_layout()
    return chosen_basis, plot_df, ax

