"""Preparation pipeline for scRNA + spatial analysis."""

from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc

from scrt_agent.preprocess import (
    _annotate_clusters_with_llm,
    _discover_from_directory,
    _extract_marker_table,
    _is_linc_like,
    _read_10x_sample,
    _sample_qc_summary,
    _save_umap_figure,
)

from .logger import AgentLogger
from .utils import read_text


@dataclass
class PreparationResult:
    output_dir: Path
    rna_h5ad_path: Path
    spatial_h5ad_path: Path
    rna_markers_path: Path
    spatial_markers_path: Path
    rna_annotations_path: Path
    spatial_annotations_path: Path
    qc_summary_path: Path
    manifest_path: Path
    rna_umap_path: Path
    spatial_umap_path: Path
    spatial_map_path: Path


def _stage_input(raw_input_path: str | Path, stage_dir: Path, logger: AgentLogger) -> Path:
    raw_path = Path(raw_input_path).resolve()
    if raw_path.is_dir():
        return raw_path
    if tarfile.is_tarfile(raw_path):
        extract_dir = stage_dir / f"{raw_path.stem}_extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        if any(extract_dir.iterdir()):
            return extract_dir
        logger.info(f"Extracting {raw_path} to {extract_dir}")
        with tarfile.open(raw_path, "r") as handle:
            handle.extractall(extract_dir)
        return extract_dir
    raise ValueError(f"Unsupported input path: {raw_input_path}")


def _parse_visium_key(name: str) -> tuple[str, str]:
    tokens = name.replace("-", "_").split("_")
    if len(tokens) <= 1:
        return name, "unknown"
    return tokens[0], "_".join(tokens[1:])


def _infer_standard_sample_key(path: Path) -> str:
    generic_names = {"filtered_feature_bc_matrix", "raw_feature_bc_matrix", "outs"}
    if path.name in generic_names and path.parent.name:
        return path.parent.name
    return path.name


def _load_standard_rna_input(rna_root: Path) -> ad.AnnData | None:
    if (rna_root / "matrix.mtx.gz").exists() and (rna_root / "features.tsv.gz").exists() and (rna_root / "barcodes.tsv.gz").exists():
        sample_key = _infer_standard_sample_key(rna_root)
        adata = sc.read_10x_mtx(rna_root, var_names="gene_symbols")
        adata.var_names_make_unique()
        sample_id, tissue = _parse_visium_key(sample_key)
        adata.obs["sample_key"] = sample_key
        adata.obs["sample_id"] = sample_id
        adata.obs["tissue"] = tissue
        adata.obs["barcode"] = adata.obs_names.astype(str)
        adata.obs_names = [f"{sample_key}:{idx}" for idx in adata.obs_names.astype(str)]
        return adata
    h5_candidates = sorted(rna_root.glob("*.h5")) + sorted(rna_root.glob("*.h5ad"))
    if not h5_candidates:
        return None
    path = h5_candidates[0]
    sample_key = _infer_standard_sample_key(path.parent if path.name.startswith("filtered_feature_bc_matrix") else path.with_suffix(""))
    if path.suffix == ".h5ad":
        adata = sc.read_h5ad(path)
    else:
        adata = sc.read_10x_h5(path)
    adata.var_names_make_unique()
    sample_id, tissue = _parse_visium_key(sample_key)
    adata.obs["sample_key"] = sample_key
    adata.obs["sample_id"] = sample_id
    adata.obs["tissue"] = tissue
    if "barcode" not in adata.obs.columns:
        adata.obs["barcode"] = adata.obs_names.astype(str)
    adata.obs_names = [f"{sample_key}:{idx}" for idx in adata.obs_names.astype(str)]
    return adata


def _discover_visium_dirs(root: Path) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    potential_dirs = [root] + [path for path in root.iterdir() if path.is_dir()]
    for path in potential_dirs:
        if (path / "spatial").is_dir() and ((path / "filtered_feature_bc_matrix.h5").exists() or (path / "filtered_feature_bc_matrix").is_dir()):
            candidates.append((path.name, path))
    return candidates


def _load_spatial_objects(spatial_root: Path) -> ad.AnnData:
    visium_dirs = _discover_visium_dirs(spatial_root)
    if visium_dirs:
        adatas = []
        for sample_key, path in visium_dirs:
            adata = sc.read_visium(path)
            adata.var_names_make_unique()
            sample_id, tissue = _parse_visium_key(sample_key)
            adata.obs["sample_key"] = sample_key
            adata.obs["sample_id"] = sample_id
            adata.obs["tissue"] = tissue
            adata.obs_names = [f"{sample_key}:{idx}" for idx in adata.obs_names.astype(str)]
            adatas.append(adata)
        if len(adatas) == 1:
            return adatas[0]
        return ad.concat(adatas, join="outer", merge="same")

    h5ad_files = sorted(spatial_root.glob("*.h5ad"))
    if h5ad_files:
        adatas = []
        for path in h5ad_files:
            adata = sc.read_h5ad(path)
            adata.var_names_make_unique()
            sample_id, tissue = _parse_visium_key(path.stem)
            if "sample_key" not in adata.obs.columns:
                adata.obs["sample_key"] = path.stem
            if "sample_id" not in adata.obs.columns:
                adata.obs["sample_id"] = sample_id
            if "tissue" not in adata.obs.columns:
                adata.obs["tissue"] = tissue
            adatas.append(adata)
        if len(adatas) == 1:
            return adatas[0]
        return ad.concat(adatas, join="outer", merge="same")
    raise ValueError(f"No Visium directories or spatial .h5ad files were found under {spatial_root}")


def _save_spatial_map(adata: ad.AnnData, color: str, output_path: Path, title: str) -> None:
    if "spatial" not in adata.obsm:
        _save_umap_figure(adata, color, output_path, title)
        return
    coords = adata.obsm["spatial"]
    frame = pd.DataFrame(coords[:, :2], columns=["x", "y"], index=adata.obs_names)
    frame[color] = adata.obs[color].astype(str).values
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    categories = frame[color].astype(str).unique().tolist()
    for category in categories:
        subset = frame.loc[frame[color].astype(str) == category]
        ax.scatter(subset["x"], subset["y"], s=8, alpha=0.8, label=category)
    ax.set_title(title)
    ax.set_xlabel("spatial_x")
    ax.set_ylabel("spatial_y")
    ax.invert_yaxis()
    if len(categories) <= 12:
        ax.legend(loc="best", fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close("all")


def prepare_dataset(
    *,
    rna_raw_input_path: str,
    spatial_raw_input_path: str,
    output_dir: str,
    annotation_model: str = "gpt-4o",
    annotation_notes_path: str | None = None,
    min_genes: int = 200,
    min_cells: int = 3,
    max_pct_mt: float = 15.0,
    n_top_genes: int = 3000,
    n_pcs: int = 30,
    n_neighbors: int = 15,
    leiden_resolution: float = 0.8,
    marker_top_n: int = 100,
    annotation_marker_top_n: int = 50,
    log_prompts: bool = False,
) -> PreparationResult:
    outdir = Path(output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    logger = AgentLogger("prepare_spatial_data", outdir / "logs", log_prompts=log_prompts)
    sc.settings.verbosity = 2
    sc.settings.set_figure_params(dpi=120, facecolor="white", frameon=False)

    rna_input = _stage_input(rna_raw_input_path, outdir / "work", logger)
    spatial_input = _stage_input(spatial_raw_input_path, outdir / "work", logger)
    rna_samples = [sample for sample in _discover_from_directory(rna_input).values() if sample.barcodes_path and sample.features_path and sample.matrix_path]
    if rna_samples:
        rna = ad.concat([_read_10x_sample(sample) for sample in sorted(rna_samples, key=lambda item: item.sample_key)], join="outer", merge="same")
    else:
        rna = _load_standard_rna_input(rna_input)
        if rna is None:
            raise ValueError("No complete scRNA input was found. Supported inputs are GEO-style triplets, standard 10x matrix directories, .h5, or .h5ad.")
    rna.var_names_make_unique()
    rna.var["mt"] = rna.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(rna, qc_vars=["mt"], inplace=True)
    pre_qc = _sample_qc_summary(rna, "rna_before_filter")
    rna = rna[(rna.obs["n_genes_by_counts"] >= min_genes) & (rna.obs["pct_counts_mt"] <= max_pct_mt)].copy()
    sc.pp.filter_genes(rna, min_cells=min_cells)
    post_qc = _sample_qc_summary(rna, "rna_after_filter")
    rna.layers["counts"] = rna.X.copy()
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    rna.raw = rna.copy()
    sc.pp.highly_variable_genes(rna, n_top_genes=n_top_genes, flavor="seurat", batch_key="sample_key", subset=False)
    rna_hvg = rna[:, rna.var["highly_variable"].fillna(False).to_numpy()].copy()
    sc.pp.scale(rna_hvg, max_value=10)
    sc.tl.pca(rna_hvg, svd_solver="arpack")
    sc.pp.neighbors(rna_hvg, n_neighbors=n_neighbors, n_pcs=min(n_pcs, rna_hvg.obsm["X_pca"].shape[1]))
    sc.tl.umap(rna_hvg)
    sc.tl.leiden(rna_hvg, resolution=leiden_resolution, key_added="leiden")
    rna.obs["leiden"] = rna_hvg.obs["leiden"].astype("category")
    rna.obsm["X_pca"] = rna_hvg.obsm["X_pca"]
    rna.obsm["X_umap"] = rna_hvg.obsm["X_umap"]
    rna.uns["neighbors"] = rna_hvg.uns["neighbors"]
    if "umap" in rna_hvg.uns:
        rna.uns["umap"] = rna_hvg.uns["umap"]
    sc.tl.rank_genes_groups(rna, groupby="leiden", method="wilcoxon", use_raw=True)
    rna_marker_df = _extract_marker_table(rna, top_n=marker_top_n)
    rna_marker_df["used_for_annotation"] = False
    for cluster in rna_marker_df["cluster"].astype(str).unique():
        mask = (rna_marker_df["cluster"].astype(str) == cluster) & (~rna_marker_df["is_linc_like"])
        rna_marker_df.loc[rna_marker_df.loc[mask].head(annotation_marker_top_n).index, "used_for_annotation"] = True

    spatial = _load_spatial_objects(spatial_input)
    spatial.var_names_make_unique()
    if "barcode" not in spatial.obs.columns:
        spatial.obs["barcode"] = [name.split(":")[-1] for name in spatial.obs_names.astype(str)]
    spatial.var["mt"] = spatial.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(spatial, qc_vars=["mt"], inplace=True)
    spatial = spatial[(spatial.obs["n_genes_by_counts"] >= min_genes) & (spatial.obs["pct_counts_mt"] <= max_pct_mt)].copy()
    sc.pp.filter_genes(spatial, min_cells=min_cells)
    spatial.layers["counts"] = spatial.X.copy()
    sc.pp.normalize_total(spatial, target_sum=1e4)
    sc.pp.log1p(spatial)
    spatial.raw = spatial.copy()
    batch_key = "sample_key" if "sample_key" in spatial.obs.columns and spatial.obs["sample_key"].nunique() > 1 else None
    sc.pp.highly_variable_genes(spatial, n_top_genes=n_top_genes, flavor="seurat", batch_key=batch_key, subset=False)
    spatial_hvg = spatial[:, spatial.var["highly_variable"].fillna(False).to_numpy()].copy()
    sc.pp.scale(spatial_hvg, max_value=10)
    sc.tl.pca(spatial_hvg, svd_solver="arpack")
    sc.pp.neighbors(spatial_hvg, n_neighbors=n_neighbors, n_pcs=min(n_pcs, spatial_hvg.obsm["X_pca"].shape[1]))
    sc.tl.umap(spatial_hvg)
    sc.tl.leiden(spatial_hvg, resolution=leiden_resolution, key_added="leiden")
    spatial.obs["leiden"] = spatial_hvg.obs["leiden"].astype("category")
    spatial.obsm["X_pca"] = spatial_hvg.obsm["X_pca"]
    spatial.obsm["X_umap"] = spatial_hvg.obsm["X_umap"]
    if "spatial" in spatial_hvg.obsm:
        spatial.obsm["spatial"] = spatial_hvg.obsm["spatial"]
    sc.tl.rank_genes_groups(spatial, groupby="leiden", method="wilcoxon", use_raw=True)
    spatial_marker_df = _extract_marker_table(spatial, top_n=marker_top_n)
    spatial_marker_df["used_for_annotation"] = False
    for cluster in spatial_marker_df["cluster"].astype(str).unique():
        mask = (spatial_marker_df["cluster"].astype(str) == cluster) & (~spatial_marker_df["is_linc_like"])
        spatial_marker_df.loc[spatial_marker_df.loc[mask].head(annotation_marker_top_n).index, "used_for_annotation"] = True

    annotation_notes = read_text(annotation_notes_path) if annotation_notes_path else ""
    rna_annotation_df = _annotate_clusters_with_llm(rna_marker_df, model_name=annotation_model, logger=logger, annotation_notes=annotation_notes)
    spatial_annotation_df = _annotate_clusters_with_llm(
        spatial_marker_df,
        model_name=annotation_model,
        logger=logger,
        annotation_notes=(annotation_notes + "\nLabel broad spatial spot programs or domains; avoid overclaiming single pure cell types.").strip(),
    )
    rna.obs["cluster_cell_type"] = rna.obs["leiden"].astype(str).map(dict(zip(rna_annotation_df["cluster_id"].astype(str), rna_annotation_df["cell_type"].astype(str)))).fillna("unresolved")
    spatial.obs["cluster_cell_type"] = spatial.obs["leiden"].astype(str).map(dict(zip(spatial_annotation_df["cluster_id"].astype(str), spatial_annotation_df["cell_type"].astype(str)))).fillna("unresolved")
    rna.obs["cluster_cell_type"] = rna.obs["cluster_cell_type"].astype("category")
    spatial.obs["cluster_cell_type"] = spatial.obs["cluster_cell_type"].astype("category")

    figures_dir = outdir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    rna_umap_path = figures_dir / "rna_umap_cluster_cell_type.png"
    spatial_umap_path = figures_dir / "spatial_umap_leiden.png"
    spatial_map_path = figures_dir / "spatial_map_cluster_cell_type.png"
    _save_umap_figure(rna, "cluster_cell_type", rna_umap_path, "scRNA UMAP by cluster annotation")
    _save_umap_figure(spatial, "leiden", spatial_umap_path, "Spatial UMAP by Leiden")
    _save_spatial_map(spatial, "cluster_cell_type", spatial_map_path, "Spatial map by cluster annotation")

    rna_h5ad_path = outdir / "processed_rna.h5ad"
    spatial_h5ad_path = outdir / "processed_spatial.h5ad"
    rna_markers_path = outdir / "rna_cluster_markers.csv"
    spatial_markers_path = outdir / "spatial_cluster_markers.csv"
    rna_annotations_path = outdir / "rna_cluster_annotations.csv"
    spatial_annotations_path = outdir / "spatial_cluster_annotations.csv"
    qc_summary_path = outdir / "qc_summary.txt"
    manifest_path = outdir / "prep_manifest.json"

    rna.write_h5ad(rna_h5ad_path)
    spatial.write_h5ad(spatial_h5ad_path)
    rna_marker_df.to_csv(rna_markers_path, index=False)
    spatial_marker_df.to_csv(spatial_markers_path, index=False)
    rna_annotation_df.to_csv(rna_annotations_path, index=False)
    spatial_annotation_df.to_csv(spatial_annotations_path, index=False)

    qc_lines = [
        "RNA QC before filtering",
        pre_qc.to_string(index=False),
        "",
        "RNA QC after filtering",
        post_qc.to_string(index=False),
        "",
        f"Spatial spots after filtering: {spatial.n_obs}",
        f"Spatial genes after filtering: {spatial.n_vars}",
        f"Spatial samples: {spatial.obs['sample_key'].nunique() if 'sample_key' in spatial.obs.columns else 1}",
    ]
    qc_summary_path.write_text("\n".join(qc_lines), encoding="utf-8")

    manifest = {
        "rna_raw_input_path": str(Path(rna_raw_input_path).resolve()),
        "spatial_raw_input_path": str(Path(spatial_raw_input_path).resolve()),
        "rna_h5ad_path": str(rna_h5ad_path),
        "spatial_h5ad_path": str(spatial_h5ad_path),
        "rna_cells_after_filter": int(rna.n_obs),
        "spatial_spots_after_filter": int(spatial.n_obs),
        "shared_genes": int(len(set(map(str, rna.var_names)).intersection(map(str, spatial.var_names)))),
        "rna_clusters": int(rna.obs["leiden"].nunique()),
        "spatial_clusters": int(spatial.obs["leiden"].nunique()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"Spatial preparation complete. Outputs written to {outdir}")

    return PreparationResult(
        output_dir=outdir,
        rna_h5ad_path=rna_h5ad_path,
        spatial_h5ad_path=spatial_h5ad_path,
        rna_markers_path=rna_markers_path,
        spatial_markers_path=spatial_markers_path,
        rna_annotations_path=rna_annotations_path,
        spatial_annotations_path=spatial_annotations_path,
        qc_summary_path=qc_summary_path,
        manifest_path=manifest_path,
        rna_umap_path=rna_umap_path,
        spatial_umap_path=spatial_umap_path,
        spatial_map_path=spatial_map_path,
    )
