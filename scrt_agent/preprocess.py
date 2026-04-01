"""Raw-data preparation pipeline for scRT-agent."""

from __future__ import annotations

import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from matplotlib import pyplot as plt
from pydantic import BaseModel, Field
from scipy.io import mmread

from .logger import AgentLogger
from .utils import normalize_tcr_columns, read_text

try:
    import gzip
except Exception:  # pragma: no cover
    gzip = None

try:
    import instructor
    import litellm

    litellm.drop_params = True
except Exception:  # pragma: no cover
    instructor = None
    litellm = None


RAW_BARCODES_SUFFIX = "_barcodes.tsv.gz"
RAW_FEATURES_SUFFIX = "_features.tsv.gz"
RAW_MATRIX_SUFFIX = "_matrix.mtx.gz"
RAW_TCR_SUFFIX = "_filtered_contig_annotations.csv.gz"


@dataclass
class SampleFiles:
    sample_key: str
    sample_id: str
    tissue: str
    gsm_accession: str | None = None
    barcodes_path: Path | None = None
    features_path: Path | None = None
    matrix_path: Path | None = None
    tcr_path: Path | None = None
    tcr_gsm_accession: str | None = None


@dataclass
class PreparationResult:
    output_dir: Path
    rna_h5ad_path: Path
    tcr_table_path: Path
    cluster_markers_path: Path
    cluster_annotations_path: Path
    qc_summary_path: Path
    manifest_path: Path
    umap_cluster_path: Path
    umap_annotation_path: Path


class ClusterAnnotationRecord(BaseModel):
    cluster_id: str = Field(description="Cluster identifier.")
    cell_type: str = Field(description="Best label for this cluster.")
    confidence: str = Field(description="One of: high, medium, low.")
    rationale: str = Field(description="Short explanation using the marker genes.")
    supporting_markers: list[str] = Field(description="Marker genes supporting the label.")


class ClusterAnnotationResponse(BaseModel):
    overall_notes: str = Field(description="Short notes about uncertainty across clusters.")
    annotations: list[ClusterAnnotationRecord] = Field(description="One annotation per cluster.")


def _normalize_model_name(model: str) -> str:
    if "/" in model:
        return model
    return model


def _is_linc_like(gene_name: object) -> bool:
    text = str(gene_name or "").strip().upper()
    return text.startswith("LINC")


def _parse_sample_key(sample_key: str) -> tuple[str, str]:
    tokens = sample_key.split("_")
    if len(tokens) <= 1:
        return sample_key, "unknown"
    return tokens[0], "_".join(tokens[1:])


def _discover_from_directory(input_dir: Path) -> dict[str, SampleFiles]:
    samples: dict[str, SampleFiles] = {}
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        name = path.name
        suffix = None
        if name.endswith(RAW_BARCODES_SUFFIX):
            suffix = RAW_BARCODES_SUFFIX
            slot = "barcodes_path"
        elif name.endswith(RAW_FEATURES_SUFFIX):
            suffix = RAW_FEATURES_SUFFIX
            slot = "features_path"
        elif name.endswith(RAW_MATRIX_SUFFIX):
            suffix = RAW_MATRIX_SUFFIX
            slot = "matrix_path"
        elif name.endswith(RAW_TCR_SUFFIX):
            suffix = RAW_TCR_SUFFIX
            slot = "tcr_path"
        else:
            continue

        stem = name[: -len(suffix)]
        if "_" not in stem:
            continue
        gsm_accession, sample_key = stem.split("_", 1)
        sample_id, tissue = _parse_sample_key(sample_key)
        sample = samples.setdefault(
            sample_key,
            SampleFiles(sample_key=sample_key, sample_id=sample_id, tissue=tissue),
        )
        if slot == "tcr_path":
            sample.tcr_gsm_accession = gsm_accession
        else:
            sample.gsm_accession = gsm_accession
        setattr(sample, slot, path)
    return samples


def _stage_raw_input(raw_input_path: str | Path, stage_dir: Path, logger: AgentLogger) -> Path:
    raw_path = Path(raw_input_path).resolve()
    if raw_path.is_dir():
        logger.info(f"Using raw input directory: {raw_path}")
        return raw_path
    if tarfile.is_tarfile(raw_path):
        extract_dir = stage_dir / f"{raw_path.stem}_extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        expected = list(extract_dir.iterdir())
        if expected:
            logger.info(f"Using existing extracted raw files in {extract_dir}")
            return extract_dir
        logger.info(f"Extracting raw archive {raw_path} to {extract_dir}")
        with tarfile.open(raw_path, "r") as handle:
            handle.extractall(extract_dir)
        return extract_dir
    raise ValueError(f"Unsupported raw input path: {raw_input_path}")


def _read_table(path: Path, sep: str = "\t", header: int | None = None) -> pd.DataFrame:
    compression = "gzip" if path.suffix == ".gz" or path.name.endswith(".gz") else None
    return pd.read_csv(path, sep=sep, header=header, compression=compression)


def _read_10x_sample(sample: SampleFiles) -> ad.AnnData:
    if not sample.barcodes_path or not sample.features_path or not sample.matrix_path:
        raise ValueError(f"RNA files are incomplete for sample {sample.sample_key}")

    barcodes = _read_table(sample.barcodes_path, sep="\t", header=None).iloc[:, 0].astype(str).tolist()
    features = _read_table(sample.features_path, sep="\t", header=None)
    if gzip is None:
        raise RuntimeError("gzip module is unavailable")
    with gzip.open(sample.matrix_path, "rb") as handle:
        matrix = mmread(handle).tocsr()
    if matrix.shape[1] != len(barcodes):
        raise ValueError(
            f"Barcode count does not match matrix columns for {sample.sample_key}: "
            f"{len(barcodes)} vs {matrix.shape[1]}"
        )

    feature_columns = ["gene_id", "gene_name", "feature_type"][: features.shape[1]]
    features.columns = feature_columns
    if "gene_name" not in features.columns:
        features["gene_name"] = features.iloc[:, 0].astype(str)
    if "feature_type" in features.columns:
        gene_mask = features["feature_type"].astype(str).eq("Gene Expression").to_numpy()
        matrix = matrix[gene_mask, :]
        features = features.loc[gene_mask].reset_index(drop=True)

    adata = ad.AnnData(X=matrix.T.tocsr())
    adata.var["gene_id"] = features["gene_id"].astype(str).to_numpy() if "gene_id" in features.columns else features["gene_name"].astype(str).to_numpy()
    adata.var["gene_name"] = features["gene_name"].astype(str).to_numpy()
    if "feature_type" in features.columns:
        adata.var["feature_type"] = features["feature_type"].astype(str).to_numpy()
    adata.var_names = pd.Index(adata.var["gene_name"].astype(str))
    adata.var_names_make_unique()
    adata.obs_names = pd.Index([f"{sample.sample_key}:{barcode}" for barcode in barcodes])
    adata.obs["barcode"] = barcodes
    adata.obs["sample_key"] = sample.sample_key
    adata.obs["sample_id"] = sample.sample_id
    adata.obs["tissue"] = sample.tissue
    adata.obs["gsm_accession_rna"] = sample.gsm_accession or "unknown"
    return adata


def _load_and_merge_tcr(samples: Iterable[SampleFiles]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for sample in samples:
        if not sample.tcr_path:
            continue
        df = pd.read_csv(sample.tcr_path, compression="gzip")
        df = normalize_tcr_columns(df)
        if "barcode" not in df.columns:
            continue
        df["barcode"] = df["barcode"].astype(str)
        df["sample_key"] = sample.sample_key
        df["sample_id"] = sample.sample_id
        df["tissue"] = sample.tissue
        df["gsm_accession_tcr"] = sample.tcr_gsm_accession or "unknown"
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["barcode", "sample_key", "sample_id", "tissue"])
    merged = pd.concat(frames, ignore_index=True)
    return normalize_tcr_columns(merged)


def _sample_qc_summary(adata: ad.AnnData, stage_name: str) -> pd.DataFrame:
    grouped = adata.obs.groupby("sample_key", observed=True)
    rows = []
    for sample_key, frame in grouped:
        rows.append(
            {
                "stage": stage_name,
                "sample_key": sample_key,
                "sample_id": frame["sample_id"].iloc[0] if "sample_id" in frame.columns else "unknown",
                "tissue": frame["tissue"].iloc[0] if "tissue" in frame.columns else "unknown",
                "cells": int(len(frame)),
                "median_counts": float(frame["total_counts"].median()),
                "median_genes": float(frame["n_genes_by_counts"].median()),
                "median_pct_mt": float(frame["pct_counts_mt"].median()),
            }
        )
    return pd.DataFrame(rows)


def _extract_marker_table(adata: ad.AnnData, top_n: int = 100) -> pd.DataFrame:
    groups = [str(item) for item in adata.obs["leiden"].cat.categories]
    frames: list[pd.DataFrame] = []
    for group in groups:
        frame = sc.get.rank_genes_groups_df(adata, group=group)
        if frame.empty:
            continue
        frame = frame.head(top_n).copy()
        frame["cluster"] = group
        frame["rank"] = np.arange(1, len(frame) + 1)
        frame["is_linc_like"] = frame["names"].map(_is_linc_like)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["cluster", "rank", "names", "logfoldchanges", "pvals_adj", "is_linc_like"])
    return pd.concat(frames, ignore_index=True)


def _annotation_marker_summary(marker_df: pd.DataFrame, top_n: int = 50) -> str:
    lines: list[str] = []
    for cluster in sorted(marker_df["cluster"].astype(str).unique(), key=lambda x: (len(x), x)):
        subset = marker_df.loc[marker_df["cluster"].astype(str) == cluster].copy()
        subset = subset.loc[~subset["is_linc_like"]].head(top_n)
        genes = subset["names"].astype(str).tolist()
        lines.append(f"Cluster {cluster}: {', '.join(genes) if genes else 'no markers available'}")
    return "\n".join(lines)


def _annotate_clusters_with_llm(
    marker_df: pd.DataFrame,
    *,
    model_name: str,
    logger: AgentLogger,
    annotation_notes: str = "",
) -> pd.DataFrame:
    clusters = sorted(marker_df["cluster"].astype(str).unique(), key=lambda x: (len(x), x))
    if not clusters:
        return pd.DataFrame(columns=["cluster_id", "cell_type", "confidence", "rationale", "supporting_markers"])
    if instructor is None or litellm is None:
        logger.warning("LLM annotation libraries are unavailable; using placeholder cluster labels.")
        return pd.DataFrame(
            [
                {
                    "cluster_id": cluster,
                    "cell_type": f"Cluster {cluster}",
                    "confidence": "low",
                    "rationale": "LLM annotation was unavailable during preprocessing.",
                    "supporting_markers": "",
                }
                for cluster in clusters
            ]
        )

    prompt = (
        "You label human single-cell RNA-seq clusters using marker genes.\n"
        "Return one label per cluster.\n"
        "Use plain cell type or cell state names.\n"
        "Do not invent long titles.\n"
        "If uncertain, say unresolved and keep the label broad.\n\n"
        f"Extra notes from the user:\n{annotation_notes or 'No extra notes.'}\n\n"
        "Cluster markers:\n"
        f"{_annotation_marker_summary(marker_df)}"
    )
    logger.log_prompt("user", prompt, "cluster_annotation")
    client = instructor.from_litellm(litellm.completion)
    try:
        response = client.chat.completions.create(
            model=_normalize_model_name(model_name),
            messages=[
                {
                    "role": "system",
                    "content": "You annotate clusters from marker genes for scRNA-seq preprocessing.",
                },
                {"role": "user", "content": prompt},
            ],
            response_model=ClusterAnnotationResponse,
        )
        logger.log_response(response.model_dump_json(indent=2), "cluster_annotation")
        by_cluster = {item.cluster_id: item for item in response.annotations}
        records = []
        for cluster in clusters:
            item = by_cluster.get(cluster)
            if item is None:
                records.append(
                    {
                        "cluster_id": cluster,
                        "cell_type": f"Cluster {cluster}",
                        "confidence": "low",
                        "rationale": "The model did not return a label for this cluster.",
                        "supporting_markers": "",
                    }
                )
                continue
            records.append(
                {
                    "cluster_id": item.cluster_id,
                    "cell_type": item.cell_type,
                    "confidence": item.confidence,
                    "rationale": item.rationale,
                    "supporting_markers": "|".join(item.supporting_markers),
                }
            )
        return pd.DataFrame(records)
    except Exception as exc:
        logger.warning(f"LLM cluster annotation failed; using placeholder labels. Error: {exc}")
        return pd.DataFrame(
            [
                {
                    "cluster_id": cluster,
                    "cell_type": f"Cluster {cluster}",
                    "confidence": "low",
                    "rationale": f"LLM annotation failed: {exc}",
                    "supporting_markers": "",
                }
                for cluster in clusters
            ]
        )


def _write_qc_summary_text(
    *,
    pre_qc: pd.DataFrame,
    post_qc: pd.DataFrame,
    sample_table: pd.DataFrame,
    unmatched_rna_samples: list[str],
    unmatched_tcr_samples: list[str],
) -> str:
    lines = [
        "Sample table",
        sample_table.to_string(index=False) if not sample_table.empty else "No samples found.",
        "",
        "QC summary before filtering",
        pre_qc.to_string(index=False) if not pre_qc.empty else "No cells before filtering.",
        "",
        "QC summary after filtering",
        post_qc.to_string(index=False) if not post_qc.empty else "No cells after filtering.",
        "",
        f"RNA-only samples: {', '.join(unmatched_rna_samples) if unmatched_rna_samples else 'none'}",
        f"TCR-only samples: {', '.join(unmatched_tcr_samples) if unmatched_tcr_samples else 'none'}",
    ]
    return "\n".join(lines)


def _save_qc_figures(adata: ad.AnnData, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    qc_path = figures_dir / "qc_violin.png"
    sc.pl.violin(
        adata,
        ["total_counts", "n_genes_by_counts", "pct_counts_mt"],
        groupby="sample_key",
        rotation=90,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(qc_path, bbox_inches="tight")
    plt.close("all")


def _save_umap_figure(adata: ad.AnnData, color: str, output_path: Path, title: str) -> None:
    sc.pl.umap(adata, color=color, show=False, title=title)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close("all")


def prepare_dataset(
    *,
    raw_input_path: str,
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
    log_dir = outdir / "logs"
    logger = AgentLogger("prepare_data", log_dir, log_prompts=log_prompts)
    sc.settings.verbosity = 2
    sc.settings.set_figure_params(dpi=120, facecolor="white", frameon=False)

    staged_input = _stage_raw_input(raw_input_path, outdir / "work", logger)
    discovered = _discover_from_directory(staged_input)
    if not discovered:
        raise ValueError(f"No supported raw files were found under {staged_input}")

    sample_table = pd.DataFrame(
        [
            {
                "sample_key": sample.sample_key,
                "sample_id": sample.sample_id,
                "tissue": sample.tissue,
                "has_rna": bool(sample.barcodes_path and sample.features_path and sample.matrix_path),
                "has_tcr": bool(sample.tcr_path),
            }
            for sample in discovered.values()
        ]
    ).sort_values(["sample_id", "tissue", "sample_key"])
    logger.info("Discovered samples:\n" + sample_table.to_string(index=False))

    rna_samples = [sample for sample in discovered.values() if sample.barcodes_path and sample.features_path and sample.matrix_path]
    tcr_samples = [sample for sample in discovered.values() if sample.tcr_path]
    unmatched_rna_samples = sorted(sample.sample_key for sample in rna_samples if not sample.tcr_path)
    unmatched_tcr_samples = sorted(sample.sample_key for sample in tcr_samples if not sample.barcodes_path)

    if not rna_samples:
        raise ValueError("No complete RNA 10x triplets were found.")

    adatas = []
    for sample in sorted(rna_samples, key=lambda item: item.sample_key):
        logger.info(f"Loading RNA sample {sample.sample_key}")
        adatas.append(_read_10x_sample(sample))
    adata = ad.concat(adatas, join="outer", merge="same")
    adata.obs["sample_key"] = adata.obs["sample_key"].astype("category")
    adata.obs["sample_id"] = adata.obs["sample_id"].astype("category")
    adata.obs["tissue"] = adata.obs["tissue"].astype("category")

    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    pre_qc = _sample_qc_summary(adata, "before_filter")

    keep_cells = (adata.obs["n_genes_by_counts"] >= min_genes) & (adata.obs["pct_counts_mt"] <= max_pct_mt)
    adata = adata[keep_cells].copy()
    sc.pp.filter_genes(adata, min_cells=min_cells)
    post_qc = _sample_qc_summary(adata, "after_filter")
    logger.info(f"Filtered RNA object: {adata.n_obs} cells x {adata.n_vars} genes")

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata.copy()

    batch_key = "sample_key" if adata.obs["sample_key"].nunique() > 1 else None
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor="seurat",
        batch_key=batch_key,
        subset=False,
    )
    hvg_mask = adata.var["highly_variable"].fillna(False).to_numpy()
    if int(hvg_mask.sum()) == 0:
        raise RuntimeError("No highly variable genes were selected.")

    adata_hvg = adata[:, hvg_mask].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, svd_solver="arpack")
    pcs = min(n_pcs, adata_hvg.obsm["X_pca"].shape[1])
    sc.pp.neighbors(adata_hvg, n_neighbors=n_neighbors, n_pcs=pcs)
    sc.tl.umap(adata_hvg)
    sc.tl.leiden(adata_hvg, resolution=leiden_resolution, key_added="leiden")

    adata.obs["leiden"] = adata_hvg.obs["leiden"].astype("category")
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
    adata.obsm["X_umap"] = adata_hvg.obsm["X_umap"]
    adata.obsp["connectivities"] = adata_hvg.obsp["connectivities"]
    adata.obsp["distances"] = adata_hvg.obsp["distances"]
    adata.uns["neighbors"] = adata_hvg.uns["neighbors"]
    if "umap" in adata_hvg.uns:
        adata.uns["umap"] = adata_hvg.uns["umap"]

    sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon", use_raw=True)
    marker_df = _extract_marker_table(adata, top_n=marker_top_n)
    marker_df["used_for_annotation"] = False
    for cluster in marker_df["cluster"].astype(str).unique():
        mask = (marker_df["cluster"].astype(str) == cluster) & (~marker_df["is_linc_like"])
        selected_index = marker_df.loc[mask].head(annotation_marker_top_n).index
        marker_df.loc[selected_index, "used_for_annotation"] = True

    annotation_notes = read_text(annotation_notes_path) if annotation_notes_path else ""
    annotation_df = _annotate_clusters_with_llm(
        marker_df,
        model_name=annotation_model,
        logger=logger,
        annotation_notes=annotation_notes,
    )
    annotation_map = dict(zip(annotation_df["cluster_id"].astype(str), annotation_df["cell_type"].astype(str)))
    confidence_map = dict(zip(annotation_df["cluster_id"].astype(str), annotation_df["confidence"].astype(str)))
    adata.obs["cluster_cell_type"] = adata.obs["leiden"].astype(str).map(annotation_map).fillna("unresolved")
    adata.obs["cluster_annotation_confidence"] = adata.obs["leiden"].astype(str).map(confidence_map).fillna("low")
    adata.obs["cluster_cell_type"] = adata.obs["cluster_cell_type"].astype("category")
    adata.obs["cluster_annotation_confidence"] = adata.obs["cluster_annotation_confidence"].astype("category")

    figures_dir = outdir / "figures"
    _save_qc_figures(adata, figures_dir)
    umap_cluster_path = figures_dir / "umap_leiden.png"
    umap_annotation_path = figures_dir / "umap_cluster_cell_type.png"
    _save_umap_figure(adata, "leiden", umap_cluster_path, "UMAP by leiden")
    _save_umap_figure(adata, "cluster_cell_type", umap_annotation_path, "UMAP by cluster annotation")

    rna_h5ad_path = outdir / "processed_rna.h5ad"
    tcr_table_path = outdir / "merged_tcr_annotations.tsv.gz"
    cluster_markers_path = outdir / "cluster_markers.csv"
    cluster_annotations_path = outdir / "cluster_annotations.csv"
    qc_summary_path = outdir / "qc_summary.txt"
    manifest_path = outdir / "prep_manifest.json"
    sample_qc_csv = outdir / "sample_qc_summary.csv"

    adata.write_h5ad(rna_h5ad_path)
    _load_and_merge_tcr(discovered.values()).to_csv(tcr_table_path, sep="\t", index=False, compression="gzip")
    marker_df.to_csv(cluster_markers_path, index=False)
    annotation_df.to_csv(cluster_annotations_path, index=False)
    pd.concat([pre_qc, post_qc], ignore_index=True).to_csv(sample_qc_csv, index=False)
    qc_summary_path.write_text(
        _write_qc_summary_text(
            pre_qc=pre_qc,
            post_qc=post_qc,
            sample_table=sample_table,
            unmatched_rna_samples=unmatched_rna_samples,
            unmatched_tcr_samples=unmatched_tcr_samples,
        ),
        encoding="utf-8",
    )

    manifest = {
        "raw_input_path": str(Path(raw_input_path).resolve()),
        "staged_input_path": str(staged_input),
        "rna_h5ad_path": str(rna_h5ad_path),
        "tcr_table_path": str(tcr_table_path),
        "cluster_markers_path": str(cluster_markers_path),
        "cluster_annotations_path": str(cluster_annotations_path),
        "qc_summary_path": str(qc_summary_path),
        "sample_qc_csv": str(sample_qc_csv),
        "umap_cluster_path": str(umap_cluster_path),
        "umap_annotation_path": str(umap_annotation_path),
        "cells_after_filter": int(adata.n_obs),
        "genes_after_filter": int(adata.n_vars),
        "n_clusters": int(adata.obs["leiden"].nunique()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"Preparation complete. Outputs written to {outdir}")

    return PreparationResult(
        output_dir=outdir,
        rna_h5ad_path=rna_h5ad_path,
        tcr_table_path=tcr_table_path,
        cluster_markers_path=cluster_markers_path,
        cluster_annotations_path=cluster_annotations_path,
        qc_summary_path=qc_summary_path,
        manifest_path=manifest_path,
        umap_cluster_path=umap_cluster_path,
        umap_annotation_path=umap_annotation_path,
    )
