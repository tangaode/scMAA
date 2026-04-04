"""Utility helpers for scMAA."""

from __future__ import annotations

import ast
import importlib
import inspect
import math
import textwrap
from pathlib import Path
from typing import Iterable

import pandas as pd


SAMPLE_COLUMN_CANDIDATES = (
    "sample_key",
    "sample_id",
    "sample",
    "orig.ident",
    "donor",
    "patient",
    "library_id",
)


def read_text(path: str | Path, default: str = "") -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return default


def truncate_text(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 32] + "\n...[truncated]..."


def load_tcr_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".gz", ".bz2"}:
        if path.name.endswith(".csv.gz"):
            return pd.read_csv(path)
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported TCR table format: {path}")


def normalize_tcr_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common TCR annotation columns onto a consistent naming scheme."""
    aliases = {
        "barcode": ["barcode", "cell_id", "cell_barcode", "raw_clonotype_id"],
        "sample_id": ["sample", "sample_id", "orig.ident", "donor", "patient", "library_id"],
        "sample_key": ["sample_key", "sample_name", "sample_label"],
        "clonotype_id": ["clonotype_id", "raw_clonotype_id", "clone_id", "clonotype"],
        "chain": ["chain", "locus"],
        "cdr3": ["cdr3", "cdr3_aa", "cdr3s_aa", "cdr3_nt"],
        "v_gene": ["v_gene", "v_call", "v_segment", "trav", "trbv"],
        "j_gene": ["j_gene", "j_call", "j_segment", "traj", "trbj"],
        "productive": ["productive", "high_confidence", "is_productive"],
        "reads": ["reads", "umis", "consensus_count", "duplicate_count"],
    }
    renamed = df.copy()
    lower_to_original = {str(col).lower(): col for col in renamed.columns}
    for target, candidates in aliases.items():
        if target in renamed.columns:
            continue
        for candidate in candidates:
            if candidate.lower() in lower_to_original:
                renamed[target] = renamed[lower_to_original[candidate.lower()]]
                break
    return renamed


def normalize_barcode(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    if ":" in text:
        text = text.rsplit(":", 1)[-1]
    return text


def barcode_core(value: object) -> str:
    text = normalize_barcode(value)
    if not text:
        return ""
    return text.split("-")[0]


def normalize_sample_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def infer_sample_column(columns: Iterable[object]) -> str | None:
    normalized = {str(col).lower(): str(col) for col in columns}
    for candidate in SAMPLE_COLUMN_CANDIDATES:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None


def make_merge_key(barcode_value: object, sample_value: object = None, *, use_core: bool = False) -> str:
    barcode = barcode_core(barcode_value) if use_core else normalize_barcode(barcode_value)
    sample = normalize_sample_value(sample_value)
    if not barcode:
        return ""
    return f"{sample}::{barcode}" if sample else barcode


def extract_call_names(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except (IndentationError, SyntaxError):
        try:
            tree = ast.parse(textwrap.dedent(source))
        except (IndentationError, SyntaxError):
            return []

    calls: set[str] = set()

    def full_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = full_name(node.value)
            return f"{parent}.{node.attr}" if parent else None
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = full_name(node.func)
            if name:
                calls.add(name)
    return sorted(calls)


def load_namespace(source: str, filename: str = "<string>") -> dict:
    namespace: dict = {}
    try:
        exec(compile(source, filename, "exec"), namespace)
        return namespace
    except Exception:
        pass

    try:
        tree = ast.parse(textwrap.dedent(source))
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        import_module = ast.Module(body=imports, type_ignores=[])
        exec(compile(import_module, filename, "exec"), namespace)
    except Exception:
        pass
    return namespace


def resolve_obj(fqname: str, namespace: dict) -> object:
    parts = fqname.split(".")
    if parts[0] in namespace:
        obj = namespace[parts[0]]
    else:
        obj = importlib.import_module(parts[0])
    for attr in parts[1:]:
        obj = getattr(obj, attr)
    return obj


def _normalize_doc_name(name: str) -> str:
    replacements = {
        "sc.": "scanpy.",
        "ad.": "anndata.",
        "ir.": "scirpy.",
    }
    for prefix, replacement in replacements.items():
        if name.startswith(prefix):
            return replacement + name[len(prefix) :]
    return name


def get_documentation(code: str, max_characters: int = 12000) -> str:
    allowed_prefixes = (
        "scanpy.",
        "scirpy.",
        "anndata.",
        "pandas.",
        "numpy.",
        "scipy.",
        "sklearn.",
    )

    calls = extract_call_names(code)
    namespace = load_namespace(code)
    docs: list[str] = []
    for raw_name in calls:
        name = _normalize_doc_name(raw_name)
        if not name.startswith(allowed_prefixes):
            continue
        try:
            obj = resolve_obj(name, namespace)
            doc = inspect.getdoc(obj) or "<no docstring available>"
        except Exception as exc:
            doc = f"<could not resolve: {exc}>"
        docs.append(f"{name}:\n{doc}")
    return truncate_text("\n\n".join(docs), max_characters)


def _output_text(output: object) -> str:
    output_type = getattr(output, "output_type", None) or output.get("output_type")
    if output_type == "stream":
        return str(getattr(output, "text", None) or output.get("text", ""))
    if output_type == "execute_result":
        data = getattr(output, "data", None) or output.get("data", {})
        return str(data.get("text/plain", ""))
    if output_type == "display_data":
        data = getattr(output, "data", None) or output.get("data", {})
        if "text/plain" in data:
            return str(data.get("text/plain", ""))
        if "image/png" in data:
            return "[image/png output]"
        return "[display output]"
    if output_type == "error":
        ename = getattr(output, "ename", None) or output.get("ename", "")
        evalue = getattr(output, "evalue", None) or output.get("evalue", "")
        return f"{ename}: {evalue}"
    return ""


def summarize_notebook_cells(notebook_cells: Iterable[object], max_chars: int = 12000) -> str:
    parts: list[str] = []
    for idx, cell in enumerate(notebook_cells, start=1):
        cell_type = getattr(cell, "cell_type", None) or cell.get("cell_type")
        source = getattr(cell, "source", None) or cell.get("source", "")
        parts.append(f"[Cell {idx} | {cell_type}]\n{source}")
        if cell_type == "code":
            outputs = getattr(cell, "outputs", None) or cell.get("outputs", [])
            output_preview = "\n".join(
                part for part in (_output_text(output) for output in outputs) if part
            ).strip()
            if output_preview:
                parts.append(f"[Cell {idx} Outputs]\n{truncate_text(output_preview, 1200)}")
    return truncate_text("\n\n".join(parts), max_chars)

