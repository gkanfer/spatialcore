"""Shared utilities for optional imports, validation, saving, and matrices."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse


def optional_import(module_name: str, package_hint: str | None = None) -> Any:
    """Import an optional dependency with a concise installation hint."""
    try:
        return import_module(module_name)
    except ImportError as exc:
        if package_hint == "core":
            install_hint = "`pip install -e .`"
        elif package_hint:
            install_hint = f"`pip install -e '.[{package_hint}]'`"
        else:
            install_hint = f"`pip install {module_name}`"
        raise ImportError(
            f"`{module_name}` is required for this workflow. Install it with "
            f"{install_hint}, or install the package directly."
        ) from exc


def require_columns(df: pd.DataFrame, columns: Iterable[str], df_name: str = "DataFrame") -> None:
    """Raise a clear error if required columns are missing."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"{df_name} is missing required columns: {missing}")


def require_obsm(adata: Any, key: str = "spatial") -> np.ndarray:
    """Return `adata.obsm[key]` or raise a useful error."""
    if not hasattr(adata, "obsm") or key not in adata.obsm:
        raise KeyError(f"`adata.obsm['{key}']` is required.")
    return np.asarray(adata.obsm[key])


def get_layer_or_x(adata: Any, layer: str | None = None) -> Any:
    """Return a requested AnnData layer or `adata.X`."""
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"`adata.layers['{layer}']` was not found.")
    return adata.layers[layer]


def to_dense_array(matrix: Any) -> np.ndarray:
    """Convert a sparse or dense matrix to a NumPy array."""
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def matrix_sum(matrix: Any, axis: int) -> np.ndarray:
    """Sum dense or sparse matrices and always return a flat NumPy array."""
    values = matrix.sum(axis=axis)
    return np.asarray(values).ravel()


def matrix_nnz(matrix: Any, axis: int) -> np.ndarray:
    """Count non-zero entries in dense or sparse matrices."""
    if sparse.issparse(matrix):
        values = (matrix > 0).sum(axis=axis)
    else:
        values = (np.asarray(matrix) > 0).sum(axis=axis)
    return np.asarray(values).ravel()


def ensure_directory(path: str | Path) -> Path:
    """Create and return a directory path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_figure_if_requested(fig: Any, save_path: str | Path | None, dpi: int = 300) -> None:
    """Save a Matplotlib figure only when `save_path` is provided."""
    if save_path is not None:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")


def save_table_if_requested(df: pd.DataFrame, save_path: str | Path | None, **kwargs: Any) -> None:
    """Save a table only when `save_path` is provided."""
    if save_path is None:
        return
    path = Path(save_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=kwargs.pop("index", False), **kwargs)
    elif suffix in {".tsv", ".txt"}:
        df.to_csv(path, sep="\t", index=kwargs.pop("index", False), **kwargs)
    elif suffix in {".xlsx", ".xls"}:
        df.to_excel(path, index=kwargs.pop("index", False), **kwargs)
    else:
        raise ValueError("Unsupported table extension. Use .csv, .tsv, .txt, .xlsx, or .xls.")
