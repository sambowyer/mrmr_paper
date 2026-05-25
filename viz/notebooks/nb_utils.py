from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib as jbl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter
from tqdm.auto import tqdm

try:
    from data_utils import (
        continuous_cat_main_datasets,
        helm_datasets,
        openllm_datasets,
        pass_at_k_v3_benchmarks,
    )
except Exception:
    openllm_datasets = [
        "ifeval",
        "openllm_math",
        "mmlu_pro",
        "arc_challenge",
        "bbh",
        "gpqa",
        "musr",
    ]
    helm_datasets = ["commonsense", "gsm", "legalbench", "math", "med_qa", "mmlu"]
    continuous_cat_main_datasets = [
        "biolaysumm_rougel",
        "biolaysumm_bertscore",
        "biolaysumm_fkgl",
        "govreport_rougel",
        "govreport_bertscore",
        "truthfulqa_judge",
        "nemotron_pii",
    ]
    pass_at_k_v3_benchmarks = [
        "LBPP_Python",
        "HumanEvalPack_PythonPlus",
        "MBPPPlus_mbpp_plus",
        "MBPP_mbpp",
    ]


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "mrmr").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate project root containing mrmr/")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
MRMR_DIR = PROJECT_ROOT / "mrmr"
RESULTS_ROOT = MRMR_DIR / "results"
CONSOLIDATED_RESULTS_ROOT = MRMR_DIR / "consolidated_results"
SCORES_DIR = MRMR_DIR / "data" / "scores"
CONTINUOUS_CAT_SCORES_JSON = MRMR_DIR / "data" / "continuous_cat_scores" / "scores.json"
PASSK_OPEN_SCORE_DIRS = [
    MRMR_DIR / "data" / "pass_at_k_code_v3" / "open",
    MRMR_DIR / "data" / "pass_at_k_code_v3" / "scores" / "open",
]
MI_CACHE_DIR = MRMR_DIR / "mi_cache"
NOTEBOOKS_DIR = MRMR_DIR / "viz" / "notebooks"
PLOTS_ROOT = MRMR_DIR / "viz" / "plots"
TABLES_DIR = MRMR_DIR / "viz" / "tables"
FONT_DIR = MRMR_DIR / "viz" / "styles" / "fonts"
CACHE_DIR = NOTEBOOKS_DIR / "cache"

_CONSOLIDATED_RESULTS = "_consolidated_results.parquet"
_CONSOLIDATED_CROSSK = "_consolidated_crossk.parquet"
_CONSOLIDATED_JBL = "_consolidated_jbl.parquet"

PASS_AT_K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]
PASS_AT_K_TARGET_KS_DEFAULT = [1, 2, 4, 8, 16, 32, 64]

BINARY_SPLIT_METHOD = "binned_interpolation"
CONTINUOUS_SPLIT_METHOD = "stratified"

BINARY_DATASETS_DEFAULT = [
    "ifeval",
    "openllm_math",
    "mmlu_pro",
    "arc_challenge",
    "bbh",
    "gpqa",
    "musr",
    "commonsense",
    "gsm",
    "legalbench",
    "math",
    "med_qa",
    "mmlu",
]

CONTINUOUS_DATASETS_DEFAULT = [
    "biolaysumm_rougel",
    "biolaysumm_bertscore",
    "biolaysumm_fkgl",
    "govreport_rougel",
    "govreport_bertscore",
    "truthfulqa_judge",
]
CONTINUOUS_DATASETS_OPTIONAL = [
    "nemotron_pii",
]

PASSK_BENCHMARKS_DEFAULT = [
    "LBPP_Python",
    "HumanEvalPack_PythonPlus",
    "MBPPPlus_mbpp_plus",
    "MBPP_mbpp",
]

PASSK_ALLOWED_SUFFIXES_DEFAULT = {"_v3", "_v3_open"}

METRIC_SPECS = {
    "rmse": {
        "column": "rmse",
        "label": r"RMSE (%) $\downarrow$",
        "higher_is_better": False,
        "scale": 100.0,
        "sigma_normalize": True,
    },
    "mae": {
        "column": "error",
        "label": r"MAE (%) $\downarrow$",
        "higher_is_better": False,
        "scale": 100.0,
        "sigma_normalize": True,
    },
    "tau": {
        "column": "corr_kendall",
        "label": r"Kendall $\tau$ $\uparrow$",
        "higher_is_better": True,
        "scale": 1.0,
        "sigma_normalize": False,
    },
    "rho": {
        "column": "corr_spearman",
        "label": r"Spearman $\rho$ $\uparrow$",
        "higher_is_better": True,
        "scale": 1.0,
        "sigma_normalize": False,
    },
    "pearson": {
        "column": "corr_pearson",
        "label": r"Pearson $r$ $\uparrow$",
        "higher_is_better": True,
        "scale": 1.0,
        "sigma_normalize": False,
    },
}

METHOD_ALIASES = {
    "random_sample": "random_sampling",
    "random_sample_and_learn": "random_sampling_and_learn",
    "random_search_and_learn": "random_search_and_learn",
    "anchor_points": "anchor_points_weighted",
}

LINESTYLE_BY_DEGREE = {
    1: "-",
    2: "--",
    3: ":",
    4: "-.",
}

LINESTYLE_BY_IRT_DIM = {
    1: "-",
    5: "--",
    10: ":",
}


def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    r = int(hex_color[1:3], 16) / 255.0
    g = int(hex_color[3:5], 16) / 255.0
    b = int(hex_color[5:7], 16) / 255.0
    return r, g, b


def _rgb01_to_hex(r: float, g: float, b: float) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        int(np.clip(r, 0.0, 1.0) * 255),
        int(np.clip(g, 0.0, 1.0) * 255),
        int(np.clip(b, 0.0, 1.0) * 255),
    )


def _lighten(hex_color: str, amount: float) -> str:
    r, g, b = _hex_to_rgb01(hex_color)
    return _rgb01_to_hex(
        r + (1 - r) * amount,
        g + (1 - g) * amount,
        b + (1 - b) * amount,
    )


def _darken(hex_color: str, amount: float) -> str:
    r, g, b = _hex_to_rgb01(hex_color)
    return _rgb01_to_hex(r * (1 - amount), g * (1 - amount), b * (1 - amount))


FAMILY_STYLE = {
    "mrmr": {"color": "#e91e63", "marker": "^"},
    "irt": {"color": "#2e7d32", "marker": "s"},
    "baseline": {"color": "#616161", "marker": "*"},
    "metabench": {"color": "#7b1fa2", "marker": "D"},
    "anchor": {"color": "#6d4c41", "marker": "p"},
    "other": {"color": "#607d8b", "marker": "o"},
}

OBJECTIVE_COLORS = {
    "MIQ": "#e91e63",
    "MID": "#7e57c2",
    "MI": "#006064",
    "PMI": "#3949ab",
    "FCQ": "#fb8c00",
    "FCD": "#43a047",
    "PMIQ": "#ec407a",
    "PMID": "#8e24aa",
    "QGMIQ": "#d81b60",
    "QGMID": "#5e35b1",
    "KSGMIQ": "#00799b", 
    "KSGMID": "#4db6ac",
}

BASELINE_COLORS = {
    "random_sampling": "#c7c7c7",
    "random_sampling_and_learn": "#7f7f7f",
    "krandom_sampling_and_learn": "#7f7f7f",
    "random_search_and_learn": "#000000",
    "krandom_search_and_learn": "#000000",
    "small_search_and_learn": "#4f4f4f",
    "sample_first_and_learn": "#9e9e9e",
    "lasso": "#64b5f6",
}

IRT_MODEL_COLORS = {
    "IRT": "#2e7d32",
    "gp-IRT": "#2e7d32",
    "LEGO-IRT": "#1b5e20",
    r"$\beta$-IRT": "#2e7d32",
    r"$\beta^3$-IRT": "#26a69a",
    "G-IRT": "#66bb6a",
    "pIRT": "#43a047",
}

IRT_PREFIX_VARIANT_COLORS = {
    ("base", "gp"): "#1b5e20",
    ("base", "p"): "#2e7d32",
    ("LEGO", "gp"): "#42a5f5",
    ("LEGO", "p"): "#90caf9",
    ("B", "gp"): "#1b5e20",
    ("B", "p"): "#2e7d32",
    ("G", "gp"): "#ef6c00",
    ("G", "p"): "#ffb74d",
    ("B3_v2", "gp"): "#8e24aa", ##00897b",
    ("B3_v2", "p"): "#ce93d8", #"#4db6ac",
    ("B3", "gp"): "#6d4c41",
    ("B3", "p"): "#a1887f",
}

_PASS_AT_K_RE = re.compile(
    r"^("
    + "|".join(re.escape(b) for b in sorted(set(pass_at_k_v3_benchmarks + PASSK_BENCHMARKS_DEFAULT)))
    + r")_pass_at_(\d+)(_v3_open|_v3|_v2)?$"
)
_PRED_PASS_AT_RE = re.compile(r"^pred_pass_at_(\d+)$")

_MRMR_RE = re.compile(
    r"^(l?)"
    r"(gpirt_|pirt_|anti_|raw_|syn_|k[34]|cv|rf|k|g)?"
    r"mrmr"
    r"(\d*)"
    r"_((?:(?:Beta|B3|LEGO|G)?IRT)\d+)?"
    r"(QGMID|QGMIQ|GMID|GMIQ|PMIQ|PMID|PMI|MID|MIQ|MI|FCD2|FCQ2|FCD|FCQ)"
    r"_(.+?)"
    r"(_aipw)?"
    r"(\+)?$"
)

_font_registered = False


@dataclass(frozen=True)
class Setting:
    split_method: str
    coreset_size: str
    num_train_models: str
    nmodels_dir: Path


@dataclass
class StyleInfo:
    family: str
    color: str
    marker: str
    linestyle: str
    linewidth: float = 1.0


_SETTING_RESULTS_CACHE: dict[tuple[str, str, str], pd.DataFrame] = {}
_SETTING_CROSSK_CACHE: dict[tuple[str, str, str], pd.DataFrame] = {}
_SETTING_JBL_CACHE: dict[tuple[str, str, str], pd.DataFrame] = {}
_SETTING_CORESET_CACHE: dict[tuple[str, str, str], pd.DataFrame] = {}
_DATASET_SCORE_STATS_MEMO: dict[str, dict[str, float]] | None = None
_SCORE_STATS_PATH = CACHE_DIR / "dataset_score_stats.json"
_SCORE_STATS_META_PATH = CACHE_DIR / "dataset_score_stats_meta.json"
_SCORE_STATS_CACHE_VERSION = 2
LINE_PLOT_MARKER_SIZE = 2.0 # 3.2
SCATTER_LABEL_FONT_SIZE = 6.2
FIG_WIDTH_IN = 6.8
FIG_HEIGHT_TWO_ROW_IN = 3.6
FIG_HEIGHT_ONE_ROW_IN = 2.4

_SAVE_OUTPUTS_ENABLED = True
_DISPLAY_SAVED_PLOTS_ENABLED = True
_CAPTURED_FIGURES: dict[str, plt.Figure] | None = None


def ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)


def register_fonts() -> None:
    global _font_registered
    if _font_registered:
        return
    if FONT_DIR.is_dir():
        patterns = ("*.ttf", "*.otf", "*.ttc", "*.TTF", "*.OTF", "*.TTC")
        for pattern in patterns:
            for font_path in sorted(FONT_DIR.glob(pattern)):
                try:
                    font_manager.fontManager.addfont(str(font_path))
                except Exception:
                    pass
    _font_registered = True


def _dmmono_local_font_name() -> str | None:
    # Prefer an explicit DMMono-Regular file from the local fonts directory.
    for pattern in ("DMMono-Regular.*", "DMMono*.ttf", "DMMono*.otf", "DMMono*.ttc"):
        for font_path in sorted(FONT_DIR.glob(pattern)):
            try:
                return font_manager.FontProperties(fname=str(font_path)).get_name()
            except Exception:
                continue
    return None


def _preferred_font_family() -> list[str]:
    def _unique(names: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            out.append(name)
        return out

    available = {f.name for f in font_manager.fontManager.ttflist}

    dmmono_local = _dmmono_local_font_name()
    if dmmono_local:
        return _unique([dmmono_local, "DM Mono", "DejaVu Sans Mono", "DejaVu Sans"])

    for preferred_name in ("DMMono-Regular", "DM Mono", "DM Mono Regular", "DMMono"):
        if preferred_name in available:
            return _unique([preferred_name, "DejaVu Sans Mono", "DejaVu Sans"])

    # Last chance: match available names by a normalized DMMono token.
    for name in sorted(available):
        normalized = re.sub(r"[-_\s]+", "", name).lower()
        if normalized in {"dmmono", "dmmonoregular"}:
            return _unique([name, "DejaVu Sans Mono", "DejaVu Sans"])

    return ["DejaVu Sans", "DejaVu Sans Mono"]


def setup_matplotlib(base_font_size: int = 8) -> None:
    register_fonts()
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 400,
            "font.size": base_font_size,
            "axes.titlesize": base_font_size + 1,
            "axes.labelsize": base_font_size,
            "legend.fontsize": base_font_size - 1,
            "xtick.labelsize": base_font_size - 1,
            "ytick.labelsize": base_font_size - 1,
            "font.family": _preferred_font_family(),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.color": "#111827",
            "axes.labelcolor": "#111827",
            "xtick.color": "#111827",
            "ytick.color": "#111827",
            "axes.edgecolor": "#111827",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.color": "#CBD5E1",
            "grid.alpha": 0.6,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.0,
            "axes.axisbelow": True,
            "mathtext.default": "regular",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def parse_coreset_size(value: str) -> float:
    text = str(value)
    if text.endswith("%"):
        return float(text[:-1])
    return float(text)


def parse_num_train_models(value: str) -> float:
    text = str(value)
    if text == "default":
        return -1.0
    return float(text)


def nearest_available_label(
    df: pd.DataFrame,
    column: str,
    target: str,
    parser_fn,
) -> str:
    if df.empty or column not in df.columns:
        return str(target)
    labels = sorted(set(df[column].astype(str)))
    if str(target) in labels:
        return str(target)
    numeric = []
    for label in labels:
        try:
            numeric.append((abs(parser_fn(label) - parser_fn(target)), label))
        except Exception:
            continue
    if numeric:
        numeric.sort(key=lambda x: x[0])
        return numeric[0][1]
    return labels[0]


def _setting_sort_key(setting: Setting) -> tuple[str, float, float]:
    return (
        setting.split_method,
        parse_coreset_size(setting.coreset_size),
        parse_num_train_models(setting.num_train_models),
    )


def discover_settings(split_method: str | None = None) -> list[Setting]:
    settings: list[Setting] = []
    if not RESULTS_ROOT.is_dir():
        return settings
    for split_dir in sorted(p for p in RESULTS_ROOT.iterdir() if p.is_dir()):
        if split_method is not None and split_dir.name != split_method:
            continue
        for cs_dir in sorted(p for p in split_dir.iterdir() if p.is_dir() and p.name.startswith("coreset_")):
            coreset_size = cs_dir.name.replace("coreset_", "", 1)
            for nm_dir in sorted(p for p in cs_dir.iterdir() if p.is_dir() and p.name.startswith("nmodels_")):
                num_train_models = nm_dir.name.replace("nmodels_", "", 1)
                settings.append(
                    Setting(
                        split_method=split_dir.name,
                        coreset_size=coreset_size,
                        num_train_models=num_train_models,
                        nmodels_dir=nm_dir,
                    )
                )
    settings.sort(key=_setting_sort_key)
    return settings


def get_setting(split_method: str, coreset_size: str, num_train_models: str) -> Setting:
    for setting in discover_settings(split_method=split_method):
        if setting.coreset_size == str(coreset_size) and setting.num_train_models == str(num_train_models):
            return setting
    raise ValueError(
        f"Could not find setting split={split_method}, coreset={coreset_size}, nmodels={num_train_models}"
    )


def get_setting_flexible(split_method: str, coreset_size: str, num_train_models: str) -> Setting:
    """Return the exact setting when available, otherwise nearest nmodels."""
    settings = [
        setting
        for setting in discover_settings(split_method=split_method)
        if setting.coreset_size == str(coreset_size)
    ]
    if not settings:
        raise ValueError(f"No settings found for split={split_method}, coreset={coreset_size}")

    for setting in settings:
        if setting.num_train_models == str(num_train_models):
            return setting

    numeric_settings = []
    for setting in settings:
        try:
            numeric_settings.append((abs(float(setting.num_train_models) - float(num_train_models)), setting))
        except Exception:
            continue
    if numeric_settings:
        numeric_settings.sort(key=lambda x: x[0])
        return numeric_settings[0][1]
    return settings[0]


def _parse_record(record_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    with record_path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line == "$END$":
                break
            try:
                result.update(json.loads(line))
            except json.JSONDecodeError:
                continue
    return result


def _load_results_fallback(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not results_dir.is_dir():
        return pd.DataFrame(rows)
    for dataset_dir in sorted(p for p in results_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for method_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            for seed_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
                record_path = seed_dir / "record.result"
                if not record_path.is_file():
                    continue
                rec = _parse_record(record_path)
                if not rec:
                    continue
                rmse_val = rec.get("error_RMSE", np.nan)
                if pd.isna(rmse_val):
                    error_mse = rec.get("error_MSE", np.nan)
                    rmse_val = np.sqrt(error_mse) if not pd.isna(error_mse) else np.nan
                rows.append(
                    {
                        "dataset": dataset_dir.name,
                        "method": method_dir.name,
                        "seed": seed_dir.name,
                        "error": rec.get("error_MAE", rec.get("error", np.nan)),
                        "rmse": rmse_val,
                        "training_time": rec.get("training_time", np.nan),
                        "inference_time": rec.get("inference_time", np.nan),
                        "corr_spearman": rec.get("corr_spearman", np.nan),
                        "corr_kendall": rec.get("corr_kendall", np.nan),
                        "corr_pearson": rec.get("corr_pearson", np.nan),
                    }
                )
    return pd.DataFrame(rows)


def _load_crossk_fallback(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not results_dir.is_dir():
        return pd.DataFrame(rows)
    for dataset_dir in sorted(p for p in results_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
        parsed = parse_pass_at_k_dataset_name(dataset_dir.name)
        if parsed is None:
            continue
        benchmark, source_k, _suffix = parsed
        for method_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            for seed_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
                for pred_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
                    m = _PRED_PASS_AT_RE.match(pred_dir.name)
                    if m is None:
                        continue
                    pred_k = int(m.group(1))
                    record_path = pred_dir / "record.result"
                    if not record_path.is_file():
                        continue
                    rec = _parse_record(record_path)
                    if not rec:
                        continue
                    rmse_val = rec.get("error_RMSE", np.nan)
                    if pd.isna(rmse_val):
                        error_mse = rec.get("error_MSE", np.nan)
                        rmse_val = np.sqrt(error_mse) if not pd.isna(error_mse) else np.nan
                    rows.append(
                        {
                            "dataset": dataset_dir.name,
                            "method": method_dir.name,
                            "seed": seed_dir.name,
                            "benchmark": benchmark,
                            "source_k": source_k,
                            "pred_k": pred_k,
                            "error": rec.get("error_MAE", rec.get("error", np.nan)),
                            "rmse": rmse_val,
                            "corr_spearman": rec.get("corr_spearman", np.nan),
                            "corr_kendall": rec.get("corr_kendall", np.nan),
                            "corr_pearson": rec.get("corr_pearson", np.nan),
                        }
                    )
    return pd.DataFrame(rows)


def _decode_json_column(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _load_jbl_fallback(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not results_dir.is_dir():
        return pd.DataFrame(rows)
    for dataset_dir in sorted(p for p in results_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for method_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            for seed_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
                result_path = seed_dir / "result.jbl"
                if not result_path.is_file():
                    continue
                try:
                    result = jbl.load(result_path)
                except Exception:
                    continue
                rows.append(
                    {
                        "dataset": dataset_dir.name,
                        "method": method_dir.name,
                        "seed": seed_dir.name,
                        "coreset_indices": result.get("coreset_indices"),
                        "train_model_indices": result.get("train_model_indices"),
                        "test_model_indices": result.get("test_model_indices"),
                        "true_acc_train": result.get("true_acc_train"),
                        "pred_acc_train": result.get("pred_acc_train"),
                        "true_acc_test": result.get("true_acc_test"),
                        "pred_acc_test": result.get("pred_acc_test"),
                        "selection_metrics": result.get("selection_metrics"),
                    }
                )
    return pd.DataFrame(rows)


def _setting_cache_key(setting: Setting) -> tuple[str, str, str]:
    return (setting.split_method, setting.coreset_size, setting.num_train_models)


def _with_setting_columns(frame: pd.DataFrame, setting: Setting) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["coreset_size"] = setting.coreset_size
    out["num_train_models"] = setting.num_train_models
    return out


def load_setting_results(setting: Setting) -> pd.DataFrame:
    key = _setting_cache_key(setting)
    if key in _SETTING_RESULTS_CACHE:
        return _SETTING_RESULTS_CACHE[key].copy()

    path = setting.nmodels_dir / _CONSOLIDATED_RESULTS
    frame = pd.read_parquet(path) if path.is_file() else _load_results_fallback(setting.nmodels_dir)
    frame = _with_setting_columns(frame, setting)
    _SETTING_RESULTS_CACHE[key] = frame
    return frame.copy()


def load_setting_crossk(setting: Setting) -> pd.DataFrame:
    key = _setting_cache_key(setting)
    if key in _SETTING_CROSSK_CACHE:
        return _SETTING_CROSSK_CACHE[key].copy()

    path = setting.nmodels_dir / _CONSOLIDATED_CROSSK
    frame = pd.read_parquet(path) if path.is_file() else _load_crossk_fallback(setting.nmodels_dir)
    frame = _with_setting_columns(frame, setting)
    _SETTING_CROSSK_CACHE[key] = frame
    return frame.copy()


def _read_jbl_parquet(setting: Setting, columns: list[str] | None = None) -> pd.DataFrame:
    path = setting.nmodels_dir / _CONSOLIDATED_JBL
    if not path.is_file():
        return _load_jbl_fallback(setting.nmodels_dir)
    if columns is None:
        return pd.read_parquet(path)
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        frame = pd.read_parquet(path)
        keep = [col for col in columns if col in frame.columns]
        return frame[keep]


def load_setting_jbl(setting: Setting) -> pd.DataFrame:
    key = _setting_cache_key(setting)
    if key in _SETTING_JBL_CACHE:
        return _SETTING_JBL_CACHE[key].copy()

    frame = _read_jbl_parquet(setting)
    for col in [
        "coreset_indices",
        "train_model_indices",
        "test_model_indices",
        "true_acc_train",
        "pred_acc_train",
        "true_acc_test",
        "pred_acc_test",
        "selection_metrics",
    ]:
        if col in frame.columns:
            frame[col] = frame[col].apply(_decode_json_column)
    frame = _with_setting_columns(frame, setting)
    _SETTING_JBL_CACHE[key] = frame
    return frame.copy()


def load_setting_coresets(setting: Setting) -> pd.DataFrame:
    key = _setting_cache_key(setting)
    if key in _SETTING_CORESET_CACHE:
        return _SETTING_CORESET_CACHE[key].copy()

    frame = _read_jbl_parquet(setting, columns=["dataset", "method", "seed", "coreset_indices"])
    if "coreset_indices" in frame.columns:
        frame["coreset_indices"] = frame["coreset_indices"].apply(_decode_json_column)
    frame = _with_setting_columns(frame, setting)
    _SETTING_CORESET_CACHE[key] = frame
    return frame.copy()


def load_split_results(split_method: str) -> pd.DataFrame:
    frames = [load_setting_results(setting) for setting in discover_settings(split_method=split_method)]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_split_crossk(split_method: str) -> pd.DataFrame:
    frames = [load_setting_crossk(setting) for setting in discover_settings(split_method=split_method)]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_split_jbl(split_method: str) -> pd.DataFrame:
    frames = [load_setting_jbl(setting) for setting in discover_settings(split_method=split_method)]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _cache_key(prefix: str, payload: dict[str, Any]) -> Path:
    payload_text = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{prefix}_{digest}.parquet"


def parse_pass_at_k_dataset_name(dataset_name: str) -> tuple[str, int, str] | None:
    match = _PASS_AT_K_RE.match(dataset_name)
    if match is None:
        return None
    benchmark = match.group(1)
    k = int(match.group(2))
    suffix = match.group(3) or ""
    return benchmark, k, suffix


def _safe_numeric_array(values: Any) -> np.ndarray:
    if isinstance(values, dict):
        iterable = values.values()
    elif isinstance(values, (list, tuple, np.ndarray, pd.Series)):
        iterable = values
    else:
        iterable = [values]
    out: list[float] = []
    for value in iterable:
        if isinstance(value, (dict, list, tuple, np.ndarray, pd.Series)):
            nested = _safe_numeric_array(value)
            if nested.size:
                out.extend(nested.tolist())
            continue
        try:
            num = float(value)
        except Exception:
            continue
        if np.isfinite(num):
            out.append(num)
    return np.asarray(out, dtype=float)


def _compute_score_stats_from_matrix(scores: Any) -> dict[str, float] | None:
    try:
        arr = np.asarray(scores, dtype=float)
    except Exception:
        return None
    if arr.size == 0 or arr.ndim < 2:
        return None
    model_means = np.mean(arr, axis=1)
    model_std = float(np.std(model_means))
    all_std = float(np.std(arr))
    if not np.isfinite(model_std) or model_std <= 0.0:
        return None
    if not np.isfinite(all_std):
        all_std = model_std
    return {
        "model_mean_std": model_std,
        "all_scores_std": all_std,
    }


def _load_continuous_json_score_stats() -> dict[str, dict[str, float]]:
    if not CONTINUOUS_CAT_SCORES_JSON.is_file():
        return {}
    try:
        payload = json.loads(CONTINUOUS_CAT_SCORES_JSON.read_text())
    except Exception:
        return {}

    datasets_payload = payload.get("datasets", {})
    if not isinstance(datasets_payload, dict):
        return {}

    out: dict[str, dict[str, float]] = {}
    for dataset, dataset_payload in datasets_payload.items():
        if not isinstance(dataset_payload, dict):
            continue
        model_payload = dataset_payload.get("scores", {})
        if not isinstance(model_payload, dict):
            continue

        model_means: list[float] = []
        all_scores: list[float] = []
        for _, model_scores in model_payload.items():
            chosen_scores: Any = None
            if isinstance(model_scores, dict):
                # Prefer the canonical deterministic temperature when available.
                if "temp_0.0" in model_scores:
                    chosen_scores = model_scores.get("temp_0.0")
                elif model_scores and all(
                    not isinstance(v, (dict, list, tuple, np.ndarray, pd.Series))
                    for v in model_scores.values()
                ):
                    chosen_scores = model_scores
                elif model_scores:
                    first_key = sorted(model_scores.keys())[0]
                    chosen_scores = model_scores.get(first_key)
            else:
                chosen_scores = model_scores

            vals = _safe_numeric_array(chosen_scores)
            if vals.size == 0:
                continue
            model_means.append(float(np.mean(vals)))
            all_scores.extend(vals.tolist())

        if not model_means:
            continue
        model_means_arr = np.asarray(model_means, dtype=float)
        all_scores_arr = np.asarray(all_scores, dtype=float)
        model_std = float(np.std(model_means_arr))
        all_std = float(np.std(all_scores_arr)) if all_scores_arr.size else model_std
        if not np.isfinite(model_std) or model_std <= 0.0:
            continue
        out[str(dataset)] = {
            "model_mean_std": model_std,
            "all_scores_std": all_std,
        }
    return out


def _iter_passk_open_score_dirs() -> list[Path]:
    return [path for path in PASSK_OPEN_SCORE_DIRS if path.is_dir()]


def _load_passk_open_score_stats() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for passk_dir in _iter_passk_open_score_dirs():
        for score_path in sorted(passk_dir.glob("*.jbl")):
            if score_path.name.endswith("_models.jbl") or score_path.name.endswith("_info.jbl"):
                continue
            try:
                scores = jbl.load(score_path)
            except Exception:
                continue
            stats = _compute_score_stats_from_matrix(scores)
            if stats is None:
                continue
            dataset = score_path.stem
            out[dataset] = stats
            if dataset.endswith("_v3"):
                out[f"{dataset}_open"] = stats
            elif dataset.endswith("_v3_open"):
                out[dataset.removesuffix("_open")] = stats
    return out


def _score_file_digest(score_dir: Path, *, skip_suffixes: tuple[str, ...] = ()) -> dict[str, Any]:
    if not score_dir.is_dir():
        return {"exists": False}
    hasher = hashlib.sha256()
    count = 0
    for score_path in sorted(score_dir.glob("*.jbl")):
        if any(score_path.name.endswith(suffix) for suffix in skip_suffixes):
            continue
        try:
            stat = score_path.stat()
        except Exception:
            continue
        hasher.update(
            f"{score_path.name}:{int(stat.st_mtime_ns)}:{int(stat.st_size)}\n".encode("utf-8")
        )
        count += 1
    return {
        "exists": True,
        "count": count,
        "digest": hasher.hexdigest(),
    }


def _score_sources_signature() -> dict[str, Any]:
    signature: dict[str, Any] = {
        "cache_version": _SCORE_STATS_CACHE_VERSION,
        "continuous_json": {"exists": False},
        "passk_open_dirs": [],
        "legacy_scores": {"exists": False},
    }

    if CONTINUOUS_CAT_SCORES_JSON.is_file():
        try:
            stat = CONTINUOUS_CAT_SCORES_JSON.stat()
            signature["continuous_json"] = {
                "exists": True,
                "mtime_ns": int(stat.st_mtime_ns),
                "size": int(stat.st_size),
            }
        except Exception:
            signature["continuous_json"] = {"exists": True, "mtime_ns": 0, "size": 0}

    for passk_dir in _iter_passk_open_score_dirs():
        digest = _score_file_digest(
            passk_dir,
            skip_suffixes=("_models.jbl", "_info.jbl"),
        )
        digest["path"] = str(passk_dir)
        signature["passk_open_dirs"].append(digest)

    signature["legacy_scores"] = _score_file_digest(
        SCORES_DIR,
        skip_suffixes=("_models.jbl",),
    )
    return signature


def _read_cached_score_stats(path: Path) -> dict[str, dict[str, float]]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text())
    except Exception:
        return {}
    if not isinstance(loaded, dict):
        return {}
    out: dict[str, dict[str, float]] = {}
    for dataset, values in loaded.items():
        if not isinstance(values, dict):
            continue
        model_std = values.get("model_mean_std")
        all_std = values.get("all_scores_std", model_std)
        try:
            model_std_f = float(model_std)
            all_std_f = float(all_std)
        except Exception:
            continue
        if not np.isfinite(model_std_f) or model_std_f <= 0.0:
            continue
        if not np.isfinite(all_std_f):
            all_std_f = model_std_f
        out[str(dataset)] = {
            "model_mean_std": model_std_f,
            "all_scores_std": all_std_f,
        }
    return out


def _score_stats_cache_is_fresh(source_signature: dict[str, Any]) -> bool:
    if not _SCORE_STATS_META_PATH.is_file():
        return False
    try:
        meta = json.loads(_SCORE_STATS_META_PATH.read_text())
    except Exception:
        return False
    if not isinstance(meta, dict):
        return False
    if int(meta.get("cache_version", -1)) != _SCORE_STATS_CACHE_VERSION:
        return False
    return meta.get("source_signature") == source_signature


def _write_score_stats_cache(stats: dict[str, dict[str, float]], source_signature: dict[str, Any]) -> None:
    _SCORE_STATS_PATH.write_text(json.dumps(stats))
    _SCORE_STATS_META_PATH.write_text(
        json.dumps(
            {
                "cache_version": _SCORE_STATS_CACHE_VERSION,
                "source_signature": source_signature,
            }
        )
    )


def _score_stats_cache_token() -> str:
    signature = _score_sources_signature()
    text = json.dumps(signature, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_dataset_score_stats() -> dict[str, dict[str, float]]:
    global _DATASET_SCORE_STATS_MEMO
    if _DATASET_SCORE_STATS_MEMO is not None:
        return _DATASET_SCORE_STATS_MEMO

    source_signature = _score_sources_signature()
    cached_stats = _read_cached_score_stats(_SCORE_STATS_PATH)
    if cached_stats and _score_stats_cache_is_fresh(source_signature):
        _DATASET_SCORE_STATS_MEMO = cached_stats
        return cached_stats

    stats = dict(cached_stats)

    # Prefer explicit score sources for sigma normalization.
    for source_stats in (_load_continuous_json_score_stats(), _load_passk_open_score_stats()):
        for dataset, values in source_stats.items():
            current = stats.get(dataset, {})
            candidate = {
                "model_mean_std": float(values.get("model_mean_std", np.nan)),
                "all_scores_std": float(values.get("all_scores_std", np.nan)),
            }
            if current != candidate:
                stats[dataset] = candidate

    # Fall back to legacy score matrices only for missing datasets.
    if SCORES_DIR.is_dir():
        for score_path in sorted(SCORES_DIR.glob("*.jbl")):
            if score_path.name.endswith("_models.jbl"):
                continue
            ds = score_path.stem
            if not _needs_score_stat_entry(stats, ds):
                continue
            try:
                scores = jbl.load(score_path)
            except Exception:
                continue
            computed = _compute_score_stats_from_matrix(scores)
            if computed is None:
                continue
            stats[ds] = computed

    _write_score_stats_cache(stats, source_signature)
    _DATASET_SCORE_STATS_MEMO = stats
    return stats


def _needs_score_stat_entry(stats: dict[str, dict[str, float]], dataset: str) -> bool:
    if dataset not in stats:
        return True
    model_std = stats.get(dataset, {}).get("model_mean_std")
    if model_std is None:
        return True
    try:
        model_std = float(model_std)
    except Exception:
        return True
    return (not np.isfinite(model_std)) or (model_std <= 0.0)


def _augment_score_stats_from_jbl(stats: dict[str, dict[str, float]], datasets: Iterable[str]) -> bool:
    missing = {str(ds) for ds in datasets if _needs_score_stat_entry(stats, str(ds))}
    if not missing:
        return False

    per_dataset_models: dict[str, dict[int, list[float]]] = {ds: {} for ds in missing}
    per_dataset_vals: dict[str, list[float]] = {ds: [] for ds in missing}
    seen: set[str] = set()

    for setting in discover_settings():
        jbl_df = load_setting_jbl(setting)
        if jbl_df.empty or "dataset" not in jbl_df.columns:
            continue
        sub = jbl_df[jbl_df["dataset"].isin(missing)]
        if sub.empty:
            continue

        for _, row in sub.iterrows():
            dataset = str(row["dataset"])
            model_map = per_dataset_models.setdefault(dataset, {})
            all_vals = per_dataset_vals.setdefault(dataset, [])
            for idx_col, acc_col in [
                ("train_model_indices", "true_acc_train"),
                ("test_model_indices", "true_acc_test"),
            ]:
                idxs = row.get(idx_col)
                accs = row.get(acc_col)
                if not isinstance(idxs, list) or not isinstance(accs, list):
                    continue
                if len(idxs) != len(accs):
                    continue
                for model_idx, acc in zip(idxs, accs):
                    try:
                        model_i = int(model_idx)
                        acc_f = float(acc)
                    except Exception:
                        continue
                    if not np.isfinite(acc_f):
                        continue
                    model_map.setdefault(model_i, []).append(acc_f)
                    all_vals.append(acc_f)
            if model_map:
                seen.add(dataset)
        if seen == missing:
            break

    updated = False
    for dataset in missing:
        model_map = per_dataset_models.get(dataset, {})
        if not model_map:
            continue
        model_means = np.asarray(
            [np.mean(vals) for vals in model_map.values() if len(vals) > 0],
            dtype=float,
        )
        if len(model_means) == 0:
            continue
        all_vals = np.asarray(per_dataset_vals.get(dataset, []), dtype=float)
        model_std = float(np.std(model_means))
        all_std = float(np.std(all_vals)) if len(all_vals) > 0 else model_std
        if not np.isfinite(model_std) or model_std <= 0.0:
            continue
        stats[dataset] = {
            "model_mean_std": model_std,
            "all_scores_std": all_std,
        }
        updated = True
    return updated


def ensure_dataset_score_stats(datasets: Iterable[str]) -> dict[str, dict[str, float]]:
    global _DATASET_SCORE_STATS_MEMO
    stats = load_dataset_score_stats()
    if _augment_score_stats_from_jbl(stats, datasets):
        _write_score_stats_cache(stats, _score_sources_signature())
        _DATASET_SCORE_STATS_MEMO = stats
    return stats


def load_num_questions() -> dict[str, int]:
    num_questions: dict[str, int] = {}
    if not SCORES_DIR.is_dir():
        return num_questions
    for score_path in sorted(SCORES_DIR.glob("*.jbl")):
        if score_path.name.endswith("_models.jbl"):
            continue
        try:
            scores = jbl.load(score_path)
        except Exception:
            continue
        num_questions[score_path.stem] = int(scores.shape[1])
    return num_questions


def load_score_matrix(dataset: str) -> np.ndarray | None:
    path = SCORES_DIR / f"{dataset}.jbl"
    if path.is_file():
        return jbl.load(path)
    return None


def _resolve_method_alias(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


def resolve_method_candidates(candidates: Iterable[str], available_methods: set[str]) -> str | None:
    for candidate in candidates:
        method = _resolve_method_alias(candidate)
        if method in available_methods:
            return method
    return None


def _weighted_mean_and_se(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), 0.0
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    sum_w = float(np.sum(weights))
    sum_w2 = float(np.sum(weights ** 2))
    eff_n = (sum_w ** 2) / sum_w2 if sum_w2 > 0 else len(values)
    se = math.sqrt(max(variance, 0.0) / max(eff_n, 1.0))
    return mean, se


def aggregate_metric(
    df: pd.DataFrame,
    metric_key: str,
    datasets: Iterable[str],
    methods: Iterable[str] | None = None,
) -> pd.DataFrame:
    if metric_key not in METRIC_SPECS:
        raise ValueError(f"Unknown metric key: {metric_key}")
    spec = METRIC_SPECS[metric_key]
    metric_col = spec["column"]
    datasets = list(datasets)

    if df.empty:
        return pd.DataFrame(columns=["method", "value", "se", "n_units"])

    sub = df[df["dataset"].isin(datasets)].copy()
    if methods is not None:
        methods = [_resolve_method_alias(m) for m in methods]
        sub = sub[sub["method"].isin(methods)]
    if sub.empty or metric_col not in sub.columns:
        return pd.DataFrame(columns=["method", "value", "se", "n_units"])

    unit_df = (
        sub.dropna(subset=[metric_col])
        .groupby(["dataset", "method"], as_index=False)[metric_col]
        .mean()
    )
    if unit_df.empty:
        return pd.DataFrame(columns=["method", "value", "se", "n_units"])

    score_stats = ensure_dataset_score_stats(unit_df["dataset"].tolist())
    rows = []
    for method, group in unit_df.groupby("method"):
        values = group[metric_col].to_numpy(dtype=float)
        ds_names = group["dataset"].tolist()
        if spec["sigma_normalize"]:
            raw = np.array(
                [1.0 / max(score_stats.get(ds, {}).get("model_mean_std", 1.0), 1e-9) for ds in ds_names],
                dtype=float,
            )
            weights = raw / raw.sum() if raw.sum() > 0 else np.ones_like(raw) / len(raw)
        else:
            weights = np.ones_like(values) / len(values)
        mean, se = _weighted_mean_and_se(values, weights)
        rows.append(
            {
                "method": method,
                "value": mean * spec["scale"],
                "se": se * spec["scale"],
                "n_units": len(values),
            }
        )
    return pd.DataFrame(rows)


def aggregate_multi_metric(
    df: pd.DataFrame,
    datasets: Iterable[str],
    methods: Iterable[str] | None = None,
    metric_keys: Iterable[str] = ("rmse", "mae", "tau", "rho", "pearson"),
) -> pd.DataFrame:
    metric_keys = list(metric_keys)
    merged: pd.DataFrame | None = None
    for metric_key in metric_keys:
        agg = aggregate_metric(df=df, metric_key=metric_key, datasets=datasets, methods=methods)
        agg = agg.rename(columns={"value": metric_key, "se": f"{metric_key}_se"})
        keep_cols = ["method", metric_key, f"{metric_key}_se", "n_units"]
        agg = agg[keep_cols]
        merged = agg if merged is None else merged.merge(agg.drop(columns=["n_units"]), on="method", how="outer")
    return merged if merged is not None else pd.DataFrame(columns=["method"])


def build_crossk_combined(
    native_df: pd.DataFrame,
    crossk_df: pd.DataFrame,
    allowed_suffixes: set[str] | None = None,
) -> pd.DataFrame:
    allowed_suffixes = allowed_suffixes or PASSK_ALLOWED_SUFFIXES_DEFAULT
    parts: list[pd.DataFrame] = []

    if not native_df.empty:
        ndf = native_df.copy()
        ndf["_parsed"] = ndf["dataset"].map(parse_pass_at_k_dataset_name)
        ndf = ndf[ndf["_parsed"].notna()].copy()
        if not ndf.empty:
            ndf["benchmark"] = ndf["_parsed"].str[0]
            ndf["source_k"] = ndf["_parsed"].str[1].astype(int)
            ndf["suffix"] = ndf["_parsed"].str[2]
            ndf = ndf[ndf["suffix"].isin(allowed_suffixes)]
            ndf["pred_k"] = ndf["source_k"]
            parts.append(
                ndf[
                    [
                        "dataset",
                        "benchmark",
                        "method",
                        "seed",
                        "source_k",
                        "pred_k",
                        "error",
                        "rmse",
                        "corr_spearman",
                        "corr_kendall",
                        "corr_pearson",
                    ]
                ]
            )

    if not crossk_df.empty:
        cdf = crossk_df.copy()
        if "benchmark" not in cdf.columns or "source_k" not in cdf.columns:
            cdf["_parsed"] = cdf["dataset"].map(parse_pass_at_k_dataset_name)
            cdf = cdf[cdf["_parsed"].notna()].copy()
            cdf["benchmark"] = cdf["_parsed"].str[0]
            cdf["source_k"] = cdf["_parsed"].str[1].astype(int)
            cdf["suffix"] = cdf["_parsed"].str[2]
        else:
            cdf["_parsed"] = cdf["dataset"].map(parse_pass_at_k_dataset_name)
            cdf["suffix"] = cdf["_parsed"].str[2]
        cdf = cdf[cdf["suffix"].isin(allowed_suffixes)]
        parts.append(
            cdf[
                [
                    "dataset",
                    "benchmark",
                    "method",
                    "seed",
                    "source_k",
                    "pred_k",
                    "error",
                    "rmse",
                    "corr_spearman",
                    "corr_kendall",
                    "corr_pearson",
                ]
            ]
        )

    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    combined["source_k"] = pd.to_numeric(combined["source_k"], errors="coerce")
    combined["pred_k"] = pd.to_numeric(combined["pred_k"], errors="coerce")
    combined = combined.dropna(subset=["source_k", "pred_k"])
    combined["source_k"] = combined["source_k"].astype(int)
    combined["pred_k"] = combined["pred_k"].astype(int)
    combined = combined[
        (~combined["source_k"].isin([128])) & (~combined["pred_k"].isin([128]))
    ].copy()
    return combined


def aggregate_passk_metric(
    combined_df: pd.DataFrame,
    metric_key: str,
    benchmarks: Iterable[str],
    methods: Iterable[str] | None = None,
    source_mode: str = "k1",
    target_ks: Iterable[int] = PASS_AT_K_TARGET_KS_DEFAULT,
    optimization_metric_key: str = "rmse",
) -> pd.DataFrame:
    if combined_df.empty:
        return pd.DataFrame(columns=["method", "value", "se", "n_units"])
    if metric_key not in METRIC_SPECS:
        raise ValueError(f"Unknown metric key: {metric_key}")
    metric_col = METRIC_SPECS[metric_key]["column"]
    optimize_col = METRIC_SPECS[optimization_metric_key]["column"]

    benchmarks = set(benchmarks)
    target_ks = set(target_ks)

    sub = combined_df[
        combined_df["benchmark"].isin(benchmarks) & combined_df["pred_k"].isin(target_ks)
    ].copy()
    if methods is not None:
        methods = {_resolve_method_alias(m) for m in methods}
        sub = sub[sub["method"].isin(methods)]
    if sub.empty:
        return pd.DataFrame(columns=["method", "value", "se", "n_units"])

    seed_avg = (
        sub.groupby(["dataset", "benchmark", "method", "source_k", "pred_k"], as_index=False)[
            ["error", "rmse", "corr_spearman", "corr_kendall", "corr_pearson"]
        ]
        .mean()
    )

    score_stats = ensure_dataset_score_stats(sub["dataset"].astype(str).tolist())

    def _sigma_raw_weights(ds_values: Iterable[str]) -> np.ndarray:
        return np.asarray(
            [
                1.0 / max(score_stats.get(str(ds), {}).get("model_mean_std", 1.0), 1e-9)
                for ds in ds_values
            ],
            dtype=float,
        )

    if source_mode == "k1":
        selected = seed_avg[seed_avg["source_k"] == 1].copy()
    elif source_mode == "opt":
        opt_source = seed_avg.dropna(subset=[optimize_col]).copy()
        if METRIC_SPECS[optimization_metric_key]["sigma_normalize"]:
            rows = []
            for (method, source_k), group in opt_source.groupby(["method", "source_k"]):
                values = group[optimize_col].to_numpy(dtype=float)
                raw = _sigma_raw_weights(group["dataset"].tolist())
                weights = raw / raw.sum() if raw.sum() > 0 else np.ones_like(raw) / len(raw)
                rows.append(
                    {
                        "method": method,
                        "source_k": source_k,
                        optimize_col: float(np.average(values, weights=weights)),
                    }
                )
            opt_summary = pd.DataFrame(rows)
        else:
            opt_summary = (
                opt_source.groupby(["method", "source_k"], as_index=False)[optimize_col]
                .mean()
            )
        opt_summary = opt_summary.dropna(subset=[optimize_col])
        if opt_summary.empty:
            return pd.DataFrame(columns=["method", "value", "se", "n_units"])
        best_idx = opt_summary.groupby("method")[optimize_col].idxmin()
        best_sources = opt_summary.loc[best_idx, ["method", "source_k"]]
        selected = seed_avg.merge(best_sources, on=["method", "source_k"], how="inner")
    else:
        raise ValueError(f"Unknown source mode: {source_mode}")

    if selected.empty:
        return pd.DataFrame(columns=["method", "value", "se", "n_units"])

    rows = []
    use_sigma = METRIC_SPECS[metric_key]["sigma_normalize"]
    for method, group in selected.groupby("method"):
        valid = group.dropna(subset=[metric_col]).copy()
        values = valid[metric_col].to_numpy(dtype=float)
        if len(values) == 0:
            rows.append({"method": method, "value": np.nan, "se": np.nan, "n_units": 0})
            continue
        if use_sigma:
            raw = _sigma_raw_weights(valid["dataset"].tolist())
            weights = raw / raw.sum() if raw.sum() > 0 else np.ones_like(raw) / len(raw)
        else:
            weights = np.ones_like(values) / len(values)
        mean, se = _weighted_mean_and_se(values, weights)
        rows.append(
            {
                "method": method,
                "value": mean * METRIC_SPECS[metric_key]["scale"],
                "se": se * METRIC_SPECS[metric_key]["scale"],
                "n_units": len(values),
            }
        )
    return pd.DataFrame(rows)


def _predictor_label(prefix: str, logit: bool) -> str:
    if prefix in {"rf"}:
        return "RF"
    if logit:
        return "logit ridge"
    return "ridge"


def parse_mrmr_method(method: str) -> dict[str, Any] | None:
    match = _MRMR_RE.match(method)
    if match is None:
        return None
    logit_flag, prefix, mi_k_str, irt_rep, objective, target, _aipw, plus_flag = match.groups()
    prefix = prefix or ""
    degree = 1
    if prefix == "k":
        degree = 2
    elif prefix == "k3":
        degree = 3
    elif prefix == "k4":
        degree = 4
    return {
        "method": method,
        "logit": bool(logit_flag),
        "prefix": prefix,
        "degree": degree,
        "mi_k": int(mi_k_str) if mi_k_str else 3,
        "irt_rep": irt_rep,
        "objective": objective,
        "target": target,
        "plus": bool(plus_flag),
        "predictor": _predictor_label(prefix, bool(logit_flag)),
    }


def mrmr_method_name(
    *,
    degree: int = 1,
    mi_k: int = 3,
    objective: str = "MIQ",
    target: str = "y",
    predictor: str = "ridge",
    plus: bool = False,
    logit: bool = False,
) -> str:
    prefix = ""
    if predictor == "rf":
        prefix = "rf"
    else:
        if degree == 2:
            prefix = "k"
        elif degree == 3:
            prefix = "k3"
        elif degree == 4:
            prefix = "k4"
    logit_prefix = "l" if logit else ""
    k_text = "" if mi_k == 3 else str(int(mi_k))
    method = f"{logit_prefix}{prefix}mrmr{k_text}_{objective}_{target}"
    if plus:
        method += "+"
    return method


def build_mrmr_methods(
    *,
    degrees: Iterable[int] = (1, 2),
    mi_ks: Iterable[int] = (5,),
    objectives: Iterable[str] = ("MIQ",),
    targets: Iterable[str] = ("y",),
    predictor: str = "ridge",
    plus: bool = False,
    logit: bool = False,
) -> list[str]:
    methods: list[str] = []
    for degree in degrees:
        for objective in objectives:
            if objective in {"FCQ", "FCD", "FCQ2", "FCD2"}:
                ks = [3]
            else:
                ks = list(mi_ks)
            for mi_k in ks:
                for target in targets:
                    methods.append(
                        mrmr_method_name(
                            degree=degree,
                            mi_k=mi_k,
                            objective=objective,
                            target=target,
                            predictor=predictor,
                            plus=plus,
                            logit=logit,
                        )
                    )
    return methods


def infer_method_family(method: str) -> str:
    method = _resolve_method_alias(method)
    if parse_mrmr_method(method) is not None:
        return "mrmr"
    if method in {"metabench"}:
        return "metabench"
    if method.startswith("anchor_points") or method.startswith("kanchor_points"):
        return "anchor"
    if "pirt" in method or "gpirt" in method:
        return "irt"
    if method in BASELINE_COLORS or "random_" in method or "search" in method or "lasso" in method:
        return "baseline"
    return "other"


def _irt_label_from_method(method: str) -> str:
    base = method
    for prefix in ["k3", "k4", "k"]:
        if base.startswith(prefix):
            base = base[len(prefix) :]
    if "gpirt" in base:
        variant = "gp-IRT"
    elif "pirt" in base:
        variant = "p-IRT"
    else:
        variant = "gp-IRT"
    if base.startswith("LEGO_"):
        return f"LEGO{variant}"
    if base.startswith("B3_v2_"):
        return r"$\beta^3$" + f"{variant}"
    if base.startswith("B3_"):
        return f"B3 (old){variant}"
    if base.startswith("B_"):
        return r"$\beta$ " + f"{variant}"
    if base.startswith("G_"):
        return f"G{variant}"
    return variant


def _irt_prefix_variant_from_method(method: str) -> tuple[str, str]:
    base = method
    for prefix in ["k3", "k4", "k"]:
        if base.startswith(prefix):
            base = base[len(prefix) :]
            break
    variant = "gp" if "gpirt" in base else "p"
    if base.startswith("LEGO_"):
        return "LEGO", variant
    if base.startswith("B3_v2_"):
        return "B3_v2", variant
    if base.startswith("B3_"):
        return "B3", variant
    if base.startswith("B_"):
        return "B", variant
    if base.startswith("G_"):
        return "G", variant
    return "base", variant


def _irt_dim_from_method(method: str) -> int:
    match = re.search(r"(?:gpirt|pirt)(\d+)", method)
    if match:
        return int(match.group(1))
    # Treat unnumbered aliases as model-specific defaults used in ablation-4.
    base = method
    for prefix in ["k3", "k4", "k"]:
        if base.startswith(prefix):
            base = base[len(prefix) :]
            break
    if base.startswith("LEGO_") or base.startswith("B_"):
        return 10
    if base.startswith("gpirt") or base.startswith("pirt"):
        return 10
    return 1


def _kernel_degree_for_method_name(method: str) -> tuple[str, int]:
    base = method
    if base.startswith("k3"):
        return base[2:], 3
    if base.startswith("k4"):
        return base[2:], 4
    if base.startswith("k"):
        return base[1:], 2
    return base, 1


def _plus_degree_from_method_name(method: str) -> int | None:
    if not method.endswith("+"):
        return None
    method_core = method[:-1]
    _base, degree = _kernel_degree_for_method_name(method_core)
    return degree


def pretty_method_name(
    method: str,
    *,
    mrmr_detail: str | None = None,
    compact_anchor: bool = False,
    irt_include_dim: bool = False,
) -> str:
    method = _resolve_method_alias(method)
    plus_suffix = method.endswith("+")
    method_core = method[:-1] if plus_suffix else method
    method_core, degree = _kernel_degree_for_method_name(method_core)

    baseline_names = {
        "random_sampling": "Random",
        "random_sampling_and_learn": "Random+",
        "random_search_and_learn": "Search+",
        "small_search_and_learn": "Small Search+",
        "sample_first_and_learn": "Sample First+",
        "lasso": "Lasso",
        "metabench": "MetaBench",
        "anchor_points_weighted": "AnchorPts" if compact_anchor else "AnchorPoints",
        "anchor_points_predictor": "AnchorPointsPred",
    }
    if method_core in baseline_names:
        name = baseline_names[method_core]
        if method_core in {"random_sampling_and_learn", "random_search_and_learn"}:
            name = f"{name} (d={degree})"
        elif plus_suffix and not name.endswith("+"):
            name = f"{name}+"
        return name

    parsed = parse_mrmr_method(method)
    if parsed is not None:
        degree = parsed["degree"]
        objective = parsed["objective"]
        mi_k = parsed["mi_k"]
        target = parsed["target"]
        predictor = parsed["predictor"]
        estimator = "PCA" if objective in {"PMIQ", "PMID", "PMI"} else "KSG"
        if mrmr_detail == "mi_k":
            if objective in {"FCQ", "FCD", "FCQ2", "FCD2"}:
                return f"mRMR {objective} (d={degree})"
            return f"mRMR {objective} (k={mi_k}, d={degree})"
        if mrmr_detail == "mi_k_plain":
            return f"mRMR (k={mi_k}, d={degree})"
        if mrmr_detail == "objective":
            return f"mRMR {objective} (d={degree})"
        if mrmr_detail == "target_predictor":
            details = [f"d={degree}", predictor]
            if target == "PC1":
                details.append("PC1 rel.")
            return f"mRMR ({', '.join(details)})"
        if mrmr_detail == "estimator":
            return f"mRMR {objective.replace('P', '')} (d={degree}, {estimator})"
        return f"mRMR (d={degree})"

    if "pirt" in method or "gpirt" in method:
        label = _irt_label_from_method(method)
        if irt_include_dim:
            return f"{label} (p={_irt_dim_from_method(method)})"
        return label

    return method


def method_style(method: str, *, use_irt_dim_linestyle: bool = False) -> StyleInfo:
    method = _resolve_method_alias(method)
    family = infer_method_family(method)
    base = FAMILY_STYLE[family]
    color = base["color"]
    marker = base["marker"]
    linestyle = "-"

    if family == "baseline":
        color = BASELINE_COLORS.get(method, FAMILY_STYLE["baseline"]["color"])
        # marker = "o" if method == "lasso" else "*"
        marker = "o" if method == "lasso" else "^"
        if method.startswith("k3"):
            linestyle = LINESTYLE_BY_DEGREE[3]
        elif method.startswith("k4"):
            linestyle = LINESTYLE_BY_DEGREE[4]
        elif method.startswith("k"):
            linestyle = LINESTYLE_BY_DEGREE[2]
    elif family == "metabench":
        color = FAMILY_STYLE["metabench"]["color"]
        marker = "D"
    elif family == "anchor":
        color = FAMILY_STYLE["anchor"]["color"]
        marker = "p"
    elif family == "irt":
        color = IRT_PREFIX_VARIANT_COLORS.get(_irt_prefix_variant_from_method(method), FAMILY_STYLE["irt"]["color"])
        marker = "s"
        if use_irt_dim_linestyle:
            linestyle = LINESTYLE_BY_IRT_DIM.get(_irt_dim_from_method(method), "-")
    elif family == "mrmr":
        parsed = parse_mrmr_method(method)
        assert parsed is not None
        objective = parsed["objective"]
        color = OBJECTIVE_COLORS.get(objective, FAMILY_STYLE["mrmr"]["color"])
        if parsed["predictor"] == "RF":
            color = _darken(color, 0.3)
        elif parsed["predictor"] == "logit ridge":
            color = _lighten(color, 0.2)
        if parsed["target"] == "PC1":
            color = _lighten(color, 0.25)
        marker = "*"
        # marker = "^"
        linestyle = LINESTYLE_BY_DEGREE.get(parsed["degree"], "-")

    # For methods with trailing "+", force linestyle by kernel degree encoded in
    # the method prefix (e.g., "+" -> d=1 solid, "k...+" -> d=2 dashed).
    plus_degree = _plus_degree_from_method_name(method)
    if plus_degree is not None:
        linestyle = LINESTYLE_BY_DEGREE.get(plus_degree, linestyle)
        # Make + IRT/Anchor variants visually softer than base methods.
        if family == "anchor":
            color = _lighten(color, 0.2)
        elif family == "irt":
            color = _lighten(color, 0.2)

    return StyleInfo(
        family=family,
        color=color,
        marker=marker,
        linestyle=linestyle,
    )


def _combined_grid_plus_legend_handles() -> tuple[list[Line2D], int]:
    """
    Fixed 6x2 legend layout for notebook-03 combined grids.
    Columns (top->bottom):
      [mRMR, Random], [Random+, Search+], [AnchorPts, AnchorPts+],
      [gp-IRT, gp-IRT+], [Lasso, MetaBench], [d=1, d=2]
    """
    mrmr = method_style("mrmr5_MIQ_y")
    rnd = method_style("random_sampling")
    rndp = method_style("random_sampling_and_learn")
    srchp = method_style("random_search_and_learn")
    anch = method_style("anchor_points_weighted")
    anchp = method_style("anchor_points_weighted+")
    gp = method_style("gpirt")
    gpp = method_style("gpirt+")
    lasso = method_style("lasso")
    metabench = method_style("metabench")
    d1 = LINESTYLE_BY_DEGREE.get(1, "-")
    d2 = LINESTYLE_BY_DEGREE.get(2, "--")

    # Row-major ordering for matplotlib with ncol=5 to realize the requested
    # per-column top->bottom pairing.
    handles = [
        Line2D([0], [0], color=mrmr.color, marker=mrmr.marker, linestyle="-", label="mRMR"),
        Line2D([0], [0], color=rnd.color, marker=rnd.marker, linestyle="-", label="Random"),
        Line2D([0], [0], color=rndp.color, marker=rndp.marker, linestyle="-", label="Random+"),
        Line2D([0], [0], color=srchp.color, marker=srchp.marker, linestyle="-", label="Search+"),
        Line2D([0], [0], color=anch.color, marker=anch.marker, linestyle="-", label="AnchorPts"),
        Line2D([0], [0], color=anchp.color, marker=anchp.marker, linestyle="-", label="AnchorPts+"),
        Line2D([0], [0], color=gp.color, marker=gp.marker, linestyle="-", label="gp-IRT"),
        Line2D([0], [0], color=gpp.color, marker=gpp.marker, linestyle="-", label="gp-IRT+"),
        Line2D([0], [0], color=lasso.color, marker=lasso.marker, linestyle="-", label="Lasso"),
        Line2D([0], [0], color=metabench.color, marker=metabench.marker, linestyle="-", label="MetaBench"),
        Line2D([0], [0], color="#555555", marker="None", linestyle=d1, label="d=1"),
        Line2D([0], [0], color="#555555", marker="None", linestyle=d2, label="d=2"),
    ]
    return handles, 6


def family_legend_handles(methods: Iterable[str]) -> list[Line2D]:
    family_to_style: dict[str, StyleInfo] = {}
    for method in methods:
        style = method_style(method)
        family_to_style.setdefault(style.family, style)
    legend_labels = {
        "baseline": "Baselines",
        "anchor": "AnchorPoints",
        "irt": "TinyBenchmarks",
        "mrmr": "mRMR (ours)",
        "metabench": "MetaBench",
        "other": "Other",
    }
    handles = []
    for family in ["baseline", "anchor", "irt", "mrmr", "metabench", "other"]:
        if family not in family_to_style:
            continue
        style = family_to_style[family]
        handles.append(
            Line2D(
                [0],
                [0],
                marker=style.marker,
                color=style.color,
                linestyle="None",
                markersize=8,
                label=legend_labels.get(family, family.title()),
            )
        )
    return handles


def scatter_legend_handles(methods: Iterable[str]) -> list[Line2D]:
    """
    Legend ordering for scatter plots.

    Desired order:
    mRMR, Anchor Points, TinyBenchmarks, MetaBench, Baselines, + Ridge, ++ Kernel Ridge
    """
    family_order = ["mrmr", "anchor", "irt", "metabench", "baseline"]
    family_label = {
        "mrmr": "mRMR",
        "anchor": "AnchorPoints",
        "irt": "TinyBenchmarks",
        "metabench": "MetaBench",
        "baseline": "Baselines",
    }

    # Pick one representative method per family, keeping stability based on first
    # occurrence in the input.
    family_to_style: dict[str, StyleInfo] = {}
    for method in methods:
        style = method_style(method)
        if style.family not in family_order:
            continue
        family_to_style.setdefault(style.family, style)

    handles: list[Line2D] = []
    for fam in family_order:
        style = family_to_style.get(fam)
        if style is None:
            continue
        handles.append(
            Line2D(
                [0],
                [0],
                marker=style.marker,
                color=style.color,
                linestyle="None",
                markersize=6,
                label=family_label.get(fam, fam.title()),
            )
        )

    # Explanatory entries for kernel/ridge variants used by mRMR labels.
    handles.append(
        Line2D(
            [0],
            [0],
            linestyle="None",
            # Keep the layout tight while not rendering a visible marker.
            marker="+",
            color="#111111",
            markersize=0,
            alpha=0.0,
            label="+ Ridge",
        )
    )
    handles.append(
        Line2D(
            [0],
            [0],
            linestyle="None",
            # Keep the layout tight while not rendering a visible marker.
            marker="+",
            color="#111111",
            markersize=0,
            alpha=0.0,
            label="++ Kernel Ridge",
        )
    )
    return handles


def _ensure_plot_subdir(subdir_name: str) -> Path:
    out_dir = PLOTS_ROOT / subdir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _enforce_full_subplot_boxes(fig: plt.Figure) -> None:
    for ax in fig.axes:
        ax.grid(axis="both", alpha=0.6, color="#CBD5E1", linestyle="-", linewidth=0.5, zorder=0)
        for side in ("top", "right", "bottom", "left"):
            spine = ax.spines.get(side)
            if spine is not None:
                spine.set_visible(True)
                spine.set_linewidth(0.9)
                spine.set_color("#111827")


def _style_legend_box(legend) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_edgecolor("black")
    frame.set_linewidth(0.7)
    frame.set_alpha(1.0)
    frame.set_facecolor("white")


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    global _CAPTURED_FIGURES
    if _CAPTURED_FIGURES is not None:
        _CAPTURED_FIGURES[filename] = fig
    if not _SAVE_OUTPUTS_ENABLED:
        return path
    _enforce_full_subplot_boxes(fig)
    # fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", format="pdf")
    return path


def display_saved_plots(paths: Iterable[Path], *, width: int = 900, height: int = 520) -> None:
    if not _DISPLAY_SAVED_PLOTS_ENABLED:
        return
    try:
        from IPython.display import IFrame, display
    except Exception:
        return
    for path in paths:
        p = Path(path)
        if not p.is_file():
            continue
        display(IFrame(src=p.resolve().as_uri(), width=width, height=height))


def _capture_mode_begin(
    *,
    save_outputs: bool,
    display_saved_plots_enabled: bool,
    return_figures: bool,
    close_figures: bool,
) -> dict[str, Any]:
    global _SAVE_OUTPUTS_ENABLED, _DISPLAY_SAVED_PLOTS_ENABLED, _CAPTURED_FIGURES
    state = {
        "save_outputs_enabled": _SAVE_OUTPUTS_ENABLED,
        "display_saved_plots_enabled": _DISPLAY_SAVED_PLOTS_ENABLED,
        "captured_figures": _CAPTURED_FIGURES,
        "plt_close": plt.close,
    }
    _SAVE_OUTPUTS_ENABLED = bool(save_outputs)
    _DISPLAY_SAVED_PLOTS_ENABLED = bool(display_saved_plots_enabled)
    _CAPTURED_FIGURES = {} if return_figures else None
    if (not close_figures) or return_figures:
        plt.close = lambda *args, **kwargs: None  # type: ignore[assignment]
    return state


def _capture_mode_end(state: dict[str, Any]) -> dict[str, plt.Figure]:
    global _SAVE_OUTPUTS_ENABLED, _DISPLAY_SAVED_PLOTS_ENABLED, _CAPTURED_FIGURES
    captured = _CAPTURED_FIGURES or {}
    _SAVE_OUTPUTS_ENABLED = state["save_outputs_enabled"]
    _DISPLAY_SAVED_PLOTS_ENABLED = state["display_saved_plots_enabled"]
    _CAPTURED_FIGURES = state["captured_figures"]
    plt.close = state["plt_close"]  # type: ignore[assignment]
    return captured


def _merge_metric_frames(metric_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for metric_key, frame in metric_frames.items():
        renamer = {
            "value": metric_key,
            "se": f"{metric_key}_se",
            "n_units": f"{metric_key}_n",
        }
        frame = frame.rename(columns=renamer)
        keep_cols = ["method", metric_key, f"{metric_key}_se", f"{metric_key}_n"]
        frame = frame[keep_cols]
        merged = frame if merged is None else merged.merge(frame, on="method", how="outer")
    return merged if merged is not None else pd.DataFrame(columns=["method"])


def summarize_setting_standard(
    setting: Setting,
    datasets: Iterable[str],
    methods: Iterable[str] | None = None,
    metric_keys: Iterable[str] = ("rmse", "mae", "tau", "rho", "pearson"),
    force_recompute: bool = False,
) -> pd.DataFrame:
    ensure_dirs()
    datasets_list = list(datasets)
    metric_keys_list = list(metric_keys)
    sigma_token = _score_stats_cache_token()
    setting_key = _setting_cache_key(setting)
    cache_path = _cache_key(
        "summary_setting_standard",
        {
            "setting": list(setting_key),
            "datasets": sorted(datasets_list),
            "methods": sorted(methods) if methods is not None else None,
            "metric_keys": metric_keys_list,
            "sigma_token": sigma_token,
        },
    )
    if cache_path.is_file() and not force_recompute:
        return pd.read_parquet(cache_path)

    df = load_setting_results(setting)
    metric_frames = {
        metric_key: aggregate_metric(df=df, metric_key=metric_key, datasets=datasets_list, methods=methods)
        for metric_key in metric_keys_list
    }
    summary = _merge_metric_frames(metric_frames)
    summary.insert(0, "num_train_models", setting.num_train_models)
    summary.insert(0, "coreset_size", setting.coreset_size)
    if not summary.empty:
        summary.to_parquet(cache_path, index=False)
    return summary


def summarize_setting_passk(
    setting: Setting,
    benchmarks: Iterable[str],
    methods: Iterable[str] | None = None,
    source_mode: str = "k1",
    metric_keys: Iterable[str] = ("rmse", "mae", "tau", "rho", "pearson"),
    target_ks: Iterable[int] = PASS_AT_K_TARGET_KS_DEFAULT,
    force_recompute: bool = False,
) -> pd.DataFrame:
    ensure_dirs()
    benchmarks_list = list(benchmarks)
    metric_keys_list = list(metric_keys)
    target_ks_list = list(target_ks)
    sigma_token = _score_stats_cache_token()
    setting_key = _setting_cache_key(setting)
    cache_path = _cache_key(
        "summary_setting_passk",
        {
            "setting": list(setting_key),
            "benchmarks": sorted(benchmarks_list),
            "methods": sorted(methods) if methods is not None else None,
            "source_mode": source_mode,
            "metric_keys": metric_keys_list,
            "target_ks": target_ks_list,
            "sigma_token": sigma_token,
        },
    )
    if cache_path.is_file() and not force_recompute:
        return pd.read_parquet(cache_path)

    native_df = load_setting_results(setting)
    crossk_df = load_setting_crossk(setting)
    combined = build_crossk_combined(native_df=native_df, crossk_df=crossk_df)
    metric_frames = {
        metric_key: aggregate_passk_metric(
            combined_df=combined,
            metric_key=metric_key,
            benchmarks=benchmarks_list,
            methods=methods,
            source_mode=source_mode,
            target_ks=target_ks_list,
            optimization_metric_key="rmse",
        )
        for metric_key in metric_keys_list
    }
    summary = _merge_metric_frames(metric_frames)
    summary.insert(0, "num_train_models", setting.num_train_models)
    summary.insert(0, "coreset_size", setting.coreset_size)
    if not summary.empty:
        summary.to_parquet(cache_path, index=False)
    return summary


def summarize_split_standard(
    split_method: str,
    datasets: Iterable[str],
    methods: Iterable[str] | None = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    datasets_list = list(datasets)
    sigma_token = _score_stats_cache_token()
    cache_path = _cache_key(
        "summary_standard",
        {
            "split": split_method,
            "datasets": sorted(datasets_list),
            "methods": sorted(methods) if methods is not None else None,
            "sigma_token": sigma_token,
        },
    )
    if cache_path.is_file() and not force_recompute:
        return pd.read_parquet(cache_path)

    def _matches_default_datasets(split_name: str, ds: list[str]) -> bool:
        if split_name == BINARY_SPLIT_METHOD:
            default = BINARY_DATASETS_DEFAULT
        elif split_name == CONTINUOUS_SPLIT_METHOD:
            default = CONTINUOUS_DATASETS_DEFAULT
        else:
            return False
        return sorted(str(x) for x in ds) == sorted(str(x) for x in default)

    def _load_precomputed_standard_summary(split_name: str) -> pd.DataFrame:
        if methods is not None or not _matches_default_datasets(split_name, datasets_list):
            return pd.DataFrame()
        candidates = [
            CACHE_DIR / "summary_standard_global.parquet",
            CACHE_DIR / f"summary_standard_{'binary' if split_name == BINARY_SPLIT_METHOD else 'continuous'}.parquet",
        ]
        for path in candidates:
            if not path.is_file():
                continue
            try:
                frame = pd.read_parquet(path)
            except Exception:
                continue
            if frame.empty:
                continue
            if "split_method" in frame.columns:
                frame = frame[frame["split_method"].astype(str) == str(split_name)].copy()
            if frame.empty:
                continue
            drop_cols = [c for c in ("group", "split_method") if c in frame.columns]
            if drop_cols:
                frame = frame.drop(columns=drop_cols)
            return frame
        return pd.DataFrame()

    precomputed = pd.DataFrame() if force_recompute else _load_precomputed_standard_summary(split_method)
    frames: list[pd.DataFrame] = []
    if not precomputed.empty:
        frames.append(precomputed)
        present_settings = {
            (str(cs), str(nm))
            for cs, nm in precomputed[["coreset_size", "num_train_models"]]
            .dropna()
            .astype(str)
            .itertuples(index=False, name=None)
        }
        settings = [
            setting
            for setting in discover_settings(split_method=split_method)
            if (str(setting.coreset_size), str(setting.num_train_models)) not in present_settings
        ]
    else:
        settings = discover_settings(split_method=split_method)

    frames.extend(
        summarize_setting_standard(
            setting=setting,
            datasets=datasets_list,
            methods=methods,
            force_recompute=force_recompute,
        )
        for setting in settings
    )
    frames = [frame for frame in frames if not frame.empty]
    summary = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not summary.empty:
        summary.to_parquet(cache_path, index=False)
    return summary


def summarize_split_passk(
    split_method: str,
    benchmarks: Iterable[str],
    methods: Iterable[str] | None = None,
    source_mode: str = "k1",
    force_recompute: bool = False,
) -> pd.DataFrame:
    benchmarks_list = list(benchmarks)
    sigma_token = _score_stats_cache_token()
    cache_path = _cache_key(
        "summary_passk",
        {
            "split": split_method,
            "benchmarks": sorted(benchmarks_list),
            "methods": sorted(methods) if methods is not None else None,
            "source_mode": source_mode,
            "sigma_token": sigma_token,
        },
    )
    if cache_path.is_file() and not force_recompute:
        return pd.read_parquet(cache_path)

    def _matches_default_benchmarks(split_name: str, bms: list[str]) -> bool:
        if split_name != CONTINUOUS_SPLIT_METHOD:
            return False
        return sorted(str(x) for x in bms) == sorted(str(x) for x in PASSK_BENCHMARKS_DEFAULT)

    def _load_precomputed_passk_summary(split_name: str, src_mode: str) -> pd.DataFrame:
        if methods is not None or not _matches_default_benchmarks(split_name, benchmarks_list):
            return pd.DataFrame()
        candidates = [
            CACHE_DIR / "summary_passk_global.parquet",
            CACHE_DIR / f"summary_passk_{src_mode}.parquet",
        ]
        for path in candidates:
            if not path.is_file():
                continue
            try:
                frame = pd.read_parquet(path)
            except Exception:
                continue
            if frame.empty:
                continue
            if "split_method" in frame.columns:
                frame = frame[frame["split_method"].astype(str) == str(split_name)].copy()
            if "source_mode" in frame.columns:
                frame = frame[frame["source_mode"].astype(str) == str(src_mode)].copy()
            if frame.empty:
                continue
            drop_cols = [c for c in ("group", "split_method", "source_mode") if c in frame.columns]
            if drop_cols:
                frame = frame.drop(columns=drop_cols)
            return frame
        return pd.DataFrame()

    precomputed = pd.DataFrame() if force_recompute else _load_precomputed_passk_summary(split_method, source_mode)
    frames: list[pd.DataFrame] = []
    if not precomputed.empty:
        frames.append(precomputed)
        present_settings = {
            (str(cs), str(nm))
            for cs, nm in precomputed[["coreset_size", "num_train_models"]]
            .dropna()
            .astype(str)
            .itertuples(index=False, name=None)
        }
        settings = [
            setting
            for setting in discover_settings(split_method=split_method)
            if (str(setting.coreset_size), str(setting.num_train_models)) not in present_settings
        ]
    else:
        settings = discover_settings(split_method=split_method)

    frames.extend(
        summarize_setting_passk(
            setting=setting,
            benchmarks=benchmarks_list,
            methods=methods,
            source_mode=source_mode,
            force_recompute=force_recompute,
        )
        for setting in settings
    )
    frames = [frame for frame in frames if not frame.empty]
    summary = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not summary.empty:
        summary.to_parquet(cache_path, index=False)
    return summary


def _selection_metrics_from_row(row: pd.Series) -> dict[str, float] | None:
    sm = _selection_metrics_payload_from_row(row)
    if not isinstance(sm, dict):
        return None
    rel = sm.get("coreset_relevance")
    red = sm.get("coreset_redundancy")
    if not isinstance(rel, list) or not isinstance(red, list) or len(rel) == 0 or len(red) == 0:
        return None
    last_rel = float(rel[-1])
    last_red = float(red[-1])
    return {
        "relevance": last_rel,
        "redundancy": last_red,
        "difference": last_rel - last_red,
        "quotient": last_rel / (last_red + 1e-12),
    }


def _mi_cache_hash(train_model_indices: Iterable[int]) -> str:
    arr = np.array(sorted(int(i) for i in train_model_indices), dtype=np.int32)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _load_mi_cache(dataset: str, train_model_indices: Iterable[int]) -> dict[str, Any] | None:
    h = _mi_cache_hash(train_model_indices)
    path = MI_CACHE_DIR / dataset / f"{h}.npz"
    if not path.is_file():
        return None
    data = np.load(path)
    return {
        "relevance": data["relevance"],
        "binary": bool(data["binary"][0]),
    }


def _mi_dd_pairwise(src: np.ndarray) -> np.ndarray:
    """Vectorized discrete-discrete MI for all column pairs of binary src."""
    n = float(src.shape[0])
    s = src.sum(axis=0).astype(np.float64)
    S = (src.T @ src).astype(np.float64)

    s1 = s[:, np.newaxis]
    s2 = s[np.newaxis, :]

    p_11 = S / n
    p_10 = s1 / n - p_11
    p_01 = s2 / n - p_11
    p_00 = 1.0 - s1 / n - s2 / n + p_11

    mi = np.zeros((src.shape[1], src.shape[1]), dtype=np.float64)
    for p_ij, m_ij in [
        (p_00, (1 - s1 / n) * (1 - s2 / n)),
        (p_01, (1 - s1 / n) * (s2 / n)),
        (p_10, (s1 / n) * (1 - s2 / n)),
        (p_11, (s1 / n) * (s2 / n)),
    ]:
        mask = p_ij > 0
        mi[mask] += p_ij[mask] * np.log(p_ij[mask] / (m_ij[mask] + 1e-15))
    return np.maximum(mi, 0.0)


def _selection_metrics_from_mi_cache_row(row: pd.Series) -> dict[str, Any] | None:
    dataset = row.get("dataset")
    train_model_indices = row.get("train_model_indices")
    coreset_indices = row.get("coreset_indices")
    if not isinstance(dataset, str):
        return None
    if not isinstance(train_model_indices, list) or not isinstance(coreset_indices, list):
        return None
    if len(train_model_indices) == 0:
        return None
    cache = _load_mi_cache(dataset, train_model_indices)
    if cache is None:
        return None

    rel = np.asarray(cache["relevance"], dtype=float)
    d = rel.shape[0]
    coreset = np.asarray(coreset_indices, dtype=int)
    coreset = coreset[(coreset >= 0) & (coreset < d)]
    if coreset.size == 0:
        return None

    coreset_rel = rel[coreset]
    k = int(coreset.size)
    if k <= 1:
        coreset_red = np.zeros(k, dtype=float)
    elif cache["binary"]:
        scores = load_score_matrix(dataset)
        if scores is None:
            return None
        tmi = np.asarray(train_model_indices, dtype=int)
        tmi = tmi[(tmi >= 0) & (tmi < scores.shape[0])]
        if tmi.size == 0:
            return None
        src = scores[tmi][:, coreset].astype(np.float64)
        if src.size == 0:
            return None
        mi_mat = _mi_dd_pairwise(src)
        np.fill_diagonal(mi_mat, 0.0)
        coreset_red = mi_mat.sum(axis=1) / max(k - 1, 1)
    else:
        coreset_red = np.zeros(k, dtype=float)

    return {
        "all_relevance_min": float(np.min(rel)),
        "all_relevance_max": float(np.max(rel)),
        "all_relevance_mean": float(np.mean(rel)),
        "all_relevance_median": float(np.median(rel)),
        "all_relevance_std": float(np.std(rel)),
        "coreset_relevance": coreset_rel.astype(float).tolist(),
        "coreset_redundancy": coreset_red.astype(float).tolist(),
    }


def _selection_metrics_payload_from_row(row: pd.Series) -> dict[str, Any] | None:
    sm = row.get("selection_metrics")
    if isinstance(sm, dict):
        rel = sm.get("coreset_relevance")
        red = sm.get("coreset_redundancy")
        if isinstance(rel, list) and isinstance(red, list) and len(rel) > 0 and len(red) > 0:
            return sm
    return _selection_metrics_from_mi_cache_row(row)


def selection_objective_table(
    jbl_df: pd.DataFrame,
    datasets: Iterable[str],
    methods: Iterable[str] | None = None,
) -> pd.DataFrame:
    if jbl_df.empty:
        return pd.DataFrame(columns=["method", "relevance", "redundancy", "difference", "quotient"])
    sub = jbl_df[jbl_df["dataset"].isin(list(datasets))].copy()
    if methods is not None:
        resolved = [_resolve_method_alias(m) for m in methods]
        sub = sub[sub["method"].isin(resolved)]
    if sub.empty:
        return pd.DataFrame(columns=["method", "relevance", "redundancy", "difference", "quotient"])

    rows = []
    for _, row in sub.iterrows():
        metrics = _selection_metrics_from_row(row)
        if metrics is None:
            continue
        rows.append({"dataset": row["dataset"], "method": row["method"], **metrics})
    if not rows:
        return pd.DataFrame(columns=["method", "relevance", "redundancy", "difference", "quotient"])
    df = pd.DataFrame(rows)
    out = (
        df.groupby("method", as_index=False)[["relevance", "redundancy", "difference", "quotient"]]
        .mean()
    )
    return out


def average_selection_sequences(
    jbl_df: pd.DataFrame,
    methods: Iterable[str],
    datasets: Iterable[str],
) -> dict[str, dict[str, np.ndarray]]:
    results: dict[str, dict[str, np.ndarray]] = {}
    if jbl_df.empty:
        return results
    sub = jbl_df[jbl_df["dataset"].isin(list(datasets))].copy()
    for method in methods:
        method_name = _resolve_method_alias(method)
        method_rows = sub[sub["method"] == method_name]
        rel_sequences: list[np.ndarray] = []
        red_sequences:  list[np.ndarray] = []
        for _, row in method_rows.iterrows():
            sm = _selection_metrics_payload_from_row(row)
            if not isinstance(sm, dict):
                continue
            rel = sm.get("coreset_relevance")
            red = sm.get("coreset_redundancy")
            if not isinstance(rel, list) or not isinstance(red, list):
                continue
            if len(rel) == 0 or len(red) == 0:
                continue
            rel_sequences.append(np.asarray(rel, dtype=float))
            red_sequences.append(np.asarray(red, dtype=float))
        if not rel_sequences:
            continue
        min_len = min(len(seq) for seq in rel_sequences)
        rel_arr = np.stack([seq[:min_len] for seq in rel_sequences], axis=0)
        red_arr = np.stack([seq[:min_len] for seq in red_sequences], axis=0)
        rel_mean = rel_arr.mean(axis=0)
        red_mean = red_arr.mean(axis=0)
        diff_mean = rel_mean - red_mean
        quotient = rel_mean / (red_mean + 1e-12)
        results[method_name] = {
            "relevance": rel_mean,
            "redundancy": red_mean,
            "difference": diff_mean,
            "quotient": quotient,
        }
    return results


def _coreset_binary_vector(indices: Iterable[int], d: int) -> np.ndarray:
    vec = np.zeros(d, dtype=np.int8)
    idx = np.asarray(list(indices), dtype=int)
    idx = idx[(idx >= 0) & (idx < d)]
    vec[idx] = 1
    return vec


def average_pairwise_hamming(coresets: list[list[int]], d: int) -> float | None:
    if len(coresets) < 2:
        return None
    vecs = [_coreset_binary_vector(c, d) for c in coresets]
    distances = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            raw = np.sum(vecs[i] != vecs[j])
            max_dist = int(vecs[i].sum() + vecs[j].sum())
            distances.append(raw / max(max_dist, 1))
    return float(np.mean(distances)) if distances else None


def compute_nogueira_phi(coresets: list[list[int]], d: int) -> float | None:
    m = len(coresets)
    if m < 2:
        return None
    p_hat = np.zeros(d, dtype=float)
    k_sum = 0
    for coreset in coresets:
        idx = np.asarray(coreset, dtype=int)
        idx = idx[(idx >= 0) & (idx < d)]
        p_hat[idx] += 1.0
        k_sum += len(idx)
    p_hat /= m
    k_bar = k_sum / m
    denom = (k_bar / d) * (1.0 - k_bar / d)
    if denom <= 0:
        return None
    s2 = (m / (m - 1)) * p_hat * (1.0 - p_hat)
    phi = 1.0 - np.mean(s2) / denom
    return float(phi)


def standard_error(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=0) / np.sqrt(len(values)))


def metric_axis_title(metric_key: str) -> str:
    label = METRIC_SPECS[metric_key]["label"]
    return label.replace(r" $\downarrow$", "").replace(r" $\uparrow$", "")


def _subplot_row_label(ax: plt.Axes, label: str, xshift: float = 0.0) -> None:
    ax.text(
        -0.22 + xshift,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _annotate_2x2_subplots(axes: np.ndarray, labels: tuple[str, str, str, str] = ("(a)", "(b)", "(c)", "(d)")) -> None:
    for ax, label in zip(np.asarray(axes).flatten(), labels):
        ax.text(
            -0.22,
            1.05,
            label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
            ha="left",
        )


def _apply_xy_labels(ax: plt.Axes, x_label: str, y_label: str) -> None:
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def _plot_method_lines(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    label: str | None = None,
    *,
    use_irt_dim_linestyle: bool = False,
) -> None:
    style = method_style(method, use_irt_dim_linestyle=use_irt_dim_linestyle)
    ax.plot(
        x,
        y,
        linestyle=style.linestyle,
        marker=style.marker,
        color=style.color,
        linewidth=style.linewidth,
        markersize=LINE_PLOT_MARKER_SIZE,
        label=label or pretty_method_name(method),
    )


def _select_existing_methods(
    requested_methods: Iterable[str | tuple[str, ...]],
    available_methods: set[str],
) -> list[str]:
    selected: list[str] = []
    for req in requested_methods:
        if isinstance(req, tuple):
            method = resolve_method_candidates(req, available_methods)
            if method is not None:
                selected.append(method)
        else:
            method = resolve_method_candidates([req], available_methods)
            if method is not None:
                selected.append(method)
    return selected


def _default_main_methods_binary() -> list[str | tuple[str, ...]]:
    return [
        "random_sampling",
        "lasso",
        "random_sampling_and_learn",
        "krandom_search_and_learn",
        "random_search_and_learn",
        "krandom_search_and_learn",
        "anchor_points_weighted",
        "gpirt",
        ("mrmr5_MIQ_y", "mrmr_MIQ_y"),
        ("kmrmr5_MIQ_y", "kmrmr_MIQ_y"),
        "metabench",
    ]


def _default_main_methods_continuous() -> list[str | tuple[str, ...]]:
    return [
        "random_sampling",
        "lasso",
        "random_sampling_and_learn",
        "krandom_search_and_learn",
        "random_search_and_learn",
        "krandom_search_and_learn",
        "anchor_points_weighted",
        ("LEGO_gpirt", "gpirt"),
        ("mrmr_PMIQ_y", "mrmr5_PMIQ_y"),
        ("kmrmr_PMIQ_y", "kmrmr5_PMIQ_y", "kmrmr_PMIQ_y"),
    ]


def _summary_lookup(summary: pd.DataFrame, metric_key: str) -> pd.Series:
    if summary.empty or metric_key not in summary.columns:
        return pd.Series(dtype=float)
    return summary.set_index("method")[metric_key]


def _scatter_method_label(method: str, method_label_detail: str | None) -> str:
    """
    Scatter labels use slightly different formatting than line/legend labels.
    """
    method = _resolve_method_alias(method)

    # Keep mRMR families compact across objectives (MIQ/PMIQ/etc):
    # d=1 -> mRMR+, d=2 -> mRMR++.
    parsed = parse_mrmr_method(method)
    if parsed is not None:
        degree = int(parsed["degree"])
        if degree == 1:
            return "mRMR+"
        if degree == 2:
            return "mRMR++"

    label = pretty_method_name(method, mrmr_detail=method_label_detail, compact_anchor=True)

    m = re.fullmatch(r"mRMR \(d=(\d+)\)", label)
    if m:
        degree = int(m.group(1))
        if degree == 1:
            return "mRMR+"
        if degree == 2:
            return "mRMR++"

    # Baseline labels like "Search+ (d=1)" should be compact in scatter plots.
    m = re.fullmatch(r"^(.*\+)\s*\(d=(\d+)\)$", label)
    if m:
        base = m.group(1)
        degree = int(m.group(2))
        if degree == 1:
            return base
        if degree == 2:
            return base if base.endswith("++") else f"{base}+"

    # If the method identifier starts with "k" (kernelized variant) and the label
    # already ends in "+", upgrade it to "++".
    if method.startswith("k") and label.endswith("+") and not label.endswith("++"):
        label = label[:-1] + "++"

    # pIRT / gpIRT methods can have a trailing "+" in the method identifier,
    # but pretty_method_name drops it. Re-attach it for scatter labels.
    if (("pirt" in method or "gpirt" in method) and method.endswith("+")):
        label = label.rstrip("+")
        label = f"{label}++" if method.startswith("k") else f"{label}+"

    return label


def _compact_method_label(
    method: str,
    *,
    mrmr_detail: str | None = None,
    compact_anchor: bool = True,
    irt_include_dim: bool = False,
) -> str:
    """
    Compact display label:
    - drop explicit d=1 text
    - map *+ (d=2) style labels to *++
    """
    label = pretty_method_name(
        method,
        mrmr_detail=mrmr_detail,
        compact_anchor=compact_anchor,
        irt_include_dim=irt_include_dim,
    )

    # Remove explicit d=1 annotations.
    label = re.sub(r",\s*d=1(?=[,)])", "", label)
    label = re.sub(r"\(\s*d=1\s*\)", "", label)
    label = re.sub(r"\(\s*([^)]*?),\s*\)", r"(\1)", label)
    label = re.sub(r"\s{2,}", " ", label).strip()

    # Convert labels like "Search+ (d=2)" -> "Search++".
    m = re.fullmatch(r"^(.*\+)\s*\(d=2\)$", label)
    if m:
        return m.group(1) + "+"

    return label


def _plot_scatter_with_labels(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    method_label_detail: str | None = None,
    nudge_map: dict[str, tuple[float, float]] | None = None,
) -> None:
    nudge_map = nudge_map or {}
    for _, row in frame.iterrows():
        method = row["method"]
        if pd.isna(row[x_col]) or pd.isna(row[y_col]):
            continue
        style = method_style(method)
        ax.scatter(
            row[x_col],
            row[y_col],
            color=style.color,
            marker=style.marker,
            s=75,
            edgecolor="black",
            linewidth=0.25,
            alpha=0.95,
            zorder=3,
        )
        dx, dy = nudge_map.get(method, (0.18, 0.0))
        ax.text(
            row[x_col] + dx,
            row[y_col] + dy,
            _scatter_method_label(method, method_label_detail),
            fontsize=SCATTER_LABEL_FONT_SIZE,
        )


def build_notebook_01(
    *,
    output_dir: Path | None = None,
    binary_setting: tuple[str, str, str] = (BINARY_SPLIT_METHOD, "5%", "30"),
    continuous_setting: tuple[str, str, str] = (CONTINUOUS_SPLIT_METHOD, "5%", "32"),
    passk_setting: tuple[str, str, str] = (CONTINUOUS_SPLIT_METHOD, "5%", "30"),
    binary_datasets: Iterable[str] = BINARY_DATASETS_DEFAULT,
    continuous_datasets: Iterable[str] = CONTINUOUS_DATASETS_DEFAULT,
    passk_benchmarks: Iterable[str] = PASSK_BENCHMARKS_DEFAULT,
    methods_binary: Iterable[str | tuple[str, ...]] | None = None,
    methods_continuous: Iterable[str | tuple[str, ...]] | None = None,
    passk_mrmr_mi_k: int | None = 6,
    save_outputs: bool = True,
    show_saved_plots: bool = False,
    return_figures: bool = False,
    close_figures: bool = True,
) -> list[Path] | tuple[list[Path], dict[str, plt.Figure]]:
    setup_matplotlib()
    ensure_dirs()
    output_dir = output_dir or _ensure_plot_subdir("01_pareto_rmse_tau")
    capture_state = _capture_mode_begin(
        save_outputs=save_outputs,
        display_saved_plots_enabled=show_saved_plots,
        return_figures=return_figures,
        close_figures=close_figures,
    )
    outputs: list[Path] = []

    try:
        bs = get_setting_flexible(*binary_setting)
        cs = get_setting_flexible(*continuous_setting)
        ps = get_setting_flexible(*passk_setting)

        # If the requested pass@k setting has no pass@k datasets, pick the
        # nearest nmodels setting (same split + coreset) that does.
        passk_bench_set = set(passk_benchmarks)
        ps_native = load_setting_results(ps)
        has_passk_native = False
        if not ps_native.empty and "dataset" in ps_native.columns:
            for ds_name in ps_native["dataset"].astype(str).unique():
                parsed = parse_pass_at_k_dataset_name(ds_name)
                if parsed is None:
                    continue
                bench, _k, suffix = parsed
                if bench in passk_bench_set and suffix in PASSK_ALLOWED_SUFFIXES_DEFAULT:
                    has_passk_native = True
                    break
        if not has_passk_native:
            split_name, cs_label, nm_label = passk_setting
            candidates: list[tuple[float, Setting]] = []
            for setting in discover_settings(split_method=split_name):
                if setting.coreset_size != str(cs_label):
                    continue
                native_df = load_setting_results(setting)
                if native_df.empty or "dataset" not in native_df.columns:
                    continue
                found = False
                for ds_name in native_df["dataset"].astype(str).unique():
                    parsed = parse_pass_at_k_dataset_name(ds_name)
                    if parsed is None:
                        continue
                    bench, _k, suffix = parsed
                    if bench in passk_bench_set and suffix in PASSK_ALLOWED_SUFFIXES_DEFAULT:
                        found = True
                        break
                if not found:
                    continue
                distance = abs(parse_num_train_models(setting.num_train_models) - parse_num_train_models(str(nm_label)))
                candidates.append((distance, setting))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                ps = candidates[0][1]

        b_summary = summarize_setting_standard(bs, datasets=binary_datasets)
        c_summary = summarize_setting_standard(cs, datasets=continuous_datasets)
        p_summary = summarize_setting_passk(ps, benchmarks=passk_benchmarks, source_mode="k1")

        methods_binary_req = methods_binary or _default_main_methods_binary()
        methods_cont_req = methods_continuous or _default_main_methods_continuous()
        methods_binary_sel = _select_existing_methods(methods_binary_req, set(b_summary["method"]))
        methods_cont_sel_all = _select_existing_methods(
            methods_cont_req, set(c_summary["method"]) | set(p_summary["method"])
        )
        methods_cont_sel = list(methods_cont_sel_all)
        methods_cont_sel = [m for m in methods_cont_sel if _resolve_method_alias(m) != "lasso"]

        b_plot = b_summary[b_summary["method"].isin(methods_binary_sel)].copy()
        c_plot = c_summary[c_summary["method"].isin(methods_cont_sel)].set_index("method")
        p_plot = p_summary.set_index("method")
        comb_methods = list(dict.fromkeys(methods_cont_sel))
        comb_rows = []
        c_rmse = c_plot["rmse"] if "rmse" in c_plot.columns else pd.Series(dtype=float)
        p_rmse = p_plot["rmse"] if "rmse" in p_plot.columns else pd.Series(dtype=float)
        c_tau = c_plot["tau"] if "tau" in c_plot.columns else pd.Series(dtype=float)
        p_tau = p_plot["tau"] if "tau" in p_plot.columns else pd.Series(dtype=float)
        c_rmse_n = c_plot["rmse_n"] if "rmse_n" in c_plot.columns else pd.Series(dtype=float)
        p_rmse_n = p_plot["rmse_n"] if "rmse_n" in p_plot.columns else pd.Series(dtype=float)
        c_tau_n = c_plot["tau_n"] if "tau_n" in c_plot.columns else pd.Series(dtype=float)
        p_tau_n = p_plot["tau_n"] if "tau_n" in p_plot.columns else pd.Series(dtype=float)

        def _weighted_two(a: float, na: float, b: float, nb: float) -> float | None:
            vals: list[float] = []
            wts: list[float] = []
            if not pd.isna(a):
                vals.append(float(a))
                wts.append(float(na) if not pd.isna(na) and float(na) > 0 else 1.0)
            if not pd.isna(b):
                vals.append(float(b))
                wts.append(float(nb) if not pd.isna(nb) and float(nb) > 0 else 1.0)
            if not vals:
                return None
            if len(vals) == 1:
                return vals[0]
            return float(np.average(np.asarray(vals), weights=np.asarray(wts)))

        p_available = set(p_plot.index.astype(str))
        for method in comb_methods:
            passk_method = method
            parsed = parse_mrmr_method(method)
            if (
                passk_mrmr_mi_k is not None
                and parsed is not None
                and parsed["objective"] == "PMIQ"
                and parsed["target"] == "y"
                and parsed["predictor"] == "ridge"
            ):
                passk_primary = mrmr_method_name(
                    degree=parsed["degree"],
                    mi_k=int(passk_mrmr_mi_k),
                    objective="PMIQ",
                    target="y",
                )
                passk_method = resolve_method_candidates(
                    [passk_primary, method],
                    p_available,
                ) or method
            rmse_val = _weighted_two(
                c_rmse.get(method, np.nan),
                c_rmse_n.get(method, np.nan),
                p_rmse.get(passk_method, np.nan),
                p_rmse_n.get(passk_method, np.nan),
            )
            tau_val = _weighted_two(
                c_tau.get(method, np.nan),
                c_tau_n.get(method, np.nan),
                p_tau.get(passk_method, np.nan),
                p_tau_n.get(passk_method, np.nan),
            )
            if rmse_val is None or tau_val is None:
                continue
            comb_rows.append({"method": method, "rmse": rmse_val, "tau": tau_val})
        cp_plot = pd.DataFrame(comb_rows)

        fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_ONE_ROW_IN), constrained_layout=True)
        nudge_left = {
            "random_sampling": (-0.51, -0.005),
            "random_sampling_and_learn": (0.09, 0.001),
            "krandom_sampling_and_learn": (0.09, 0.002),
            "random_search_and_learn": (0.09, 0.001),
            "krandom_search_and_learn": (0.09, 0.004),
            "anchor_points_weighted": (-0.70, 0.004),
            "anchor_points_weighted+": (0.1, 0.003),
            "kanchor_points_weighted+": (0.1, 0.003),
            "gpirt1": (-0.60, -0.005),
            "gpirt1+": (-0.50, -0.015),
            "kgpirt1+": (-0.70, -0.002),
            "mrmr5_MIQ_y": (0.1, 0.004),
            "kmrmr5_MIQ_y": (0.0, 0.009),
        }
        nudge_right = {
            "random_sampling": (0.09, 0.002),
            "random_sampling_and_learn": (0.09, 0.001),
            "krandom_sampling_and_learn": (0.09, 0.002),
            "random_search_and_learn": (0.09, 0.002),
            "krandom_search_and_learn": (-0.92, -0.001),
            "anchor_points_weighted": (-0.80, 0.004),
            "anchor_points_weighted+": (0.12, -0.002),
            "kanchor_points_weighted+": (0.12, -0.001),  
            "B_gpirt5":  (-0.90, -0.005),
            "B_gpirt5+": (-0.90, -0.005),
            "kB_gpirt5+": (0.18, 0.00),  
            "mrmr_PMIQ_y": (0.12, 0.002),
            "kmrmr_PMIQ_y": (0.12, 0.002),
        }
        _plot_scatter_with_labels(axes[0], b_plot, "rmse", "tau", nudge_map=nudge_left)
        _apply_xy_labels(axes[0], r"$\bf{\leftarrow}$ RMSE (%) ", r"Kendall $\tau$ $\bf{\rightarrow}$")
        axes[0].set_title("Binary Scores") # (n=5%, M=30)")

        _plot_scatter_with_labels(axes[1], cp_plot, "rmse", "tau", nudge_map=nudge_right)
        _apply_xy_labels(axes[1], r"$\bf{\leftarrow}$ RMSE (%) ", r"Kendall $\tau$ $\bf{\rightarrow}$")
        axes[1].set_title("Continuous Scores") # (n=5%, M=30)")

        # Add subplot labels (a) and (b) to the top left of each subplot
        axes[0].text(
            -0.21, 1.12, "(a)", transform=axes[0].transAxes, fontsize=9, fontweight='bold', va='top', ha='left'
        )
        axes[1].text(
            -0.21, 1.12, "(b)", transform=axes[1].transAxes, fontsize=9, fontweight='bold', va='top', ha='left'
        )

        legend_methods = list(dict.fromkeys(methods_binary_sel + methods_cont_sel))
        handles = scatter_legend_handles(legend_methods)
        legend = fig.legend(
            handles=handles,
            loc="lower center",
            ncol=max(3, len(handles)),
            frameon=True,
            fancybox=False,
            columnspacing=0.3,
            handletextpad=0.3,
            labelspacing=0.35,
            borderpad=0.45,
            bbox_to_anchor=(0.5, -0.06),
        )
        axes[0].set_xlim(2.7, 6.0)
        axes[0].set_ylim(0.69, 0.82)
        axes[1].set_xlim(1.2, 6.0)
        axes[1].set_ylim(0.81, 0.88)

        _style_legend_box(legend)
        fig.tight_layout()

        out_path = save_figure(fig, output_dir, "pareto_rmse_tau_binary_vs_cont_plus_passk.pdf")
        outputs = [out_path]
        plt.close(fig)
        if any(_resolve_method_alias(m) == "lasso" for m in methods_cont_sel_all):
            methods_cont_sel_lasso = list(dict.fromkeys(methods_cont_sel_all))
            c_plot_lasso = c_summary[c_summary["method"].isin(methods_cont_sel_lasso)].set_index("method")
            comb_rows_lasso = []
            c_rmse_lasso = c_plot_lasso["rmse"] if "rmse" in c_plot_lasso.columns else pd.Series(dtype=float)
            c_tau_lasso = c_plot_lasso["tau"] if "tau" in c_plot_lasso.columns else pd.Series(dtype=float)
            c_rmse_n_lasso = c_plot_lasso["rmse_n"] if "rmse_n" in c_plot_lasso.columns else pd.Series(dtype=float)
            c_tau_n_lasso = c_plot_lasso["tau_n"] if "tau_n" in c_plot_lasso.columns else pd.Series(dtype=float)
            for method in methods_cont_sel_lasso:
                passk_method = method
                parsed = parse_mrmr_method(method)
                if (
                    passk_mrmr_mi_k is not None
                    and parsed is not None
                    and parsed["objective"] == "PMIQ"
                    and parsed["target"] == "y"
                    and parsed["predictor"] == "ridge"
                ):
                    passk_primary = mrmr_method_name(
                        degree=parsed["degree"],
                        mi_k=int(passk_mrmr_mi_k),
                        objective="PMIQ",
                        target="y",
                    )
                    passk_method = resolve_method_candidates([passk_primary, method], p_available) or method
                rmse_val = _weighted_two(
                    c_rmse_lasso.get(method, np.nan),
                    c_rmse_n_lasso.get(method, np.nan),
                    p_rmse.get(passk_method, np.nan),
                    p_rmse_n.get(passk_method, np.nan),
                )
                tau_val = _weighted_two(
                    c_tau_lasso.get(method, np.nan),
                    c_tau_n_lasso.get(method, np.nan),
                    p_tau.get(passk_method, np.nan),
                    p_tau_n.get(passk_method, np.nan),
                )
                if rmse_val is None or tau_val is None:
                    continue
                comb_rows_lasso.append({"method": method, "rmse": rmse_val, "tau": tau_val})
            cp_plot_lasso = pd.DataFrame(comb_rows_lasso)

            fig_lasso, axes_lasso = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_ONE_ROW_IN), constrained_layout=True)
            _plot_scatter_with_labels(axes_lasso[0], b_plot, "rmse", "tau", nudge_map=nudge_left)
            _apply_xy_labels(axes_lasso[0], r"$\bf{\leftarrow}$ RMSE (%) ", r"Kendall $\tau$ $\bf{\rightarrow}$")
            axes_lasso[0].set_title("Binary Scores")
            _plot_scatter_with_labels(axes_lasso[1], cp_plot_lasso, "rmse", "tau", nudge_map=nudge_right)
            _apply_xy_labels(axes_lasso[1], r"$\bf{\leftarrow}$ RMSE (%) ", r"Kendall $\tau$ $\bf{\rightarrow}$")
            axes_lasso[1].set_title("Continuous Scores")
            axes_lasso[0].text(-0.21, 1.12, "(a)", transform=axes_lasso[0].transAxes, fontsize=9, fontweight="bold", va="top", ha="left")
            axes_lasso[1].text(-0.21, 1.12, "(b)", transform=axes_lasso[1].transAxes, fontsize=9, fontweight="bold", va="top", ha="left")
            legend_methods_lasso = list(dict.fromkeys(methods_binary_sel + methods_cont_sel_lasso))
            handles_lasso = scatter_legend_handles(legend_methods_lasso)
            legend_lasso = fig_lasso.legend(
                handles=handles_lasso,
                loc="lower center",
                ncol=max(3, len(handles_lasso)),
                frameon=True,
                fancybox=False,
                columnspacing=0.3,
                handletextpad=0.3,
                labelspacing=0.35,
                borderpad=0.45,
                bbox_to_anchor=(0.5, -0.06),
            )
            axes_lasso[0].set_xlim(2.7, 6.0)
            axes_lasso[0].set_ylim(0.69, 0.82)
            axes_lasso[1].set_xlim(1.2, 6.0)
            axes_lasso[1].set_ylim(0.81, 0.88)
            _style_legend_box(legend_lasso)
            fig_lasso.tight_layout()
            outputs.append(save_figure(fig_lasso, output_dir, "pareto_rmse_tau_binary_vs_cont_plus_passk_with_lasso.pdf"))
            plt.close(fig_lasso)
        display_saved_plots(outputs, height=420)
    finally:
        captured = _capture_mode_end(capture_state)
    if return_figures:
        return outputs, captured
    return outputs


def build_notebook_02(
    *,
    output_dir: Path | None = None,
    setting: tuple[str, str, str] = (BINARY_SPLIT_METHOD, "5%", "30"),
    datasets_binary: Iterable[str] = BINARY_DATASETS_DEFAULT,
    ifeval_only: bool = False,
    main_methods: Iterable[str | tuple[str, ...]] | None = None,
    objective_methods: Iterable[str | tuple[str, ...]] | None = None,
    selection_methods: Iterable[str | tuple[str, ...]] | None = None,
) -> list[Path]:
    setup_matplotlib()
    ensure_dirs()
    output_dir = output_dir or _ensure_plot_subdir("02_mrmr_selection_metrics")
    setting_obj = get_setting_flexible(*setting)
    results_df = load_setting_results(setting_obj)
    jbl_df = load_setting_jbl(setting_obj)

    methods_default_req = list(main_methods) if main_methods is not None else _default_main_methods_binary()
    methods_default = _select_existing_methods(methods_default_req, set(results_df["method"]))
    perf = summarize_setting_standard(setting_obj, datasets=datasets_binary, methods=methods_default)
    sel = selection_objective_table(jbl_df, datasets=datasets_binary, methods=methods_default)
    merged = perf.merge(sel, on="method", how="left")

    outputs: list[Path] = []
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_ONE_ROW_IN), constrained_layout=True)
    nudge_left = {
        "random_sampling": (0.1, 0.0),
        "random_sampling_and_learn": (-0.55, 0.1),
        "krandom_sampling_and_learn": (-0.1, -0.3),
        "random_search_and_learn": (0.01, 0.15),
        "krandom_search_and_learn": (-0.1, -0.3),
        'metabench': (0.0, 0.25),
        "anchor_points_weighted": (0.1, 0.0),
        "anchor_points_weighted+": (-0.79, -0.002),
        "kanchor_points_weighted+": (-0.85, -0.001),  
        "gpirt1": (0.1, -0.05),
        "gpirt1+": (0.1, 0.0),
        "kgpirt1+": (0.1, -0.15),
        "mrmr5_MIQ_y": (-0.44, 0.00),
        "kmrmr5_MIQ_y": (-0.5, -0.008),
    }
    nudge_right = {
        "random_sampling": (0.15, -0.005),
        "random_sampling_and_learn": (0.15, -0.005),
        "krandom_sampling_and_learn": (-0.05, 0.008),
        "random_search_and_learn": (0.09, 0.002),
        "krandom_search_and_learn": (-0.61, -0.00),
        "anchor_points_weighted": (-0.73, 0.00),
        "anchor_points_weighted+": (-0.79, 0.00),
        "kanchor_points_weighted+": (-0.85, 0.00),
        "lasso": (-0.5, 0.00),  
        "gpirt1":  (0.1, -0.005),
        "gpirt1+": (0.1, -0.00),
        "kgpirt1+": (0.1, 0.00),  
        "mrmr5_MIQ_y": (-0.44, -0.005),
        "kmrmr5_MIQ_y": (-0.5, 0.00),
    }
    axes[0].set_xlim(1.5, 5.2)
    axes[1].set_xlim(1.5, 5.2)
    _plot_scatter_with_labels(axes[0], merged, "quotient", "rmse", nudge_map=nudge_left)
    axes[0].set_xlabel(r"mRMR MIQ score (Rel. / Red.) $\bf{\rightarrow}$")
    axes[0].set_ylabel(r"$\bf{\leftarrow}$ RMSE (%)")
    # axes[0].set_title("Binary: MIQ score vs RMSE")
    _plot_scatter_with_labels(axes[1], merged, "quotient", "tau", nudge_map=nudge_right)
    axes[1].set_xlabel(r"mRMR MIQ score (Rel. / Red.) $\bf{\rightarrow}$")
    axes[1].set_ylabel(r"Kendall $\tau$ $\bf{\rightarrow}$")
    # axes[1].set_title("Binary: MIQ score vs Kendall")
    axes[0].text(
        -0.18, 1.12, "(a)", transform=axes[0].transAxes, fontsize=9, fontweight="bold", va="top", ha="left"
    )
    axes[1].text(
        -0.20, 1.12, "(b)", transform=axes[1].transAxes, fontsize=9, fontweight="bold", va="top", ha="left"
    )
    legend_methods = merged["method"].astype(str).tolist() if "method" in merged.columns else []
    handles = scatter_legend_handles(legend_methods)
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=max(3, len(handles)),
        frameon=True,
        fancybox=False,
        columnspacing=0.3,
        handletextpad=0.3,
        labelspacing=0.35,
        borderpad=0.45,
        bbox_to_anchor=(0.5, -0.06),
    )
    _style_legend_box(legend)
    fig.tight_layout()
    outputs.append(save_figure(fig, output_dir, "miq_score_vs_rmse_tau_binary.pdf"))
    plt.close(fig)

    obj_methods_req: list[str | tuple[str, ...]] = list(objective_methods) if objective_methods is not None else [
        ("mrmr_MIQ_y", "mrmr3_MIQ_y"),
        ("mrmr5_MIQ_y",),
        ("mrmr7_MIQ_y",),
        ("mrmr_MID_y", "mrmr3_MID_y"),
        ("mrmr5_MID_y",),
        ("mrmr7_MID_y",),
        "random_sampling",
        "random_search_and_learn",
        "krandom_search_and_learn",
        "lasso",
        "anchor_points_weighted",
        ("gpirt1", "gpirt"),
        ("gpirt5", "gpirt"),
        ("gpirt10", "gpirt5", "gpirt"),
        "metabench",
    ]
    selected_methods = _select_existing_methods(obj_methods_req, set(jbl_df["method"]))
    target_datasets = ["ifeval"] if ifeval_only else list(datasets_binary)
    obj_df = selection_objective_table(jbl_df=jbl_df, datasets=target_datasets, methods=selected_methods)
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH_IN, 4.8), constrained_layout=True)
    keys = [
        ("relevance", "Relevance"),
        ("redundancy", "Redundancy"),
        ("difference", "Difference"),
        ("quotient", "Quotient"),
    ]
    ordered_methods = [m for m in selected_methods if m in set(obj_df["method"])]
    obj_by_method = obj_df.set_index("method")
    for ax, (key, title) in zip(axes.flatten(), keys):
        temp = obj_by_method.reindex(ordered_methods).dropna(subset=[key]).reset_index()
        labels = [_compact_method_label(m, mrmr_detail="mi_k", irt_include_dim=True, compact_anchor=False) for m in temp["method"]]
        colors = [method_style(m).color for m in temp["method"]]
        x = np.arange(len(temp))
        ax.bar(x, temp[key], color=colors, alpha=0.9, edgecolor="black", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right")
        ax.set_title(title)
    outputs.append(save_figure(fig, output_dir, "selection_objective_bars_2x2.pdf"))
    plt.close(fig)

    ablation_methods = list(selection_methods) if selection_methods is not None else [
        ("mrmr3_MIQ_y", "mrmr_MIQ_y"),
        ("mrmr5_MIQ_y", "mrmr5_MIQ_y"),
        ("mrmr7_MIQ_y", "mrmr7_MIQ_y"),
        ("mrmr_FCQ_y", "mrmr_FCQ_y"),
        ("mrmr3_MID_y", "mrmr_MID_y"),
        ("mrmr5_MID_y", "mrmr5_MID_y"),
        ("mrmr7_MID_y", "mrmr7_MID_y"),
        ("mrmr_FCD_y", "mrmr_FCD_y"),
    ]
    methods_sel = _select_existing_methods(ablation_methods, set(jbl_df["method"]))
    seq = average_selection_sequences(
        jbl_df=jbl_df,
        methods=methods_sel,
        datasets=(["ifeval"] if ifeval_only else datasets_binary),
    )
    fig, axes = plt.subplots(2, 4, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_TWO_ROW_IN), constrained_layout=False)
    for ax, method in zip(axes.flatten(), methods_sel):
        s = seq.get(method)
        if s is None:
            ax.axis("off")
            continue
        x = np.arange(1, len(s["relevance"]) + 1)
        ax.plot(x, s["relevance"], color="#1976d2", linewidth=1.0, label="Relevance")
        ax.plot(x, s["redundancy"], color="#d32f2f", linewidth=1.0, label="Redundancy")
        ax.plot(x, s["difference"], color="black", linestyle=":", linewidth=1.0, label="Difference")
        ax2 = ax.twinx()
        if len(x) > 1:
            ax2.plot(x[1:], s["quotient"][1:], color="#7b1fa2", linewidth=1.0, label="Quotient")
        ax2.tick_params(axis="y", colors="#7b1fa2", labelsize=7)
        ax.set_title(_compact_method_label(method, mrmr_detail="mi_k", compact_anchor=True), fontsize=8)
        ax.set_xlabel("Step")
    handles = [
        Line2D([0], [0], color="#1976d2", label="Relevance"),
        Line2D([0], [0], color="#d32f2f", label="Redundancy"),
        Line2D([0], [0], color="black", linestyle=":", label="Difference"),
        Line2D([0], [0], color="#7b1fa2", label="Quotient"),
    ]
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=True,
        fancybox=False,
        bbox_to_anchor=(0.5, 0.02),
    )
    _style_legend_box(legend)
    outputs.append(save_figure(fig, output_dir, "selection_metrics_steps_2x4.pdf"))
    plt.close(fig)
    display_saved_plots(outputs)
    return outputs


def _prepare_curve_frame(
    summary_df: pd.DataFrame,
    metric_key: str,
    fixed_col: str,
    fixed_value: str,
    x_col: str,
) -> pd.DataFrame:
    sub = summary_df[summary_df[fixed_col] == str(fixed_value)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["method", "x_value", metric_key])
    sub["x_value"] = sub[x_col].map(parse_coreset_size if x_col == "coreset_size" else parse_num_train_models)
    keep_cols = ["method", "x_value", metric_key]
    return sub[keep_cols].dropna(subset=[metric_key])


def _build_hybrid_degree_method_legend_handles(
    methods_for_legend: list[str],
    *,
    degree_values: tuple[int, ...] = (1, 2),
    degree_label_prefix: str = "d",
    method_cols: int = 4,
    compact_anchor: bool = True,
    force_d_at_end: bool = False,
) -> tuple[list[Line2D], int]:
    # Build unique method handles by degree-stripped name, so degree lives in a
    # separate linestyle legend section.
    label_to_handle: dict[str, Line2D] = {}
    for method in methods_for_legend:
        label = pretty_method_name(method, compact_anchor=compact_anchor)
        label = re.sub(r"\s*\(d=\d+\)$", "", label)
        if label in label_to_handle:
            continue
        style = method_style(method)
        label_to_handle[label] = Line2D(
            [0],
            [0],
            color=style.color,
            marker=style.marker,
            linestyle="-",
            label=label,
        )
    method_handles = list(label_to_handle.values())
    degree_handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            marker="None",
            linestyle=LINESTYLE_BY_DEGREE.get(d, "-"),
            label=f"{degree_label_prefix}={d}",
        )
        for d in degree_values
    ]
    blank = Line2D([0], [0], color="white", marker="None", linestyle="None", label="")
    ncol = method_cols + len(degree_handles)
    method_rows = max(1, math.ceil(len(method_handles) / method_cols))
    handles: list[Line2D] = []
    for row in range(method_rows):
        chunk = method_handles[row * method_cols : (row + 1) * method_cols]
        handles.extend(chunk)
        if len(chunk) < method_cols:
            handles.extend([blank] * (method_cols - len(chunk)))
        if row == 0:
            handles.extend(degree_handles)
        else:
            handles.extend([blank] * len(degree_handles))
    # if force_d_at_end:
    #     handles.extend(degree_handles)
    # else:
    #     handles.extend([blank] * len(degree_handles))
    return handles, ncol


def _plot_combined_grid(
    *,
    summary_df: pd.DataFrame,
    methods: list[str],
    fixed_nmodels: str,
    fixed_coreset: str,
    title: str,
    output_path: Path,
    include_bottom_row: bool = True,
    row_labels: tuple[str, str] = ("(a)", "(b)"),
    use_irt_dim_linestyle: bool = False,
    method_legend_solid_only: bool = False,
    collapse_irt_variant_legend: bool = False,
    legend_single_row: bool = False,
    legend_ncol: int = 5,
    legend_handles_override: list[Line2D] | None = None,
    append_degree_linestyle_legend: bool = False,
    degree_values: tuple[int, ...] = (1, 2, 3, 4),
    degree_label_prefix: str = "d",
    hybrid_degree_method_legend: bool = False,
    ylims: dict[str | tuple[str, str], tuple[float, float]] | None = None,
) -> Path:
    metric_order = ["rmse", "mae", "tau", "rho"]
    fixed_nmodels = nearest_available_label(summary_df, "num_train_models", str(fixed_nmodels), parse_num_train_models)
    fixed_coreset = nearest_available_label(summary_df, "coreset_size", str(fixed_coreset), parse_coreset_size)
    if include_bottom_row:
        fig, axes = plt.subplots(2, 4, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_TWO_ROW_IN), constrained_layout=False)
    else:
        fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_ONE_ROW_IN), constrained_layout=False)
        axes = np.array([axes])

    top_frames = {
        metric: _prepare_curve_frame(
            summary_df=summary_df, metric_key=metric, fixed_col="num_train_models", fixed_value=fixed_nmodels, x_col="coreset_size"
        )
        for metric in metric_order
    }
    bottom_frames = {
        metric: _prepare_curve_frame(
            summary_df=summary_df, metric_key=metric, fixed_col="coreset_size", fixed_value=fixed_coreset, x_col="num_train_models"
        )
        for metric in metric_order
    }

    for col_idx, metric in enumerate(metric_order):
        ax = axes[0, col_idx]
        frame = top_frames[metric]
        for method in methods:
            mf = frame[frame["method"] == method]
            if mf.empty:
                continue
            mf = mf.sort_values("x_value")
            _plot_method_lines(
                ax,
                mf["x_value"].to_numpy(),
                mf[metric].to_numpy(),
                method,
                pretty_method_name(method),
                use_irt_dim_linestyle=use_irt_dim_linestyle,
            )
        ax.set_title(metric_axis_title(metric))
        ax.set_xlabel("Coreset Size (%)")
        if ylims is not None:
            y_lim = ylims.get(("top", metric), ylims.get(metric))
            if y_lim is not None:
                ax.set_ylim(*y_lim)
        # if col_idx == 0:
        #     ax.set_ylabel(METRIC_SPECS[metric]["label"])
        if col_idx == 0:
            _subplot_row_label(ax, row_labels[0])

        if include_bottom_row:
            ax_b = axes[1, col_idx]
            frame_b = bottom_frames[metric]
            for method in methods:
                mf = frame_b[frame_b["method"] == method]
                if mf.empty:
                    continue
                mf = mf.sort_values("x_value")
                _plot_method_lines(
                    ax_b,
                    mf["x_value"].to_numpy(),
                    mf[metric].to_numpy(),
                    method,
                    pretty_method_name(method),
                    use_irt_dim_linestyle=use_irt_dim_linestyle,
                )
            ax_b.set_xlabel("Num Source Models")
            if ylims is not None:
                y_lim_b = ylims.get(("bottom", metric), ylims.get(metric))
                if y_lim_b is not None:
                    ax_b.set_ylim(*y_lim_b)
            if col_idx == 0:
                _subplot_row_label(ax_b, row_labels[1])

    legend_ncol_effective = legend_ncol
    if legend_handles_override is not None:
        legend_handles = legend_handles_override
    elif hybrid_degree_method_legend:
        legend_handles, legend_ncol_effective = _build_hybrid_degree_method_legend_handles(
            methods,
            degree_values=degree_values,
            degree_label_prefix=degree_label_prefix,
            method_cols=max(1, legend_ncol - 2),
        )
    else:
        legend_handles = []
        seen_irt_legend: set[tuple[str, str]] = set()
        for method in methods:
            style = method_style(method)
            if collapse_irt_variant_legend and infer_method_family(method) == "irt":
                irt_key = _irt_prefix_variant_from_method(method)
                if irt_key in seen_irt_legend:
                    continue
                seen_irt_legend.add(irt_key)
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=style.color,
                    marker=style.marker,
                    linestyle="-" if method_legend_solid_only else style.linestyle,
                    label=pretty_method_name(method, compact_anchor=True),
                )
            )
        if append_degree_linestyle_legend:
            linestyle_map = LINESTYLE_BY_IRT_DIM if degree_label_prefix == "p" else LINESTYLE_BY_DEGREE
            for degree in degree_values:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color="#555555",
                        marker="None",
                        linestyle=linestyle_map.get(degree, "-"),
                        label=f"{degree_label_prefix}={degree}",
                    )
                )
    actual_ncol = max(1, len(legend_handles)) if legend_single_row else legend_ncol_effective
    legend_rows = max(1, math.ceil(len(legend_handles) / actual_ncol))
    # Reserve footer space based on legend row count.
    if include_bottom_row:
        bottom_rect = min(0.10 + 0.045 * legend_rows, 0.12)
    else:
        bottom_rect = min(0.07 + 0.04 * legend_rows, 0.8)
    fig.tight_layout(rect=(0.0, bottom_rect, 1.0, 1.0), h_pad=0.5)
    legend = fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        ncol=actual_ncol,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        frameon=True,
        fancybox=False,
    )
    _style_legend_box(legend)
    out_path = save_figure(fig, output_path.parent, output_path.name)
    plt.close(fig)
    return out_path


_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

_COMBINED_GRID_TABLE_METRICS = [
    ("RMSE", "rmse", "min"),
    ("MAE", "mae", "min"),
    ("Kendall_tau", "tau", "max"),
    ("Spearman_rho", "rho", "max"),
    ("pearson_r", "pearson", "max"),
]


def _latex_escape(text: str) -> str:
    s = str(text)
    for old, new in _LATEX_ESCAPES.items():
        s = s.replace(old, new)
    return s


def _best_and_within_se_masks(mean_vals: pd.Series, se_vals: pd.Series, direction: str) -> tuple[pd.Series, pd.Series]:
    mean_numeric = pd.to_numeric(mean_vals, errors="coerce")
    se_numeric = pd.to_numeric(se_vals, errors="coerce")

    valid = mean_numeric.notna()
    if not valid.any():
        empty_mask = pd.Series(False, index=mean_vals.index)
        return empty_mask, empty_mask

    best = mean_numeric[valid].min() if direction == "min" else mean_numeric[valid].max()
    best_mask = valid & np.isclose(mean_numeric, best, rtol=1e-12, atol=1e-12)

    tol = se_numeric.fillna(0.0)
    if direction == "min":
        within_mask = valid & ~best_mask & (mean_numeric <= best + tol)
    else:
        within_mask = valid & ~best_mask & (mean_numeric >= best - tol)

    return best_mask, within_mask


def _format_uncertainty(value: float) -> str | None:
    if pd.isna(value):
        return None
    v = float(value)
    if v == 0.0:
        return "0.0"
    magnitude = math.floor(math.log10(abs(v)))
    decimals = max(0, 1 - magnitude)
    return f"{v:.{decimals}f}"


def _format_metric_cell(mean_val: float, se_val: float, style: str) -> str:
    if pd.isna(mean_val):
        return "--"

    mean_text = f"{mean_val:.4f}"
    if style == "bold":
        mean_text = f"\\textbf{{{mean_text}}}"
    elif style == "underline":
        mean_text = f"\\underline{{{mean_text}}}"

    se_text = _format_uncertainty(se_val)
    if se_text is not None:
        return f"{mean_text} {{\\scriptsize $\\pm$ {se_text}}}"
    return mean_text


def _render_combined_grid_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    if df.empty:
        return "% Empty table\n"

    table_df = df.copy()
    for _, metric_col, _ in _COMBINED_GRID_TABLE_METRICS:
        if metric_col in table_df.columns:
            table_df[metric_col] = table_df[metric_col].astype(object)
    for (_, _), block_idx in table_df.groupby(["coreset_size", "num_train_models"], sort=False).groups.items():
        block_idx = list(block_idx)
        for _, metric_col, direction in _COMBINED_GRID_TABLE_METRICS:
            se_col = f"{metric_col}_se"
            if metric_col not in table_df.columns or se_col not in table_df.columns:
                continue
            metric_vals = pd.to_numeric(table_df.loc[block_idx, metric_col], errors="coerce")
            metric_vals = metric_vals.dropna()
            best_idx = None
            second_idx = None
            if not metric_vals.empty:
                ascending = direction == "min"
                ranked_idx = metric_vals.sort_values(ascending=ascending).index.tolist()
                if ranked_idx:
                    best_idx = ranked_idx[0]
                if len(ranked_idx) >= 2:
                    second_idx = ranked_idx[1]
            for idx in block_idx:
                style = "normal"
                if best_idx is not None and idx == best_idx:
                    style = "bold"
                elif second_idx is not None and idx == second_idx:
                    style = "underline"
                table_df.loc[idx, metric_col] = _format_metric_cell(
                    table_df.loc[idx, metric_col],
                    table_df.loc[idx, se_col],
                    style,
                )

    group_keys = ["coreset_size", "num_train_models"]
    grouped_indices = []
    for _, idxs in table_df.groupby(group_keys, sort=False).groups.items():
        idxs = list(idxs)
        grouped_indices.append(idxs)
        for idx in idxs[1:]:
            table_df.loc[idx, "coreset_size"] = ""
            table_df.loc[idx, "num_train_models"] = ""

    out_cols = ["coreset_size", "num_train_models", "method", *[m[1] for m in _COMBINED_GRID_TABLE_METRICS]]
    header_map = {
        "coreset_size": "$n$",
        "num_train_models": "$M$",
        "method": "Method",
        "rmse": "RMSE",
        "mae": "MAE",
        "tau": "Kendall $\\tau$",
        "rho": "Spearman $\\rho$",
        "pearson": "Pearson $r$",
    }
    header_row = " & ".join(header_map[c] for c in out_cols) + r" \\"

    lines: list[str] = []
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{2pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.05}")
    lines.append(r"\setlength{\LTpre}{0pt}")
    lines.append(r"\setlength{\LTpost}{0pt}")
    lines.append(r"\setlength{\LTleft}{\fill}")
    lines.append(r"\setlength{\LTright}{\fill}")
    lines.append(r"\begin{longtable}{llp{0.2\textwidth}" + "r" * (len(out_cols) - 3) + "}")
    lines.append(rf"\caption{{{_latex_escape(caption)}}}\label{{{_latex_escape(label)}}}\\")
    lines.append(r"\toprule")
    lines.append(header_row)
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(header_row)
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(rf"\multicolumn{{{len(out_cols)}}}{{r}}{{\emph{{Continued on next page}}}} \\")
    lines.append(r"\midrule")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    block_start_indices = {idxs[0] for idxs in grouped_indices if idxs}
    first_row = True
    for idx, row in table_df[out_cols].iterrows():
        if idx in block_start_indices and not first_row:
            lines.append(r"\midrule")
        first_row = False

        cell_text = []
        for col in out_cols:
            value = row[col]
            if col in [m[1] for m in _COMBINED_GRID_TABLE_METRICS]:
                cell_text.append(str(value))
            else:
                raw_value = value
                if col == "method":
                    raw_value = _scatter_method_label(str(value), None)
                escaped = _latex_escape(raw_value)
                if col == "method":
                    escaped = escaped.replace(r"\_", r"\_\allowbreak")
                cell_text.append(escaped)
        lines.append(" & ".join(cell_text) + r" \\")

    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def _save_combined_grid_table_latex(
    summary_df: pd.DataFrame,
    methods: Iterable[str],
    output_path: Path,
    caption: str,
    *,
    overwrite: bool = False,
) -> Path:
    if output_path.is_file() and not overwrite:
        return output_path

    method_order = {m: i for i, m in enumerate(methods)}
    table_df = summary_df[summary_df["method"].isin(set(method_order))].copy()
    if table_df.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("% Empty table\n")
        return output_path

    table_df["_method_order"] = table_df["method"].map(method_order).fillna(10**9)
    table_df["_cs_sort"] = table_df["coreset_size"].map(_table_coreset_sort_key)
    table_df["_nm_sort"] = table_df["num_train_models"].map(_table_nmodels_sort_key)
    table_df = table_df.sort_values(["_cs_sort", "_nm_sort", "_method_order", "method"]).drop(
        columns=["_method_order", "_cs_sort", "_nm_sort"]
    )

    tex = _render_combined_grid_latex_table(
        table_df,
        caption=caption,
        label=f"tab:{output_path.stem}",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(tex)
    return output_path


def _exclude_methods(methods: list[str], methods_to_exclude: Iterable[str]) -> list[str]:
    excluded = set(methods_to_exclude)
    return [m for m in methods if m not in excluded]


def _insert_before_gpirt_variants(methods: list[str], additions: list[str]) -> list[str]:
    """
    Insert additions before the first gp-IRT-family method while preserving order.
    """
    base = list(dict.fromkeys(methods))
    extra = [m for m in additions if m not in base]
    if not extra:
        return base
    insert_at = next((i for i, m in enumerate(base) if "gpirt" in str(m)), len(base))
    return [*base[:insert_at], *extra, *base[insert_at:]]


def _table_coreset_sort_key(value: Any) -> float:
    try:
        return parse_coreset_size(str(value))
    except Exception:
        return float("inf")


def _table_nmodels_sort_key(value: Any) -> float:
    try:
        return parse_num_train_models(str(value))
    except Exception:
        return float("inf")


def build_notebook_03(
    *,
    output_dir: Path | None = None,
    binary_datasets: Iterable[str] = BINARY_DATASETS_DEFAULT,
    continuous_datasets: Iterable[str] = CONTINUOUS_DATASETS_DEFAULT,
    passk_benchmarks: Iterable[str] = PASSK_BENCHMARKS_DEFAULT,
    methods_binary_requested: Iterable[str | tuple[str, ...]] | None = None,
    methods_cont_requested: Iterable[str | tuple[str, ...]] | None = None,
    passk_mrmr_mi_k: int | None = 6,
    fixed_nmodels_binary: str = "30",
    fixed_coreset_binary: str = "10%",
    fixed_nmodels_continuous: str = "32",
    fixed_coreset_continuous: str = "10%",
    fixed_nmodels_passk: str = "15",
    fixed_coreset_passk: str = "10%",
    binary_filtered_exclude_methods: Iterable[str] = (),
    continuous_passk_filtered_exclude_methods: Iterable[str] = (),
    save_tables_latex: bool = True,
    table_overwrite: bool = False,
) -> list[Path]:
    setup_matplotlib()
    ensure_dirs()
    output_dir = output_dir or _ensure_plot_subdir("03_combined_grids")

    b_summary = summarize_split_standard(BINARY_SPLIT_METHOD, datasets=binary_datasets)
    c_summary = summarize_split_standard(CONTINUOUS_SPLIT_METHOD, datasets=continuous_datasets)
    p1_summary = summarize_split_passk(CONTINUOUS_SPLIT_METHOD, benchmarks=passk_benchmarks, source_mode="k1")
    po_summary = summarize_split_passk(CONTINUOUS_SPLIT_METHOD, benchmarks=passk_benchmarks, source_mode="opt")

    requested_methods_binary = list(methods_binary_requested) if methods_binary_requested is not None else [
        "random_sampling",
        "lasso",
        "random_sampling_and_learn",
        "krandom_search_and_learn",
        "random_search_and_learn",
        "krandom_search_and_learn",
        "anchor_points_weighted",
        "gpirt",
        ("mrmr5_MIQ_y", "mrmr_MIQ_y"),
        ("kmrmr5_MIQ_y", "kmrmr_MIQ_y"),
    ]
    requested_methods_cont = list(methods_cont_requested) if methods_cont_requested is not None else [
        "random_sampling",
        "lasso",
        "random_sampling_and_learn",
        "krandom_search_and_learn",
        "random_search_and_learn",
        "krandom_search_and_learn",
        "anchor_points_weighted",
        ("LEGO_gpirt", "gpirt"),
        ("mrmr_PMIQ_y", "mrmr5_PMIQ_y"),
        ("kmrmr_PMIQ_y", "kmrmr5_PMIQ_y", "kmrmr_PMIQ_y"),
    ]
    methods_binary = _select_existing_methods(requested_methods_binary, set(b_summary["method"]))
    methods_cont = _select_existing_methods(requested_methods_cont, set(c_summary["method"]))
    p1_available = set(p1_summary["method"].astype(str))
    po_available = set(po_summary["method"].astype(str))
    methods_p1 = _select_existing_methods(requested_methods_cont, p1_available)
    methods_po = _select_existing_methods(requested_methods_cont, po_available)

    def _apply_passk_mrmr_mi_k(methods: list[str], available: set[str]) -> list[str]:
        if passk_mrmr_mi_k is None:
            return methods
        remapped: list[str] = []
        for method in methods:
            selected = method
            parsed = parse_mrmr_method(method)
            if (
                parsed is not None
                and parsed["objective"] == "PMIQ"
                and parsed["target"] == "y"
                and parsed["predictor"] == "ridge"
            ):
                passk_primary = mrmr_method_name(
                    degree=parsed["degree"],
                    mi_k=int(passk_mrmr_mi_k),
                    objective="PMIQ",
                    target="y",
                )
                selected = resolve_method_candidates([passk_primary, method], available) or method
            remapped.append(selected)
        return list(dict.fromkeys(remapped))

    methods_p1 = _apply_passk_mrmr_mi_k(methods_p1, p1_available)
    methods_po = _apply_passk_mrmr_mi_k(methods_po, po_available)

    outputs = []
    combined_grid_legend_handles, combined_grid_legend_ncol = _combined_grid_plus_legend_handles()
    combined_grid_legend_handles_nonbinary = [
        h for h in combined_grid_legend_handles if h.get_label() not in {"Lasso", "MetaBench"}
    ]
    combined_grid_legend_kwargs_binary = {
        "method_legend_solid_only": True,
        "legend_handles_override": combined_grid_legend_handles,
        "legend_ncol": combined_grid_legend_ncol,
    }
    combined_grid_legend_kwargs_nonbinary = {
        "method_legend_solid_only": True,
        "legend_handles_override": combined_grid_legend_handles_nonbinary,
        "legend_ncol": max(1, math.ceil(len(combined_grid_legend_handles_nonbinary) / 2)),
    }
    outputs.append(
        _plot_combined_grid(
            summary_df=b_summary,
            methods=methods_binary,
            fixed_nmodels=fixed_nmodels_binary,
            fixed_coreset=fixed_coreset_binary,
            title="Binary combined grid",
            output_path=output_dir / "combined_grid_binary_2row.pdf",
            include_bottom_row=True,
            ylims={
                ("top", "rmse"): (1.9, 4.0),
                ("bottom", "rmse"): (2.2, 3.9),
                ("top", "mae"): (1.5, 3.2),
                ("bottom", "mae"): (1.8, 3.0),
                ("top", "tau"): (0.75, 0.88),
                ("bottom", "tau"): (0.78, 0.865),
                ("top", "rho"): (0.885, 0.975),
                ("bottom", "rho"): (0.91, 0.965),
            },
            **combined_grid_legend_kwargs_binary,
        )
    )
    methods_binary_filtered = _exclude_methods(methods_binary, binary_filtered_exclude_methods)
    outputs.append(
        _plot_combined_grid(
            summary_df=b_summary,
            methods=methods_binary_filtered,
            fixed_nmodels=fixed_nmodels_binary,
            fixed_coreset=fixed_coreset_binary,
            title="Binary combined grid (filtered methods)",
            output_path=output_dir / "combined_grid_binary_2row_filtered.pdf",
            include_bottom_row=True,
            ylims={
                ("top", "rmse"): (1.9, 4.0),
                ("bottom", "rmse"): (2.2, 3.9),
                ("top", "mae"): (1.5, 3.2),
                ("bottom", "mae"): (1.8, 3.0),
                ("top", "tau"): (0.75, 0.88),
                ("bottom", "tau"): (0.78, 0.865),
                ("top", "rho"): (0.885, 0.975),
                ("bottom", "rho"): (0.91, 0.965),
            },
            **combined_grid_legend_kwargs_binary,
        )
    )
    outputs.append(
        _plot_combined_grid(
            summary_df=b_summary,
            methods=methods_binary,
            fixed_nmodels=fixed_nmodels_binary,
            fixed_coreset=fixed_coreset_binary,
            title="Binary combined grid (single row)",
            output_path=output_dir / "combined_grid_binary_single_row.pdf",
            include_bottom_row=False,
            **combined_grid_legend_kwargs_binary,
        )
    )
    outputs.append(
        _plot_combined_grid(
            summary_df=c_summary,
            methods=methods_cont,
            fixed_nmodels=fixed_nmodels_continuous,
            fixed_coreset=fixed_coreset_continuous,
            title="Continuous combined grid",
            output_path=output_dir / "combined_grid_continuous_2row.pdf",
            include_bottom_row=True,
            # ylims={
            #     ("top", "rmse"): (1.9, 4.0),
            #     ("bottom", "rmse"): (2.2, 3.9),
            #     ("top", "mae"): (1.5, 3.2),
            #     ("bottom", "mae"): (1.8, 3.0),
            #     ("top", "tau"): (0.75, 0.88),
            #     ("bottom", "tau"): (0.78, 0.865),
            #     ("top", "rho"): (0.885, 0.975),
            #     ("bottom", "rho"): (0.91, 0.965),
            # },
            **combined_grid_legend_kwargs_nonbinary,
        )
    )
    outputs.append(
        _plot_combined_grid(
            summary_df=p1_summary,
            methods=methods_p1,
            fixed_nmodels=fixed_nmodels_passk,
            fixed_coreset=fixed_coreset_passk,
            title="Pass@k cross-k (source k=1)",
            output_path=output_dir / "combined_grid_passk_k1_2row.pdf",
            include_bottom_row=True,
            **combined_grid_legend_kwargs_nonbinary,
        )
    )
    outputs.append(
        _plot_combined_grid(
            summary_df=po_summary,
            methods=methods_po,
            fixed_nmodels=fixed_nmodels_passk,
            fixed_coreset=fixed_coreset_passk,
            title="Pass@k cross-k (source k=opt)",
            output_path=output_dir / "combined_grid_passk_kopt_2row.pdf",
            include_bottom_row=True,
            **combined_grid_legend_kwargs_nonbinary,
        )
    )
    outputs.append(
        _plot_combined_grid(
            summary_df=p1_summary,
            methods=methods_p1,
            fixed_nmodels=fixed_nmodels_passk,
            fixed_coreset=fixed_coreset_passk,
            title="Pass@k cross-k (source k=1, single row)",
            output_path=output_dir / "combined_grid_passk_k1_single_row.pdf",
            include_bottom_row=False,
            **combined_grid_legend_kwargs_nonbinary,
        )
    )

    fig, axes = plt.subplots(2, 4, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_TWO_ROW_IN), constrained_layout=False)
    subplotcoords2ylims = {
        (0,2): (0.875, 0.93),
        (0,3): (0.97, 0.99),
        (1,0): (2.2, 5.0),
        (1,1): (1.5, 4.0),
        # (1,2): (0.85, 0.95),
        # (1,3): (0.96, 0.985),
    }
    for row_idx, (summary, methods, title, fixed_nmodels) in enumerate(
        [
            (c_summary, methods_cont, "(a)", fixed_nmodels_continuous),
            (p1_summary, methods_p1, "(b)", fixed_nmodels_passk),
        ]
    ):
        for col_idx, metric in enumerate(["rmse", "mae", "tau", "rho"]):
            frame = _prepare_curve_frame(
                summary_df=summary,
                metric_key=metric,
                fixed_col="num_train_models",
                fixed_value=fixed_nmodels,
                x_col="coreset_size",
            )
            ax = axes[row_idx, col_idx]
            for method in methods:
                mf = frame[frame["method"] == method].sort_values("x_value")
                if mf.empty:
                    continue
                _plot_method_lines(ax, mf["x_value"].to_numpy(), mf[metric].to_numpy(), method)

            if (row_idx, col_idx) in subplotcoords2ylims:
                ax.set_ylim(*subplotcoords2ylims[(row_idx, col_idx)])

            ax.set_xlabel("Coreset Size (%)")
            if col_idx == 0:
                _subplot_row_label(ax, title)
            if row_idx == 0:
                ax.set_title(metric_axis_title(metric))
    handles = combined_grid_legend_handles_nonbinary
    legend_ncol = max(1, math.ceil(len(handles) / 2))
    legend_rows = max(1, math.ceil(len(handles) / legend_ncol))
    # Keep footer compact for this final two-row combined figure so subplot rows
    # do not get vertically squeezed.
    bottom_rect = min(0.06 + 0.035 * legend_rows, 0.14)
    fig.tight_layout(rect=(0.0, bottom_rect, 1.0, 1.0), h_pad=0.5)
    legend = fig.legend(
        handles=handles,
        ncol=legend_ncol,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        frameon=True,
        fancybox=False,
    )
    _style_legend_box(legend)
    outputs.append(save_figure(fig, output_dir, "combined_grid_continuous_and_passk_k1_two_row.pdf"))
    plt.close(fig)

    methods_cont_filtered = _exclude_methods(methods_cont, continuous_passk_filtered_exclude_methods)
    methods_p1_filtered = _exclude_methods(methods_p1, continuous_passk_filtered_exclude_methods)
    fig, axes = plt.subplots(2, 4, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_TWO_ROW_IN), constrained_layout=False)
    for row_idx, (summary, methods, title, fixed_nmodels) in enumerate(
        [
            (c_summary, methods_cont_filtered, "(a)", fixed_nmodels_continuous),
            (p1_summary, methods_p1_filtered, "(b)", fixed_nmodels_passk),
        ]
    ):
        for col_idx, metric in enumerate(["rmse", "mae", "tau", "rho"]):
            frame = _prepare_curve_frame(
                summary_df=summary,
                metric_key=metric,
                fixed_col="num_train_models",
                fixed_value=fixed_nmodels,
                x_col="coreset_size",
            )
            ax = axes[row_idx, col_idx]
            for method in methods:
                mf = frame[frame["method"] == method].sort_values("x_value")
                if mf.empty:
                    continue
                _plot_method_lines(ax, mf["x_value"].to_numpy(), mf[metric].to_numpy(), method)

            if (row_idx, col_idx) in subplotcoords2ylims:
                ax.set_ylim(*subplotcoords2ylims[(row_idx, col_idx)])

            ax.set_xlabel("Coreset Size (%)")
            if col_idx == 0:
                _subplot_row_label(ax, title)
            if row_idx == 0:
                ax.set_title(metric_axis_title(metric))
    handles = combined_grid_legend_handles_nonbinary
    legend_ncol = max(1, math.ceil(len(handles) / 2))
    legend_rows = max(1, math.ceil(len(handles) / legend_ncol))
    bottom_rect = min(0.06 + 0.035 * legend_rows, 0.14)
    fig.tight_layout(rect=(0.0, bottom_rect, 1.0, 1.0), h_pad=0.5)
    legend = fig.legend(
        handles=handles,
        ncol=legend_ncol,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        frameon=True,
        fancybox=False,
    )
    _style_legend_box(legend)
    outputs.append(save_figure(fig, output_dir, "combined_grid_continuous_and_passk_k1_two_row_filtered.pdf"))
    plt.close(fig)

    if save_tables_latex:
        table_dir = output_dir / "tables"
        # Table-specific additions requested for notebook-03 reporting.
        methods_binary_table = _insert_before_gpirt_variants(
            list(methods_binary),
            _select_existing_methods(
                ["lasso", "metabench", "pirt1", "pirt1+", "kpirt1+"],
                set(b_summary["method"].astype(str)),
            ),
        )
        methods_cont_table = _insert_before_gpirt_variants(
            list(methods_cont),
            _select_existing_methods(
                ["lasso", "B_pirt5", "B_pirt5+", "kB_pirt5+"],
                set(c_summary["method"].astype(str)),
            ),
        )
        methods_p1_table = _insert_before_gpirt_variants(
            list(methods_p1),
            _select_existing_methods(
                ["lasso", "B_pirt5", "B_pirt5+", "kB_pirt5+"],
                set(p1_summary["method"].astype(str)),
            ),
        )
        methods_po_table = _insert_before_gpirt_variants(
            list(methods_po),
            _select_existing_methods(
                ["lasso", "B_pirt5", "B_pirt5+", "kB_pirt5+"],
                set(po_summary["method"].astype(str)),
            ),
        )
        outputs.append(
            _save_combined_grid_table_latex(
                b_summary,
                methods_binary_table,
                table_dir / "combined_grid_binary.tex",
                "Combined grid table for binary datasets",
                overwrite=table_overwrite,
            )
        )
        outputs.append(
            _save_combined_grid_table_latex(
                c_summary,
                methods_cont_table,
                table_dir / "combined_grid_continuous.tex",
                "Combined grid table for continuous datasets",
                overwrite=table_overwrite,
            )
        )
        outputs.append(
            _save_combined_grid_table_latex(
                p1_summary,
                methods_p1_table,
                table_dir / "combined_grid_passk_k1.tex",
                "Combined grid table for pass@k cross-k (source k=1)",
                overwrite=table_overwrite,
            )
        )
        outputs.append(
            _save_combined_grid_table_latex(
                po_summary,
                methods_po_table,
                table_dir / "combined_grid_passk_kopt.tex",
                "Combined grid table for pass@k cross-k (source k=opt)",
                overwrite=table_overwrite,
            )
        )
    display_saved_plots(outputs)
    return outputs


def _difficulty_hist_plot(
    datasets: list[str],
    methods: list[str],
    jbl_df: pd.DataFrame,
    output_path: Path,
    bins: int = 30,
    height_in: float | None = None,
) -> Path:
    n_rows = len(datasets)
    n_cols = len(methods)
    height = height_in or max(3.0, n_rows * 1.6)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.8, height), constrained_layout=True)
    if n_rows == 1:
        axes = np.array([axes])
    for r, dataset in enumerate(datasets):
        scores = load_score_matrix(dataset)
        if scores is None:
            for c in range(n_cols):
                axes[r, c].axis("off")
            continue
        difficulty = 1.0 - scores.mean(axis=0)
        hist_range = (float(difficulty.min()), float(difficulty.max()))
        for c, method in enumerate(methods):
            ax = axes[r, c]
            sub = jbl_df[(jbl_df["dataset"] == dataset) & (jbl_df["method"] == method)]
            pooled = []
            for _, row in sub.iterrows():
                idx = row.get("coreset_indices")
                if idx is None:
                    continue
                idx_arr = np.asarray(idx, dtype=int)
                idx_arr = idx_arr[(idx_arr >= 0) & (idx_arr < len(difficulty))]
                if len(idx_arr) > 0:
                    pooled.append(difficulty[idx_arr])
            pooled_vals = np.concatenate(pooled) if pooled else np.array([])
            ax.hist(difficulty, bins=bins, range=hist_range, color="#c0c0c0", alpha=0.65)
            if len(pooled_vals) > 0:
                overlay_color = "black" if _resolve_method_alias(method) == "random_sampling" else method_style(method).color
                ax.hist(
                    pooled_vals,
                    bins=bins,
                    range=hist_range,
                    color=overlay_color,
                    alpha=0.72,
                )
            if r == 0:
                ax.set_title(
                    _compact_method_label(
                        method,
                        mrmr_detail="mi_k",
                        compact_anchor=True,
                        irt_include_dim=True,
                    ),
                    fontsize=9,
                )
            if c == 0:
                ax.set_ylabel(f"{dataset}\nCount", fontsize=8)
            ax.set_xlabel("Difficulty")
    out = save_figure(fig, output_path.parent, output_path.name)
    plt.close(fig)
    return out


def _stability_timing_dataframe(
    *,
    binary_split: str = BINARY_SPLIT_METHOD,
    continuous_split: str = CONTINUOUS_SPLIT_METHOD,
    cs_values: Iterable[str] = ("10%",),
    binary_datasets: Iterable[str] = BINARY_DATASETS_DEFAULT,
    continuous_datasets: Iterable[str] = CONTINUOUS_DATASETS_DEFAULT,
    binary_methods: Iterable[str | tuple[str, ...]] = (
        "random_sampling",
        "random_search_and_learn",
        "krandom_search_and_learn",
        "lasso",
        "anchor_points_weighted",
        "metabench",
        ("gpirt1", "gpirt"),
        ("gpirt5", "gpirt"),
        ("gpirt10", "gpirt"),
        ("mrmr3_MIQ_y", "mrmr_MIQ_y"),
        ("mrmr5_MIQ_y",),
        ("mrmr7_MIQ_y",),
    ),
    continuous_methods: Iterable[str | tuple[str, ...]] = (),
) -> pd.DataFrame:
    rows = []
    num_q = load_num_questions()

    def _collect(
        split: str,
        datasets: Iterable[str],
        methods: Iterable[str | tuple[str, ...]],
        data_type: str,
    ) -> None:
        dataset_set = set(datasets)
        cs_set = set(cs_values)

        coresets_by_method_dataset: dict[str, dict[str, list[list[int]]]] = {}
        timing_by_method_dataset: dict[str, dict[str, list[float]]] = {}

        for setting in discover_settings(split_method=split):
            if setting.coreset_size not in cs_set:
                continue

            jbl_df = load_setting_coresets(setting)
            if not jbl_df.empty:
                requested_setting = _select_existing_methods(methods, set(jbl_df["method"]))
                jbl_df = jbl_df[
                    jbl_df["dataset"].isin(dataset_set) & jbl_df["method"].isin(requested_setting)
                ]
                for _, row in jbl_df.iterrows():
                    method = row["method"]
                    dataset = row["dataset"]
                    coreset = row.get("coreset_indices")
                    if coreset is None:
                        continue
                    method_map = coresets_by_method_dataset.setdefault(method, {})
                    method_map.setdefault(dataset, []).append(list(np.asarray(coreset, dtype=int)))

            res_df = load_setting_results(setting)
            if not res_df.empty:
                requested_setting = _select_existing_methods(methods, set(res_df["method"]))
                res_df = res_df[
                    res_df["dataset"].isin(dataset_set) & res_df["method"].isin(requested_setting)
                ]
                if not res_df.empty:
                    res_df = res_df.copy()
                    res_df["execution_time"] = res_df["training_time"].fillna(0.0) + res_df["inference_time"].fillna(0.0)
                    grouped = (
                        res_df.groupby(["dataset", "method"], as_index=False)["execution_time"]
                        .mean()
                    )
                    for _, row in grouped.iterrows():
                        method = row["method"]
                        dataset = row["dataset"]
                        timing_map = timing_by_method_dataset.setdefault(method, {})
                        timing_map.setdefault(dataset, []).append(float(row["execution_time"]))

        available = set(coresets_by_method_dataset.keys()) | set(timing_by_method_dataset.keys())
        selected = _select_existing_methods(methods, available)
        for method in selected:
            per_dataset_hamming: list[float] = []
            per_dataset_nogueira: list[float] = []
            per_dataset_time: list[float] = []
            for dataset in sorted(dataset_set):
                coreset_rows = coresets_by_method_dataset.get(method, {}).get(dataset, [])
                if len(coreset_rows) > 40:
                    idx = np.linspace(0, len(coreset_rows) - 1, 40).astype(int)
                    coreset_rows = [coreset_rows[i] for i in idx]
                d = num_q.get(dataset, 0)
                if d > 0 and len(coreset_rows) >= 2:
                    h = average_pairwise_hamming(coreset_rows, d)
                    n = compute_nogueira_phi(coreset_rows, d)
                    if h is not None:
                        per_dataset_hamming.append(1.0 - h)
                    if n is not None:
                        per_dataset_nogueira.append(n)
                dataset_times = timing_by_method_dataset.get(method, {}).get(dataset, [])
                if dataset_times:
                    per_dataset_time.append(float(np.mean(dataset_times)))

            rows.append(
                {
                    "method": method,
                    "pretty": _compact_method_label(method, mrmr_detail="mi_k", irt_include_dim=True, compact_anchor=False),
                    "data_type": data_type,
                    "hamming": float(np.mean(per_dataset_hamming)) if per_dataset_hamming else np.nan,
                    "hamming_se": standard_error(np.asarray(per_dataset_hamming, dtype=float)) if per_dataset_hamming else np.nan,
                    "nogueira": float(np.mean(per_dataset_nogueira)) if per_dataset_nogueira else np.nan,
                    "nogueira_se": standard_error(np.asarray(per_dataset_nogueira, dtype=float)) if per_dataset_nogueira else np.nan,
                    "timing": float(np.mean(per_dataset_time)) if per_dataset_time else np.nan,
                    "timing_se": standard_error(np.asarray(per_dataset_time, dtype=float)) if per_dataset_time else np.nan,
                }
            )

    _collect(binary_split, binary_datasets, binary_methods, "binary")
    _collect(continuous_split, continuous_datasets, continuous_methods, "continuous")
    return pd.DataFrame(rows)


def build_notebook_04(
    *,
    output_dir: Path | None = None,
    openllm_filtered: Iterable[str] = ("ifeval", "mmlu_pro", "arc_challenge", "bbh", "musr"),
    difficulty_setting: tuple[str, str, str] = (BINARY_SPLIT_METHOD, "10%", "30"),
    difficulty_methods: Iterable[str | tuple[str, ...]] | None = None,
    stability_cs_values: Iterable[str] = ("10%",),
    stability_binary_methods: Iterable[str | tuple[str, ...]] = (
        "random_sampling",
        "random_search_and_learn",
        "krandom_search_and_learn",
        "lasso",
        "anchor_points_weighted",
        "metabench",
        ("gpirt1", "gpirt"),
        ("gpirt5", "gpirt"),
        ("gpirt10", "gpirt"),
        ("mrmr3_MIQ_y", "mrmr_MIQ_y"),
        ("mrmr5_MIQ_y",),
        ("mrmr7_MIQ_y",),
    ),
    stability_cont_methods: Iterable[str | tuple[str, ...]] = (),
) -> list[Path]:
    setup_matplotlib()
    ensure_dirs()
    output_dir = output_dir or _ensure_plot_subdir("04_difficulty_stability_timing")
    outputs: list[Path] = []

    setting_binary = get_setting_flexible(*difficulty_setting)
    jbl_binary = load_setting_coresets(setting_binary)
    difficulty_methods_req = list(difficulty_methods) if difficulty_methods is not None else [
        "random_sampling",
        "anchor_points_weighted",
        ("gpirt1", "gpirt"),
        ("mrmr5_MIQ_y", "mrmr_MIQ_y"),
    ]
    difficulty_methods = _select_existing_methods(
        difficulty_methods_req,
        set(jbl_binary["method"]),
    )

    source_groups = {
        "openllm": [d for d in openllm_datasets if d in set(jbl_binary["dataset"])],
        "helm": [d for d in helm_datasets if d in set(jbl_binary["dataset"])],
        "continuous_cat_main": [d for d in continuous_cat_main_datasets if d in set(jbl_binary["dataset"])],
    }
    for source_name, datasets in source_groups.items():
        if not datasets:
            continue
        outputs.append(
            _difficulty_hist_plot(
                datasets=datasets,
                methods=difficulty_methods,
                jbl_df=jbl_binary,
                output_path=output_dir / f"difficulty_grid_{source_name}.pdf",
            )
        )

    outputs.append(
        _difficulty_hist_plot(
            datasets=[d for d in openllm_filtered if d in set(jbl_binary["dataset"])],
            methods=difficulty_methods,
            jbl_df=jbl_binary,
            output_path=output_dir / "difficulty_grid_openllm_filtered.pdf",
            height_in=8.6,
        )
    )

    stab_df = _stability_timing_dataframe(
        cs_values=stability_cs_values,
        binary_methods=stability_binary_methods,
        continuous_methods=stability_cont_methods,
    )
    if not stab_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_ONE_ROW_IN), constrained_layout=True)
        for ax, metric, title in [
            (axes[0], "nogueira", r"$\hat{\Phi}$ Stability"),
            (axes[1], "hamming", "Hamming Stability"),
        ]:
            temp = stab_df.dropna(subset=[metric]).copy()
            temp = temp.reset_index(drop=True)
            x = np.arange(len(temp))
            colors = [method_style(m).color for m in temp["method"]]
            ax.bar(
                x,
                temp[metric],
                yerr=temp[f"{metric}_se"],
                color=colors,
                edgecolor="black",
                linewidth=1.0,
                alpha=0.9,
                capsize=2.5,
                error_kw={"elinewidth": 0.8, "capthick": 0.8},
            )
            labels = [
                _compact_method_label(m, mrmr_detail="mi_k", irt_include_dim=True, compact_anchor=False)
                for m in temp["method"]
            ]
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=70, ha="right")
            ax.set_title(title)
        axes[0].text(
            -0.15, 1.05, "(a)", transform=axes[0].transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left"
        )
        axes[1].text(
            -0.15, 1.05, "(b)", transform=axes[1].transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left"
        )
        outputs.append(save_figure(fig, output_dir, "stability_hamming_vs_nogueira.pdf"))
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_ONE_ROW_IN), constrained_layout=True)
        temp = stab_df.dropna(subset=["timing"]).copy().sort_values(["timing"])
        x = np.arange(len(temp))
        colors = [method_style(m).color for m in temp["method"]]
        ax.bar(
            x,
            temp["timing"],
            yerr=temp["timing_se"],
            color=colors,
            edgecolor="black",
            linewidth=1.0,
            capsize=2.5,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )
        labels = [
            _compact_method_label(m, mrmr_detail="mi_k", irt_include_dim=True, compact_anchor=False)
            for m in temp["method"]
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=70, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("Time (s)")
        # ax.set_title("Timing")
        outputs.append(save_figure(fig, output_dir, "timing_bar_logscale.pdf"))
        plt.close(fig)

    mmlu_methods = difficulty_methods
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_TWO_ROW_IN), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 1.0])
    mmlu = "mmlu_pro"
    scores = load_score_matrix(mmlu)
    if scores is not None:
        difficulty = 1.0 - scores.mean(axis=0)
        for c, method in enumerate(mmlu_methods):
            ax = fig.add_subplot(gs[0, c])
            sub = jbl_binary[(jbl_binary["dataset"] == mmlu) & (jbl_binary["method"] == method)]
            pooled = []
            for _, row in sub.iterrows():
                idx = row.get("coreset_indices")
                if idx is None:
                    continue
                idx_arr = np.asarray(idx, dtype=int)
                idx_arr = idx_arr[(idx_arr >= 0) & (idx_arr < len(difficulty))]
                if len(idx_arr) > 0:
                    pooled.append(difficulty[idx_arr])
            pooled_vals = np.concatenate(pooled) if pooled else np.array([])
            ax.hist(difficulty, bins=20, color="#c0c0c0", alpha=0.65)
            if len(pooled_vals) > 0:
                overlay_color = "black" if _resolve_method_alias(method) == "random_sampling" else method_style(method).color
                ax.hist(pooled_vals, bins=20, color=overlay_color, alpha=0.72)
            ax.set_title(
                _compact_method_label(
                    method,
                    mrmr_detail="mi_k_plain",
                    compact_anchor=True,
                    irt_include_dim=True,
                ),
                fontsize=8,
            )
            ax.set_xlabel("Difficulty")
            if c == 0:
                ax.set_ylabel("Count")
                _subplot_row_label(ax, "(a)", xshift=-0.35)
    ax_stab = fig.add_subplot(gs[1, :2])
    ax_time = fig.add_subplot(gs[1, 2:])
    if not stab_df.empty:
        tmp = stab_df.dropna(subset=["nogueira"]).copy()
        x = np.arange(len(tmp))
        ax_stab.bar(
            x,
            tmp["nogueira"],
            yerr=tmp["nogueira_se"],
            color=[method_style(m).color for m in tmp["method"]],
            capsize=2,
            error_kw={"elinewidth": 0.6, "capthick": 0.6},
        )
        ax_stab.set_xticks(x)
        ax_stab.set_xticklabels(
            [_compact_method_label(m, mrmr_detail="mi_k_plain", compact_anchor=True, irt_include_dim=True) for m in tmp["method"]],
            rotation=60,
            ha="right",
        )
        ax_stab.set_ylabel(r"$\hat{\Phi}$ Stability")
        _subplot_row_label(ax_stab, "(b)")
        ttmp = stab_df.dropna(subset=["timing"]).copy()
        x2 = np.arange(len(ttmp))
        ax_time.bar(
            x2,
            ttmp["timing"],
            yerr=ttmp["timing_se"],
            color=[method_style(m).color for m in ttmp["method"]],
            capsize=2,
            error_kw={"elinewidth": 0.6, "capthick": 0.6},
        )
        ax_time.set_xticks(x2)
        ax_time.set_xticklabels(
            [_compact_method_label(m, mrmr_detail="mi_k_plain", compact_anchor=True, irt_include_dim=True) for m in ttmp["method"]],
            rotation=60,
            ha="right",
        )
        ax_time.set_yscale("log")
        ax_time.set_ylabel("Timing (s)")
        _subplot_row_label(ax_time, "(c)")
    outputs.append(save_figure(fig, output_dir, "mmlu_difficulty_plus_stability_timing.pdf"))
    plt.close(fig)

    display_saved_plots(outputs)
    return outputs


def _plot_true_vs_pred_grid(
    *,
    jbl_df: pd.DataFrame,
    dataset_list: list[str],
    methods: list[str],
    output_path: Path,
    row_height: float = 2.4,
) -> Path:
    n_rows = len(dataset_list)
    n_cols = len(methods)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_WIDTH_IN, max(2.0, row_height * n_rows)) if n_rows > 1 else (FIG_WIDTH_IN, 2.0), constrained_layout=True)
    if n_rows == 1:
        axes = np.array([axes])
    for r, dataset in enumerate(dataset_list):
        for c, method in enumerate(methods):
            ax = axes[r, c]
            sub = jbl_df[(jbl_df["dataset"] == dataset) & (jbl_df["method"] == method)]
            if sub.empty:
                ax.axis("off")
                continue
            train_true, train_pred, test_true, test_pred = [], [], [], []
            for _, row in sub.iterrows():
                if row.get("true_acc_train") is not None and row.get("pred_acc_train") is not None:
                    train_true.extend(row["true_acc_train"])
                    train_pred.extend(row["pred_acc_train"])
                if row.get("true_acc_test") is not None and row.get("pred_acc_test") is not None:
                    test_true.extend(row["true_acc_test"])
                    test_pred.extend(row["pred_acc_test"])
            if test_true:
                ax.scatter(test_true, test_pred, s=8, alpha=1.0, color="#fb8c00", label="Test", zorder=2)
            if train_true:
                ax.scatter(train_true, train_pred, s=8, alpha=0.9, color="#1e88e5", label="Train", zorder=3)

            true_all = np.asarray([*train_true, *test_true], dtype=float)
            pred_all = np.asarray([*train_pred, *test_pred], dtype=float)
            valid = np.isfinite(true_all) & np.isfinite(pred_all)
            true_all = true_all[valid]
            pred_all = pred_all[valid]

            if len(true_all) > 0:
                lo = float(min(true_all.min(), pred_all.min()))
                hi = float(max(true_all.max(), pred_all.max()))
                if hi > lo:
                    pad = 0.02 * (hi - lo)
                    lo -= pad
                    hi += pad
                ax.plot([lo, hi], [lo, hi], color="black", linestyle=":", linewidth=1.0, zorder=1)

            if len(true_all) > 0:
                diff = pred_all - true_all
                rmse = float(np.sqrt(np.mean(diff**2)))
                mae = float(np.mean(np.abs(diff)))
            else:
                rmse = np.nan
                mae = np.nan

            if len(true_all) > 1:
                corr_df = pd.DataFrame({"true": true_all, "pred": pred_all})
                tau = float(corr_df["true"].corr(corr_df["pred"], method="kendall"))
                rho = float(corr_df["true"].corr(corr_df["pred"], method="spearman"))
            else:
                tau = np.nan
                rho = np.nan
            title_line_1 = pretty_method_name(method, compact_anchor=True)
            parsed = parse_mrmr_method(method)
            if parsed is not None and int(parsed.get("degree", -1)) == 2:
                title_line_1 = "mRMR++"
            title_line_2 = f"RMSE={rmse:.3g}, MAE={mae:.3g}"
            title_line_3 = rf"$\tau$={tau:.3g}, $\rho$={rho:.3g}"
            ax.set_title(f"{title_line_1}\n{title_line_2}\n{title_line_3}", fontsize=7.5)
            if c == 0:
                ax.set_ylabel(dataset)
            ax.set_xlabel("True")
            if r == 0 and c == 0:
                handles = [
                    Line2D([0], [0], marker="o", linestyle="None", color="#1e88e5", label="Train"),
                    Line2D([0], [0], marker="o", linestyle="None", color="#fb8c00", label="Test"),
                ]
                legend = ax.legend(
                    handles=handles,
                    loc="upper left",
                    ncol=1,
                    frameon=True,
                    fancybox=False,
                    fontsize=7.5,
                    borderpad=0.35,
                    handletextpad=0.35,
                    labelspacing=0.25,
                )
                _style_legend_box(legend)
    out = save_figure(fig, output_path.parent, output_path.name)
    plt.close(fig)
    return out


def build_notebook_05(
    *,
    output_dir: Path | None = None,
    setting: tuple[str, str, str] = (BINARY_SPLIT_METHOD, "10%", "30"),
    single_datasets: Iterable[str] = ("bbh", "mmlu_pro", "arc_challenge"),
    grouped_sources: dict[str, list[str]] | None = None,
    methods_requested: Iterable[str | tuple[str, ...]] | None = None,
    include_extra_big_continuous_and_passk: bool = True,
    continuous_setting: tuple[str, str, str] = (CONTINUOUS_SPLIT_METHOD, "10%", "32"),
    passk_setting: tuple[str, str, str] = (CONTINUOUS_SPLIT_METHOD, "10%", "15"),
    continuous_datasets: Iterable[str] = CONTINUOUS_DATASETS_DEFAULT,
    passk_datasets: Iterable[str] = PASSK_BENCHMARKS_DEFAULT,
    extra_big_row_height: float = 1.7,
) -> list[Path]:
    setup_matplotlib()
    ensure_dirs()
    output_dir = output_dir or _ensure_plot_subdir("05_true_vs_pred")
    methods_req = list(methods_requested) if methods_requested is not None else [
        "random_sampling",
        "anchor_points_weighted",
        "gpirt",
        ("kmrmr5_MIQ_y", "kmrmr_MIQ_y"),
    ]
    outputs = []
    setting_obj = get_setting_flexible(*setting)
    jbl_df = load_setting_jbl(setting_obj)
    methods = _select_existing_methods(methods_req, set(jbl_df["method"]))
    for dataset in single_datasets:
        if dataset not in set(jbl_df["dataset"]):
            continue
        outputs.append(
            _plot_true_vs_pred_grid(
                jbl_df=jbl_df,
                dataset_list=[dataset],
                methods=methods,
                output_path=output_dir / f"true_vs_pred_{dataset}.pdf",
            )
        )
    grouped_sources = grouped_sources or {
        "openllm_filtered": ["ifeval", "mmlu_pro", "arc_challenge", "bbh", "musr"],
        "openllm": list(openllm_datasets),
        "helm": list(helm_datasets),
    }
    for name, datasets in grouped_sources.items():
        available_ds = [d for d in datasets if d in set(jbl_df["dataset"])]
        if not available_ds:
            continue
        outputs.append(
            _plot_true_vs_pred_grid(
                jbl_df=jbl_df,
                dataset_list=available_ds,
                methods=methods,
                output_path=output_dir / f"true_vs_pred_grid_{name}.pdf",
                row_height=1.7,
            )
        )

    if include_extra_big_continuous_and_passk:
        cont_setting_obj = get_setting_flexible(*continuous_setting)
        cont_jbl_df = load_setting_jbl(cont_setting_obj)
        cont_available_ds = [d for d in continuous_datasets if d in set(cont_jbl_df["dataset"])]
        cont_methods = _select_existing_methods(methods_req, set(cont_jbl_df["method"]))
        if cont_available_ds and cont_methods:
            outputs.append(
                _plot_true_vs_pred_grid(
                    jbl_df=cont_jbl_df,
                    dataset_list=cont_available_ds,
                    methods=cont_methods,
                    output_path=output_dir / "true_vs_pred_grid_continuous_big.pdf",
                    row_height=extra_big_row_height,
                )
            )

        passk_setting_obj = get_setting_flexible(*passk_setting)
        passk_jbl_df = load_setting_jbl(passk_setting_obj)
        passk_dataset_values = set(passk_jbl_df["dataset"].dropna().astype(str))
        passk_available_ds = [d for d in passk_datasets if d in passk_dataset_values]
        if not passk_available_ds:
            # Resolve benchmark names (e.g. "MBPP_mbpp") to dataset keys like
            # "MBPP_mbpp_pass_at_1_v3_open" present in the JBL table.
            k_order = [1, 2, 4, 8, 16, 32, 64, 128]
            for bench in passk_datasets:
                bench = str(bench)
                candidates = [d for d in passk_dataset_values if d.startswith(f"{bench}_pass_at_")]
                if not candidates:
                    continue
                preferred = None
                for k in k_order:
                    open_name = f"{bench}_pass_at_{k}_v3_open"
                    base_name = f"{bench}_pass_at_{k}_v3"
                    if open_name in passk_dataset_values:
                        preferred = open_name
                        break
                    if base_name in passk_dataset_values:
                        preferred = base_name
                        break
                passk_available_ds.append(preferred or sorted(candidates)[0])
        passk_methods = _select_existing_methods(methods_req, set(passk_jbl_df["method"]))
        if passk_available_ds and passk_methods:
            outputs.append(
                _plot_true_vs_pred_grid(
                    jbl_df=passk_jbl_df,
                    dataset_list=passk_available_ds,
                    methods=passk_methods,
                    output_path=output_dir / "true_vs_pred_grid_passk_big.pdf",
                    row_height=extra_big_row_height,
                )
            )
    display_saved_plots(outputs)
    return outputs


def build_notebook_06(
    *,
    output_dir: Path | None = None,
    binary_datasets: Iterable[str] = BINARY_DATASETS_DEFAULT,
    continuous_datasets: Iterable[str] = CONTINUOUS_DATASETS_DEFAULT,
    passk_benchmarks: Iterable[str] = PASSK_BENCHMARKS_DEFAULT,
    fixed_nmodels_binary: str = "30",
    fixed_nmodels_continuous: str = "32",
    fixed_nmodels_passk: str = "15",
    fixed_coreset: str = "10%",
    ablation_1a_methods: Iterable[str | tuple[str, ...]] | None = None,
    ablation_2_mik_values: Iterable[int] = (3, 4, 5, 6, 7, 8, 9),
    ablation_3_methods: Iterable[str | tuple[str, ...]] | None = None,
    ablation_4_binary_methods: Iterable[str | tuple[str, ...]] | None = None,
    ablation_4_cont_methods: Iterable[str | tuple[str, ...]] | None = None,
    ablation_5_binary_methods: Iterable[str | tuple[str, ...]] | None = None,
    ablation_5_cont_methods: Iterable[str | tuple[str, ...]] | None = None,
    enabled_ablations: Iterable[str] | None = None,
    save_tables_latex: bool = True,
    table_overwrite: bool = False,
) -> list[Path]:
    setup_matplotlib()
    ensure_dirs()
    output_dir = output_dir or _ensure_plot_subdir("06_ablations")
    outputs: list[Path] = []
    enabled = {str(a).lower() for a in (enabled_ablations or ("1a", "1b", "2", "3", "4", "5"))}

    summary_binary = summarize_split_standard(BINARY_SPLIT_METHOD, datasets=binary_datasets)
    summary_cont = summarize_split_standard(CONTINUOUS_SPLIT_METHOD, datasets=continuous_datasets)
    summary_p1 = summarize_split_passk(CONTINUOUS_SPLIT_METHOD, benchmarks=passk_benchmarks, source_mode="k1")
    summary_po = summarize_split_passk(CONTINUOUS_SPLIT_METHOD, benchmarks=passk_benchmarks, source_mode="opt")
    ablation_table_methods: dict[tuple[str, str], list[str]] = {}

    def _record_ablation_methods(ablation_name: str, setting_name: str, methods: Iterable[str]) -> None:
        key = (ablation_name, setting_name)
        existing = ablation_table_methods.get(key, [])
        merged = list(dict.fromkeys([*existing, *list(methods)]))
        ablation_table_methods[key] = merged

    def _ablation_methods(base: str) -> list[str]:
        return _select_existing_methods(
            [
                base,
                f"k{base}",
                f"k3{base}",
                f"k4{base}",
            ],
            set(summary_binary["method"]) | set(summary_cont["method"]) | set(summary_p1["method"]),
        )

    def _legend_layout_bottom(rows: int, *, include_bottom_row: bool) -> float:
        if include_bottom_row:
            return min(0.10 + 0.045 * rows, 0.24)
        return min(0.07 + 0.04 * rows, 0.18)

    def _hybrid_degree_method_legend_handles(
        methods_for_legend: list[str],
        *,
        degree_values: tuple[int, ...] = (1, 2),
        degree_label_prefix: str = "d",
        method_cols: int = 4,
        compact_anchor: bool = True,
    ) -> tuple[list[Line2D], int]:
        # Build unique method handles by degree-stripped name, so degree lives in linestyle legend only.
        label_to_handle: dict[str, Line2D] = {}
        for method in methods_for_legend:
            label = pretty_method_name(method, compact_anchor=compact_anchor)
            label = re.sub(r"\s*\(d=\d+\)$", "", label)
            if label in label_to_handle:
                continue
            style = method_style(method)
            label_to_handle[label] = Line2D(
                [0],
                [0],
                color=style.color,
                marker=style.marker,
                linestyle="-",
                label=label,
            )
        method_handles = list(label_to_handle.values())
        degree_handles = [
            Line2D(
                [0],
                [0],
                color="#555555",
                marker="None",
                linestyle=LINESTYLE_BY_DEGREE.get(d, "-"),
                label=f"{degree_label_prefix}={d}",
            )
            for d in degree_values
        ]
        blank = Line2D([0], [0], color="white", marker="None", linestyle="None", label="")
        ncol = method_cols + len(degree_handles)
        method_rows = max(1, math.ceil(len(method_handles) / method_cols))
        handles: list[Line2D] = []
        for row in range(method_rows):
            chunk = method_handles[row * method_cols : (row + 1) * method_cols]
            handles.extend(chunk)
            if len(chunk) < method_cols:
                handles.extend([blank] * (method_cols - len(chunk)))
            if row == 0:
                handles.extend(degree_handles)
            else:
                handles.extend([blank] * len(degree_handles))
        return handles, ncol

    if "1a" in enabled:
        methods_1a_req = list(ablation_1a_methods) if ablation_1a_methods is not None else [
                "random_sampling",
                "random_sampling_and_learn",
                "krandom_sampling_and_learn",
                "k3random_sampling_and_learn",
                "k4random_sampling_and_learn",
                "random_search_and_learn",
                "krandom_search_and_learn",
                "k3random_search_and_learn",
                "k4random_search_and_learn",
                ("mrmr5_MIQ_y", "mrmr_MIQ_y"),
                ("kmrmr5_MIQ_y", "kmrmr_MIQ_y"),
                ("k3mrmr5_MIQ_y", "k3mrmr_MIQ_y"),
                ("k4mrmr5_MIQ_y", "k4mrmr_MIQ_y"),
            ]
        methods_1a = _select_existing_methods(
            methods_1a_req,
            set(summary_binary["method"]),
        )
        handles_1a, ncol_1a = _hybrid_degree_method_legend_handles(methods_1a, degree_values=(1, 2))
        outputs.append(
            _plot_combined_grid(
                summary_df=summary_binary,
                methods=methods_1a,
                fixed_nmodels=fixed_nmodels_binary,
                fixed_coreset=fixed_coreset,
                title="Ablation 1A (binary): ridge degree",
                output_path=output_dir / "ablation_1a_binary_with_random.pdf",
                include_bottom_row=True,
                method_legend_solid_only=True,
                legend_handles_override=handles_1a,
                legend_ncol=ncol_1a,
            )
        )
        methods_1a_no_random = [m for m in methods_1a if m != "random_sampling"]
        _record_ablation_methods("1a", "binary", methods_1a)
        handles_1a_nr, ncol_1a_nr = _hybrid_degree_method_legend_handles(methods_1a_no_random, degree_values=(1, 2))
        outputs.append(
            _plot_combined_grid(
                summary_df=summary_binary,
                methods=methods_1a_no_random,
                fixed_nmodels=fixed_nmodels_binary,
                fixed_coreset=fixed_coreset,
                title="Ablation 1A (binary): ridge degree (no random)",
                output_path=output_dir / "ablation_1a_binary_no_random.pdf",
                include_bottom_row=True,
                method_legend_solid_only=True,
                legend_handles_override=handles_1a_nr,
                legend_ncol=ncol_1a_nr,
            )
        )

    def _degree_variant(base_method: str, degree: int) -> str:
        if degree == 1:
            return base_method
        if degree == 2:
            return f"k{base_method}"
        return f"k{degree}{base_method}"

    def _plot_1b_setting(
        *,
        summary: pd.DataFrame,
        setting_name: str,
        fixed_nm: str,
        include_plus_extras: bool,
    ) -> tuple[Path, list[str]]:
        fixed_nm_local = nearest_available_label(summary, "num_train_models", fixed_nm, parse_num_train_models)
        fixed_cs_local = nearest_available_label(summary, "coreset_size", fixed_coreset, parse_coreset_size)
        frame = summary[
            (summary["num_train_models"] == str(fixed_nm_local)) & (summary["coreset_size"] == str(fixed_cs_local))
        ].copy()
        available = set(frame["method"])
        plotted_methods: set[str] = set()

        mrmr_base = "mrmr5_MIQ_y" if setting_name == "binary" else "mrmr_PMIQ_y"
        family_specs: list[tuple[str, tuple[str, ...], str]] = [
            ("mRMR", (mrmr_base,), "#e91e63"),
            ("Random+", ("random_sampling_and_learn",), "#757575"),
            ("Search+", ("random_search_and_learn",), "#111111"),
        ]
        if include_plus_extras:
            family_specs.extend(
                [
                    ("AnchorPoints+", ("anchor_points_weighted+",), "#6d4c41"),
                    (
                        "gp-IRT+",
                        ("gpirt1+", "gpirt5+") if setting_name == "binary" else ("B_gpirt5+", "LEGO_gpirt5+"),
                        "#1b5e20",
                    ),
                ]
            )

        def _resolve_degree_method(base_candidates: tuple[str, ...], degree: int) -> str | None:
            candidates: list[str] = []
            for base in base_candidates:
                dv = _degree_variant(base, degree)
                candidates.extend([dv, dv.replace("5_", "_")])
            return resolve_method_candidates(candidates, available)

        fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_ONE_ROW_IN), constrained_layout=False)
        for ax, metric in zip(axes, ["rmse", "mae", "tau", "rho"]):
            for label, base_candidates, color in family_specs:
                deg_x: list[int] = []
                deg_y: list[float] = []
                for degree in [1, 2, 3, 4]:
                    selected = _resolve_degree_method(base_candidates, degree)
                    if selected is None:
                        continue
                    plotted_methods.add(selected)
                    yv = frame[frame["method"] == selected][metric].mean()
                    if pd.isna(yv):
                        continue
                    deg_x.append(degree)
                    deg_y.append(float(yv))
                linestyle = "-"
                if not deg_x:
                    continue
                ax.plot(
                    deg_x,
                    deg_y,
                    marker="o",
                    markersize=LINE_PLOT_MARKER_SIZE,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.0,
                    label=label,
                )
            ax.set_xlabel("d")
            ax.set_xticks([1, 2, 3, 4])
            ax.set_title(metric_axis_title(metric))
        family_handles = []
        for label, base_candidates, color in family_specs:
            selected = resolve_method_candidates(list(base_candidates), available)
            marker = method_style(selected).marker if selected is not None else "o"
            family_handles.append(
                Line2D([0], [0], color=color, marker=marker, markersize=LINE_PLOT_MARKER_SIZE, linestyle="-", label=label)
            )
        legend_rows = max(1, math.ceil(len(family_handles) / len(family_handles)))
        fig.tight_layout(rect=(0.0, _legend_layout_bottom(legend_rows, include_bottom_row=False), 1.0, 1.0), h_pad=0.5)
        legend = fig.legend(
            handles=family_handles,
            loc="lower center",
            ncol=len(family_handles),
            bbox_to_anchor=(0.5, 0.02),
            frameon=True,
            fancybox=False,
        )
        _style_legend_box(legend)
        suffix = "_plus" if include_plus_extras else ""
        out = save_figure(fig, output_dir, f"ablation_1b_{setting_name}_metric_vs_degree{suffix}.pdf")
        plt.close(fig)
        return out, sorted(plotted_methods)

    if "1b" in enabled:
        for setting_name, summary, fixed_nm in [
            ("binary", summary_binary, fixed_nmodels_binary),
            ("continuous", summary_cont, fixed_nmodels_continuous),
            ("passk_k1", summary_p1, fixed_nmodels_passk),
            ("passk_kopt", summary_po, fixed_nmodels_passk),
        ]:
            out_plain, methods_plain = _plot_1b_setting(
                summary=summary,
                setting_name=setting_name,
                fixed_nm=fixed_nm,
                include_plus_extras=False,
            )
            outputs.append(out_plain)
            out_plus, methods_plus = _plot_1b_setting(
                summary=summary,
                setting_name=setting_name,
                fixed_nm=fixed_nm,
                include_plus_extras=True,
            )
            outputs.append(out_plus)
            _record_ablation_methods("1b", setting_name, [*methods_plain, *methods_plus])

    if "2" in enabled:
        def _ablation2_method_candidates(degree: int, objective: str, k: int) -> list[str]:
            if objective in {"FCD2", "FCQ2"}:
                prefix = {1: "", 2: "k", 3: "k3", 4: "k4"}.get(degree, "")
                if prefix is None:
                    return []
                return [f"{prefix}mrmr_{objective}_y"]
            if objective in {"MI", "PMI"} and int(k) == 3:
                # For relevance-only MI/PMI, k=3 is the implicit base name.
                implicit_k3 = mrmr_method_name(degree=degree, mi_k=3, objective=objective, target="y")
                explicit_k3 = implicit_k3.replace("mrmr_", "mrmr3_")
                return [implicit_k3, explicit_k3]
            method = mrmr_method_name(degree=degree, mi_k=k, objective=objective, target="y")
            return [method]

        def _ablation2_methods_for_objectives(objectives: Iterable[str]) -> list[str]:
            methods: list[str] = []
            for objective in objectives:
                k_values = [3] if objective in {"FCQ", "FCD", "FCQ2", "FCD2"} else list(ablation_2_mik_values)
                for degree in [1, 2]:
                    for k in k_values:
                        methods.extend(_ablation2_method_candidates(degree, objective, k))
            return sorted(set(methods))

        def _plot_ablation2_setting(
            *,
            setting_name: str,
            summary: pd.DataFrame,
            fixed_nm_raw: str,
            objectives: list[str],
            output_suffix: str,
        ) -> None:
            fixed_nm = nearest_available_label(summary, "num_train_models", fixed_nm_raw, parse_num_train_models)
            fixed_cs = nearest_available_label(summary, "coreset_size", fixed_coreset, parse_coreset_size)
            fixed_frame = summary[
                (summary["num_train_models"] == str(fixed_nm)) & (summary["coreset_size"] == str(fixed_cs))
            ].copy()
            ablation2_methods: set[str] = set()
            plotted_objectives: set[str] = set()
            fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_ONE_ROW_IN), constrained_layout=False)
            for ax, metric in zip(axes, ["rmse", "mae", "tau", "rho"]):
                for objective in objectives:
                    if setting_name != "binary" and objective in {"MIQ", "MID"}:
                        base_color = OBJECTIVE_COLORS.get("KSG" + objective, "#455a64")
                    else:
                        base_color = OBJECTIVE_COLORS.get(
                            objective,
                            OBJECTIVE_COLORS.get(objective.replace("P", "").replace("2", ""), "#455a64"),
                        )
                    for degree in [1, 2]:
                        style = LINESTYLE_BY_DEGREE.get(degree, "-")
                        k_values = [3] if objective in {"FCQ", "FCD", "FCQ2", "FCD2"} else list(ablation_2_mik_values)
                        points_x = []
                        points_y = []
                        for k in k_values:
                            method_sel = resolve_method_candidates(
                                _ablation2_method_candidates(degree=degree, objective=objective, k=k),
                                set(fixed_frame["method"]),
                            )
                            if method_sel is None:
                                continue
                            ablation2_methods.add(method_sel)
                            value = fixed_frame[fixed_frame["method"] == method_sel][metric].mean()
                            points_x.append(k)
                            points_y.append(value)
                        if not points_x:
                            continue
                        plotted_objectives.add(objective)
                        label = f"{objective} (d={degree})"
                        if objective in {"FCQ", "FCD", "FCQ2", "FCD2"} and len(points_x) == 1:
                            ax.hlines(
                                points_y[0],
                                3,
                                9,
                                color=base_color,
                                linestyle=style,
                                linewidth=1.0,
                                label=label,
                            )
                        else:
                            ax.plot(
                                points_x,
                                points_y,
                                marker="o",
                                markersize=LINE_PLOT_MARKER_SIZE,
                                color=base_color,
                                linestyle=style,
                                linewidth=1.0,
                                label=label,
                            )
                ax.set_xlabel("k")
                ax.set_xticks(sorted(set([3, *list(ablation_2_mik_values)])))
                ax.set_title(metric_axis_title(metric))
            plotted_objective_list = [o for o in objectives if o in plotted_objectives]
            plotted_objective_list = [o for o in plotted_objective_list if o not in {"MI", "PMI"}] + [
                o for o in plotted_objective_list if o in {"MI", "PMI"}
            ]
            method_handles = []
            for obj in plotted_objective_list:
                rep_obj_k = 3 if obj in {"FCQ", "FCD", "FCQ2", "FCD2"} else 5
                rep_sel = resolve_method_candidates(
                    _ablation2_method_candidates(degree=1, objective=obj, k=rep_obj_k),
                    set(fixed_frame["method"]),
                )
                marker = method_style(rep_sel).marker if rep_sel is not None else "o"
                if setting_name != "binary" and obj in {"MIQ", "MID"}:
                    color = OBJECTIVE_COLORS.get("KSG" + obj, "#455a64")
                else:
                    color = OBJECTIVE_COLORS.get(
                        obj,
                        OBJECTIVE_COLORS.get(obj.replace("P", "").replace("2", ""), "#455a64"),
                    )

                if obj in {"MI", "PMI"}:
                    label = "Rel. Only"
                elif setting_name != "binary" and obj not in {"FCQ", "FCD", "FCQ2", "FCD2"}:
                    if obj in {"PMIQ", "PMID"}:
                        label = obj.replace("P", "") + " (PCA)"
                    else:
                        label = obj + " (KSG)"
                elif obj in {"FCQ2", "FCD2"}:
                    label = obj.replace("2", "")
                else:
                    label = obj.replace("P", "")

                method_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        marker=marker,
                        markersize=LINE_PLOT_MARKER_SIZE,
                        linestyle="-",
                        label=label,
                    )
                )
            degree_handles = [
                Line2D([0], [0], color="#555555", marker="None", linestyle=LINESTYLE_BY_DEGREE.get(1, "-"), label="d=1"),
                Line2D([0], [0], color="#555555", marker="None", linestyle=LINESTYLE_BY_DEGREE.get(2, "-"), label="d=2"),
            ]
            blank = Line2D([0], [0], color="white", marker="None", linestyle="None", label="")
            legend_handles = degree_handles + method_handles
            legend_ncol = 4 if setting_name == "binary" else 5
            target_slots = 2 * legend_ncol
            if len(legend_handles) < target_slots:
                legend_handles.extend([blank] * (target_slots - len(legend_handles)))
            legend_rows = 2
            fig.tight_layout(rect=(0.0, _legend_layout_bottom(legend_rows, include_bottom_row=False), 1.0, 1.0), h_pad=0.5)
            legend = fig.legend(
                legend_handles,
                [h.get_label() for h in legend_handles],
                loc="lower center",
                ncol=legend_ncol,
                bbox_to_anchor=(0.5, 0.02),
                frameon=True,
                fancybox=False,
            )
            _style_legend_box(legend)
            outputs.append(save_figure(fig, output_dir, f"ablation_2_mik_{output_suffix}.pdf"))
            plt.close(fig)
            _record_ablation_methods("2", output_suffix, sorted(ablation2_methods))

        ablation2_objectives_by_setting = {
            "binary": ["MI", "MIQ", "MID", "FCQ", "FCD"],
            "continuous": ["PMI", "PMIQ", "PMID", "MIQ", "MID", "FCQ", "FCD"],
            "passk_k1": ["PMI", "PMIQ", "PMID", "MIQ", "MID", "FCQ", "FCD"],
            "passk_kopt": ["PMI", "PMIQ", "PMID", "MIQ", "MID", "FCQ", "FCD"],
        }
        summary_binary_ab2 = summarize_split_standard(
            BINARY_SPLIT_METHOD,
            datasets=binary_datasets,
            methods=_ablation2_methods_for_objectives(ablation2_objectives_by_setting["binary"]),
            force_recompute=True,
        )
        summary_cont_ab2 = summarize_split_standard(
            CONTINUOUS_SPLIT_METHOD,
            datasets=continuous_datasets,
            methods=_ablation2_methods_for_objectives(ablation2_objectives_by_setting["continuous"]),
            force_recompute=True,
        )
        passk_ab2_methods = _ablation2_methods_for_objectives(ablation2_objectives_by_setting["passk_k1"])
        summary_p1_ab2 = summarize_split_passk(
            CONTINUOUS_SPLIT_METHOD,
            benchmarks=passk_benchmarks,
            methods=passk_ab2_methods,
            source_mode="k1",
            force_recompute=True,
        )
        summary_po_ab2 = summarize_split_passk(
            CONTINUOUS_SPLIT_METHOD,
            benchmarks=passk_benchmarks,
            methods=passk_ab2_methods,
            source_mode="opt",
            force_recompute=True,
        )

        for setting_name, summary, fixed_nm in [
            ("binary", summary_binary_ab2, fixed_nmodels_binary),
            ("continuous", summary_cont_ab2, fixed_nmodels_continuous),
            ("passk_k1", summary_p1_ab2, fixed_nmodels_passk),
            ("passk_kopt", summary_po_ab2, fixed_nmodels_passk),
        ]:
            objectives = ablation2_objectives_by_setting[setting_name]
            if summary.empty:
                continue
            _plot_ablation2_setting(
                setting_name=setting_name,
                summary=summary,
                fixed_nm_raw=fixed_nm,
                objectives=objectives,
                output_suffix=setting_name,
            )

        # Extra Ablation-2 figure for continuous_cat_main using FCD2/FCQ2
        # in place of FCD/FCQ.
        fc2_objectives = ["PMI", "PMIQ", "PMID", "MIQ", "MID", "FCQ2", "FCD2"]
        fc2_methods = []
        for _obj in fc2_objectives:
            for _deg in [1, 2]:
                _ks = [3] if _obj in {"FCQ2", "FCD2"} else list(ablation_2_mik_values)
                for _k in _ks:
                    fc2_methods.extend(_ablation2_method_candidates(_deg, _obj, _k))
        fc2_methods = sorted(set(fc2_methods))
        if "dataset" in summary_cont.columns:
            cont_main_set = set(continuous_cat_main_datasets)
            summary_cont_main = summary_cont[summary_cont["dataset"].isin(cont_main_set)].copy()
        else:
            # Some cached/precomputed summaries are already dataset-aggregated and
            # omit per-dataset columns; in that case, build a dedicated summary.
            summary_cont_main = summarize_split_standard(
                CONTINUOUS_SPLIT_METHOD,
                datasets=continuous_cat_main_datasets,
                methods=fc2_methods,
                force_recompute=True,
            )
        if not summary_cont_main.empty:
            _plot_ablation2_setting(
                setting_name="continuous",
                summary=summary_cont_main,
                fixed_nm_raw=fixed_nmodels_continuous,
                objectives=fc2_objectives,
                output_suffix="continuous_cat_main_fc2",
            )

        # Extra Ablation-2 figures for pass@k (k1 + k-opt) using FCD2/FCQ2 instead of FCD/FCQ.
        passk_benchmarks_list = list(passk_benchmarks)
        for src_mode, sk, out_suffix in (
            ("k1", "passk_k1", "passk_k1_fc2"),
            ("opt", "passk_kopt", "passk_kopt_fc2"),
        ):
            summary_passk_fc2 = summarize_split_passk(
                CONTINUOUS_SPLIT_METHOD,
                benchmarks=passk_benchmarks_list,
                methods=fc2_methods,
                source_mode=src_mode,
                force_recompute=True,
            )
            if not summary_passk_fc2.empty:
                _plot_ablation2_setting(
                    setting_name=sk,
                    summary=summary_passk_fc2,
                    fixed_nm_raw=fixed_nmodels_passk,
                    objectives=fc2_objectives,
                    output_suffix=out_suffix,
                )

    if "3" in enabled:
        target_pred_req = list(ablation_3_methods) if ablation_3_methods is not None else [
                ("mrmr5_MIQ_y", "mrmr_MIQ_y"),
                ("mrmr5_MIQ_PC1", "mrmr_MIQ_PC1"),
                ("kmrmr5_MIQ_y", "kmrmr_MIQ_y"),
                ("kmrmr5_MIQ_PC1", "kmrmr_MIQ_PC1"),
                ("rfmrmr5_MIQ_y", "rfmrmr_MIQ_y"),
                ("rfmrmr5_MIQ_PC1", "rfmrmr_MIQ_PC1"),
                ("lmrmr5_MIQ_y", "lmrmr_MIQ_y"),
                ("lmrmr5_MIQ_PC1", "lmrmr_MIQ_PC1"),
            ]
        target_pred_methods = _select_existing_methods(
            target_pred_req,
            set(summary_binary["method"]),
        )
        _record_ablation_methods("3", "binary", target_pred_methods)
        tp_fixed_nm = nearest_available_label(summary_binary, "num_train_models", fixed_nmodels_binary, parse_num_train_models)
        tp_fixed_cs = nearest_available_label(summary_binary, "coreset_size", fixed_coreset, parse_coreset_size)
        tp_combo_colors = {
            ("ridge", "y"): "#e91e63",
            ("ridge", "PC1"): "#f57c00",
            ("RF", "y"): "#7b1fa2",
            ("RF", "PC1"): "#d8aaff",
            ("logit ridge", "y"): "#1565c0",
            ("logit ridge", "PC1"): "#00838f",
        }
        fig, axes = plt.subplots(2, 4, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_TWO_ROW_IN), constrained_layout=False)
        for col_idx, metric in enumerate(["rmse", "mae", "tau", "rho"]):
            top_frame = _prepare_curve_frame(
                summary_df=summary_binary,
                metric_key=metric,
                fixed_col="num_train_models",
                fixed_value=tp_fixed_nm,
                x_col="coreset_size",
            )
            bot_frame = _prepare_curve_frame(
                summary_df=summary_binary,
                metric_key=metric,
                fixed_col="coreset_size",
                fixed_value=tp_fixed_cs,
                x_col="num_train_models",
            )
            ax_top = axes[0, col_idx]
            ax_bot = axes[1, col_idx]
            for method in target_pred_methods:
                parsed = parse_mrmr_method(method)
                if parsed is None:
                    continue
                color = tp_combo_colors.get((parsed["predictor"], parsed["target"]), "#455a64")
                linestyle = LINESTYLE_BY_DEGREE.get(parsed["degree"], "-")
                tsub = top_frame[top_frame["method"] == method].sort_values("x_value")
                bsub = bot_frame[bot_frame["method"] == method].sort_values("x_value")
                if not tsub.empty:
                    ax_top.plot(
                        tsub["x_value"],
                        tsub[metric],
                        color=color,
                        linestyle=linestyle,
                        marker="o",
                        markersize=LINE_PLOT_MARKER_SIZE,
                        linewidth=1.0,
                    )
                if not bsub.empty:
                    ax_bot.plot(
                        bsub["x_value"],
                        bsub[metric],
                        color=color,
                        linestyle=linestyle,
                        marker="o",
                        markersize=LINE_PLOT_MARKER_SIZE,
                        linewidth=1.0,
                    )
            ax_top.set_xlabel("Coreset Size (%)")
            ax_bot.set_xlabel("Num Source Models")
            ax_top.set_title(metric_axis_title(metric))
            if col_idx == 0:
                ax_top.set_ylabel(METRIC_SPECS[metric]["label"])
                _subplot_row_label(ax_top, "(a)")
                ax_bot.set_ylabel(METRIC_SPECS[metric]["label"])
                _subplot_row_label(ax_bot, "(b)")
        combo_handles = []
        seen_combo: set[tuple[str, str]] = set()
        for method in target_pred_methods:
            parsed = parse_mrmr_method(method)
            if parsed is None:
                continue
            combo = (parsed["predictor"], parsed["target"])
            if combo in seen_combo:
                continue
            seen_combo.add(combo)
            combo_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=tp_combo_colors.get(combo, "#455a64"),
                    linestyle="-",
                    marker="o",
                    markersize=LINE_PLOT_MARKER_SIZE,
                    label=f"{combo[0]}{', PC1 rel.' if combo[1] == 'PC1' else ''}",
                )
            )
        combo_handles.extend(
            [
                Line2D([0], [0], color="#555555", linestyle=LINESTYLE_BY_DEGREE[1], label="d=1"),
                Line2D([0], [0], color="#555555", linestyle=LINESTYLE_BY_DEGREE[2], label="d=2"),
            ]
        )
        legend_rows = max(1, math.ceil(len(combo_handles) / 4))
        fig.tight_layout(rect=(0.0, max(0.12, _legend_layout_bottom(legend_rows, include_bottom_row=True) - 0.03), 1.0, 1.0), h_pad=0.5)
        legend = fig.legend(handles=combo_handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.02), frameon=True, fancybox=False)
        _style_legend_box(legend)
        outputs.append(save_figure(fig, output_dir, "ablation_3_target_predictor_binary.pdf"))
        plt.close(fig)

    if "4" in enabled:
        def _ablation4_mixed_legend_handles(methods_for_legend: list[str]) -> list[Line2D] | None:
            rep: dict[tuple[str, str], str] = {}
            for m in methods_for_legend:
                if infer_method_family(m) != "irt":
                    continue
                rep.setdefault(_irt_prefix_variant_from_method(m), m)
            base_order = ["LEGO", "G", "B", "B3_v2"]
            if any((base, "p") not in rep for base in base_order):
                return None
            if any((base, "gp") not in rep for base in base_order):
                return None
            handles: list[Line2D] = []
            for base in base_order:
                gm = rep[(base, "gp")]
                gs = method_style(gm)
                handles.append(
                    Line2D([0], [0], color=gs.color, marker=gs.marker, linestyle="-", label=pretty_method_name(gm, compact_anchor=True))
                )
                pm = rep[(base, "p")]
                ps = method_style(pm)
                handles.append(
                    Line2D([0], [0], color=ps.color, marker=ps.marker, linestyle="-", label=pretty_method_name(pm, compact_anchor=True))
                )

            handles.append(Line2D([0], [0], color="#555555", marker="None", linestyle=LINESTYLE_BY_IRT_DIM[1], label="p=1"))
            handles.append(Line2D([0], [0], color="#555555", marker="None", linestyle=LINESTYLE_BY_IRT_DIM[5], label="p=5"))
            handles.append(Line2D([0], [0], color="#555555", marker="None", linestyle=LINESTYLE_BY_IRT_DIM[10], label="p=10"))
            handles.append(Line2D([0], [0], color="white", marker="None", linestyle="None", label=""))
            return handles

        irt_binary_req = list(ablation_4_binary_methods) if ablation_4_binary_methods is not None else [
            ("gpirt", "gpirt10"),
            ("gpirt5",),
            ("gpirt1",),
            ("pirt", "pirt10"),
            ("pirt5",),
            ("pirt1",),
        ]
        irt_binary_methods = _select_existing_methods(
            irt_binary_req,
            set(summary_binary["method"]),
        )
        _record_ablation_methods("4", "binary", irt_binary_methods)
        irt_cont_req = list(ablation_4_cont_methods) if ablation_4_cont_methods is not None else [
            ("LEGO_gpirt", "LEGO_gpirt10"),
            ("LEGO_gpirt5",),
            ("LEGO_gpirt1",),
            ("LEGO_pirt", "LEGO_pirt10"),
            ("LEGO_pirt5",),
            ("LEGO_pirt1",),
            ("B_gpirt", "B_gpirt10"),
            ("B_gpirt5",),
            ("B_gpirt1",),
            ("B_pirt", "B_pirt10"),
            ("B_pirt5",),
            ("B_pirt1",),
            ("G_gpirt", "G_gpirt1"),
            ("G_pirt", "G_pirt1"),
            ("B3_v2_gpirt", "B3_v2_gpirt1"),
            ("B3_v2_pirt", "B3_v2_pirt1"),
            ]
        irt_cont_methods = _select_existing_methods(
            irt_cont_req,
            set(summary_cont["method"]),
        )
        _record_ablation_methods("4", "continuous", irt_cont_methods)
        outputs.append(
            _plot_combined_grid(
            summary_df=summary_binary,
            methods=irt_binary_methods,
            fixed_nmodels=fixed_nmodels_binary,
            fixed_coreset=fixed_coreset,
            title="Ablation 4: IRT dimension (binary)",
            output_path=output_dir / "ablation_4_irt_binary.pdf",
            include_bottom_row=True,
            use_irt_dim_linestyle=True,
            method_legend_solid_only=True,
            collapse_irt_variant_legend=True,
            append_degree_linestyle_legend=True,
            degree_values=(1, 5, 10),
            degree_label_prefix="p",
            legend_ncol=6,
            legend_handles_override=_ablation4_mixed_legend_handles(irt_binary_methods),
            )
        )
        irt_binary_gp_only_methods = [m for m in irt_binary_methods if "gpirt" in m and "pirt" not in m.replace("gpirt", "")]
        if irt_binary_gp_only_methods:
            outputs.append(
                _plot_combined_grid(
                summary_df=summary_binary,
                methods=irt_binary_gp_only_methods,
                fixed_nmodels=fixed_nmodels_binary,
                fixed_coreset=fixed_coreset,
                title="Ablation 4: IRT dimension (binary, gp-IRT only)",
                output_path=output_dir / "ablation_4_irt_binary_gpirt_only.pdf",
                include_bottom_row=True,
                use_irt_dim_linestyle=True,
                method_legend_solid_only=True,
                collapse_irt_variant_legend=True,
                append_degree_linestyle_legend=True,
                degree_values=(1, 5, 10),
                degree_label_prefix="p",
                legend_single_row=True,
                )
            )
        outputs.append(
            _plot_combined_grid(
            summary_df=summary_cont,
            methods=irt_cont_methods,
            fixed_nmodels=fixed_nmodels_continuous,
            fixed_coreset=fixed_coreset,
            title="Ablation 4: IRT variants (continuous)",
            output_path=output_dir / "ablation_4_irt_continuous.pdf",
            include_bottom_row=True,
            use_irt_dim_linestyle=True,
            method_legend_solid_only=True,
            collapse_irt_variant_legend=True,
            append_degree_linestyle_legend=True,
            degree_values=(1, 5, 10),
            degree_label_prefix="p",
            legend_ncol=6,
            legend_handles_override=_ablation4_mixed_legend_handles(irt_cont_methods),
            )
        )
        irt_cont_gp_only_methods = [m for m in irt_cont_methods if "gpirt" in m and "pirt" not in m.replace("gpirt", "")]
        if irt_cont_gp_only_methods:
            outputs.append(
                _plot_combined_grid(
                summary_df=summary_cont,
                methods=irt_cont_gp_only_methods,
                fixed_nmodels=fixed_nmodels_continuous,
                fixed_coreset=fixed_coreset,
                title="Ablation 4: IRT variants (continuous, gp-IRT only)",
                output_path=output_dir / "ablation_4_irt_continuous_gpirt_only.pdf",
                include_bottom_row=True,
                use_irt_dim_linestyle=True,
                method_legend_solid_only=True,
                collapse_irt_variant_legend=True,
                append_degree_linestyle_legend=True,
                degree_values=(1, 5, 10),
                degree_label_prefix="p",
                legend_single_row=True,
                )
            )
        irt_passk_k1_methods = _select_existing_methods(
            irt_cont_req,
            set(summary_p1["method"]),
        )
        _record_ablation_methods("4", "passk_k1", irt_passk_k1_methods)
        if irt_passk_k1_methods:
            outputs.append(
                _plot_combined_grid(
                summary_df=summary_p1,
                methods=irt_passk_k1_methods,
                fixed_nmodels=fixed_nmodels_passk,
                fixed_coreset=fixed_coreset,
                title="Ablation 4: IRT variants (pass@k k=1)",
                output_path=output_dir / "ablation_4_irt_passk_k1.pdf",
                include_bottom_row=True,
                use_irt_dim_linestyle=True,
                method_legend_solid_only=True,
                collapse_irt_variant_legend=True,
                append_degree_linestyle_legend=True,
                degree_values=(1, 5, 10),
                degree_label_prefix="p",
                legend_ncol=6,
                legend_handles_override=_ablation4_mixed_legend_handles(irt_passk_k1_methods),
                )
            )
            irt_passk_k1_gp_only_methods = [m for m in irt_passk_k1_methods if "gpirt" in m and "pirt" not in m.replace("gpirt", "")]
            if irt_passk_k1_gp_only_methods:
                outputs.append(
                    _plot_combined_grid(
                    summary_df=summary_p1,
                    methods=irt_passk_k1_gp_only_methods,
                    fixed_nmodels=fixed_nmodels_passk,
                    fixed_coreset=fixed_coreset,
                    title="Ablation 4: IRT variants (pass@k k=1, gp-IRT only)",
                    output_path=output_dir / "ablation_4_irt_passk_k1_gpirt_only.pdf",
                    include_bottom_row=True,
                    use_irt_dim_linestyle=True,
                    method_legend_solid_only=True,
                    collapse_irt_variant_legend=True,
                    append_degree_linestyle_legend=True,
                    degree_values=(1, 5, 10),
                    degree_label_prefix="p",
                    legend_single_row=True,
                    )
                )
        irt_passk_opt_methods = _select_existing_methods(
            irt_cont_req,
            set(summary_po["method"]),
        )
        _record_ablation_methods("4", "passk_kopt", irt_passk_opt_methods)
        if irt_passk_opt_methods:
            outputs.append(
                _plot_combined_grid(
                summary_df=summary_po,
                methods=irt_passk_opt_methods,
                fixed_nmodels=fixed_nmodels_passk,
                fixed_coreset=fixed_coreset,
                title="Ablation 4: IRT variants (pass@k k=opt)",
                output_path=output_dir / "ablation_4_irt_passk_kopt.pdf",
                include_bottom_row=True,
                use_irt_dim_linestyle=True,
                method_legend_solid_only=True,
                collapse_irt_variant_legend=True,
                append_degree_linestyle_legend=True,
                degree_values=(1, 5, 10),
                degree_label_prefix="p",
                legend_ncol=6,
                legend_handles_override=_ablation4_mixed_legend_handles(irt_passk_opt_methods),
                )
            )
            irt_passk_opt_gp_only_methods = [m for m in irt_passk_opt_methods if "gpirt" in m and "pirt" not in m.replace("gpirt", "")]
            if irt_passk_opt_gp_only_methods:
                outputs.append(
                    _plot_combined_grid(
                    summary_df=summary_po,
                    methods=irt_passk_opt_gp_only_methods,
                    fixed_nmodels=fixed_nmodels_passk,
                    fixed_coreset=fixed_coreset,
                    title="Ablation 4: IRT variants (pass@k k=opt, gp-IRT only)",
                    output_path=output_dir / "ablation_4_irt_passk_kopt_gpirt_only.pdf",
                    include_bottom_row=True,
                    use_irt_dim_linestyle=True,
                    method_legend_solid_only=True,
                    collapse_irt_variant_legend=True,
                    append_degree_linestyle_legend=True,
                    degree_values=(1, 5, 10),
                    degree_label_prefix="p",
                    legend_single_row=True,
                    )
                )
    if "5" in enabled:
        ablation_5_binary_req = list(ablation_5_binary_methods) if ablation_5_binary_methods is not None else _default_main_methods_binary()
        ablation_5_cont_req = list(ablation_5_cont_methods) if ablation_5_cont_methods is not None else _default_main_methods_continuous()
        pearson_specs = [
            ("binary", summary_binary, fixed_nmodels_binary, ablation_5_binary_req, "ablation_5_pearson_binary.pdf", ((0.88, 0.99), (0.90, 0.98))),
            ("continuous", summary_cont, fixed_nmodels_continuous, ablation_5_cont_req, "ablation_5_pearson_continuous.pdf", ((0.96, 0.993), (0.965, 1.00))),
            ("passk_k1", summary_p1, fixed_nmodels_passk, ablation_5_cont_req, "ablation_5_pearson_passk_k1.pdf", ((0.972, 1.00), (0.98, 1.00))),
            ("passk_kopt", summary_po, fixed_nmodels_passk, ablation_5_cont_req, "ablation_5_pearson_passk_kopt.pdf", ((0.988, 1.00), (0.985, 1.00))),
        ]
        for _name, summary_df, fixed_nm, methods_req, filename, (ylim1, ylim2) in pearson_specs:
            if "pearson" not in summary_df.columns:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_ONE_ROW_IN), constrained_layout=True)
            frame_cs = _prepare_curve_frame(
                summary_df=summary_df,
                metric_key="pearson",
                fixed_col="num_train_models",
                fixed_value=fixed_nm,
                x_col="coreset_size",
            )
            frame_nm = _prepare_curve_frame(
                summary_df=summary_df,
                metric_key="pearson",
                fixed_col="coreset_size",
                fixed_value=fixed_coreset,
                x_col="num_train_models",
            )
            methods_sel = _select_existing_methods(methods_req, set(summary_df["method"]))
            _record_ablation_methods("5", _name, methods_sel)
            for method in methods_sel:
                m1 = frame_cs[frame_cs["method"] == method].sort_values("x_value")
                m2 = frame_nm[frame_nm["method"] == method].sort_values("x_value")
                if not m1.empty:
                    _plot_method_lines(axes[0], m1["x_value"].to_numpy(), m1["pearson"].to_numpy(), method)
                if not m2.empty:
                    _plot_method_lines(axes[1], m2["x_value"].to_numpy(), m2["pearson"].to_numpy(), method)
            axes[0].set_xlabel("Coreset Size (%)")
            axes[0].set_ylabel(r"Pearson $r$ $\uparrow$")
            axes[1].set_xlabel("Num Source Models")
            axes[1].set_ylabel(r"Pearson $r$ $\uparrow$")
            axes[0].set_ylim(*ylim1)
            axes[1].set_ylim(*ylim2)
            methods_sel = methods_sel[:3] + methods_sel[-1] + methods_sel[3] + methods_sel[6:7] + 'B_gpirt+' + methods_sel[4:6]
            legend_handles, legend_ncol = _hybrid_degree_method_legend_handles(
                methods_sel,
                degree_values=(1, 2),
                degree_label_prefix="d",
                method_cols=4,
                compact_anchor=True,
                force_d_at_end=True,
            )
            if legend_handles:
                legend_rows = max(1, math.ceil(len(legend_handles) / max(1, legend_ncol)))
                bottom_rect = min(0.10 + 0.045 * legend_rows, 0.24)
                fig.tight_layout(rect=(0.0, bottom_rect, 1.0, 1.0), h_pad=0.5)
                legend = fig.legend(
                    handles=legend_handles,
                    loc="lower center",
                    ncol=legend_ncol,
                    bbox_to_anchor=(0.5, 0.02),
                    frameon=True,
                    fancybox=False,
                )
                _style_legend_box(legend)
            else:
                fig.tight_layout()
            outputs.append(save_figure(fig, output_dir, filename))
            plt.close(fig)

    if save_tables_latex:
        table_dir = output_dir / "tables"
        summary_by_setting = {
            "binary": summary_binary,
            "continuous": summary_cont,
            "passk_k1": summary_p1,
            "passk_kopt": summary_po,
        }
        caption_by_setting = {
            "binary": "binary",
            "continuous": "continuous",
            "passk_k1": "pass@k (source k=1)",
            "passk_kopt": "pass@k (source k=opt)",
        }
        for (ablation_name, setting_name), methods in sorted(ablation_table_methods.items()):
            if not methods or setting_name not in summary_by_setting:
                continue
            outputs.append(
                _save_combined_grid_table_latex(
                    summary_by_setting[setting_name],
                    methods,
                    table_dir / f"ablation_{ablation_name}_{setting_name}.tex",
                    f"Ablation {ablation_name} table for {caption_by_setting[setting_name]}",
                    overwrite=table_overwrite,
                )
            )

    display_saved_plots(outputs)
    return outputs


def build_notebook_06_ablation_1a(**kwargs) -> list[Path]:
    return build_notebook_06(enabled_ablations=("1a",), **kwargs)


def build_notebook_06_ablation_1b(**kwargs) -> list[Path]:
    return build_notebook_06(enabled_ablations=("1b",), **kwargs)


def build_notebook_06_ablation_2(**kwargs) -> list[Path]:
    return build_notebook_06(enabled_ablations=("2",), **kwargs)


def build_notebook_06_ablation_3(**kwargs) -> list[Path]:
    return build_notebook_06(enabled_ablations=("3",), **kwargs)


def build_notebook_06_ablation_4(**kwargs) -> list[Path]:
    return build_notebook_06(enabled_ablations=("4",), **kwargs)


def build_notebook_06_ablation_5(**kwargs) -> list[Path]:
    return build_notebook_06(enabled_ablations=("5",), **kwargs)


def _dataset_method_metrics_standard(
    split_method: str,
    datasets: Iterable[str],
    *,
    force_recompute: bool = False,
    progress_desc: str | None = None,
) -> pd.DataFrame:
    cache_path = _cache_key(
        "winner_standard",
        {"split": split_method, "datasets": sorted(set(datasets))},
    )
    if cache_path.is_file() and not force_recompute:
        cached = pd.read_parquet(cache_path)
        if "mae" not in cached.columns:
            cached["mae"] = cached["error"] if "error" in cached.columns else np.nan
        if "tau" not in cached.columns:
            cached["tau"] = cached["corr_kendall"] if "corr_kendall" in cached.columns else np.nan
        if "rho" not in cached.columns:
            cached["rho"] = cached["corr_spearman"] if "corr_spearman" in cached.columns else np.nan
        return cached

    dataset_set = set(datasets)
    frames: list[pd.DataFrame] = []
    settings = discover_settings(split_method=split_method)
    settings_iter = (
        tqdm(settings, desc=progress_desc, unit="setting") if progress_desc else settings
    )
    for setting in settings_iter:
        df = load_setting_results(setting)
        if df.empty:
            continue
        sub = df[df["dataset"].isin(dataset_set)].copy()
        if sub.empty:
            continue
        grouped = (
            sub.groupby(["dataset", "method"], as_index=False)[
                ["rmse", "error", "corr_kendall", "corr_spearman"]
            ]
            .mean()
        )
        grouped["mae"] = grouped["error"]
        grouped["tau"] = grouped["corr_kendall"]
        grouped["rho"] = grouped["corr_spearman"]
        grouped["coreset_size"] = setting.coreset_size
        grouped["num_train_models"] = setting.num_train_models
        frames.append(grouped)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out.to_parquet(cache_path, index=False)
    return out


def _dataset_method_metrics_passk(
    split_method: str,
    benchmarks: Iterable[str],
    source_mode: str,
    target_ks: Iterable[int] = PASS_AT_K_TARGET_KS_DEFAULT,
    *,
    force_recompute: bool = False,
    progress_desc: str | None = None,
) -> pd.DataFrame:
    cache_path = _cache_key(
        "winner_passk",
        {
            "split": split_method,
            "benchmarks": sorted(set(benchmarks)),
            "source_mode": source_mode,
            "target_ks": sorted(set(target_ks)),
        },
    )
    if cache_path.is_file() and not force_recompute:
        cached = pd.read_parquet(cache_path)
        if "mae" not in cached.columns:
            cached["mae"] = cached["error"] if "error" in cached.columns else np.nan
        if "tau" not in cached.columns:
            cached["tau"] = cached["corr_kendall"] if "corr_kendall" in cached.columns else np.nan
        if "rho" not in cached.columns:
            cached["rho"] = cached["corr_spearman"] if "corr_spearman" in cached.columns else np.nan
        return cached

    benchmark_set = set(benchmarks)
    target_k_set = set(target_ks)
    frames: list[pd.DataFrame] = []
    settings = discover_settings(split_method=split_method)
    settings_iter = (
        tqdm(settings, desc=progress_desc, unit="setting") if progress_desc else settings
    )
    for setting in settings_iter:
        native = load_setting_results(setting)
        crossk = load_setting_crossk(setting)
        combined = build_crossk_combined(native_df=native, crossk_df=crossk)
        if combined.empty:
            continue
        sub = combined[
            combined["benchmark"].isin(benchmark_set) & combined["pred_k"].isin(target_k_set)
        ].copy()
        if sub.empty:
            continue

        metric_cols = ["rmse", "error", "corr_kendall", "corr_spearman"]
        seed_avg = (
            sub.groupby(["benchmark", "method", "source_k", "pred_k"], as_index=False)[metric_cols]
            .mean()
        )
        src_avg = (
            seed_avg.groupby(["benchmark", "method", "source_k"], as_index=False)[metric_cols]
            .mean()
        )

        if source_mode == "k1":
            selected = src_avg[src_avg["source_k"] == 1].copy()
        else:
            selector = src_avg.dropna(subset=["rmse"]).copy()
            if selector.empty:
                continue
            best_idx = selector.groupby(["benchmark", "method"])["rmse"].idxmin()
            best_source = selector.loc[best_idx, ["benchmark", "method", "source_k"]]
            selected = src_avg.merge(best_source, on=["benchmark", "method", "source_k"], how="inner")
        if selected.empty:
            continue

        selected = selected.rename(columns={"benchmark": "dataset"})
        selected["mae"] = selected["error"]
        selected["tau"] = selected["corr_kendall"]
        selected["rho"] = selected["corr_spearman"]
        selected["coreset_size"] = setting.coreset_size
        selected["num_train_models"] = setting.num_train_models
        frames.append(selected)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out.to_parquet(cache_path, index=False)
    return out


def precompute_notebook_07_cache(
    *,
    binary_datasets: Iterable[str] = BINARY_DATASETS_DEFAULT,
    continuous_datasets: Iterable[str] = CONTINUOUS_DATASETS_DEFAULT,
    passk_benchmarks: Iterable[str] = PASSK_BENCHMARKS_DEFAULT,
    include_continuous: bool = False,
    include_passk: bool = False,
    force_recompute: bool = False,
) -> dict[str, Path]:
    ensure_dirs()
    datasets_binary = list(binary_datasets)
    datasets_continuous = list(continuous_datasets)
    benchmarks_passk = list(passk_benchmarks)
    num_questions = load_num_questions()
    passk_num_questions = _passk_benchmark_num_questions(benchmarks_passk, prefer_k1=True)

    cache_paths: dict[str, Path] = {}
    tasks = [
        (
            "binary",
            lambda: _dataset_method_metrics_standard(
                BINARY_SPLIT_METHOD,
                datasets_binary,
                force_recompute=force_recompute,
                progress_desc="winner cache: binary",
            ),
            _cache_key(
                "winner_standard",
                {"split": BINARY_SPLIT_METHOD, "datasets": sorted(set(datasets_binary))},
            ),
        )
    ]
    if include_continuous:
        tasks.append(
            (
                "continuous",
                lambda: _dataset_method_metrics_standard(
                    CONTINUOUS_SPLIT_METHOD,
                    datasets_continuous,
                    force_recompute=force_recompute,
                    progress_desc="winner cache: continuous",
                ),
                _cache_key(
                    "winner_standard",
                    {"split": CONTINUOUS_SPLIT_METHOD, "datasets": sorted(set(datasets_continuous))},
                ),
            )
        )
    if include_passk:
        for source_mode in ("k1", "opt"):
            tasks.append(
                (
                    f"passk_{source_mode}",
                    lambda src_mode=source_mode: _dataset_method_metrics_passk(
                        CONTINUOUS_SPLIT_METHOD,
                        benchmarks_passk,
                        src_mode,
                        force_recompute=force_recompute,
                        progress_desc=f"winner cache: passk ({src_mode})",
                    ),
                    _cache_key(
                        "winner_passk",
                        {
                            "split": CONTINUOUS_SPLIT_METHOD,
                            "benchmarks": sorted(set(benchmarks_passk)),
                            "source_mode": source_mode,
                            "target_ks": sorted(set(PASS_AT_K_TARGET_KS_DEFAULT)),
                        },
                    ),
                )
            )

    for name, compute_fn, cache_path in tqdm(tasks, desc="plot 07 cache groups", unit="group"):
        metrics_frame = compute_fn()
        cache_paths[name] = cache_path
        winner_points, metric_points = _notebook_07_prepare_plot_points(
            frame=metrics_frame,
            setting_name=name,
            num_questions=num_questions,
            passk_num_questions=passk_num_questions,
        )
        winner_path, metric_path = _notebook_07_plot_cache_paths(name)
        if force_recompute or not winner_path.is_file():
            winner_points.to_parquet(winner_path, index=False)
        if force_recompute or not metric_path.is_file():
            metric_points.to_parquet(metric_path, index=False)
        cache_paths[f"{name}_plot_winners"] = winner_path
        cache_paths[f"{name}_plot_metric_points"] = metric_path
    return cache_paths


def _notebook_07_plot_cache_paths(setting_name: str) -> tuple[Path, Path]:
    winner_path = _cache_key(
        "winner_plot07_winners",
        {"setting_name": setting_name, "version": 1},
    )
    metric_path = _cache_key(
        "winner_plot07_metric_points",
        {"setting_name": setting_name, "version": 1},
    )
    return winner_path, metric_path


def _notebook_07_prepare_plot_points(
    *,
    frame: pd.DataFrame,
    setting_name: str,
    num_questions: dict[str, int],
    passk_num_questions: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        empty_winners = pd.DataFrame(columns=["metric", "x", "y", "method"])
        empty_points = pd.DataFrame(columns=["metric", "x", "y", "method"])
        return empty_winners, empty_points

    def _raw_n(dataset_key: str, cs_text: str) -> float:
        pct = parse_coreset_size(cs_text)
        if setting_name.startswith("passk"):
            q = passk_num_questions.get(dataset_key)
        else:
            q = num_questions.get(dataset_key)
        if q is None:
            return max(float(pct), 1.0)
        return max(float(round(q * pct / 100.0)), 1.0)

    working = frame.copy()
    working["n_raw"] = working.apply(
        lambda row: _raw_n(str(row["dataset"]), str(row["coreset_size"])),
        axis=1,
    )
    working["m_raw"] = working["num_train_models"].apply(lambda x: parse_num_train_models(str(x)))
    working["x_winner"] = working["m_raw"]

    metric_cfg = [("rmse", False), ("mae", False), ("tau", True), ("rho", True)]
    winner_frames: list[pd.DataFrame] = []
    for metric, higher_is_better in metric_cfg:
        metric_df = working.dropna(subset=[metric]).copy()
        if metric_df.empty:
            continue
        grouped = metric_df.groupby(["dataset", "coreset_size", "num_train_models"])[metric]
        idx = grouped.idxmax() if higher_is_better else grouped.idxmin()
        wins = metric_df.loc[idx, ["x_winner", "n_raw", "method"]].copy()
        wins = wins.rename(columns={"x_winner": "x", "n_raw": "y"})
        wins["metric"] = metric
        winner_frames.append(wins)
    winner_points = (
        pd.concat(winner_frames, ignore_index=True)
        if winner_frames
        else pd.DataFrame(columns=["metric", "x", "y", "method"])
    )

    scatter_frames: list[pd.DataFrame] = []
    for metric in ["rmse", "mae", "tau", "rho"]:
        metric_df = working.dropna(subset=[metric]).copy()
        if metric_df.empty:
            continue
        metric_df = metric_df[metric_df["m_raw"] > 0].copy()
        if metric_df.empty:
            continue
        metric_df["x"] = metric_df["n_raw"] / metric_df["m_raw"]
        metric_df["y"] = metric_df[metric].astype(float) * float(METRIC_SPECS[metric]["scale"])
        points = metric_df[["x", "y", "method"]].copy()
        points["metric"] = metric
        scatter_frames.append(points)
    metric_points = (
        pd.concat(scatter_frames, ignore_index=True)
        if scatter_frames
        else pd.DataFrame(columns=["metric", "x", "y", "method"])
    )
    return winner_points, metric_points


def _notebook_07_focus_method(method: str) -> str | None:
    resolved = _resolve_method_alias(str(method))
    if resolved == "kmrmr5_MIQ_y":
        return "mRMR++"
    if resolved == "gpirt1":
        return "gpirt"
    if resolved == "anchor_points_weighted":
        return "anchor_points_weighted"
    if resolved == "random_sampling":
        return "random_sampling"
    if resolved == "krandom_sampling_and_learn":
        return "Random++"
    if resolved == "lasso":
        return "lasso"
    return None


def _notebook_07_focus_style_method(focus_method: str) -> str:
    style_map = {
        "mRMR++": "kmrmr5_MIQ_y",
        "gpirt": "gpirt1",
        "anchor_points_weighted": "anchor_points_weighted",
        "random_sampling": "random_sampling",
        "Random++": "krandom_sampling_and_learn",
        "lasso": "lasso",
    }
    return style_map.get(focus_method, focus_method)


def _notebook_07_focus_winner_bins(
    *,
    frame: pd.DataFrame,
    setting_name: str,
    num_questions: dict[str, int],
    passk_num_questions: dict[str, int],
    binning_mode: str,
    fixed_bin_count: int,
    quantile_bins: int,
    focus_methods: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    focus_set = set(focus_methods)
    if frame.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    def _raw_n(dataset_key: str, cs_text: str) -> float:
        pct = parse_coreset_size(cs_text)
        if setting_name.startswith("passk"):
            q = passk_num_questions.get(dataset_key)
        else:
            q = num_questions.get(dataset_key)
        if q is None:
            return max(float(pct), 1.0)
        return max(float(round(q * pct / 100.0)), 1.0)

    working = frame.copy()
    working["focus_method"] = working["method"].apply(_notebook_07_focus_method)
    working = working[working["focus_method"].isin(focus_set)].copy()
    if working.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    working["n_raw"] = working.apply(
        lambda row: _raw_n(str(row["dataset"]), str(row["coreset_size"])),
        axis=1,
    )
    working["m_raw"] = working["num_train_models"].apply(lambda x: parse_num_train_models(str(x)))
    working = working[working["m_raw"] > 0].copy()
    if working.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    working["n_over_m"] = working["n_raw"] / working["m_raw"]

    metric_cfg = [("rmse", False), ("mae", False), ("tau", True), ("rho", True)]
    winner_frames: list[pd.DataFrame] = []
    for metric, higher_is_better in metric_cfg:
        metric_df = working.dropna(subset=[metric]).copy()
        if metric_df.empty:
            continue
        grouped = metric_df.groupby(["dataset", "coreset_size", "num_train_models"])[metric]
        idx = grouped.idxmax() if higher_is_better else grouped.idxmin()
        wins = metric_df.loc[idx, ["m_raw", "n_over_m", "focus_method"]].copy()
        wins["metric"] = metric
        wins = wins.rename(columns={"focus_method": "method"})
        winner_frames.append(wins)
    winners = pd.concat(winner_frames, ignore_index=True) if winner_frames else pd.DataFrame()
    if winners.empty:
        empty = pd.DataFrame()
        return winners, empty, empty

    binned_frames: list[pd.DataFrame] = []
    dominant_frames: list[pd.DataFrame] = []
    for (metric, m_raw), chunk in winners.groupby(["metric", "m_raw"], sort=True):
        chunk = chunk.copy()
        n_unique = int(chunk["n_over_m"].nunique())
        if binning_mode == "fixed":
            n_bins = max(1, int(fixed_bin_count))
        else:
            n_bins = max(1, min(int(quantile_bins), n_unique))

        if n_bins <= 1 or n_unique <= 1:
            chunk["bin_idx"] = 0
            bin_centers = {0: float(chunk["n_over_m"].mean())}
            bin_left = dict(bin_centers)
            bin_right = dict(bin_centers)
        else:
            if binning_mode == "fixed":
                lo = float(chunk["n_over_m"].min())
                hi = float(chunk["n_over_m"].max())
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    chunk["bin_idx"] = 0
                    bin_centers = {0: float(chunk["n_over_m"].mean())}
                    bin_left = dict(bin_centers)
                    bin_right = dict(bin_centers)
                else:
                    edges = np.linspace(lo, hi, n_bins + 1)
                    binned = pd.cut(chunk["n_over_m"], bins=edges, include_lowest=True)
                    chunk["bin_idx"] = binned.cat.codes
                    valid_codes = sorted([int(v) for v in chunk["bin_idx"].unique() if int(v) >= 0])
                    bin_centers = {c: float((edges[c] + edges[c + 1]) * 0.5) for c in valid_codes}
                    bin_left = {c: float(edges[c]) for c in valid_codes}
                    bin_right = {c: float(edges[c + 1]) for c in valid_codes}
            else:
                binned = pd.qcut(chunk["n_over_m"], q=n_bins, duplicates="drop")
                chunk["bin_idx"] = binned.cat.codes
                bin_centers = {
                    int(code): float(interval.mid)
                    for code, interval in enumerate(binned.cat.categories)
                }
                bin_left = {
                    int(code): float(interval.left)
                    for code, interval in enumerate(binned.cat.categories)
                }
                bin_right = {
                    int(code): float(interval.right)
                    for code, interval in enumerate(binned.cat.categories)
                }
        chunk = chunk[chunk["bin_idx"] >= 0].copy()
        if chunk.empty:
            continue

        totals = chunk.groupby("bin_idx").size().rename("bin_total").reset_index()
        per_method = chunk.groupby(["bin_idx", "method"]).size().rename("win_count").reset_index()
        stats = per_method.merge(totals, on="bin_idx", how="left")
        stats["win_rate"] = stats["win_count"] / stats["bin_total"].clip(lower=1)
        stats["metric"] = metric
        stats["m_raw"] = m_raw
        stats["bin_x"] = stats["bin_idx"].map(lambda x: bin_centers.get(int(x), np.nan))
        stats["bin_center"] = stats["bin_idx"].map(lambda x: bin_centers.get(int(x), np.nan))
        stats["bin_left"] = stats["bin_idx"].map(lambda x: bin_left.get(int(x), np.nan))
        stats["bin_right"] = stats["bin_idx"].map(lambda x: bin_right.get(int(x), np.nan))
        binned_frames.append(stats)

        top = stats.sort_values(["bin_idx", "win_rate", "method"], ascending=[True, False, True]).copy()
        top = top.groupby("bin_idx", as_index=False).head(1)
        top["metric"] = metric
        top["m_raw"] = m_raw
        top["bin_x"] = top["bin_idx"].map(lambda x: bin_centers.get(int(x), np.nan))
        dominant_frames.append(top)

    binned_stats = pd.concat(binned_frames, ignore_index=True) if binned_frames else pd.DataFrame()
    dominant_stats = pd.concat(dominant_frames, ignore_index=True) if dominant_frames else pd.DataFrame()
    return winners, binned_stats, dominant_stats


def _notebook_07_pairwise_matrix_from_binned(
    *,
    binned_stats: pd.DataFrame,
    metric: str,
    methods: Iterable[str],
) -> np.ndarray:
    method_list = list(methods)
    n = len(method_list)
    out = np.full((n, n), np.nan, dtype=float)
    sub = binned_stats[binned_stats["metric"] == metric].copy()
    if sub.empty:
        return out
    counts = (
        sub[["m_raw", "bin_idx", "method", "win_count"]]
        .copy()
        .groupby(["m_raw", "bin_idx", "method"], as_index=False)["win_count"]
        .sum()
    )
    for i, mi in enumerate(method_list):
        for j, mj in enumerate(method_list):
            if i == j:
                out[i, j] = 0.5
                continue
            pivot = counts[counts["method"].isin([mi, mj])].copy()
            if pivot.empty:
                continue
            wide = (
                pivot.pivot_table(
                    index=["m_raw", "bin_idx"],
                    columns="method",
                    values="win_count",
                    aggfunc="sum",
                    fill_value=0,
                )
                .reset_index()
            )
            if mi not in wide.columns:
                wide[mi] = 0.0
            if mj not in wide.columns:
                wide[mj] = 0.0
            denom = wide[mi].astype(float) + wide[mj].astype(float)
            valid = denom > 0
            if not valid.any():
                continue
            numer = wide.loc[valid, mi].astype(float)
            denom_valid = denom.loc[valid].astype(float)
            # Weighted average across (M, bin) groups.
            out[i, j] = float(numer.sum() / denom_valid.sum())
    return out


def _load_consolidated_setting_results(
    *,
    split_method: str,
    coreset_size: str,
    num_train_models: str,
    root_dir: Path = CONSOLIDATED_RESULTS_ROOT,
) -> pd.DataFrame:
    path = (
        root_dir
        / str(split_method)
        / f"coreset_{coreset_size}"
        / f"nmodels_{num_train_models}"
        / _CONSOLIDATED_RESULTS
    )
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if frame.empty:
        return frame
    out = frame.copy()
    out["coreset_size"] = str(coreset_size)
    out["num_train_models"] = str(num_train_models)
    return out


def _dataset_method_metrics_binary_fixed_n(
    *,
    datasets: Iterable[str],
    fixed_n_values: Iterable[int] = (50, 100, 250),
    split_method: str = BINARY_SPLIT_METHOD,
) -> pd.DataFrame:
    dataset_set = set(str(d) for d in datasets)
    frames: list[pd.DataFrame] = []
    for n_value in fixed_n_values:
        cs_label = str(int(n_value))
        split_dir = CONSOLIDATED_RESULTS_ROOT / str(split_method) / f"coreset_{cs_label}"
        if not split_dir.is_dir():
            continue
        for nm_dir in sorted(p for p in split_dir.iterdir() if p.is_dir() and p.name.startswith("nmodels_")):
            nm_label = nm_dir.name.replace("nmodels_", "", 1)
            raw = _load_consolidated_setting_results(
                split_method=split_method,
                coreset_size=cs_label,
                num_train_models=nm_label,
                root_dir=CONSOLIDATED_RESULTS_ROOT,
            )
            if raw.empty:
                continue
            sub = raw[raw["dataset"].isin(dataset_set)].copy()
            if sub.empty:
                continue
            grouped = (
                sub.groupby(["dataset", "method"], as_index=False)[
                    ["rmse", "error", "corr_kendall", "corr_spearman"]
                ]
                .mean()
            )
            grouped["mae"] = grouped["error"]
            grouped["tau"] = grouped["corr_kendall"]
            grouped["rho"] = grouped["corr_spearman"]
            grouped["coreset_size"] = cs_label
            grouped["num_train_models"] = nm_label
            frames.append(grouped)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _notebook_07_aggregated_metric_vs_ratio(
    *,
    frame: pd.DataFrame,
    setting_name: str,
    num_questions: dict[str, int],
    passk_num_questions: dict[str, int],
    focus_methods: Iterable[str],
    coreset_filter: Any = None,
) -> pd.DataFrame:
    focus_set = set(focus_methods)
    if frame.empty:
        return pd.DataFrame()

    def _raw_n(dataset_key: str, cs_text: str) -> float:
        txt = str(cs_text)
        if txt.endswith("%"):
            pct = parse_coreset_size(txt)
            q = passk_num_questions.get(dataset_key) if setting_name.startswith("passk") else num_questions.get(dataset_key)
            if q is None:
                return max(float(pct), 1.0)
            return max(float(round(q * pct / 100.0)), 1.0)
        try:
            return max(float(txt), 1.0)
        except Exception:
            return 1.0

    work = frame.copy()
    work["focus_method"] = work["method"].apply(_notebook_07_focus_method)
    work = work[work["focus_method"].isin(focus_set)].copy()
    if coreset_filter is not None:
        work = work[work["coreset_size"].apply(lambda x: bool(coreset_filter(str(x))))].copy()
    if work.empty:
        return pd.DataFrame()

    work["m_raw"] = work["num_train_models"].apply(lambda x: parse_num_train_models(str(x)))
    work = work[work["m_raw"] > 0].copy()
    if work.empty:
        return pd.DataFrame()
    work["n_raw"] = work.apply(lambda r: _raw_n(str(r["dataset"]), str(r["coreset_size"])), axis=1)
    work["n_over_m"] = work["n_raw"] / work["m_raw"]
    work["m_over_n"] = work["m_raw"] / work["n_raw"].clip(lower=1e-9)

    metric_cols = ["rmse", "mae", "tau", "rho"]
    agg = (
        work.groupby(["focus_method", "num_train_models", "coreset_size"], as_index=False)[
            metric_cols + ["n_over_m", "m_over_n"]
        ]
        .mean()
        .rename(columns={"focus_method": "method"})
    )
    pieces: list[pd.DataFrame] = []
    for metric in metric_cols:
        d = agg[["method", "num_train_models", "coreset_size", "n_over_m", "m_over_n", metric]].copy()
        d = d.rename(columns={"num_train_models": "m_label", metric: "value"})
        d["metric"] = metric
        pieces.append(d)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _passk_benchmark_num_questions(
    benchmarks: Iterable[str],
    *,
    prefer_k1: bool = True,
) -> dict[str, int]:
    n_questions = load_num_questions()
    bench_set = set(benchmarks)
    bench_to_k: dict[str, list[tuple[int, int]]] = {}
    for ds_name, q_count in n_questions.items():
        parsed = parse_pass_at_k_dataset_name(ds_name)
        if parsed is None:
            continue
        bench, k, suffix = parsed
        if bench not in bench_set or suffix not in PASSK_ALLOWED_SUFFIXES_DEFAULT:
            continue
        bench_to_k.setdefault(bench, []).append((k, int(q_count)))
    result: dict[str, int] = {}
    for bench, vals in bench_to_k.items():
        vals_sorted = sorted(vals, key=lambda x: x[0])
        if prefer_k1:
            for k, q in vals_sorted:
                if k == 1:
                    result[bench] = q
                    break
        if bench not in result:
            result[bench] = vals_sorted[0][1]
    return result


def build_notebook_07(
    *,
    output_dir: Path | None = None,
    binary_datasets: Iterable[str] = BINARY_DATASETS_DEFAULT,
    continuous_datasets: Iterable[str] = CONTINUOUS_DATASETS_DEFAULT,
    passk_benchmarks: Iterable[str] = PASSK_BENCHMARKS_DEFAULT,
    include_continuous: bool = False,
    include_passk: bool = False,
    include_metric_points: bool = False,
    include_metric_density: bool = True,
    metric_density_gridsize: int = 36,
    include_winner_ratio_views: bool = True,
    include_pairwise_dominance: bool = True,
    winner_ratio_binning_mode: str = "fixed",
    winner_ratio_fixed_bins: int = 10,
    winner_ratio_quantile_bins: int = 10,
    winner_ratio_smooth_window: int = 11,
    winner_ratio_show_raw_points: bool = True,
    include_aggregated_metric_ratio_plots: bool = True,
) -> list[Path]:
    setup_matplotlib()
    ensure_dirs()
    output_dir = output_dir or _ensure_plot_subdir("07_winner_maps")
    outputs: list[Path] = []

    dataset_metrics = {
        "binary": _dataset_method_metrics_standard(BINARY_SPLIT_METHOD, binary_datasets),
    }
    if include_continuous:
        dataset_metrics["continuous"] = _dataset_method_metrics_standard(
            CONTINUOUS_SPLIT_METHOD,
            continuous_datasets,
        )
    if include_passk:
        dataset_metrics["passk_k1"] = _dataset_method_metrics_passk(
            CONTINUOUS_SPLIT_METHOD,
            passk_benchmarks,
            source_mode="k1",
        )
        dataset_metrics["passk_kopt"] = _dataset_method_metrics_passk(
            CONTINUOUS_SPLIT_METHOD,
            passk_benchmarks,
            source_mode="opt",
        )
    num_questions = load_num_questions()
    passk_num_questions = _passk_benchmark_num_questions(passk_benchmarks, prefer_k1=True)

    for setting_name, frame in dataset_metrics.items():
        if frame.empty:
            continue
        prep_t0 = time.perf_counter()
        winner_path, metric_path = _notebook_07_plot_cache_paths(setting_name)
        if winner_path.is_file() and metric_path.is_file():
            winners_df = pd.read_parquet(winner_path)
            metric_points_df = pd.read_parquet(metric_path)
            prep_elapsed = time.perf_counter() - prep_t0
            print(
                f"[07] {setting_name}: loaded plot-ready cache "
                f"in {prep_elapsed:.2f}s (winners={len(winners_df)}, metric_points={len(metric_points_df)})"
            )
        else:
            winners_df, metric_points_df = _notebook_07_prepare_plot_points(
                frame=frame,
                setting_name=setting_name,
                num_questions=num_questions,
                passk_num_questions=passk_num_questions,
            )
            winners_df.to_parquet(winner_path, index=False)
            metric_points_df.to_parquet(metric_path, index=False)
            prep_elapsed = time.perf_counter() - prep_t0
            print(
                f"[07] {setting_name}: recomputed + saved plot-ready cache "
                f"in {prep_elapsed:.2f}s (winners={len(winners_df)}, metric_points={len(metric_points_df)})"
            )

        fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_TWO_ROW_IN), constrained_layout=True)
        metric_keys = [("rmse", False), ("mae", False), ("tau", True), ("rho", True)]
        y_values: list[float] = winners_df["y"].tolist() if not winners_df.empty else []
        winner_methods: list[str] = []
        for ax, (metric, _higher_is_better) in zip(axes.flatten(), metric_keys):
            metric_winners = winners_df[winners_df["metric"] == metric]
            for _, row in metric_winners.iterrows():
                method = str(row["method"])
                winner_methods.append(method)
                style = method_style(method)
                ax.scatter(
                    float(row["x"]),
                    float(row["y"]),
                    marker=style.marker,
                    color=style.color,
                    s=95,
                    edgecolor="black",
                    linewidth=1.0,
                    alpha=0.9,
                )
            ax.set_xlabel("Num Source Models $M$")
            ax.set_ylabel("Coreset size $n$ (raw questions)")
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10.0))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_title(metric_axis_title(metric))
        _annotate_2x2_subplots(axes)
        if y_values:
            y_min = min(y_values)
            y_max = max(y_values)
            y_lo = max(1.0, y_min * 0.9)
            y_hi = max(y_max * 1.1, y_lo * 1.2)
            for ax in axes.flatten():
                ax.set_ylim(y_lo, y_hi)
        handles = scatter_legend_handles(winner_methods)
        legend = fig.legend(
            handles=handles,
            loc="lower center",
            ncol=max(3, len(handles)),
            frameon=True,
            fancybox=False,
            columnspacing=0.3,
            handletextpad=0.3,
        )
        _style_legend_box(legend)
        outputs.append(save_figure(fig, output_dir, f"winner_map_{setting_name}.pdf"))
        plt.close(fig)

        if include_metric_points:
            fig2, axes2 = plt.subplots(
                2, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_TWO_ROW_IN), constrained_layout=True
            )
            all_methods: list[str] = []
            for ax, metric in zip(axes2.flatten(), ["rmse", "mae", "tau", "rho"]):
                metric_points = metric_points_df[metric_points_df["metric"] == metric]
                for _, row in metric_points.iterrows():
                    method = str(row["method"])
                    all_methods.append(method)
                    style = method_style(method)
                    ax.scatter(
                        float(row["x"]),
                        float(row["y"]),
                        marker=style.marker,
                        color=style.color,
                        s=24,
                        edgecolor="black",
                        linewidth=1.0,
                        alpha=0.35,
                    )
                ax.set_xlabel(r"$n/M$")
                ax.set_ylabel(METRIC_SPECS[metric]["label"])
                ax.set_title(metric_axis_title(metric))
            _annotate_2x2_subplots(axes2)
            handles2 = scatter_legend_handles(all_methods)
            legend2 = fig2.legend(
                handles=handles2,
                loc="lower center",
                ncol=max(3, len(handles2)),
                frameon=True,
                fancybox=False,
                columnspacing=0.3,
                handletextpad=0.3,
            )
            _style_legend_box(legend2)
            outputs.append(save_figure(fig2, output_dir, f"metric_vs_n_over_M_{setting_name}.pdf"))
            plt.close(fig2)
        if include_metric_density:
            fig3, axes3 = plt.subplots(
                2, 2, figsize=(6.8, 6.8), constrained_layout=True
            )
            for ax, metric in zip(axes3.flatten(), ["rmse", "mae", "tau", "rho"]):
                metric_points = metric_points_df[metric_points_df["metric"] == metric]
                if metric_points.empty:
                    ax.set_xlabel(r"$n/M$")
                    ax.set_ylabel(METRIC_SPECS[metric]["label"])
                    ax.set_title(f"{metric_axis_title(metric)} (density)")
                    continue
                hb = ax.hexbin(
                    metric_points["x"].to_numpy(dtype=float),
                    metric_points["y"].to_numpy(dtype=float),
                    gridsize=metric_density_gridsize,
                    mincnt=1,
                    cmap="viridis",
                    linewidths=0.0,
                )
                cbar = fig3.colorbar(hb, ax=ax)
                cbar.set_label("Count")
                ax.set_xlabel(r"$n/M$")
                ax.set_ylabel(METRIC_SPECS[metric]["label"])
                ax.set_title(f"{metric_axis_title(metric)} (density)")
            _annotate_2x2_subplots(axes3)
            outputs.append(save_figure(fig3, output_dir, f"metric_vs_n_over_M_hexbin_{setting_name}.pdf"))
            plt.close(fig3)
        if include_winner_ratio_views and setting_name == "binary":
            focus_methods = (
                "mRMR++",
                "gpirt",
                "anchor_points_weighted",
                "random_sampling",
                "Random++",
                "lasso",
            )
            _focus_winners, focus_binned, focus_dominant = _notebook_07_focus_winner_bins(
                frame=frame,
                setting_name=setting_name,
                num_questions=num_questions,
                passk_num_questions=passk_num_questions,
                binning_mode=winner_ratio_binning_mode,
                fixed_bin_count=winner_ratio_fixed_bins,
                quantile_bins=winner_ratio_quantile_bins,
                focus_methods=focus_methods,
            )
            if not focus_binned.empty and not focus_dominant.empty:
                metric_order = ["rmse", "mae", "tau", "rho"]
                m_order = sorted(focus_binned["m_raw"].dropna().unique().tolist())

                fig4, axes4 = plt.subplots(
                    max(len(m_order), 1),
                    len(metric_order),
                    figsize=(6.8, 6.8),
                    constrained_layout=True,
                )
                axes4_arr = np.atleast_2d(axes4)
                for r, m_val in enumerate(m_order):
                    for c, metric in enumerate(metric_order):
                        ax = axes4_arr[r, c]
                        sub_raw = _focus_winners[
                            (_focus_winners["m_raw"] == m_val) & (_focus_winners["metric"] == metric)
                        ].copy()
                        if sub_raw.empty:
                            ax.set_visible(False)
                            continue
                        sub_raw = sub_raw[(sub_raw["n_over_m"] >= 0.0) & (sub_raw["n_over_m"] <= 12.0)].copy()
                        if sub_raw.empty:
                            ax.set_visible(False)
                            continue
                        for method in focus_methods:
                            msub = sub_raw.copy()
                            msub["is_win"] = (msub["method"] == method).astype(float)
                            curve_raw = (
                                msub.groupby("n_over_m", as_index=False)["is_win"]
                                .mean()
                                .sort_values("n_over_m")
                            )
                            if curve_raw.empty:
                                continue
                            if winner_ratio_show_raw_points:
                                style = method_style(_notebook_07_focus_style_method(method))
                                ax.scatter(
                                    curve_raw["n_over_m"].to_numpy(dtype=float),
                                    curve_raw["is_win"].to_numpy(dtype=float),
                                    s=14,
                                    alpha=0.12,
                                    color=style.color,
                                    edgecolor="none",
                                )
                            window = max(3, int(winner_ratio_smooth_window))
                            if window % 2 == 0:
                                window += 1
                            curve = curve_raw.copy()
                            curve["y_smooth"] = (
                                curve["is_win"]
                                .rolling(window=window, center=True, min_periods=max(3, window // 2))
                                .mean()
                            )
                            curve = curve.dropna(subset=["y_smooth"])
                            if curve.empty:
                                continue
                            style = method_style(_notebook_07_focus_style_method(method))
                            ax.plot(
                                curve["n_over_m"].to_numpy(dtype=float),
                                curve["y_smooth"].to_numpy(dtype=float),
                                marker=style.marker,
                                color=style.color,
                                linewidth=1.8,
                                alpha=0.95,
                                label=method,
                            )
                        ax.set_ylim(0.0, 1.0)
                        ax.set_xlim(0.0, 12.0)
                        ax.set_xlabel("n/M")
                        if c == 0:
                            ax.set_ylabel(f"P(win), M={int(m_val)}")
                        ax.set_title(f"{metric_axis_title(metric)} (no bins)")
                handles4 = scatter_legend_handles(list(focus_methods))
                legend4 = fig4.legend(
                    handles=handles4,
                    loc="lower center",
                    ncol=max(3, len(handles4)),
                    frameon=True,
                    fancybox=False,
                    columnspacing=0.3,
                    handletextpad=0.3,
                )
                _style_legend_box(legend4)
                outputs.append(save_figure(fig4, output_dir, f"winner_rate_vs_ratio_by_M_{setting_name}.pdf"))
                plt.close(fig4)

                fig5, axes5 = plt.subplots(
                    2, 2, figsize=(6.8, 6.8), constrained_layout=True
                )
                for ax, metric in zip(axes5.flatten(), metric_order):
                    sub = focus_dominant[focus_dominant["metric"] == metric].copy()
                    if sub.empty:
                        ax.set_title(f"{metric_axis_title(metric)} (dominant winner by ratio bin and M)")
                        ax.set_xlabel("n/M quantile bin")
                        ax.set_ylabel("M")
                        continue
                    for _, row in sub.iterrows():
                        method = str(row["method"])
                        style = method_style(_notebook_07_focus_style_method(method))
                        ax.scatter(
                            float(row["bin_x"]),
                            float(row["m_raw"]),
                            marker="s",
                            s=260,
                            color=style.color,
                            edgecolor="black",
                            linewidth=0.7,
                            alpha=0.9,
                        )
                        ax.text(
                            float(row["bin_x"]),
                            float(row["m_raw"]),
                            f"{100.0 * float(row['win_rate']):.0f}%",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="black",
                        )
                    ax.set_xlabel("n/M bin center")
                    ax.set_ylabel("M")
                    ax.set_title(f"{metric_axis_title(metric)} (dominant winner)")
                    ax.set_yticks(m_order)
                    ax.grid(True, axis="both", alpha=0.2)
                    xticks = sorted(sub["bin_x"].dropna().unique().tolist())
                    ax.set_xticks(xticks)
                    ax.set_xticklabels([f"{x:.2f}" for x in xticks], rotation=35, ha="right")
                handles5 = scatter_legend_handles(list(focus_methods))
                legend5 = fig5.legend(
                    handles=handles5,
                    loc="lower center",
                    ncol=max(3, len(handles5)),
                    frameon=True,
                    fancybox=False,
                    columnspacing=0.3,
                    handletextpad=0.3,
                )
                _style_legend_box(legend5)
                outputs.append(save_figure(fig5, output_dir, f"winner_dominance_ratio_bins_{setting_name}.pdf"))
                plt.close(fig5)
                if include_pairwise_dominance:
                    fig6, axes6 = plt.subplots(2, 2, figsize=(6.8, 6.8), constrained_layout=True)
                    im = None
                    for ax, metric in zip(axes6.flatten(), metric_order):
                        mat = _notebook_07_pairwise_matrix_from_binned(
                            binned_stats=focus_binned,
                            metric=metric,
                            methods=focus_methods,
                        )
                        if np.isnan(mat).all():
                            ax.set_title(f"{metric_axis_title(metric)} pairwise")
                            ax.set_xticks([])
                            ax.set_yticks([])
                            continue
                        im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis")
                        ax.set_title(f"{metric_axis_title(metric)} pairwise")
                        ax.set_xticks(np.arange(len(focus_methods)))
                        ax.set_yticks(np.arange(len(focus_methods)))
                        ax.set_xticklabels(list(focus_methods), rotation=40, ha="right", fontsize=7)
                        ax.set_yticklabels(list(focus_methods), fontsize=7)
                        for i in range(len(focus_methods)):
                            for j in range(len(focus_methods)):
                                val = mat[i, j]
                                if np.isnan(val):
                                    continue
                                ax.text(
                                    j,
                                    i,
                                    f"{val:.2f}",
                                    ha="center",
                                    va="center",
                                    fontsize=6,
                                    color="white" if val < 0.35 or val > 0.65 else "black",
                                )
                    _annotate_2x2_subplots(axes6)
                    if im is not None:
                        cbar6 = fig6.colorbar(im, ax=axes6.ravel().tolist(), shrink=0.8)
                        cbar6.set_label("P(A beats B)")
                    outputs.append(save_figure(fig6, output_dir, f"winner_pairwise_dominance_{setting_name}.pdf"))
                    plt.close(fig6)
        if include_aggregated_metric_ratio_plots and setting_name == "binary":
            focus_methods = ("mRMR++", "gpirt", "anchor_points_weighted", "random_sampling", "Random++")
            agg_pct = _notebook_07_aggregated_metric_vs_ratio(
                frame=frame,
                setting_name=setting_name,
                num_questions=num_questions,
                passk_num_questions=passk_num_questions,
                focus_methods=focus_methods,
                coreset_filter=lambda cs: cs.endswith("%"),
            )
            fixed_frame = _dataset_method_metrics_binary_fixed_n(datasets=binary_datasets)
            agg_fixed = _notebook_07_aggregated_metric_vs_ratio(
                frame=fixed_frame,
                setting_name=setting_name,
                num_questions=num_questions,
                passk_num_questions=passk_num_questions,
                focus_methods=focus_methods,
                coreset_filter=lambda cs: cs in {"50", "100", "250"},
            )

            def _plot_agg_ratio(
                df: pd.DataFrame,
                filename: str,
                title_suffix: str,
                *,
                connect_mode: str,
            ) -> None:
                if df.empty:
                    return
                metric_order = ["rmse", "mae", "tau", "rho"]
                fig, axes = plt.subplots(2, 2, figsize=(6.8, 6.8), constrained_layout=False)
                m_values = sorted({str(v) for v in df["m_label"].dropna().astype(str)}, key=parse_num_train_models)
                n_values = sorted({str(v) for v in df["coreset_size"].dropna().astype(str)}, key=parse_coreset_size)
                linestyles = ["-", "--", ":", "-."]
                group_values = m_values if connect_mode == "m" else n_values
                if len(group_values) <= 1:
                    alpha_levels = [0.95 for _ in group_values]
                else:
                    alpha_levels = np.linspace(0.95, 0.5, num=len(group_values)).tolist()
                for ax, metric in zip(axes.flatten(), metric_order):
                    sub = df[df["metric"] == metric].copy()
                    for method in focus_methods:
                        msub = sub[sub["method"] == method]
                        if msub.empty:
                            continue
                        group_col = "m_label" if connect_mode == "m" else "coreset_size"
                        for i, group_val in enumerate(group_values):
                            g = msub[msub[group_col].astype(str) == str(group_val)].sort_values("m_over_n")
                            if g.empty:
                                continue
                            style = method_style(_notebook_07_focus_style_method(method))
                            ax.plot(
                                g["m_over_n"].to_numpy(dtype=float),
                                g["value"].to_numpy(dtype=float) * float(METRIC_SPECS[metric]["scale"]),
                                color=style.color,
                                marker=style.marker,
                                linewidth=1.4,
                                linestyle=linestyles[i % len(linestyles)],
                                alpha=float(alpha_levels[i]),
                            )
                    ax.set_xlabel("M/n")
                    y_label_map = {
                        "rmse": r"$\mathbf{\leftarrow}$ RMSE (%)",
                        "mae": r"$\mathbf{\leftarrow}$ MAE (%)",
                        "tau": r"Kendall $\tau$ $\mathbf{\rightarrow}$",
                        "rho": r"Spearman $\rho$ $\mathbf{\rightarrow}$",
                    }
                    ax.set_ylabel(y_label_map.get(metric, METRIC_SPECS[metric]["label"]))
                    connect_label = "connect M" if connect_mode == "m" else "connect n"
                    ax.set_xlim(left=0.0)
                _annotate_2x2_subplots(axes)
                method_handles: list[Line2D] = []
                pretty_map = {
                    "mRMR++": "mRMR++",
                    "gpirt": "gpirt",
                    "anchor_points_weighted": "Anchor Points",
                    "random_sampling": "Random",
                    "Random++": "Random++",
                }
                for method in focus_methods:
                    style = method_style(_notebook_07_focus_style_method(method))
                    method_handles.append(
                        Line2D(
                            [0],
                            [0],
                            color=style.color,
                            marker=style.marker,
                            linestyle="-",
                            linewidth=1.4,
                            label=pretty_map.get(method, method),
                        )
                    )
                extra_handles: list[Line2D] = []
                extra_kind = "M" if connect_mode == "m" else "n"
                for i, group_val in enumerate(group_values[:3]):
                    extra_handles.append(
                        Line2D(
                            [0],
                            [0],
                            color="#374151",
                            marker="",
                            linestyle=linestyles[i % len(linestyles)],
                            linewidth=1.8,
                            alpha=float(alpha_levels[i]),
                            label=f"{extra_kind}={group_val}",
                        )
                    )
                handles = method_handles + extra_handles
                legend = fig.legend(
                    handles=handles,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0.035),
                    ncol=4,
                    frameon=True,
                    fancybox=False,
                    columnspacing=1.0,
                    handletextpad=0.3,
                )
                _style_legend_box(legend)
                fig.tight_layout(rect=(0.0, 0.09, 1.0, 1.0))
                outputs.append(save_figure(fig, output_dir, filename))
                plt.close(fig)

            _plot_agg_ratio(
                agg_pct,
                "combined_grid_binary_ratio_percent_agg_connectM.pdf",
                "(percent coreset)",
                connect_mode="m",
            )
            _plot_agg_ratio(
                agg_fixed,
                "combined_grid_binary_ratio_fixedn_agg_connectM.pdf",
                "(fixed n in {50,100,250})",
                connect_mode="m",
            )
            _plot_agg_ratio(
                agg_pct,
                "combined_grid_binary_ratio_percent_agg_connectN.pdf",
                "(percent coreset)",
                connect_mode="n",
            )
            _plot_agg_ratio(
                agg_fixed,
                "combined_grid_binary_ratio_fixedn_agg_connectN.pdf",
                "(fixed n in {50,100,250})",
                connect_mode="n",
            )
    display_saved_plots(outputs)
    return outputs


def run_notebook_with_capture(
    build_fn,
    *,
    save_outputs: bool = True,
    show_saved_plots: bool = False,
    return_figures: bool = True,
    close_figures: bool = False,
    **build_kwargs,
):
    capture_state = _capture_mode_begin(
        save_outputs=save_outputs,
        display_saved_plots_enabled=show_saved_plots,
        return_figures=return_figures,
        close_figures=close_figures,
    )
    outputs: list[Path] = []
    try:
        outputs = build_fn(**build_kwargs)
    finally:
        captured = _capture_mode_end(capture_state)
    if return_figures:
        return outputs, captured
    return outputs


def run_notebook_01(*, save_outputs: bool = True, show_saved_plots: bool = False, return_figures: bool = True, close_figures: bool = False, **kwargs):
    return build_notebook_01(
        save_outputs=save_outputs,
        show_saved_plots=show_saved_plots,
        return_figures=return_figures,
        close_figures=close_figures,
        **kwargs,
    )


def run_notebook_02(*, save_outputs: bool = True, show_saved_plots: bool = False, return_figures: bool = True, close_figures: bool = False, **kwargs):
    return run_notebook_with_capture(
        build_notebook_02,
        save_outputs=save_outputs,
        show_saved_plots=show_saved_plots,
        return_figures=return_figures,
        close_figures=close_figures,
        **kwargs,
    )


def run_notebook_03(*, save_outputs: bool = True, show_saved_plots: bool = False, return_figures: bool = True, close_figures: bool = False, **kwargs):
    return run_notebook_with_capture(
        build_notebook_03,
        save_outputs=save_outputs,
        show_saved_plots=show_saved_plots,
        return_figures=return_figures,
        close_figures=close_figures,
        **kwargs,
    )


def run_notebook_04(*, save_outputs: bool = True, show_saved_plots: bool = False, return_figures: bool = True, close_figures: bool = False, **kwargs):
    return run_notebook_with_capture(
        build_notebook_04,
        save_outputs=save_outputs,
        show_saved_plots=show_saved_plots,
        return_figures=return_figures,
        close_figures=close_figures,
        **kwargs,
    )


def run_notebook_05(*, save_outputs: bool = True, show_saved_plots: bool = False, return_figures: bool = True, close_figures: bool = False, **kwargs):
    return run_notebook_with_capture(
        build_notebook_05,
        save_outputs=save_outputs,
        show_saved_plots=show_saved_plots,
        return_figures=return_figures,
        close_figures=close_figures,
        **kwargs,
    )


def run_notebook_06(*, save_outputs: bool = True, show_saved_plots: bool = False, return_figures: bool = True, close_figures: bool = False, **kwargs):
    return run_notebook_with_capture(
        build_notebook_06,
        save_outputs=save_outputs,
        show_saved_plots=show_saved_plots,
        return_figures=return_figures,
        close_figures=close_figures,
        **kwargs,
    )


def run_notebook_06_ablation_1a(*, save_outputs: bool = True, show_saved_plots: bool = False, return_figures: bool = True, close_figures: bool = False, **kwargs):
    return run_notebook_with_capture(
        build_notebook_06_ablation_1a,
        save_outputs=save_outputs,
        show_saved_plots=show_saved_plots,
        return_figures=return_figures,
        close_figures=close_figures,
        **kwargs,
    )


def run_notebook_06_ablation_1b(*, save_outputs: bool = True, show_saved_plots: bool = False, return_figures: bool = True, close_figures: bool = False, **kwargs):
    return run_notebook_with_capture(
        build_notebook_06_ablation_1b,
        save_outputs=save_outputs,
        show_saved_plots=show_saved_plots,
        return_figures=return_figures,
        close_figures=close_figures,
        **kwargs,
    )


def run_notebook_06_ablation_2(*, save_outputs: bool = True, show_saved_plots: bool = False, return_figures: bool = True, close_figures: bool = False, **kwargs):
    return run_notebook_with_capture(
        build_notebook_06_ablation_2,
        save_outputs=save_outputs,
        show_saved_plots=show_saved_plots,
        return_figures=return_figures,
        close_figures=close_figures,
        **kwargs,
    )


def run_notebook_06_ablation_3(*, save_outputs: bool = True, show_saved_plots: bool = False, return_figures: bool = True, close_figures: bool = False, **kwargs):
    return run_notebook_with_capture(
        build_notebook_06_ablation_3,
        save_outputs=save_outputs,
        show_saved_plots=show_saved_plots,
        return_figures=return_figures,
        close_figures=close_figures,
        **kwargs,
    )


def run_notebook_06_ablation_4(*, save_outputs: bool = True, show_saved_plots: bool = False, return_figures: bool = True, close_figures: bool = False, **kwargs):
    return run_notebook_with_capture(
        build_notebook_06_ablation_4,
        save_outputs=save_outputs,
        show_saved_plots=show_saved_plots,
        return_figures=return_figures,
        close_figures=close_figures,
        **kwargs,
    )


def run_notebook_06_ablation_5(*, save_outputs: bool = True, show_saved_plots: bool = False, return_figures: bool = True, close_figures: bool = False, **kwargs):
    return run_notebook_with_capture(
        build_notebook_06_ablation_5,
        save_outputs=save_outputs,
        show_saved_plots=show_saved_plots,
        return_figures=return_figures,
        close_figures=close_figures,
        **kwargs,
    )


def run_notebook_07(*, save_outputs: bool = True, show_saved_plots: bool = False, return_figures: bool = True, close_figures: bool = False, **kwargs):
    return run_notebook_with_capture(
        build_notebook_07,
        save_outputs=save_outputs,
        show_saved_plots=show_saved_plots,
        return_figures=return_figures,
        close_figures=close_figures,
        **kwargs,
    )


def build_all_notebooks() -> dict[str, list[Path]]:
    outputs: dict[str, list[Path]] = {}
    outputs["01"] = build_notebook_01()
    outputs["02"] = build_notebook_02()
    outputs["03"] = build_notebook_03()
    outputs["04"] = build_notebook_04()
    outputs["05"] = build_notebook_05()
    outputs["06"] = build_notebook_06()
    outputs["07"] = build_notebook_07()
    return outputs

