from .anchor_points import AnchorPointsWeightedPred, AnchorPointPredictorPred
from .double_optimize import DoubleOptimizePred
from .lasso import LassoPred
from .pca import PCAPred
from .random import RandomSampling, RandomSamplingAndLearn, SampleFirstAndLearn, RandomSearchAndLearn, SmallSearchAndLearn, _make_krr_random_variant
from .aipw import AIPWPred
from .tiny_bench import PIRTPred, GPIRTPred
from .tiny_bench import (
    BetaPIRTPred, BetaGPIRTPred,
    B3PIRTPred, B3GPIRTPred,
    B3V2PIRTPred, B3V2GPIRTPred,
    LEGOPIRTPred, LEGOGPIRTPred,
    GaussianPIRTPred, GaussianGPIRTPred,
    PIRTPred1, PIRTPred5,
    GPIRTPred1, GPIRTPred5,
    BetaPIRTPred1, BetaPIRTPred5,
    BetaGPIRTPred1, BetaGPIRTPred5,
    LEGOPIRTPred1, LEGOPIRTPred5,
    LEGOGPIRTPred1, LEGOGPIRTPred5,
)
from .metabench import MetaBench
from .sort_search import SortAndSearchSum, SortAndSearchRecursiveSum
from . import mrmr as _mrmr_module
from .mrmr import _make_glm_variant, _make_raw_variant, _make_krr_variant, _make_cv_variant, _make_syn_variant, _make_logit_variant, _make_rf_variant
from .mrmr import (
    PIRTMRMRPred,
    PIRTMRMRPred_MID_y,
    PIRTMRMRPred_MID_PC1,
    PIRTMRMRPred_MIQ_y,
    PIRTMRMRPred_MIQ_PC1,
    GPIRTMRMRPred,
    GPIRTMRMRPred_MID_y,
    GPIRTMRMRPred_MID_PC1,
    GPIRTMRMRPred_MIQ_y,
    GPIRTMRMRPred_MIQ_PC1,
)
from .mrmr import (
    MRMRPred,
    MRMRPred_MID_y,
    MRMRPred_MID_PC1,
    MRMRPred_MIQ_y,
    MRMRPred_MIQ_PC1,
    MRMRPred_MI_y,
    MRMRPred_MID_yPC1,
    MRMRPred_MIQ_yPC1,
    MRMRPred_MID_y_aipw,
    MRMRPred_MID_PC1_aipw,
    MRMRPred_MIQ_y_aipw,
    MRMRPred_MIQ_PC1_aipw,
    MRMRPred_MID_IRT1,
    MRMRPred_MID_IRT5,
    MRMRPred_MIQ_IRT1,
    MRMRPred_MIQ_IRT5,
    MRMRPred_MID_IRT1_aipw,
    MRMRPred_MID_IRT5_aipw,
    MRMRPred_MIQ_IRT1_aipw,
    MRMRPred_MIQ_IRT5_aipw,
    AntiMRMRPred_MID_y,
    AntiMRMRPred_MIQ_y,
    MRMRPred_FCD_y,
    MRMRPred_FCD_PC1,
    MRMRPred_FCD_IRT1,
    MRMRPred_FCQ_y,
    MRMRPred_FCQ_PC1,
    MRMRPred_FCQ_IRT1,
    MRMRPred_FCD2_y,
    MRMRPred_FCD2_PC1,
    MRMRPred_FCD2_IRT1,
    MRMRPred_FCQ2_y,
    MRMRPred_FCQ2_PC1,
    MRMRPred_FCQ2_IRT1,
    MRMRGMIPred,
    MRMRPred_GMID_y,
    MRMRPred_GMID_PC1,
    MRMRPred_GMIQ_y,
    MRMRPred_GMIQ_PC1,
    MRMRPred_GMID_yPC1,
    MRMRPred_GMIQ_yPC1,
    MRMRPMIPred,
    MRMRPred_PMID_y,
    MRMRPred_PMID_PC1,
    MRMRPred_PMIQ_y,
    MRMRPred_PMIQ_PC1,
    MRMRPred_PMI_y,
    MRMRPred_PMID_yPC1,
    MRMRPred_PMIQ_yPC1,
    MRMRQGMIPred,
    MRMRPred_QGMID_y,
    MRMRPred_QGMID_PC1,
    MRMRPred_QGMIQ_y,
    MRMRPred_QGMIQ_PC1,
    MRMRPred_QGMID_yPC1,
    MRMRPred_QGMIQ_yPC1,
)

all_methods = {
    "random_sampling": RandomSampling,
    "random_sampling_and_learn": RandomSamplingAndLearn,
    "sample_first_and_learn": SampleFirstAndLearn,
    "random_search_and_learn": RandomSearchAndLearn,
    "small_search_and_learn": SmallSearchAndLearn,
    "aipw": AIPWPred,
    "pca": PCAPred,
    "anchor_points_weighted": AnchorPointsWeightedPred,
    "anchor_points_predictor": AnchorPointPredictorPred,
    "double_optimize": DoubleOptimizePred,
    "lasso": LassoPred,
    "pirt": PIRTPred,
    "gpirt": GPIRTPred,
    "B_pirt": BetaPIRTPred,
    "B_gpirt": BetaGPIRTPred,
    "B3_pirt": B3PIRTPred,
    "B3_gpirt": B3GPIRTPred,
    "B3_v2_pirt": B3V2PIRTPred,
    "B3_v2_gpirt": B3V2GPIRTPred,
    "LEGO_pirt": LEGOPIRTPred,
    "LEGO_gpirt": LEGOGPIRTPred,
    "G_pirt": GaussianPIRTPred,
    "G_gpirt": GaussianGPIRTPred,
    "pirt1": PIRTPred1,
    "pirt5": PIRTPred5,
    "gpirt1": GPIRTPred1,
    "gpirt5": GPIRTPred5,
    "B_pirt1": BetaPIRTPred1,
    "B_pirt5": BetaPIRTPred5,
    "B_gpirt1": BetaGPIRTPred1,
    "B_gpirt5": BetaGPIRTPred5,
    "LEGO_pirt1": LEGOPIRTPred1,
    "LEGO_pirt5": LEGOPIRTPred5,
    "LEGO_gpirt1": LEGOGPIRTPred1,
    "LEGO_gpirt5": LEGOGPIRTPred5,
    "metabench": MetaBench,
    "sort_search_sum": SortAndSearchSum,
    "sort_search_recursive_sum": SortAndSearchRecursiveSum,
    "mrmr": MRMRPred,
    "mrmr_MID_y": MRMRPred_MID_y,
    "mrmr_MID_PC1": MRMRPred_MID_PC1,
    "mrmr_MIQ_y": MRMRPred_MIQ_y,
    "mrmr_MIQ_PC1": MRMRPred_MIQ_PC1,
    "mrmr_MI_y": MRMRPred_MI_y,
    "mrmr_MID_y+PC1": MRMRPred_MID_yPC1,
    "mrmr_MIQ_y+PC1": MRMRPred_MIQ_yPC1,
    "mrmr_MID_y_aipw": MRMRPred_MID_y_aipw,
    "mrmr_MID_PC1_aipw": MRMRPred_MID_PC1_aipw,
    "mrmr_MIQ_y_aipw": MRMRPred_MIQ_y_aipw,
    "mrmr_MIQ_PC1_aipw": MRMRPred_MIQ_PC1_aipw,
    "mrmr_MID_IRT1": MRMRPred_MID_IRT1,
    "mrmr_MID_IRT5": MRMRPred_MID_IRT5,
    "mrmr_MIQ_IRT1": MRMRPred_MIQ_IRT1,
    "mrmr_MIQ_IRT5": MRMRPred_MIQ_IRT5,
    "mrmr_MID_IRT1_aipw": MRMRPred_MID_IRT1_aipw,
    "mrmr_MID_IRT5_aipw": MRMRPred_MID_IRT5_aipw,
    "mrmr_MIQ_IRT1_aipw": MRMRPred_MIQ_IRT1_aipw,
    "mrmr_MIQ_IRT5_aipw": MRMRPred_MIQ_IRT5_aipw,
    "anti_mrmr_MID_y": AntiMRMRPred_MID_y,
    "anti_mrmr_MIQ_y": AntiMRMRPred_MIQ_y,
    "mrmr_GMID_y": MRMRPred_GMID_y,
    "mrmr_GMID_PC1": MRMRPred_GMID_PC1,
    "mrmr_GMIQ_y": MRMRPred_GMIQ_y,
    "mrmr_GMIQ_PC1": MRMRPred_GMIQ_PC1,
    "mrmr_GMID_y+PC1": MRMRPred_GMID_yPC1,
    "mrmr_GMIQ_y+PC1": MRMRPred_GMIQ_yPC1,
    "mrmr_PMID_y": MRMRPred_PMID_y,
    "mrmr_PMID_PC1": MRMRPred_PMID_PC1,
    "mrmr_PMIQ_y": MRMRPred_PMIQ_y,
    "mrmr_PMIQ_PC1": MRMRPred_PMIQ_PC1,
    "mrmr_PMI_y": MRMRPred_PMI_y,
    "mrmr_PMID_y+PC1": MRMRPred_PMID_yPC1,
    "mrmr_PMIQ_y+PC1": MRMRPred_PMIQ_yPC1,
    "mrmr_QGMID_y": MRMRPred_QGMID_y,
    "mrmr_QGMID_PC1": MRMRPred_QGMID_PC1,
    "mrmr_QGMIQ_y": MRMRPred_QGMIQ_y,
    "mrmr_QGMIQ_PC1": MRMRPred_QGMIQ_PC1,
    "mrmr_QGMID_y+PC1": MRMRPred_QGMID_yPC1,
    "mrmr_QGMIQ_y+PC1": MRMRPred_QGMIQ_yPC1,
}

# Register KRR variants of random sampling+learn methods.
# Key naming: random_* -> krandom_* / k3random_* / k4random_*
for _key in ("random_sampling_and_learn", "sample_first_and_learn", "random_search_and_learn", "small_search_and_learn"):
    _cls = all_methods[_key]
    for _deg, _prefix in ((2, "k"), (3, "k3"), (4, "k4")):
        all_methods[_prefix + _key] = _make_krr_random_variant(_cls, degree=_deg)

# Register k=5 and k=7 MI nearest-neighbour variants for every MI-based MRMR method.
# The variant classes are generated dynamically in mrmr.py; we look them up
# by name (e.g. MRMRPred_MID_y -> MRMR5Pred_MID_y / MRMR7Pred_MID_y) and
# register them under keys like mrmr5_MID_y / mrmr7_MID_y.
for _key, _cls in list(all_methods.items()):
    if not _key.startswith("mrmr"):
        continue
    for _k in (4, 5, 6, 7, 8, 9):
        _variant_key = _key.replace("mrmr", f"mrmr{_k}", 1)
        _variant_class_name = _cls.__name__.replace("MRMR", f"MRMR{_k}", 1)
        all_methods[_variant_key] = getattr(_mrmr_module, _variant_class_name)

# F-statistic / Pearson-correlation variants (added after k-variant loop
# because FC methods do not use the Ross MI estimator and have no k-variants).
all_methods.update({
    "mrmr_FCD_y": MRMRPred_FCD_y,
    "mrmr_FCD_PC1": MRMRPred_FCD_PC1,
    "mrmr_FCD_IRT1": MRMRPred_FCD_IRT1,
    "mrmr_FCQ_y": MRMRPred_FCQ_y,
    "mrmr_FCQ_PC1": MRMRPred_FCQ_PC1,
    "mrmr_FCQ_IRT1": MRMRPred_FCQ_IRT1,
    "mrmr_FCD2_y": MRMRPred_FCD2_y,
    "mrmr_FCD2_PC1": MRMRPred_FCD2_PC1,
    "mrmr_FCD2_IRT1": MRMRPred_FCD2_IRT1,
    "mrmr_FCQ2_y": MRMRPred_FCQ2_y,
    "mrmr_FCQ2_PC1": MRMRPred_FCQ2_PC1,
    "mrmr_FCQ2_IRT1": MRMRPred_FCQ2_IRT1,
})

# IRT-representation MRMR variants (IRT-predicted scores as question reps)
_IRT_CONFIGS_REG = [
    ("IRT", 10), ("BetaIRT", 10), ("B3IRT", 1), ("LEGOIRT", 10), ("GIRT", 1),
]
_IRT_STRATEGIES_REG = [
    "MID", "MIQ", "FCD", "FCQ", "PMID", "PMIQ", "GMID", "GMIQ", "QGMID", "QGMIQ",
]
for _irt_prefix, _irt_dims in _IRT_CONFIGS_REG:
    for _strategy in _IRT_STRATEGIES_REG:
        _full_prefix = f"{_irt_prefix}{_irt_dims}"
        _key = f"mrmr_{_full_prefix}{_strategy}_y"
        _cls_name = f"MRMRPred_{_full_prefix}{_strategy}_y"
        all_methods[_key] = getattr(_mrmr_module, _cls_name)

# k-variants for IRT-representation MRMR methods
for _key, _cls in list(all_methods.items()):
    if not _key.startswith("mrmr_"):
        continue
    # Only process IRT-rep keys (contain "IRT" followed by digits then a strategy)
    _suffix = _key[len("mrmr_"):]
    if not any(_suffix.startswith(f"{p}{d}") for p, d in _IRT_CONFIGS_REG):
        continue
    for _k in (4, 5, 6, 7, 8, 9):
        _variant_key = _key.replace("mrmr", f"mrmr{_k}", 1)
        _variant_class_name = _cls.__name__.replace("MRMR", f"MRMR{_k}", 1)
        all_methods[_variant_key] = getattr(_mrmr_module, _variant_class_name)

# Register predictor variants for every mrmr method.
# Key naming: mrmr_* -> gmrmr_* / raw_mrmr_* / kmrmr_*
for _key, _cls in list(all_methods.items()):
    if not _key.startswith("mrmr"):
        continue
    all_methods["g" + _key] = _make_glm_variant(_cls)        # gmrmr_*
    all_methods["raw_" + _key] = _make_raw_variant(_cls)      # raw_mrmr_*
    all_methods["k" + _key] = _make_krr_variant(_cls)         # kmrmr_*
    all_methods["k3" + _key] = _make_krr_variant(_cls, degree=3)  # kmrmr3_*
    all_methods["k4" + _key] = _make_krr_variant(_cls, degree=4)  # kmrmr4_*
    all_methods["syn_" + _key] = _make_syn_variant(_cls)      # syn_mrmr_*
    all_methods["cv" + _key] = _make_cv_variant(_cls)          # cvmrmr_*
    all_methods["rf" + _key] = _make_rf_variant(_cls)          # rfmrmr_*

# Register logit-space regression variants for mrmr, cvmrmr, kernel ridge, and RF mrmr methods.
# Key naming: mrmr_* -> lmrmr_* / cvmrmr_* -> lcvmrmr_* / kmrmr_* -> lkmrmr_* / rfmrmr_* -> lrfmrmr_* etc.
for _key, _cls in list(all_methods.items()):
    if _key.startswith(("mrmr", "cvmrmr", "kmrmr", "k3mrmr", "k4mrmr", "rfmrmr")):
        all_methods["l" + _key] = _make_logit_variant(_cls)

# PIRT / GPIRT with MRMR coreset selection
all_methods.update({
    "pirt_mrmr_MID_y": PIRTMRMRPred_MID_y,
    "pirt_mrmr_MID_PC1": PIRTMRMRPred_MID_PC1,
    "pirt_mrmr_MIQ_y": PIRTMRMRPred_MIQ_y,
    "pirt_mrmr_MIQ_PC1": PIRTMRMRPred_MIQ_PC1,
    "gpirt_mrmr_MID_y": GPIRTMRMRPred_MID_y,
    "gpirt_mrmr_MID_PC1": GPIRTMRMRPred_MID_PC1,
    "gpirt_mrmr_MIQ_y": GPIRTMRMRPred_MIQ_y,
    "gpirt_mrmr_MIQ_PC1": GPIRTMRMRPred_MIQ_PC1,
})

# ---------------------------------------------------------------------------
# Backfill methods: reuse coresets from IRT / anchor-points methods and refit
# a Ridge or Kernel Ridge regressor (same pipeline as mrmr / kmrmr).
# Key naming: base_key+ / kbase_key+ / k3base_key+ / k4base_key+
# ---------------------------------------------------------------------------
from .backfill import _make_backfill_ridge, _make_backfill_krr

_backfill_base_methods = [
    "pirt", "gpirt",
    "pirt1", "gpirt1", "pirt5", "gpirt5",
    "B_pirt", "B_gpirt",
    "B_pirt1", "B_gpirt1", "B_pirt5", "B_gpirt5",
    "B3_pirt", "B3_gpirt",
    "B3_v2_pirt", "B3_v2_gpirt",
    "LEGO_pirt", "LEGO_gpirt",
    "LEGO_pirt1", "LEGO_gpirt1", "LEGO_pirt5", "LEGO_gpirt5",
    "G_pirt", "G_gpirt",
    "anchor_points_weighted", "anchor_points_predictor",
    "lasso",
]
for _base_key in _backfill_base_methods:
    all_methods[f"{_base_key}+"] = _make_backfill_ridge(_base_key)
    all_methods[f"k{_base_key}+"] = _make_backfill_krr(_base_key, degree=2)
    all_methods[f"k3{_base_key}+"] = _make_backfill_krr(_base_key, degree=3)
    all_methods[f"k4{_base_key}+"] = _make_backfill_krr(_base_key, degree=4)
