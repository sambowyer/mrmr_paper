import os
import sys
import time
import logging
import warnings
import threading
import traceback
import numpy as np
import joblib as jbl
from joblib import Parallel, delayed
from typing import Dict, List, Tuple
from multiprocessing import cpu_count, Manager
from tqdm import tqdm

from scipy.stats import spearmanr, pearsonr, kendalltau
from zarth_utils.config import Config
from zarth_utils.recorder import Recorder
from zarth_utils.nn_utils import set_random_seed, get_all_paths
from zarth_utils.timer import Timer

from benchpred import all_methods
from data_loader import LoaderRegistry
from data_utils import parse_pass_at_k_dataset, get_available_k_values, get_pass_at_k_paths

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def _get_error_logger():
    """Return a module-level logger that writes ERROR-level messages to logs/errors.log."""
    logger = logging.getLogger("method_runner_errors")
    if not logger.handlers:
        os.makedirs(_LOG_DIR, exist_ok=True)
        fh = logging.FileHandler(os.path.join(_LOG_DIR, "errors.log"))
        fh.setLevel(logging.ERROR)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        ))
        logger.addHandler(fh)
        logger.setLevel(logging.ERROR)
    return logger


class _TqdmProxy:
    """Lightweight tqdm stand-in for worker processes.

    Instead of writing progress bars to the terminal (which causes ANSI
    escape-code conflicts across processes), this records progress into a
    shared ``Manager().dict()`` that the main process polls to render bars.
    Updates to the shared dict are rate-limited to avoid IPC overhead.
    """
    _MIN_INTERVAL = 0.1

    def __init__(self, iterable=None, desc='', total=None,
                 _progress_dict=None, _worker_id=None, **kwargs):
        self.iterable = iterable
        self.total = total
        if self.total is None and hasattr(iterable, '__len__'):
            self.total = len(iterable)
        self.desc = desc
        self.n = 0
        self._postfix = ''
        self._progress_dict = _progress_dict
        self._worker_id = _worker_id
        self._last_report = 0
        self._report(force=True)

    def _report(self, force=False):
        if self._progress_dict is None or self._worker_id is None:
            return
        now = time.monotonic()
        if not force and (now - self._last_report) < self._MIN_INTERVAL:
            return
        self._last_report = now
        try:
            self._progress_dict[self._worker_id] = {
                'n': self.n, 'total': self.total, 'desc': self.desc,
                'postfix': self._postfix,
            }
        except Exception:
            pass

    def __iter__(self):
        if self.iterable is None:
            return
        for obj in self.iterable:
            yield obj
            self.n += 1
            self._report()
        self._report(force=True)

    def update(self, n=1):
        self.n += n
        self._report()

    def set_description(self, desc=None, refresh=True):
        self.desc = desc or ''
        if refresh:
            self._report(force=True)

    def set_postfix(self, *args, **kwargs):
        pass

    def set_postfix_str(self, s='', refresh=True):
        self._postfix = s
        if refresh:
            self._report(force=True)

    def close(self):
        self._report(force=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class MethodRunner:
    """Runner for benchmark methods across datasets"""

    def __init__(self, config: Config):
        self.config = config
        self.loader = LoaderRegistry.get_loader(config.data_source)

        # Build the results sub-directory that encodes the experiment setting:
        #   {dir_results}/{split_method}/coreset_{size}/nmodels_{label}/
        num_train_models = getattr(config, "num_train_models", "default")
        nmodels_label = (
            "nmodels_default"
            if num_train_models == "default"
            else f"nmodels_{num_train_models}"
        )
        self.results_base = os.path.join(
            config.dir_results,
            config.model_split_method,
            f"coreset_{config.coreset_size}",
            nmodels_label,
        )

    @staticmethod
    def resolve_coreset_size(coreset_size, num_data: int) -> int:
        """Resolve coreset_size to an integer.

        Accepts an int, a numeric string (e.g. "100"), or a percentage string
        (e.g. "10%") which is interpreted as a fraction of num_data.
        """
        coreset_str = str(coreset_size)
        if coreset_str.endswith("%"):
            pct = float(coreset_str[:-1])
            return max(1, int(pct / 100.0 * num_data))
        return int(coreset_str)

    @staticmethod
    def _binned_interpolation_split(num_models, true_acc, num_train_models,
                                    num_bins=10):
        """Select train (source) models with equal representation across
        accuracy bins, and assign the rest as test (target).

        25% of models become train by default; the train set is drawn in
        roughly equal proportions from *num_bins* equal-width bins over the
        true-accuracy range.
        """
        if num_train_models == "default":
            total_train = int(0.25 * num_models)
        else:
            total_train = int(num_train_models)

        min_acc, max_acc = float(true_acc.min()), float(true_acc.max())
        bin_edges = np.linspace(min_acc, max_acc + 1e-10, num_bins + 1)
        bin_ids = np.digitize(true_acc, bin_edges) - 1
        bin_ids = np.clip(bin_ids, 0, num_bins - 1)

        bins = [[] for _ in range(num_bins)]
        for idx in range(num_models):
            bins[bin_ids[idx]].append(idx)
        for b in bins:
            np.random.shuffle(b)

        non_empty = [i for i in range(num_bins) if bins[i]]
        per_bin = total_train // len(non_empty)
        remainder = total_train - per_bin * len(non_empty)

        bonus_bins = set(np.random.choice(
            len(non_empty), size=remainder, replace=False
        )) if remainder > 0 else set()
        quotas = {}
        for j, bi in enumerate(non_empty):
            extra = 1 if j in bonus_bins else 0
            quotas[bi] = min(per_bin + extra, len(bins[bi]))

        allocated = sum(quotas.values())
        while allocated < total_train:
            expandable = [i for i in non_empty if quotas[i] < len(bins[i])]
            if not expandable:
                break
            deficit = total_train - allocated
            extra_per = deficit // len(expandable)
            extra_rem = deficit % len(expandable)
            bonus_exp = set(np.random.choice(
                len(expandable), size=extra_rem, replace=False
            )) if extra_rem > 0 else set()
            for j, bi in enumerate(expandable):
                add = extra_per + (1 if j in bonus_exp else 0)
                add = min(add, len(bins[bi]) - quotas[bi])
                quotas[bi] += add
            allocated = sum(quotas.values())

        source_models = []
        for bi in non_empty:
            source_models.extend(bins[bi][: quotas[bi]])
        source_set = set(source_models)
        target_models = [i for i in range(num_models) if i not in source_set]
        return source_models, target_models

    @staticmethod
    def _stratified_split(num_models, true_acc, model_names, num_train_models,
                           num_bins=10):
        """Like binned_interpolation but groups model-temperature variants so
        that all temperatures of a base model are assigned to the same split.

        Model names are expected to have the format ``{base_model}__{temp}``.
        When a name does not contain ``__``, the entire name is treated as the
        base model (i.e. it forms a group of one), so this degenerates to
        regular binned_interpolation for non-continuous-cat datasets.
        """
        group_map = {}
        for idx, name in enumerate(model_names):
            base = name.split("__")[0] if "__" in name else name
            group_map.setdefault(base, []).append(idx)

        groups = sorted(group_map.keys())
        num_groups = len(groups)
        group_indices = [group_map[g] for g in groups]

        group_acc = np.array([
            true_acc[group_indices[i]].mean() for i in range(num_groups)
        ])

        if num_train_models == "default":
            total_train_individuals = int(0.25 * num_models)
        else:
            total_train_individuals = int(num_train_models)
        mean_group_size = num_models / num_groups
        if num_train_models == "default":
            total_train_groups = max(1, round(total_train_individuals / mean_group_size))
        else:
            total_train_groups = max(1, int(np.ceil(total_train_individuals / mean_group_size)))
        total_train_groups = min(total_train_groups, num_groups - 1)

        min_acc, max_acc = float(group_acc.min()), float(group_acc.max())
        bin_edges = np.linspace(min_acc, max_acc + 1e-10, num_bins + 1)
        bin_ids = np.digitize(group_acc, bin_edges) - 1
        bin_ids = np.clip(bin_ids, 0, num_bins - 1)

        bins = [[] for _ in range(num_bins)]
        for gidx in range(num_groups):
            bins[bin_ids[gidx]].append(gidx)
        for b in bins:
            np.random.shuffle(b)

        non_empty = [i for i in range(num_bins) if bins[i]]
        per_bin = total_train_groups // len(non_empty)
        remainder = total_train_groups - per_bin * len(non_empty)

        bonus_bins = set(np.random.choice(
            len(non_empty), size=remainder, replace=False
        )) if remainder > 0 else set()
        quotas = {}
        for j, bi in enumerate(non_empty):
            extra = 1 if j in bonus_bins else 0
            quotas[bi] = min(per_bin + extra, len(bins[bi]))

        allocated = sum(quotas.values())
        while allocated < total_train_groups:
            expandable = [i for i in non_empty if quotas[i] < len(bins[i])]
            if not expandable:
                break
            deficit = total_train_groups - allocated
            extra_per = deficit // len(expandable)
            extra_rem = deficit % len(expandable)
            bonus_exp = set(np.random.choice(
                len(expandable), size=extra_rem, replace=False
            )) if extra_rem > 0 else set()
            for j, bi in enumerate(expandable):
                add = extra_per + (1 if j in bonus_exp else 0)
                add = min(add, len(bins[bi]) - quotas[bi])
                quotas[bi] += add
            allocated = sum(quotas.values())

        source_group_ids = []
        for bi in non_empty:
            source_group_ids.extend(bins[bi][:quotas[bi]])
        source_group_set = set(source_group_ids)

        source_models = []
        for gi in source_group_ids:
            source_models.extend(group_indices[gi])

        if num_train_models != "default" and len(source_models) > total_train_individuals:
            source_models = list(
                np.random.choice(source_models, size=total_train_individuals, replace=False)
            )

        target_models = []
        for gi in range(num_groups):
            if gi not in source_group_set:
                target_models.extend(group_indices[gi])

        return source_models, target_models

    def _compute_split(self, num_models, true_acc, num_train_models,
                       model_names=None):
        """Compute source/target model split for the current RNG state."""
        if self.config.model_split_method == "interpolation":
            num_target_models = int(0.25 * num_models)
            target_models = list(
                np.random.permutation(num_models)[:num_target_models]
            )
            source_models = [
                i for i in range(num_models) if i not in target_models
            ]
        elif self.config.model_split_method == "extrapolation":
            order = np.argsort(true_acc)
            num_source = int(0.5 * num_models)
            num_target = int(0.3 * num_models)
            source_pool = list(order[:num_source])
            target_pool = list(order[-num_target:])
            np.random.shuffle(source_pool)
            np.random.shuffle(target_pool)
            keep_src = max(
                2, num_source - np.random.randint(0, max(1, num_source // 10 + 1))
            )
            keep_tgt = max(
                2, num_target - np.random.randint(0, max(1, num_target // 10 + 1))
            )
            source_models = source_pool[:keep_src]
            target_models = target_pool[:keep_tgt]
        elif self.config.model_split_method == "easier_extrapolation":
            order = np.argsort(true_acc)
            num_source = int(0.7 * num_models)
            num_target = int(0.3 * num_models)
            source_pool = list(order[:num_source])
            target_pool = list(order[-num_target:])
            np.random.shuffle(source_pool)
            np.random.shuffle(target_pool)
            keep_src = max(
                2, num_source - np.random.randint(0, max(1, num_source // 10 + 1))
            )
            keep_tgt = max(
                2, num_target - np.random.randint(0, max(1, num_target // 10 + 1))
            )
            source_models = source_pool[:keep_src]
            target_models = target_pool[:keep_tgt]
        elif self.config.model_split_method == "binned_interpolation":
            source_models, target_models = self._binned_interpolation_split(
                num_models, true_acc, num_train_models
            )
        elif self.config.model_split_method in ("stratified", "timing"):
            if model_names is None:
                raise ValueError(
                    "model_names must be provided for the 'stratified' split"
                )
            source_models, target_models = self._stratified_split(
                num_models, true_acc, model_names, num_train_models
            )
        else:
            raise NotImplementedError(
                f"Unknown split method: {self.config.model_split_method}"
            )

        if (
            num_train_models != "default"
            and self.config.model_split_method not in (
                "binned_interpolation", "stratified", "timing",
            )
        ):
            n = int(num_train_models)
            if n < len(source_models):
                source_models = list(
                    np.random.choice(source_models, size=n, replace=False)
                )

        return source_models, target_models

    @staticmethod
    def _run_single_trial(method_name, seed, scores, source_models, target_models,
                          true_acc, coreset_size, results_base, dataset_name,
                          exp_suffix, config_dict, use_git,
                          no_pass_at_k_fill=False,
                          tqdm_position_queue=None, progress_dict=None):
        """Execute a single (method, seed) trial. Safe for multiprocessing.

        When *tqdm_position_queue* and *progress_dict* are provided, tqdm is
        replaced with a lightweight proxy in the worker process so that no
        ANSI escape codes are written to the terminal.  Progress is reported
        through *progress_dict* and rendered by a polling thread in the main
        process.
        """
        tqdm_pos = None
        _tqdm_func = None
        if tqdm_position_queue is not None:
            tqdm_pos = tqdm_position_queue.get()
            _trial_label = f"[W{tqdm_pos}] {method_name}"

            def _make_proxy(*args, **kwargs):
                inner_desc = kwargs.get('desc', '')
                kwargs['desc'] = (
                    f"{_trial_label}: {inner_desc}" if inner_desc
                    else _trial_label
                )
                kwargs['_progress_dict'] = progress_dict
                kwargs['_worker_id'] = tqdm_pos
                return _TqdmProxy(*args, **kwargs)

            _tqdm_func = _make_proxy

            for _name, _mod in list(sys.modules.items()):
                if (_mod is not None
                        and _name.startswith('benchpred')
                        and callable(getattr(_mod, 'tqdm', None))):
                    _mod.tqdm = _make_proxy

        try:
            return MethodRunner._do_single_trial(
                method_name, seed, scores, source_models, target_models,
                true_acc, coreset_size, results_base, dataset_name,
                exp_suffix, config_dict, use_git, no_pass_at_k_fill,
                _tqdm_func=_tqdm_func)
        except Exception:
            _get_error_logger().error(
                "Trial FAILED | method=%s | dataset=%s | seed=%s | "
                "coreset_size=%s | results_base=%s\n%s",
                method_name, dataset_name, seed, coreset_size, results_base,
                traceback.format_exc(),
            )
            return None
        finally:
            if tqdm_pos is not None:
                tqdm_position_queue.put(tqdm_pos)

    @staticmethod
    def _run_kprime_fills(method, method_name, dir_exp, benchmark,
                          kprime_values, coreset_indices,
                          source_models, target_models,
                          config_dict, use_git, seed,
                          scores_dir=None, name_suffix="",
                          _tqdm_func=None):
        """Run cross-k regressions for every k' in *kprime_values*.

        For each k', loads the k' score matrix, refits the method's regressor
        on the source models' k' scores (using the same coreset), evaluates on
        the target models, and saves results into
        ``{dir_exp}/pred_pass_at_{k'}/``.
        """
        _tqdm = _tqdm_func or tqdm
        if scores_dir is None:
            scores_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "data", "scores",
            )
        coreset_indices = np.asarray(coreset_indices)

        pbar = _tqdm(total=len(kprime_values), desc=f"k' fill ({method_name})",
                     unit="k'")
        for kp in kprime_values:
            kp_dir = os.path.join(dir_exp, f"pred_pass_at_{kp}")
            if os.path.exists(os.path.join(kp_dir, "result.jbl")):
                pbar.update(1)
                pbar.set_postfix_str(f"k'={kp} (cached)")
                continue

            kp_scores = jbl.load(
                os.path.join(scores_dir, f"{benchmark}_pass_at_{kp}{name_suffix}.jbl")
            )
            kp_true_acc = kp_scores.mean(axis=1)

            rgs = method.refit_regressor(kp_scores[source_models])

            pred_acc_test = np.atleast_1d(
                rgs.predict(
                    kp_scores[target_models][:, coreset_indices]
                )
            ).ravel()
            pred_acc_train = np.atleast_1d(
                rgs.predict(
                    kp_scores[source_models][:, coreset_indices]
                )
            ).ravel()

            test_residuals = pred_acc_test - kp_true_acc[target_models]
            error_MAE = float(np.fabs(test_residuals).mean())
            error_MSE = float((test_residuals ** 2).mean())
            error_RMSE = np.sqrt(error_MSE)

            true_all = np.concatenate([
                kp_true_acc[source_models], kp_true_acc[target_models],
            ])
            pred_all = np.concatenate([pred_acc_train, pred_acc_test])
            corr_spearman = corr_kendall = corr_pearson = np.nan
            if len(true_all) >= 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rho, _ = spearmanr(true_all, pred_all)
                    if np.isfinite(rho):
                        corr_spearman = float(rho)
                    tau, _ = kendalltau(true_all, pred_all)
                    if np.isfinite(tau):
                        corr_kendall = float(tau)
                    r, _ = pearsonr(true_all, pred_all)
                    if np.isfinite(r):
                        corr_pearson = float(r)

            os.makedirs(kp_dir, exist_ok=True)

            recorder = Recorder(
                os.path.join(kp_dir, "record"),
                config=config_dict,
                use_git=use_git,
            )
            recorder["error_MAE"] = error_MAE
            recorder["error_RMSE"] = error_RMSE
            recorder["error"] = error_MAE
            if not np.isnan(corr_spearman):
                recorder["corr_spearman"] = corr_spearman
            if not np.isnan(corr_kendall):
                recorder["corr_kendall"] = corr_kendall
            if not np.isnan(corr_pearson):
                recorder["corr_pearson"] = corr_pearson
            recorder.end_recording()

            kp_result = {
                "seed": seed,
                "coreset_indices": coreset_indices.tolist(),
                "train_model_indices": [int(i) for i in source_models],
                "test_model_indices": [int(i) for i in target_models],
                "pred_acc_train": np.asarray(pred_acc_train).tolist(),
                "pred_acc_test": np.asarray(pred_acc_test).tolist(),
                "true_acc_train": kp_true_acc[source_models].tolist(),
                "true_acc_test": kp_true_acc[target_models].tolist(),
                "error_MAE": error_MAE,
                "error_RMSE": error_RMSE,
            }
            jbl.dump(kp_result, os.path.join(kp_dir, "result.jbl"))
            jbl.dump(rgs, os.path.join(kp_dir, "ckpt.jbl"))

            pbar.update(1)
            pbar.set_postfix_str(f"k'={kp} MAE={error_MAE:.4f}")
        pbar.close()

    @staticmethod
    def _do_single_trial(method_name, seed, scores, source_models, target_models,
                         true_acc, coreset_size, results_base, dataset_name,
                         exp_suffix, config_dict, use_git,
                         no_pass_at_k_fill=False, _tqdm_func=None):
        """Inner implementation of a single trial (no tqdm management)."""
        dir_exp = os.path.join(
            results_base,
            f"{dataset_name}{exp_suffix}",
            method_name,
            str(seed),
        )
        os.makedirs(dir_exp, exist_ok=True)

        parsed = parse_pass_at_k_dataset(dataset_name)
        do_kprime_fill = parsed is not None and not no_pass_at_k_fill
        other_k_values = []
        kprime_scores_dir = None
        kprime_suffix = ""
        if do_kprime_fill:
            benchmark, current_k = parsed
            paths_info = get_pass_at_k_paths(dataset_name)
            if paths_info is not None:
                kprime_scores_dir, kprime_suffix = paths_info
            other_k_values = get_available_k_values(
                benchmark, exclude_k=current_k,
                scores_dir=kprime_scores_dir, name_suffix=kprime_suffix,
            )
            do_kprime_fill = len(other_k_values) > 0

        main_result_path = os.path.join(dir_exp, "result.jbl")
        if os.path.exists(main_result_path):
            if not do_kprime_fill:
                return None
            missing_kprimes = [
                kp for kp in other_k_values
                if not os.path.exists(
                    os.path.join(dir_exp, f"pred_pass_at_{kp}", "result.jbl")
                )
            ]
            if not missing_kprimes:
                return None
            main_result = jbl.load(main_result_path)
            coreset_indices = main_result["coreset_indices"]
            source_models = main_result["train_model_indices"]
            target_models = main_result["test_model_indices"]
            method = all_methods[method_name]()
            method.load(os.path.join(dir_exp, "ckpt.jbl"))
            MethodRunner._run_kprime_fills(
                method, method_name, dir_exp, benchmark,
                missing_kprimes, coreset_indices,
                source_models, target_models,
                config_dict, use_git, seed,
                scores_dir=kprime_scores_dir,
                name_suffix=kprime_suffix,
                _tqdm_func=_tqdm_func,
            )
            return None

        set_random_seed(seed)

        method = all_methods[method_name]()

        # Backfill methods reuse coresets from a base method's checkpoint.
        extra_fit_kwargs = {}
        base_key = getattr(method, '_base_method_key', None)
        if base_key is not None:
            base_ckpt = os.path.join(
                results_base, f"{dataset_name}{exp_suffix}",
                base_key, str(seed), "ckpt.jbl"
            )
            if not os.path.exists(base_ckpt):
                return None
            extra_fit_kwargs['base_ckpt_path'] = base_ckpt

        timer = Timer()
        timer.start()
        method.fit(
            source_full_scores=scores[source_models],
            coreset_size=coreset_size,
            seed=seed,
            **extra_fit_kwargs,
        )
        training_time = timer.get_last_duration()

        compressed_indices = method.get_coreset()

        timer.start()
        pred_acc_test = method.predict(
            scores[target_models][:, compressed_indices]
        )
        inference_time = timer.get_last_duration()

        pred_acc_train = method.predict(
            scores[source_models][:, compressed_indices]
        )

        test_residuals = pred_acc_test - true_acc[target_models]
        error_MAE = float(np.fabs(test_residuals).mean())
        error_MSE = float((test_residuals ** 2).mean())
        error_RMSE = np.sqrt(error_MSE)

        true_all = np.concatenate([
            true_acc[source_models], true_acc[target_models],
        ])
        pred_all = np.concatenate([pred_acc_train, pred_acc_test])
        corr_spearman = corr_kendall = corr_pearson = np.nan
        if len(true_all) >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, _ = spearmanr(true_all, pred_all)
                if np.isfinite(rho):
                    corr_spearman = float(rho)
                tau, _ = kendalltau(true_all, pred_all)
                if np.isfinite(tau):
                    corr_kendall = float(tau)
                r, _ = pearsonr(true_all, pred_all)
                if np.isfinite(r):
                    corr_pearson = float(r)

        recorder = Recorder(
            os.path.join(dir_exp, "record"),
            config=config_dict,
            use_git=use_git,
        )
        recorder["training_time"] = training_time
        recorder["inference_time"] = inference_time
        recorder["error_MAE"] = error_MAE
        recorder["error_RMSE"] = error_RMSE
        recorder["error"] = error_MAE
        if not np.isnan(corr_spearman):
            recorder["corr_spearman"] = corr_spearman
        if not np.isnan(corr_kendall):
            recorder["corr_kendall"] = corr_kendall
        if not np.isnan(corr_pearson):
            recorder["corr_pearson"] = corr_pearson
        recorder.end_recording()

        selection_metrics = getattr(method, "selection_metrics", None)

        result_dict = {
            "seed": seed,
            "coreset_indices": np.asarray(compressed_indices).tolist(),
            "train_model_indices": [int(i) for i in source_models],
            "test_model_indices": [int(i) for i in target_models],
            "pred_acc_train": np.asarray(pred_acc_train).tolist(),
            "pred_acc_test": np.asarray(pred_acc_test).tolist(),
            "true_acc_train": true_acc[source_models].tolist(),
            "true_acc_test": true_acc[target_models].tolist(),
            "error_MAE": error_MAE,
            "error_RMSE": error_RMSE,
        }
        if selection_metrics is not None:
            result_dict["selection_metrics"] = selection_metrics

        jbl.dump(result_dict, os.path.join(dir_exp, "result.jbl"))

        method.save(os.path.join(dir_exp, "ckpt.jbl"))

        if do_kprime_fill:
            MethodRunner._run_kprime_fills(
                method, method_name, dir_exp, benchmark,
                other_k_values, compressed_indices,
                source_models, target_models,
                config_dict, use_git, seed,
                scores_dir=kprime_scores_dir,
                name_suffix=kprime_suffix,
                _tqdm_func=_tqdm_func,
            )

        ret = {
            "method_name": method_name,
            "seed": seed,
            "source_models": source_models,
            "target_models": target_models,
            "true_acc_train": true_acc[source_models].tolist(),
            "true_acc_test": true_acc[target_models].tolist(),
            "pred_acc_train": pred_acc_train.tolist(),
            "pred_acc_test": pred_acc_test.tolist(),
            "coreset_indices": np.asarray(compressed_indices).tolist(),
            "error_MAE": error_MAE,
            "error_RMSE": error_RMSE,
            "error": error_MAE,
        }
        if selection_metrics is not None:
            ret["selection_metrics"] = selection_metrics
        return ret

    def run_for_dataset(self, dataset_name: str) -> Dict:
        """Run all methods for a single dataset

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary with results for all methods
        """
        print(f"\n=== Running dataset: {dataset_name} ===")

        scores, model_names, true_acc = self.loader.load(dataset_name)
        num_models, num_data = scores.shape

        coreset_size = self.resolve_coreset_size(self.config.coreset_size, num_data)
        print(f"  Coreset size: {self.config.coreset_size} -> {coreset_size} (num_data={num_data})")

        num_train_models = getattr(self.config, "num_train_models", "default")
        print(f"  Num train models: {num_train_models}")

        config_dict = {k: self.config[k] for k in list(self.config.keys())}
        no_pass_at_k_fill = getattr(self.config, "no_pass_at_k_fill", False)

        # Pre-compute splits and build (seed, method) task list
        tasks = []
        for seed in range(
            self.config.seed_start, self.config.seed_start + self.config.num_run
        ):
            set_random_seed(seed)
            source_models, target_models = self._compute_split(
                num_models, true_acc, num_train_models,
                model_names=model_names,
            )
            for method_name in self.config.methods:
                tasks.append((
                    method_name, seed, scores,
                    source_models, target_models, true_acc,
                    coreset_size, self.results_base, dataset_name,
                    self.config.exp_suffix, config_dict, self.config.use_git,
                    no_pass_at_k_fill,
                ))

        multi_process = getattr(self.config, "multi_process", False)
        bar_desc = f"  {dataset_name}"
        if multi_process:
            n_cpus = cpu_count()
            n_workers = min(n_cpus, len(tasks))
            print(f"  Running {len(tasks)} trials in parallel "
                  f"({n_workers} workers)")

            mgr = Manager()
            position_queue = mgr.Queue()
            progress_dict = mgr.dict()
            for i in range(n_workers):
                position_queue.put(i + 1)

            # Worker bars are rendered exclusively by the main process;
            # workers only write to progress_dict (no terminal I/O).
            # Compute uniform description width so all bars align.
            _w_digits = len(str(n_workers))
            _max_method = max(len(m) for m in self.config.methods)
            _desc_width = (
                3 + _w_digits       # "[W" + digits + "] "
                + _max_method + 2   # method name + ": "
                + 25                # covers longest inner desc
            )

            worker_bars = {}
            for i in range(1, n_workers + 1):
                worker_bars[i] = tqdm(
                    total=1, position=i, leave=False,
                )

            stop_event = threading.Event()

            def _poll_worker_progress():
                while not stop_event.is_set():
                    try:
                        snapshot = dict(progress_dict)
                    except Exception:
                        snapshot = {}
                    for pos, info in snapshot.items():
                        if info is not None and pos in worker_bars:
                            bar = worker_bars[pos]
                            bar.total = info.get('total') or 1
                            bar.n = min(info.get('n', 0), bar.total)
                            bar.set_description_str(
                                info.get('desc', '').ljust(_desc_width))
                            postfix = info.get('postfix', '')
                            if postfix:
                                bar.set_postfix_str(postfix)
                            bar.refresh()
                    stop_event.wait(0.15)

            poller = threading.Thread(target=_poll_worker_progress,
                                      daemon=True)
            poller.start()

            parallel_tasks = [t + (position_queue, progress_dict)
                              for t in tasks]
            gen = Parallel(n_jobs=n_workers, return_as="generator")(
                delayed(MethodRunner._run_single_trial)(*t)
                for t in parallel_tasks
            )
            trial_results = []
            pbar = tqdm(gen, total=len(parallel_tasks), desc=bar_desc,
                        position=0, unit="trial")
            for result in pbar:
                trial_results.append(result)
                if result is not None:
                    pbar.set_postfix_str(
                        f"{result['method_name']} s={result['seed']} "
                        f"MAE={result['error_MAE']:.4f}"
                    )
                else:
                    pbar.set_postfix_str("(cached)")
            pbar.close()

            stop_event.set()
            poller.join(timeout=1.0)
            for bar in worker_bars.values():
                bar.close()
            print('\n' * n_workers)
            mgr.shutdown()
        else:
            trial_results = [
                MethodRunner._run_single_trial(*t)
                for t in tqdm(tasks, desc=bar_desc)
            ]

        # Collect results by method
        dataset_results = {method_name: [] for method_name in self.config.methods}
        for result in trial_results:
            if result is not None:
                dataset_results[result["method_name"]].append(result)

        # Aggregate results per method
        dataset_results = {
            method_name: {
                "results": method_results,
                "mean_error": np.mean(
                    [r.get("error_MAE", r.get("error")) for r in method_results]
                ),
                "std_error": np.std(
                    [r.get("error_MAE", r.get("error")) for r in method_results]
                ),
                "mean_error_RMSE": np.mean(
                    [r["error_RMSE"] for r in method_results if "error_RMSE" in r]
                ) if any("error_RMSE" in r for r in method_results) else np.nan,
                "std_error_RMSE": np.std(
                    [r["error_RMSE"] for r in method_results if "error_RMSE" in r]
                ) if any("error_RMSE" in r for r in method_results) else np.nan,
            }
            for method_name, method_results in dataset_results.items()
        }

        return dataset_results

    def run_all_datasets(self) -> Dict:
        """Run all datasets with configured methods

        Returns:
            Dictionary with results for all datasets
        """
        all_datasets = self.config.datasets

        # Create results directory
        os.makedirs(self.results_base, exist_ok=True)

        results = {}
        for dataset in all_datasets:
            dataset_results = self.run_for_dataset(dataset)
            results[dataset] = dataset_results

            # Save intermediate results
            jbl.dump(results, os.path.join(self.results_base, "results.jbl"))

        self._write_results_updated_sentinel()

        return results

    def _write_results_updated_sentinel(self):
        """Write a sentinel file so the Streamlit dashboard can detect that
        consolidated Parquet files may be stale."""
        from datetime import datetime, timezone
        sentinel_path = os.path.join(self.results_base, "_results_updated")
        try:
            with open(sentinel_path, "w") as f:
                f.write(datetime.now(timezone.utc).isoformat() + "\n")
        except OSError:
            pass
