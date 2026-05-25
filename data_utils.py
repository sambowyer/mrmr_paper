import functools
import json
import os
from abc import ABC, abstractmethod
from collections import defaultdict

import joblib
import joblib as jbl
import numpy as np
import pandas as pd
import requests
from datasets import Dataset
from datasets import load_dataset

from zarth_utils.logger import logging_info
from zarth_utils.general_utils import makedir_if_not_exist

dir_data_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(dir_data_base, exist_ok=True)
dir_data_raw = os.path.join(dir_data_base, "raw")
os.makedirs(dir_data_raw, exist_ok=True)

helm_datasets = [
    "commonsense", "gsm", "legalbench",
    "math", "med_qa", "mmlu",
    # "narrative_qa", "natural_qa", "wmt_14", # not binary accuracy
]
glue_datasets = [
    # "cola", # not binary accuracy
    "mrpc", "rte", "sst2", "mnli", "qnli",
    # "qqp", # this dataset always throws an error when loading
]
openllm_datasets = [
    "ifeval",
    "openllm_math",
    "mmlu_pro", "arc_challenge",
    "bbh", "gpqa", "musr"
]
continuous_cat_datasets = [
    "biolaysumm_rougel", "biolaysumm_bertscore", "biolaysumm_fkgl",
    "govreport_rougel", "govreport_bertscore",
    "truthfulqa_judge", "truthfulqa_bertscore",
    "flores_bleu", "flores_comet",
    "nemotron_pii",
]
continuous_cat_main_datasets = [
    "biolaysumm_rougel", "biolaysumm_bertscore", "biolaysumm_fkgl",
    "govreport_rougel", "govreport_bertscore",
    "truthfulqa_judge",
    "nemotron_pii",
]
continuous_cat_extra_datasets = [
    "truthfulqa_bertscore",
    "flores_bleu", "flores_comet",
]

pass_at_k_code_benchmarks = [
    "MBPP_mbpp", "MBPPPlus_mbpp_plus",
    "HumanEvalPack_Rust", "HumanEvalPack_Java", "HumanEvalPack_Js",
    "HumanEvalPack_Go", "HumanEvalPack_Cpp", "HumanEvalPack_PythonPlus",
    "LBPP_Cpp", "LBPP_Java", "LBPP_Js",
    "LBPP_Go", "LBPP_Python", "LBPP_Rust",
]
pass_at_k_values = [1, 2, 4, 8, 16, 32, 64, 128]
pass_at_k_code_datasets = [
    f"{bench}_pass_at_{k}"
    for bench in pass_at_k_code_benchmarks
    for k in pass_at_k_values
    if os.path.exists(os.path.join(dir_data_base, "scores", f"{bench}_pass_at_{k}.jbl"))
]

pass_at_k_code_v2_datasets = [
    f"{bench}_pass_at_{k}_v2"
    for bench in pass_at_k_code_benchmarks
    for k in pass_at_k_values
    if os.path.exists(os.path.join(dir_data_base, "scores", f"{bench}_pass_at_{k}_v2.jbl"))
]

pass_at_k_v3_benchmarks = [
    "LBPP_Python", "HumanEvalPack_PythonPlus",
    "MBPPPlus_mbpp_plus", "MBPP_mbpp",
]
dir_pass_at_k_v3_full = os.path.join(dir_data_base, "pass_at_k_code_v3", "scores", "full")
_dir_pass_at_k_v3_open_primary = os.path.join(
    dir_data_base, "pass_at_k_code_v3", "scores", "open"
)
_dir_pass_at_k_v3_open_fallback = os.path.join(
    dir_data_base, "pass_at_k_code_v3", "open"
)
dir_pass_at_k_v3_open = (
    _dir_pass_at_k_v3_open_primary
    if os.path.exists(_dir_pass_at_k_v3_open_primary)
    else _dir_pass_at_k_v3_open_fallback
)

pass_at_k_full_datasets = [
    f"{bench}_pass_at_{k}_v3"
    for bench in pass_at_k_v3_benchmarks
    for k in pass_at_k_values
    if os.path.exists(os.path.join(dir_pass_at_k_v3_full, f"{bench}_pass_at_{k}_v3.jbl"))
]
pass_at_k_open_datasets = [
    f"{bench}_pass_at_{k}_v3_open"
    for bench in pass_at_k_v3_benchmarks
    for k in pass_at_k_values
    if os.path.exists(os.path.join(dir_pass_at_k_v3_open, f"{bench}_pass_at_{k}_v3.jbl"))
]

_all_pass_at_k_datasets = (
    pass_at_k_code_datasets + pass_at_k_code_v2_datasets
    + pass_at_k_full_datasets + pass_at_k_open_datasets
)

all_datasets = (
    helm_datasets + glue_datasets + openllm_datasets
    + continuous_cat_datasets + _all_pass_at_k_datasets + ["imagenet"]
)


def parse_pass_at_k_dataset(dataset_name):
    """Parse a pass@k dataset name into (benchmark, k).

    Returns ``(benchmark, k)`` if *dataset_name* matches
    ``{benchmark}_pass_at_{k}``, ``{benchmark}_pass_at_{k}_v2``,
    ``{benchmark}_pass_at_{k}_v3``, or ``{benchmark}_pass_at_{k}_v3_open``
    for a known benchmark, otherwise ``None``.
    """
    name = dataset_name
    for sfx in ("_v3_open", "_v3", "_v2"):
        if name.endswith(sfx):
            name = name[: -len(sfx)]
            break

    _suffix = "_pass_at_"
    all_benchmarks = list(dict.fromkeys(pass_at_k_code_benchmarks + pass_at_k_v3_benchmarks))
    for bench in all_benchmarks:
        prefix = bench + _suffix
        if name.startswith(prefix):
            try:
                k = int(name[len(prefix):])
            except ValueError:
                continue
            return bench, k
    return None


def get_available_k_values(benchmark, exclude_k=None, v2=False,
                           scores_dir=None, name_suffix=None):
    """Return sorted list of k values that have score files on disk.

    If *exclude_k* is given it is omitted from the result.
    *scores_dir* and *name_suffix* override the directory and file suffix
    used to locate score files.  When omitted the legacy ``v2`` flag is
    used (``data/scores/`` with ``_v2`` or no suffix).
    """
    if scores_dir is None:
        scores_dir = os.path.join(dir_data_base, "scores")
    if name_suffix is None:
        name_suffix = "_v2" if v2 else ""
    available = []
    for k in pass_at_k_values:
        if k == exclude_k:
            continue
        if os.path.exists(os.path.join(scores_dir, f"{benchmark}_pass_at_{k}{name_suffix}.jbl")):
            available.append(k)
    return sorted(available)


def get_pass_at_k_paths(dataset_name):
    """Return ``(scores_dir, name_suffix)`` for a pass@k dataset name.

    Returns ``None`` when *dataset_name* is not a recognised pass@k dataset.
    """
    if dataset_name.endswith("_v3_open"):
        return dir_pass_at_k_v3_open, "_v3"
    if dataset_name.endswith("_v3"):
        return dir_pass_at_k_v3_full, "_v3"
    if dataset_name.endswith("_v2"):
        return os.path.join(dir_data_base, "scores"), "_v2"
    if parse_pass_at_k_dataset(dataset_name) is not None:
        return os.path.join(dir_data_base, "scores"), ""
    return None


def get_scores(dataset_name):
    dir_scores = os.path.join(dir_data_base, "scores")
    os.makedirs(dir_scores, exist_ok=True)
    path_scores = os.path.join(dir_scores, "%s.jbl" % dataset_name)

    if os.path.exists(path_scores):
        scores = jbl.load(path_scores)
    else:
        if dataset_name in helm_datasets:
            helm = HelmLite(tasks=[dataset_name])
            helm.download_and_check()
            dataset = helm.get_datasets()
            scores = np.array(dataset["acc"]).T
            jbl.dump(helm.models, os.path.join(dir_scores, "%s_models.jbl" % dataset_name))

        elif dataset_name in openllm_datasets:
            openllm = OpenLLM(tasks=[dataset_name])
            openllm.download_and_check()
            dataset = openllm.get_datasets()
            scores = np.array(dataset["acc"]).T
            jbl.dump(openllm.models, os.path.join(dir_scores, "%s_models.jbl" % dataset_name))

        elif dataset_name in glue_datasets:
            all_preds, gold_labels = load_glue_predictions(dataset_name)
            scores = np.zeros([all_preds.shape[0], all_preds.shape[1]])
            for m in range(all_preds.shape[0]):
                scores[m] = (all_preds[m].argmax(-1) == gold_labels)

        elif dataset_name in continuous_cat_datasets:
            scores, model_names = load_continuous_cat_scores(dataset_name)
            jbl.dump(model_names, os.path.join(dir_scores, "%s_models.jbl" % dataset_name))

        elif dataset_name in _all_pass_at_k_datasets:
            raise FileNotFoundError(
                f"Score file for {dataset_name} not found. "
                "Run preprocess_pass_at_k_code_data.py first."
            )

        else:
            raise NotImplementedError

        jbl.dump(scores, path_scores)

    if dataset_name == "imagenet":
        real_acc = scores.mean(-1)
        scores = scores[real_acc > 0.3]

    scores = scores.astype(np.float32)

    if dataset_name not in continuous_cat_datasets + _all_pass_at_k_datasets:
        assert len(np.unique(scores)) == 2

    real_acc = scores.mean(-1)
    scores = scores[real_acc > 1e-8]

    return scores


def load_glue_predictions(task="cola", model_family=None):
    all_tasks = ["cola", "mnli", "mrpc", "qnli", "rte", "sst2"]
    assert task in all_tasks
    all_model_family = ["BERT", "GPT", "INSTRUCT_GPT", "OPENAI"]
    if model_family is None:
        model_family = all_model_family
    elif isinstance(model_family, str):
        model_family = [model_family]
    else:
        assert isinstance(model_family, list)
    if task == "mnli":
        gold_label = np.array(load_dataset("SetFit/mnli", split="validation")["label"])
    else:
        gold_label = np.array(load_dataset("glue", task, split="validation")["label"])

    all_pred = []
    for f in model_family:
        assert f in all_model_family
        pred_f = np.load(
            os.path.join(dir_data_raw, "model_preds_by_family_glue", f, "%s_preds.npy" % task),
            allow_pickle=True
        )
        all_pred.append(pred_f)
    all_pred = np.concatenate(all_pred, axis=0)
    assert all_pred.shape[1] == len(gold_label), (all_pred.shape, len(gold_label))
    return all_pred, gold_label


_continuous_cat_json_paths = [
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "continuous-cat", "data", "scores.json",
    ),
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data", "continuous_cat_scores", "scores.json",
    ),
]


@functools.lru_cache(maxsize=1)
def _load_continuous_cat_json():
    for path in _continuous_cat_json_paths:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError(
        "Could not find continuous_cat scores.json in any known location: "
        f"{_continuous_cat_json_paths}"
    )


def load_continuous_cat_scores(dataset_name):
    """Load scores from continuous-cat, treating each (model, temp) pair as a
    separate model.

    Returns:
        (scores, model_names) where scores is a float32 array of shape
        (n_model_temp_combos, n_items) and model_names is a list of strings
        like ``"gemini-2.0-flash__temp_0.0"``.
    """
    data = _load_continuous_cat_json()
    ds = data["datasets"][dataset_name]
    item_ids = sorted(ds["items"].keys(), key=int)
    n_items = len(item_ids)

    # Deterministic ordering: models sorted, then temps sorted within each
    base_models = sorted(ds["scores"].keys())
    temps = sorted(next(iter(ds["scores"].values())).keys())

    model_names = []
    scores_rows = []
    for base_model in base_models:
        for temp in temps:
            temp_scores = ds["scores"][base_model][temp]
            row = np.array(
                [temp_scores.get(str(item_id), np.nan) for item_id in item_ids],
                dtype=np.float32,
            )
            if np.any(np.isnan(row)):
                continue
            
            # if loading flores_bleu, we need to convert from percentage to 0-1 scale
            if dataset_name == "flores_bleu":
                row = row / 100.0

            model_names.append(f"{base_model}__{temp}")
            scores_rows.append(row)

    scores = np.stack(scores_rows)
    assert scores.shape == (len(model_names), n_items)
    return scores, model_names


class Benchmark(ABC):
    def __init__(self):
        self.dir_data = None
        self.models = None
        self.tasks = None

    def download_and_check(self):
        all_task_texts = dict()
        all_task_labels = dict()
        all_task_model_perf = defaultdict(dict)
        for task in self.tasks:
            for i, model in enumerate(self.models):
                try:
                    meta_info = self.get_meta_infos(model, task)
                except Exception as e:
                    self.models = [m for m in self.models if m != model]
                    logging_info(str(e))
                    logging_info("Model %s not found for task %s" % (model, task))
                    logging_info("Remove model %s" % model)
                    continue
                all_task_texts[task] = meta_info["text"]
                all_task_labels[task] = meta_info["label"]
                all_task_model_perf[task][model] = meta_info["acc"]

        for task_name in self.tasks:
            task_texts = all_task_texts[task_name]
            for model_name in self.models:
                for idx in task_texts.index:
                    if idx not in all_task_model_perf[task_name][model_name].index:
                        self.models = [model for model in self.models if model != model_name]
                        logging_info("Remove model %s" % model_name)
                        break

    @abstractmethod
    def get_meta_infos(self, model_name, task_name, *args, **kwargs) -> pd.DataFrame:
        pass

    def get_texts(self, task_name) -> pd.Series:
        return self.get_meta_infos(self.models[0], task_name)["text"]

    def get_labels(self, task_name) -> pd.Series:
        return self.get_meta_infos(self.models[0], task_name)["label"]

    def get_accuracies(self, model_name, task_name) -> pd.Series:
        return self.get_meta_infos(model_name, task_name)["acc"]

    def get_datasets(self) -> Dataset:
        all_task_texts = dict()
        all_task_labels = dict()
        all_task_model_perf = defaultdict(dict)
        for task in self.tasks:
            for i, model in enumerate(self.models):
                try:
                    meta_info = self.get_meta_infos(model, task)
                except FileNotFoundError:
                    logging_info("Model %s not found for task %s" % (model, task))
                    continue
                all_task_texts[task] = meta_info["text"]
                all_task_labels[task] = meta_info["label"]
                all_task_model_perf[task][model] = meta_info["acc"]

        ret_texts, ret_perfs, ret_labels, ret_tasks = [], [], [], []
        for task_name in self.tasks:
            task_texts = all_task_texts[task_name]
            task_labels = all_task_labels[task_name]

            ret_texts += list(task_texts.values)
            ret_labels += list(task_labels.values)
            ret_tasks += [task_name] * len(task_texts)
            ret_perfs += [
                [
                    all_task_model_perf[task_name][model_name].loc[idx]
                    for model_name in self.models
                ]
                for idx in task_texts.index
            ]

        ret_dataset = Dataset.from_dict(
            {
                "text": ret_texts,
                "acc": ret_perfs,
                "label": ret_labels,
                "task": ret_tasks,
            }
        )

        return ret_dataset

    def get_leaderboard_all(self):
        leaderboard = {task: np.zeros(len(self.models)) for task in self.tasks}
        task_size = defaultdict(int)
        for task_name in self.tasks:
            for i, model_name in enumerate(self.models):
                acc = self.get_accuracies(model_name, task_name).values
                leaderboard[task_name][i] = acc.mean()
                task_size[task_name] = len(acc)

        leaderboard = pd.DataFrame(leaderboard, index=self.models)
        leaderboard["mean-task"] = leaderboard.mean(axis=1)
        mean_sample = []
        for model in self.models:
            tot, n = 0, 0
            for task in self.tasks:
                tot += leaderboard.loc[model][task] * task_size[task]
                n += task_size[task]
            mean_sample.append(tot / n)
        leaderboard["mean-sample"] = mean_sample

        return leaderboard


def download_json(url, filename):
    response = requests.get(url)

    if response.status_code == 200:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        data = response.json()

        with open(filename, "w") as file:
            json.dump(data, file, indent=4)
        return True
    else:
        return False


class HelmLite(Benchmark):
    def __init__(self, models=None, tasks=None):
        super().__init__()
        self.dir_data = os.path.join(dir_data_raw, "helm_lite")
        if models is None:
            self.models = [
                "01-ai_yi-34b",
                "01-ai_yi-6b",
                "AlephAlpha_luminous-base",
                "AlephAlpha_luminous-extended",
                "AlephAlpha_luminous-supreme",
                "ai21_j2-grande",
                "ai21_j2-jumbo",
                "anthropic_claude-2.0",
                "anthropic_claude-2.1",
                "anthropic_claude-instant-1.2",
                "anthropic_claude-v1.3",
                "cohere_command",
                "cohere_command-light",
                "google_text-bison@001",
                "google_text-unicorn@001",
                "meta_llama-2-13b",
                "meta_llama-2-70b",
                "meta_llama-2-7b",
                "meta_llama-65b",
                "mistralai_mistral-7b-v0.1",
                "mistralai_mixtral-8x7b-32kseqlen",
                "openai_gpt-3.5-turbo-0613",
                "openai_gpt-4-0613",
                "openai_gpt-4-1106-preview",
                "openai_text-davinci-002",
                "openai_text-davinci-003",
                "tiiuae_falcon-40b",
                "tiiuae_falcon-7b",
                "writer_palmyra-x-v2",
                "writer_palmyra-x-v3",
                "microsoft_phi-2",
                "mistralai_mistral-medium-2312",
                "allenai_olmo-7b",
                "google_gemma-7b",
                "meta_llama-3-70b",
                "meta_llama-3-8b",
                "mistralai_mixtral-8x22b",
                "qwen_qwen1.5-14b",
                "qwen_qwen1.5-32b",
                "qwen_qwen1.5-72b",
                "qwen_qwen1.5-7b",
                "anthropic_claude-3-haiku-20240307",
                "anthropic_claude-3-opus-20240229",
                "anthropic_claude-3-sonnet-20240229",
                "databricks_dbrx-instruct",
                "deepseek-ai_deepseek-llm-67b-chat",
                "mistralai_mistral-large-2402",
                "mistralai_mistral-small-2402",
                "cohere_command-r",
                "cohere_command-r-plus",
                "mistralai_mistral-7b-instruct-v0.3",
                "openai_gpt-4-turbo-2024-04-09",
                "openai_gpt-4o-2024-05-13",
                "qwen_qwen1.5-110b-chat",
                "qwen_qwen2-72b-instruct",
                "snowflake_snowflake-arctic-instruct",
                "01-ai_yi-large-preview",
                "anthropic_claude-3-5-sonnet-20240620",
                "google_gemini-1.0-pro-002",
                "google_gemini-1.5-flash-001",
                "google_gemini-1.5-pro-001",
                "ai21_jamba-instruct",
                "google_gemma-2-27b-it",
                "google_gemma-2-9b-it",
                "microsoft_phi-3-medium-4k-instruct",
                "mistralai_mistral-large-2407",
                "mistralai_open-mistral-nemo-2407",
                "openai_gpt-4o-mini-2024-07-18",
                "meta_llama-3.1-405b-instruct-turbo",
                "meta_llama-3.1-70b-instruct-turbo",
                "meta_llama-3.1-8b-instruct-turbo",
                "ai21_jamba-1.5-large",
                "ai21_jamba-1.5-mini",
                "meta_llama-3.2-11b-vision-instruct-turbo",
                "meta_llama-3.2-90b-vision-instruct-turbo",
                "qwen_qwen2.5-72b-instruct-turbo",
                "qwen_qwen2.5-7b-instruct-turbo",
                "anthropic_claude-3-5-sonnet-20241022",
                "google_gemini-1.5-flash-002",
                "google_gemini-1.5-pro-002",
                "openai_gpt-4o-2024-08-06",
                "meta_llama-3.3-70b-instruct-turbo",
                "upstage_solar-pro-241126",
            ]
        else:
            self.models = models
        if tasks is None:
            self.tasks = [
                "commonsense",
                "gsm",
                "legalbench",
                "math",
                "med_qa",
                "mmlu",
                "narrative_qa",
                "natural_qa",
                "wmt_14",
            ]
        else:
            self.tasks = tasks
        self.dir_predictions = os.path.join(self.dir_data, "predictions")

    def download(self, model_name, task_name, json_name):
        task2metric = {
            "narrative_qa": "f1_score",
            "natural_qa": "f1_score",
            "commonsense": "exact_match",
            "mmlu": "exact_match",
            "math": "math_equiv_chain_of_thought",
            "gsm": "final_number_exact_match",
            "legalbench": "quasi_exact_match",
            "med_qa": "quasi_exact_match",
            "wmt_14": "bleu_4",
        }
        if ":" in task_name and task_name.split(":")[0] in task2metric.keys():
            subtasks = [task_name]
            task_name = task_name.split(":")[0]
        else:
            assert task_name in task2metric.keys()
            subtasks = {
                "commonsense": [
                    "commonsense:dataset=openbookqa,method=multiple_choice_joint,"
                ],
                "gsm": [
                    "gsm:",
                ],
                "legalbench": [
                    "legalbench:subset=abercrombie,",
                    "legalbench:subset=corporate_lobbying,",
                    "legalbench:subset=function_of_decision_section,",
                    "legalbench:subset=international_citizenship_questions,",
                    "legalbench:subset=proa,",
                ],
                "math": [
                    "math:subject=algebra,level=1,use_official_examples=False,use_chain_of_thought=True,",
                    "math:subject=counting_and_probability,level=1,use_official_examples=False,use_chain_of_thought=True,",
                    "math:subject=geometry,level=1,use_official_examples=False,use_chain_of_thought=True,",
                    "math:subject=intermediate_algebra,level=1,use_official_examples=False,use_chain_of_thought=True,",
                    "math:subject=number_theory,level=1,use_official_examples=False,use_chain_of_thought=True,",
                    "math:subject=prealgebra,level=1,use_official_examples=False,use_chain_of_thought=True,",
                    "math:subject=precalculus,level=1,use_official_examples=False,use_chain_of_thought=True,",
                ],
                "med_qa": ["med_qa:"],
                "mmlu": [
                    "mmlu:subject=abstract_algebra,method=multiple_choice_joint,",
                    "mmlu:subject=college_chemistry,method=multiple_choice_joint,",
                    "mmlu:subject=computer_security,method=multiple_choice_joint,",
                    "mmlu:subject=econometrics,method=multiple_choice_joint,",
                    "mmlu:subject=us_foreign_policy,method=multiple_choice_joint,",
                ],
                "narrative_qa": ["narrative_qa:"],
                "natural_qa": [
                    "natural_qa:mode=closedbook,",
                    "natural_qa:mode=openbook_longans,",
                ],
                "wmt_14": [
                    "wmt_14:language_pair=cs-en,",
                    "wmt_14:language_pair=de-en,",
                    "wmt_14:language_pair=fr-en,",
                    "wmt_14:language_pair=hi-en,",
                    "wmt_14:language_pair=ru-en,",
                ],
            }[task_name]

        for subtask in subtasks:
            file_name = subtask + "model=%s" % model_name
            path_save = os.path.join(self.dir_predictions, file_name, json_name)
            if not os.path.exists(path_save):
                for version in range(20):
                    url = (
                            "https://storage.googleapis.com/crfm-helm-public/lite/benchmark_output/runs/v1.%d.0/%s/%s"
                            % (version, file_name, json_name)
                    )
                    if download_json(url, path_save):
                        break
            if task_name == "gsm" and not os.path.exists(path_save):
                file_name = subtask + "model=%s" % model_name
                path_save = os.path.join(self.dir_predictions, file_name, json_name)
                if not os.path.exists(path_save):
                    for version in range(20):
                        url = (
                                "https://storage.googleapis.com/crfm-helm-public/lite/benchmark_output/runs/v1.%d.0/%s/%s"
                                % (
                                    version,
                                    subtask + "model=%s,stop=none" % model_name,
                                    json_name,
                                )
                        )
                        if download_json(url, path_save):
                            break

        n = 0
        for subtask in subtasks:
            file_name = subtask + "model=%s" % model_name
            path_save = os.path.join(self.dir_predictions, file_name, json_name)
            res = json.load(open(path_save, "r"))
            n += len(res)

        ret = []
        for subtask in subtasks:
            file_name = subtask + "model=%s" % model_name
            path_save = os.path.join(self.dir_predictions, file_name, json_name)
            res = json.load(open(path_save, "r"))
            for i in range(len(res)):
                if "instance_id" in res[i].keys():
                    res[i]["instance_id"] = "%s-%s" % (subtask, res[i]["instance_id"])
                if "id" in res[i].keys():
                    res[i]["id"] = "%s-%s" % (subtask, res[i]["id"])
                metric_name = task2metric[task_name]
                if "stats" in res[i].keys() and metric_name in res[i]["stats"]:
                    res[i]["stats"]["acc"] = (
                            1.0
                            * res[i]["stats"][metric_name]
                        # without following lines, acc will be different from the original aggregated score
                        # * n
                        # / len(res)
                        # / len(subtasks)
                    )
            ret += res
        return ret

    def get_meta_infos(self, model_name, task_name) -> pd.DataFrame:
        dir_save = os.path.join(self.dir_data, "perfs")
        makedir_if_not_exist(dir_save)
        path_save = os.path.join(dir_save, "%s_%s.jbl" % (model_name, task_name))

        if os.path.exists(path_save):
            ret = jbl.load(path_save)
        else:
            instances = self.download(model_name, task_name, "instances.json")
            predictions = self.download(
                model_name, task_name, "display_predictions.json"
            )
            ret = {
                "id": [],
                "text": [],
                "label": [],
                "acc": [],
            }
            for ins, pred in zip(instances, predictions):
                assert ins["id"] == pred["instance_id"]
                idx = "%s-%s" % (task_name, ins["id"])
                ret["id"].append(idx)
                # todo: label is not loaded
                ret["label"].append(-1)
                ret["acc"].append(pred["stats"]["acc"])
                ref_text = " ".join(
                    [
                        "%s. %s" % (chr(i + 65), ref["output"]["text"])
                        for i, ref in enumerate(ins["references"])
                    ]
                )
                ret["text"].append(ins["input"]["text"] + "\n" + ref_text)
            ret = pd.DataFrame(ret)
            ret.set_index("id", inplace=True)
            joblib.dump(ret, path_save)

        return ret


class OpenLLM(Benchmark):
    def __init__(self, models=None, tasks=None):
        super(OpenLLM, self).__init__()
        self.dir_data = os.path.join(dir_data_raw, "openllm")

        if models is None:
            official_providers_dataset = load_dataset("open-llm-leaderboard/official-providers")
            official_providers = official_providers_dataset["train"][0]["CURATED_SET"]
            leaderboard_dataset = load_dataset("open-llm-leaderboard/contents")
            all_models = leaderboard_dataset["train"]["fullname"]
            self.models = [m for m in all_models if m.split("/")[0] in official_providers]
        else:
            self.models = models
        if tasks is None:
            self.tasks = [
                "ifeval",
                "openllm_math",
                "mmlu_pro",
                "arc_challenge",
                "bbh",
                "gpqa",
                "musr"
            ]
        else:
            self.tasks = tasks
        self.dir_predictions = os.path.join(self.dir_data, "predictions")

    def get_meta_infos(self, model_name, task_name) -> pd.DataFrame:
        task2metric = {
            "ifeval": "prompt_level_strict_acc",
            "openllm_math": "exact_match",
            "mmlu_pro": "acc",
            "gpqa": "acc_norm",
            "musr": "acc_norm",
            "bbh": "acc_norm",
            "arc_challenge": "acc_norm",
        }
        task2subtasks = {
            "ifeval": ["ifeval"],
            "openllm_math": [
                'math_geometry_hard',
                'math_intermediate_algebra_hard',
                'math_num_theory_hard',
                'math_prealgebra_hard',
                'math_precalculus_hard'
            ],
            "mmlu_pro": ["mmlu_pro"],
            "arc_challenge": ["arc_challenge"],
            "bbh": [
                "bbh_boolean_expressions",
                "bbh_causal_judgement",
                "bbh_date_understanding",
                "bbh_disambiguation_qa",
                "bbh_formal_fallacies",
                "bbh_geometric_shapes",
                "bbh_hyperbaton",
                "bbh_logical_deduction_five_objects",
                "bbh_logical_deduction_seven_objects",
                "bbh_logical_deduction_three_objects",
                "bbh_movie_recommendation",
                "bbh_navigate",
                "bbh_object_counting",
                "bbh_penguins_in_a_table",
                "bbh_reasoning_about_colored_objects",
                "bbh_ruin_names",
                "bbh_salient_translation_error_detection",
                "bbh_snarks",
                "bbh_sports_understanding",
                "bbh_temporal_sequences",
                "bbh_tracking_shuffled_objects_five_objects",
                "bbh_tracking_shuffled_objects_seven_objects",
                "bbh_tracking_shuffled_objects_three_objects",
                "bbh_web_of_lies"],
            "gpqa": [
                "gpqa_diamond",
                "gpqa_extended",
                "gpqa_main"],
            "musr": [
                "musr_murder_mysteries",
                "musr_object_placements",
                "musr_team_allocation", ]
        }
        model_name = model_name.replace("/", "__")
        ret = {
            "id": [],
            "text": [],
            "label": [],
            "acc": [],
        }
        for subtask in task2subtasks[task_name]:
            data = load_dataset(
                "open-llm-leaderboard/%s-details" % model_name,
                name="%s__leaderboard_%s" % (model_name, subtask),
                split="latest",
            )
            ret["id"] += ["%s-%d" % (subtask, i) for i in data["doc_id"]]
            ret["label"] += data["target"]
            ret["acc"] += data[task2metric[task_name]]
            # todo: text is not loaded
            ret["text"] += ["" for _ in data["doc"]]

        ret = pd.DataFrame(ret)
        ret.set_index("id", inplace=True)

        return ret
