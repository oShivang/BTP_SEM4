#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import random
import numpy as np

import torch
from vllm import LLM
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM

from sal.config import Config
from sal.models.reward_models import ScorerRegistry
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score
from sal.search import (
    best_of_n,
    smart_best_of_n,
    beam_search,
    smart_beam_search,
)
from datasets import Dataset

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Approach dispatch table
# ---------------------------------------------------------------------------
# Maps the computed approach_name string → the search function to call.
# Add new approaches here as the project grows.
APPROACHES = {
    "best_of_n":        best_of_n,
    "best_of_n_smart":  smart_best_of_n,
    "beam_search":      beam_search,
    "beam_search_smart": smart_beam_search,
}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


def _build_groq_llm(model_path: str):
    """
    Instantiate a GroqClient when model_path has the 'groq:<model>' prefix.

    Example model_path values:
        groq:llama-3.1-8b-instant
        groq:llama-3.3-70b-versatile
    """
    from sal.utils.groq_client import GroqClient
    model_name = model_path.split("groq:", 1)[1].strip()
    if not model_name:
        model_name = "llama-3.1-8b-instant"
    logger.info(f"Using Groq instructor LLM: {model_name}")
    return GroqClient(model=model_name)


def _build_hf_llm(model_path: str):
    """Load a local/HF instructor LLM via HuggingFace transformers."""
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()


def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    num_gpus = torch.cuda.device_count()
    print("=" * 20)
    print(f"Available GPUs: {num_gpus}")

    # ------------------------------------------------------------------
    # Resolve approach name
    # "smart" suffix → SMART pipeline (SLM + instructor LLM)
    # no suffix     → baseline pipeline (single LLM only)
    # ------------------------------------------------------------------
    approach_suffix = "_smart" if config.smart_search else ""
    approach_name = config.approach + approach_suffix

    if approach_name not in APPROACHES:
        raise ValueError(
            f"Unknown approach '{approach_name}'. "
            f"Available: {list(APPROACHES.keys())}"
        )
    approach_fn = APPROACHES[approach_name]

    # ------------------------------------------------------------------
    # Load scorer via the ScorerRegistry strategy pattern.
    # New scoring methods can be added by registering them in
    # reward_models.py — no changes needed here.
    # ------------------------------------------------------------------
    scorer = ScorerRegistry.get(config.score_method, config)

    print(
        f"\nSearch mode : {'SMART' if config.smart_search else 'Baseline'}\n"
        f"Score method: {config.score_method} (scorer: {type(scorer).__name__})\n"
        f"Approach    : {approach_name}\n"
        f"N           : {config.n}\n"
        f"Beam width  : {config.beam_width}\n"
        + ("=" * 20)
    )

    dataset = get_dataset(config)

    if config.smart_search:
        # ------------------------------------------------------------------
        # SMART pipeline: SLM generates drafts; instructor LLM fixes low-
        # confidence steps identified by the TLC scorer.
        # ------------------------------------------------------------------
        mp.set_start_method("spawn", force=True)

        slm = LLM(
            model=config.draft_model_path,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_prefix_caching=True,
            seed=config.seed,
            tensor_parallel_size=num_gpus,
            max_model_len=2048,
        )

        # Instructor LLM: either a Groq API client or a local HF model.
        if config.model_path.startswith("groq:"):
            llm = _build_groq_llm(config.model_path)
        else:
            llm = _build_hf_llm(config.model_path)

        dataset = dataset.map(
            approach_fn,
            batched=True,
            batch_size=config.search_batch_size,
            fn_kwargs={"config": config, "slm": slm, "scorer": scorer, "llm": llm},
            desc="Running SMART search",
            load_from_cache_file=False,
        )

    else:
        # ------------------------------------------------------------------
        # Baseline pipeline: single vLLM model, TLC scoring via logprobs.
        # ------------------------------------------------------------------
        llm = LLM(
            model=config.model_path,
            revision="main",
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_prefix_caching=True,
            seed=config.seed,
            tensor_parallel_size=num_gpus,
            max_model_len=2048,
        )

        dataset = dataset.map(
            approach_fn,
            batched=True,
            batch_size=config.search_batch_size,
            fn_kwargs={"config": config, "llm": llm, "scorer": scorer},
            desc="Running baseline search",
            load_from_cache_file=False,
        )

    # ------------------------------------------------------------------
    # Post-processing: aggregate scores → evaluate → save
    # ------------------------------------------------------------------
    dataset = score(dataset, config)
    save_dataset(dataset, config)

    import sys
    sys.path.append("src/evaluation")
    from evaluation.evaluate import evaluate

    if config.approach in ("best_of_n", "beam_search"):
        subsets = [2 ** i for i in range(config.n) if 2 ** i <= config.n]
        keys = []
        for n in subsets:
            keys.extend([f"pred_weighted@{n}", f"pred_maj@{n}", f"pred_naive@{n}"])
    else:
        keys = ["pred"]

    dataset, result = evaluate(
        data_name="math", prompt_type=None, samples=dataset, pred_keys=keys
    )
    dataset = Dataset.from_list(
        [{k: v for k, v in dict(sample).items() if k != "pred_completions"}
         for sample in dataset]
    )

    save_dataset(dataset, config)
    logger.info(result)
    logger.info("Done 🔥!")


if __name__ == "__main__":
    main()
