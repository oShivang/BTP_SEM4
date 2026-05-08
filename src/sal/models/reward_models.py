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

from typing import List, Dict, Any
from sal.config import Config
from sal.utils.score import calculate_step_confidence_scores

class Scorer:
    """Base class for all scoring methods."""
    def __init__(self, config: Config, **kwargs):
        self.config = config

    def score(self, questions: List[str], completions: List[List[str]], **kwargs) -> List[List[float]]:
        raise NotImplementedError

    def step_score(self, output: Any) -> List[float]:
        """
        Calculate step-level scores. Defaults to NotImplemented.
        """
        raise NotImplementedError

class ScorerRegistry:
    """Registry to load scoring methods via configuration."""
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(scorer_class):
            cls._registry[name] = scorer_class
            return scorer_class
        return wrapper

    @classmethod
    def get(cls, name: str, config: Config, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Scorer '{name}' not found. Available scorers: {list(cls._registry.keys())}")
        return cls._registry[name](config, **kwargs)

@ScorerRegistry.register("tlc")
class TLCScorer(Scorer):
    """
    Tree Level Compute (TLC) Scorer that uses generation probabilities to score.
    """
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, **kwargs)
        self.threshold = getattr(config, 'scoring_threshold', 0.9)

    def score(self, questions: List[str], completions: List[List[str]], **kwargs) -> List[List[float]]:
        # TLC natively operates on logprobs rather than raw text outputs,
        # so sequence-level TLC scores are calculated dynamically during generation
        # (e.g. inside beam_search_conf or best_of_n_conf).
        # We return a dummy empty list here as fallback.
        return []

    def step_score(self, output: Any) -> List[float]:
        """
        Compute step-level confidence scores using the probability mean.
        Requires the vllm output to have `logprobs` populated.
        """
        if not hasattr(output, 'logprobs') or output.logprobs is None:
            raise ValueError("TLC scoring requires logprobs to be enabled during generation.")
        return calculate_step_confidence_scores(output.logprobs)
