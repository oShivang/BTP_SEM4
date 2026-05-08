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

import numpy as np
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import Scorer
from sal.utils.score import aggregate_scores
from transformers.generation.stopping_criteria import StopStringCriteria
from transformers import AutoTokenizer


def convert_to_chat_template(problem_str, partial_completion: None, config: Config, tokenizer: AutoTokenizer):
    if partial_completion is None:
        conv = [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": problem_str}
        ]
        return tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    else:
        conv = [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": problem_str},
            {"role": "assistant", "content": partial_completion}
        ]
        return tokenizer.apply_chat_template(conv, tokenize=False, continue_final_message=True)
        


def smart_best_of_n(x, config: Config, slm: LLM, scorer: Scorer, llm: None):
    tokenizer = slm.get_tokenizer()
    tokenizer.padding_side = "left"
    # TODO: set the augmented template from a file
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template
        
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        n=1,  # Since we've already duplicated the prompt_token_ids, we only need to generate 1 completion per prompt
        logprobs=1, # We need logprobs for TLC step scores
    )
    
    templated_convs = [convert_to_chat_template(problem, None, config, tokenizer) for problem in x["problem"]]

    # Duplicate convs to generate config.n completions per prompt so we can do continous batching
    # This makes [p1, p2, p3, p4] become [p1, p1, p2, p2, p3, p3, p4, p4] for e.g. config.n=2
    templated_convs = [c for conv in templated_convs for c in [conv] * config.n]

    # Initialize empty lists for completions and completion tokens
    completions = [[] for _ in range(len(x["problem"]))]
    completion_tokens = [[] for _ in range(len(x["problem"]))]
    
    responses = slm.generate(
        templated_convs,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    print("Draft model finished generating initial draft completions")
    
    if len(responses) != len(x["problem"]) * config.n:
        raise ValueError(
            f"Generated {len(responses)} responses instead of {len(x['problem'] * config.n)}"
        )

    for i in range(len(completions)):
        completions[i] = [
            output.text
            for r in responses[i * config.n : (i + 1) * config.n]
            for output in r.outputs
        ]
        completion_tokens[i] = [
            len(output.token_ids)
            for r in responses[i * config.n : (i + 1) * config.n]
            for output in r.outputs
        ]

    # Check we generated the correct number of completions for each prompt
    for c in completions:
        if len(c) != config.n:
            raise ValueError(f"Generated {len(c)} completions instead of {config.n}")

    stopping_criteria = StopStringCriteria(stop_strings="\n\n", tokenizer=tokenizer)
    # while True:
    
    # Calculate step scores using TLC scorer
    scores = []
    for i in range(len(x["problem"])):
        candidate_scores = []
        for r in responses[i * config.n : (i + 1) * config.n]:
            candidate_scores.append(scorer.step_score(r.outputs[0]))
        scores.append(candidate_scores)
        
    # Find all steps that need fixing
    fixes_needed = []
    for problem_idx, candidate_scores in enumerate(scores):
        for candidate_idx, score in enumerate(candidate_scores):
            for step_idx, step_score in enumerate(score):
                if step_score < config.scoring_threshold:
                    fixes_needed.append((problem_idx, candidate_idx, step_idx))
                    break # Only fix first failing step per candidate
    
    if fixes_needed:
        print(f"Fixing {len(fixes_needed)} draft completions")
        
        # Prepare batched inputs for large LLM
        llm_inputs = []
        for problem_idx, candidate_idx, step_idx in fixes_needed:
            completion_steps = completions[problem_idx][candidate_idx].split('\n\n')
            valid_steps = completion_steps[:step_idx]
            partial_completion = '\n\n'.join(valid_steps)
            if partial_completion:
                partial_completion += '\n\n'
            templated_conv = convert_to_chat_template(x["problem"][problem_idx], partial_completion, config, tokenizer)
            llm_inputs.append(templated_conv)
            
        # Batch generate next steps with large LLM
        if hasattr(llm, "generate_batch"):
            messages_batch = []
            for problem_idx, candidate_idx, step_idx in fixes_needed:
                completion_steps = completions[problem_idx][candidate_idx].split('\n\n')
                valid_steps = completion_steps[:step_idx]
                partial_completion = '\n\n'.join(valid_steps)
                if partial_completion:
                    partial_completion += '\n\n'
                
                messages = [
                    {"role": "system", "content": config.system_prompt},
                    {"role": "user", "content": x["problem"][problem_idx]},
                ]
                if partial_completion:
                    messages.append({"role": "assistant", "content": partial_completion})
                messages_batch.append(messages)
                
            new_steps = llm.generate_batch(
                messages_batch,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
                stop=["\n\n"]
            )
        else:
            input_ids = tokenizer(llm_inputs, return_tensors="pt", padding=True).to(llm.device)
            new_ids = llm.generate(
                input_ids["input_ids"],
                attention_mask=input_ids["attention_mask"],
                temperature=config.temperature,
                top_p=config.top_p,
                stopping_criteria=[stopping_criteria],
                max_new_tokens=config.max_tokens,
            )[:, input_ids["input_ids"].shape[1]:]
            new_steps = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        
        # Prepare batched inputs for small LLM
        slm_inputs = []
        for i, (problem_idx, candidate_idx, step_idx) in enumerate(fixes_needed):
            completion_steps = completions[problem_idx][candidate_idx].split('\n\n')
            valid_steps = completion_steps[:step_idx]
            partial_completion = '\n\n'.join(valid_steps)
            if partial_completion:
                partial_completion += '\n\n'
            new_partial = partial_completion + new_steps[i]
            if not new_partial.endswith('\n\n'):
                new_partial += '\n\n'
            templated_conv = convert_to_chat_template(x["problem"][problem_idx], new_partial, config, tokenizer)
            slm_inputs.append(templated_conv)
            
        # Batch generate remaining steps with small LLM
        responses = slm.generate(
            slm_inputs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        
        # Update all completions
        for i, (problem_idx, candidate_idx, step_idx) in enumerate(fixes_needed):
            new_partial = slm_inputs[i].split(x["problem"][problem_idx])[-1].strip()
            remaining_steps = responses[i].outputs[0].text
            completions[problem_idx][candidate_idx] = new_partial + remaining_steps
            
            # Reconstruct scores
            old_scores = scores[problem_idx][candidate_idx]
            trusted_score = 1.0 # Groq API doesn't return logprobs, so we trust it
            new_step_scores = scorer.step_score(responses[i].outputs[0])
            scores[problem_idx][candidate_idx] = old_scores[:step_idx] + [trusted_score] + new_step_scores

    # Get sequence-level scores by aggregating step scores, or use sequence likelihood
    # TLC aggregate is the mean of step probabilities. 
    # Let's just use the product or mean. The original used aggregate_scores(s).
    # Since TLC is sequence level in best_of_n, we'll calculate the aggregate here:
    agg_scores = [
        [aggregate_scores(s, config.agg_strategy) for s in score] for score in scores
    ]

    # Select the completion with the highest score
    pred = [completion[np.argmax(s)] for completion, s in zip(completions, agg_scores)]

    x["completions"] = completions
    x["pred"] = pred

    return x
