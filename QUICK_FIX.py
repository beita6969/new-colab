#!/usr/bin/env python3
"""
QUICK FIX GUIDE - Copy-paste solutions for each dataset

Replace your existing code with the correct version below.
"""

# ============================================================================
# QUICK FIX #1: MATH Dataset
# ============================================================================
# BEFORE (WRONG):
# ❌ dataset = load_dataset("lighteval/MATH")
# Error: "Dataset 'lighteval/MATH' doesn't exist on the Hub"

# AFTER (CORRECT):
# ✓ dataset = load_dataset("EleutherAI/hendrycks_math")

def fix_math_dataset():
    from datasets import load_dataset
    import json

    # CORRECT: Use EleutherAI/hendrycks_math with specific config
    # The dataset has multiple configs: algebra, counting_and_probability, geometry, etc.
    configs = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']

    total_samples = 0
    with open("math_dataset.jsonl", "w") as f:
        for config in configs:
            try:
                dataset = load_dataset("EleutherAI/hendrycks_math", config)
                for sample in dataset["train"]:
                    record = {
                        "question": sample["problem"],
                        "difficulty": sample["level"],
                        "category": sample["type"],
                        "config": config,
                        "solution": sample["solution"]
                    }
                    f.write(json.dumps(record) + "\n")
                    total_samples += 1
            except Exception as e:
                print(f"  ⚠️  Failed to load config '{config}': {e}")

    print(f"✓ Saved {total_samples} MATH samples to math_dataset.jsonl")


# ============================================================================
# QUICK FIX #2: HotpotQA Dataset
# ============================================================================
# BEFORE (WRONG):
# ❌ dataset = load_dataset("hotpot_qa", "distractor")
# ❌ context = sample["context"]  # This is a dict, not a string!
# Error: "string indices must be integers"

# AFTER (CORRECT):
# ✓ dataset = load_dataset("hotpotqa/hotpot_qa", "distractor")
# ✓ context = sample["context"]  # Dict with "title" and "sentences"
# ✓ context["sentences"]  # Access nested 2D array

def fix_hotpotqa_dataset():
    from datasets import load_dataset
    import json

    # CORRECT: Use hotpotqa/hotpot_qa with "distractor" config
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor")

    # Save to JSONL
    with open("hotpotqa_dataset.jsonl", "w") as f:
        for sample in dataset["train"]:
            # Extract context properly - it's a dict with nested structure
            context = sample["context"]
            context_text = ""
            if context and "sentences" in context:
                # sentences is a 2D array - flatten it
                context_text = " ".join([
                    " ".join(sent_list) for sent_list in context["sentences"]
                ])

            record = {
                "question": sample["question"],
                "answer": sample["answer"],
                "type": sample["type"],
                "level": sample["level"],
                "context": context_text,
                "id": sample["id"]
            }
            f.write(json.dumps(record) + "\n")

    print(f"✓ Saved {len(dataset['train'])} HotpotQA samples to hotpotqa_dataset.jsonl")


# ============================================================================
# QUICK FIX #3: DROP Dataset
# ============================================================================
# BEFORE (WRONG):
# ❌ answer = sample["answers_spans"]  # This is a dict, not a string!
# Error: "string indices must be integers"

# AFTER (CORRECT):
# ✓ answers_spans = sample["answers_spans"]  # Dict with "spans" and "types"
# ✓ answer = answers_spans["spans"][0]  # Access the actual answer text

def fix_drop_dataset():
    from datasets import load_dataset
    import json

    # CORRECT: Use ucinlp/drop
    dataset = load_dataset("ucinlp/drop")

    # Save to JSONL
    with open("drop_dataset.jsonl", "w") as f:
        for sample in dataset["train"]:
            # Extract answers properly - answers_spans is a dict
            answers_spans = sample["answers_spans"]
            answer_text = answers_spans["spans"][0] if answers_spans["spans"] else ""
            answer_type = answers_spans["types"][0] if answers_spans["types"] else ""

            record = {
                "question": sample["question"],
                "passage": sample["passage"],
                "answer": answer_text,
                "answer_type": answer_type,
                "all_answers": answers_spans["spans"]
            }
            f.write(json.dumps(record) + "\n")

    print(f"✓ Saved {len(dataset['train'])} DROP samples to drop_dataset.jsonl")


# ============================================================================
# QUICK FIX #4: MBPP Dataset
# ============================================================================
# BEFORE (WRONG):
# ❌ problem = sample["text"]
# ❌ dataset = load_dataset("mbpp", "full")  # Wrong config
# Error: KeyError: 'text' (or field not found)

# AFTER (CORRECT):
# ✓ problem = sample["prompt"]  # Use "prompt" NOT "text" for sanitized
# ✓ dataset = load_dataset("mbpp", "sanitized")  # MUST use sanitized config

def fix_mbpp_dataset():
    from datasets import load_dataset
    import json

    # CORRECT: Use mbpp with "sanitized" config (not "full")
    dataset = load_dataset("mbpp", "sanitized")

    # Save to JSONL
    with open("mbpp_dataset.jsonl", "w") as f:
        for sample in dataset["train"]:
            record = {
                "task_id": sample["task_id"],
                "question": sample["prompt"],  # Use "prompt" NOT "text"
                "solution": sample["code"],
                "test_cases": sample["test_list"],
                "test_imports": sample["test_imports"]
            }
            f.write(json.dumps(record) + "\n")

    print(f"✓ Saved {len(dataset['train'])} MBPP samples to mbpp_dataset.jsonl")


# ============================================================================
# RUN ALL FIXES
# ============================================================================

def run_all_fixes():
    """Run all four dataset fixes in sequence"""
    print("\n" + "="*70)
    print("RUNNING ALL DATASET FIXES")
    print("="*70 + "\n")

    try:
        print("1. Fixing MATH dataset...")
        fix_math_dataset()
    except Exception as e:
        print(f"✗ MATH failed: {e}")

    try:
        print("\n2. Fixing HotpotQA dataset...")
        fix_hotpotqa_dataset()
    except Exception as e:
        print(f"✗ HotpotQA failed: {e}")

    try:
        print("\n3. Fixing DROP dataset...")
        fix_drop_dataset()
    except Exception as e:
        print(f"✗ DROP failed: {e}")

    try:
        print("\n4. Fixing MBPP dataset...")
        fix_mbpp_dataset()
    except Exception as e:
        print(f"✗ MBPP failed: {e}")

    print("\n" + "="*70)
    print("DONE - All datasets saved to JSONL files")
    print("="*70)


if __name__ == "__main__":
    run_all_fixes()
