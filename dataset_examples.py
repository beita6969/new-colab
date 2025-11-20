#!/usr/bin/env python3
"""
Minimal example code snippets for accessing each dataset correctly.
Use these as quick reference for your code.
"""

# ============================================================================
# 1. MATH DATASET - EleutherAI/hendrycks_math
# ============================================================================

def example_math():
    """Load and access MATH dataset"""
    from datasets import load_dataset

    # CORRECT: Use EleutherAI/hendrycks_math (not lighteval/MATH)
    dataset = load_dataset("EleutherAI/hendrycks_math")

    train_data = dataset["train"]
    sample = train_data[0]

    # Access fields
    problem = sample["problem"]           # String: the math problem
    solution = sample["solution"]         # String: worked solution
    level = sample["level"]               # String: difficulty level
    category = sample["type"]             # String: problem category

    print(f"Problem: {problem}")
    print(f"Level: {level}")
    print(f"Category: {category}")
    print(f"Solution: {solution}")


# ============================================================================
# 2. HOTPOTQA DATASET - hotpotqa/hotpot_qa
# ============================================================================

def example_hotpotqa():
    """Load and access HotpotQA dataset"""
    from datasets import load_dataset

    # CORRECT: Use hotpotqa/hotpot_qa with "distractor" config
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor")

    train_data = dataset["train"]
    sample = train_data[0]

    # Top-level fields (direct access)
    question = sample["question"]         # String
    answer = sample["answer"]             # String
    question_type = sample["type"]        # String: "comparison" or "bridge"
    difficulty = sample["level"]          # String: "easy", "medium", "hard"

    # Nested field: supporting_facts
    supporting = sample["supporting_facts"]
    fact_titles = supporting["title"]     # List[str]
    fact_sent_ids = supporting["sent_id"] # List[int]

    # Nested field: context (2D array structure)
    context = sample["context"]
    context_titles = context["title"]     # List[str]
    context_sentences = context["sentences"]  # List[List[str]]

    # Flatten context if needed
    context_text = " ".join([
        " ".join(sent_list)
        for sent_list in context_sentences
    ])

    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Type: {question_type}")
    print(f"Level: {difficulty}")
    print(f"Supporting facts: {fact_titles}")
    print(f"Context: {context_text[:200]}...")


# ============================================================================
# 3. DROP DATASET - ucinlp/drop
# ============================================================================

def example_drop():
    """Load and access DROP dataset"""
    from datasets import load_dataset

    # CORRECT: Use ucinlp/drop
    dataset = load_dataset("ucinlp/drop")

    train_data = dataset["train"]
    sample = train_data[0]

    # Top-level fields (direct access)
    question = sample["question"]         # String
    passage = sample["passage"]           # String
    section_id = sample["section_id"]     # String
    query_id = sample["query_id"]         # String

    # Nested field: answers_spans
    # IMPORTANT: answers_spans is a dict with "spans" and "types" keys
    answers_spans = sample["answers_spans"]
    answer_texts = answers_spans["spans"] # List[str]
    answer_types = answers_spans["types"] # List[str]

    # Get primary answer
    primary_answer = answer_texts[0] if answer_texts else ""
    primary_type = answer_types[0] if answer_types else ""

    # Get all answers
    all_answers = list(zip(answer_texts, answer_types))

    print(f"Question: {question}")
    print(f"Passage: {passage}")
    print(f"Primary Answer: {primary_answer}")
    print(f"Answer Type: {primary_type}")
    print(f"All Answers: {all_answers}")
    print(f"Section: {section_id}, Query: {query_id}")


# ============================================================================
# 4. MBPP DATASET - mbpp (sanitized split)
# ============================================================================

def example_mbpp():
    """Load and access MBPP dataset"""
    from datasets import load_dataset

    # CORRECT: Use mbpp with "sanitized" split
    dataset = load_dataset("mbpp", "sanitized")

    train_data = dataset["train"]
    sample = train_data[0]

    # Fields - NOTE: Use "prompt" NOT "text" for sanitized split
    task_id = sample["task_id"]           # Integer
    problem_desc = sample["prompt"]       # String: CORRECT FIELD (not "text")
    solution_code = sample["code"]        # String
    test_cases = sample["test_list"]      # List[str]
    test_imports = sample["test_imports"] # String
    source_file = sample["source_file"]   # String

    print(f"Task ID: {task_id}")
    print(f"Problem: {problem_desc}")
    print(f"Solution:\n{solution_code}")
    print(f"Test Cases: {test_cases}")
    print(f"Test Imports: {test_imports}")
    print(f"Source: {source_file}")


# ============================================================================
# BATCH PROCESSING EXAMPLE - For large datasets
# ============================================================================

def batch_process_example():
    """Process datasets in batches to save memory"""
    from datasets import load_dataset
    import json

    # Example with DROP dataset
    dataset = load_dataset("ucinlp/drop")
    train_data = dataset["train"]

    batch_size = 100
    output_file = "drop_batch_example.jsonl"

    with open(output_file, 'w') as f:
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]

            for sample in batch:
                answers_spans = sample["answers_spans"]
                record = {
                    "question": sample["question"],
                    "passage": sample["passage"][:500],  # Truncate for example
                    "answer": answers_spans["spans"][0] if answers_spans["spans"] else "",
                }
                f.write(json.dumps(record) + "\n")

    print(f"Processed {len(train_data)} samples, saved to {output_file}")


# ============================================================================
# ERROR DIAGNOSIS HELPER
# ============================================================================

def diagnose_dataset_structure(dataset_name, config=None):
    """Helper to inspect dataset structure and avoid errors"""
    from datasets import load_dataset

    print(f"\nDiagnosing {dataset_name}...")

    if config:
        dataset = load_dataset(dataset_name, config)
        print(f"Config: {config}")
    else:
        dataset = load_dataset(dataset_name)

    print(f"Available splits: {list(dataset.keys())}")

    for split_name in dataset.keys():
        split_data = dataset[split_name]
        print(f"\nSplit: {split_name}")
        print(f"  Number of samples: {len(split_data)}")
        print(f"  Fields: {list(split_data.features.keys())}")

        # Show first sample structure
        sample = split_data[0]
        for field_name, field_value in sample.items():
            field_type = type(field_value).__name__
            if field_type == "dict":
                print(f"    {field_name}: dict with keys {list(field_value.keys())}")
            elif field_type == "list" and len(field_value) > 0:
                inner_type = type(field_value[0]).__name__
                print(f"    {field_name}: list of {inner_type} (len={len(field_value)})")
            else:
                print(f"    {field_name}: {field_type} (len={len(str(field_value))})")


# ============================================================================
# MAIN EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MATH Dataset Example")
    print("=" * 70)
    try:
        example_math()
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 70)
    print("HotpotQA Dataset Example")
    print("=" * 70)
    try:
        example_hotpotqa()
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 70)
    print("DROP Dataset Example")
    print("=" * 70)
    try:
        example_drop()
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 70)
    print("MBPP Dataset Example")
    print("=" * 70)
    try:
        example_mbpp()
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 70)
    print("Dataset Diagnostics")
    print("=" * 70)
    # diagnose_dataset_structure("EleutherAI/hendrycks_math")
    # diagnose_dataset_structure("hotpotqa/hotpot_qa", "distractor")
    # diagnose_dataset_structure("ucinlp/drop")
    # diagnose_dataset_structure("mbpp", "sanitized")
