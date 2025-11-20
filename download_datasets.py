#!/usr/bin/env python3
"""
Download and process datasets with correct field mappings.
Outputs JSONL files for each dataset.

Datasets:
1. MATH - EleutherAI/hendrycks_math
2. HotpotQA - hotpotqa/hotpot_qa
3. DROP - ucinlp/drop
4. MBPP - mbpp
"""

import json
import os
from datasets import load_dataset
from tqdm import tqdm


def download_and_process_math(output_path="math_dataset.jsonl"):
    """
    Download MATH dataset from EleutherAI/hendrycks_math

    Field mapping:
    - problem -> question
    - level -> difficulty
    - type -> category
    - solution -> answer
    """
    print("\n" + "="*60)
    print("Downloading MATH dataset (EleutherAI/hendrycks_math)...")
    print("="*60)

    try:
        dataset = load_dataset("EleutherAI/hendrycks_math")
        print(f"Successfully loaded MATH dataset")
        print(f"Available splits: {list(dataset.keys())}")

        # Dataset is typically in a single split, let's check the structure
        train_split = dataset.get("train", dataset.get(list(dataset.keys())[0]))
        print(f"Number of samples: {len(train_split)}")
        print(f"Sample fields: {train_split[0].keys()}")

        # Process and save to JSONL
        with open(output_path, 'w') as f:
            for sample in tqdm(train_split, desc="Processing MATH samples"):
                record = {
                    "question": sample.get("problem", ""),
                    "category": sample.get("type", ""),
                    "difficulty": sample.get("level", ""),
                    "solution": sample.get("solution", ""),
                    "split": sample.get("split", "")
                }
                f.write(json.dumps(record) + "\n")

        print(f"Successfully saved {len(train_split)} samples to {output_path}")
        return True

    except Exception as e:
        print(f"Error downloading MATH dataset: {e}")
        return False


def download_and_process_hotpotqa(output_path="hotpotqa_dataset.jsonl"):
    """
    Download HotpotQA dataset with distractor split

    Field mapping:
    - question -> question
    - answer -> answer
    - context.sentences -> passage (flattened)
    - context.title -> context_titles
    - supporting_facts -> supporting_facts
    """
    print("\n" + "="*60)
    print("Downloading HotpotQA dataset (hotpotqa/hotpot_qa)...")
    print("="*60)

    try:
        dataset = load_dataset("hotpotqa/hotpot_qa", "distractor")
        print(f"Successfully loaded HotpotQA dataset")
        print(f"Available splits: {list(dataset.keys())}")

        train_split = dataset["train"]
        print(f"Number of samples: {len(train_split)}")
        print(f"Sample fields: {train_split[0].keys()}")

        # Process and save to JSONL
        with open(output_path, 'w') as f:
            for sample in tqdm(train_split, desc="Processing HotpotQA samples"):
                # Extract context - it's a nested dict with title and sentences
                context = sample.get("context", {})
                context_text = ""
                if isinstance(context, dict):
                    titles = context.get("title", [])
                    sentences = context.get("sentences", [])
                    # Flatten sentences (2D array into text)
                    if sentences and isinstance(sentences[0], list):
                        context_text = " ".join([" ".join(sent_list) for sent_list in sentences])
                    else:
                        context_text = " ".join(sentences) if sentences else ""

                record = {
                    "question": sample.get("question", ""),
                    "answer": sample.get("answer", ""),
                    "type": sample.get("type", ""),
                    "level": sample.get("level", ""),
                    "context": context_text,
                    "supporting_facts": sample.get("supporting_facts", {}),
                    "id": sample.get("id", "")
                }
                f.write(json.dumps(record) + "\n")

        print(f"Successfully saved {len(train_split)} samples to {output_path}")
        return True

    except Exception as e:
        print(f"Error downloading HotpotQA dataset: {e}")
        return False


def download_and_process_drop(output_path="drop_dataset.jsonl"):
    """
    Download DROP dataset

    Field mapping:
    - passage -> passage
    - question -> question
    - answers_spans.spans -> answer (first span)
    - answers_spans.types -> answer_type
    """
    print("\n" + "="*60)
    print("Downloading DROP dataset (ucinlp/drop)...")
    print("="*60)

    try:
        dataset = load_dataset("ucinlp/drop")
        print(f"Successfully loaded DROP dataset")
        print(f"Available splits: {list(dataset.keys())}")

        train_split = dataset["train"]
        print(f"Number of samples: {len(train_split)}")
        print(f"Sample fields: {train_split[0].keys()}")

        # Process and save to JSONL
        with open(output_path, 'w') as f:
            for sample in tqdm(train_split, desc="Processing DROP samples"):
                # answers_spans is a nested dict with spans and types arrays
                answers_spans = sample.get("answers_spans", {})
                spans = answers_spans.get("spans", []) if isinstance(answers_spans, dict) else []
                types = answers_spans.get("types", []) if isinstance(answers_spans, dict) else []

                record = {
                    "question": sample.get("question", ""),
                    "passage": sample.get("passage", ""),
                    "answer": spans[0] if spans else "",
                    "answer_type": types[0] if types else "",
                    "all_answers": spans,
                    "all_answer_types": types,
                    "section_id": sample.get("section_id", ""),
                    "query_id": sample.get("query_id", "")
                }
                f.write(json.dumps(record) + "\n")

        print(f"Successfully saved {len(train_split)} samples to {output_path}")
        return True

    except Exception as e:
        print(f"Error downloading DROP dataset: {e}")
        return False


def download_and_process_mbpp(output_path="mbpp_dataset.jsonl"):
    """
    Download MBPP dataset (sanitized split)

    Field mapping:
    - prompt -> question (problem description)
    - code -> solution (sample code)
    - task_id -> task_id
    - test_list -> test_cases
    - test_imports -> test_imports
    """
    print("\n" + "="*60)
    print("Downloading MBPP dataset (mbpp, sanitized)...")
    print("="*60)

    try:
        dataset = load_dataset("mbpp", "sanitized")
        print(f"Successfully loaded MBPP dataset")
        print(f"Available splits: {list(dataset.keys())}")

        train_split = dataset["train"]
        print(f"Number of samples: {len(train_split)}")
        print(f"Sample fields: {train_split[0].keys()}")

        # Process and save to JSONL
        with open(output_path, 'w') as f:
            for sample in tqdm(train_split, desc="Processing MBPP samples"):
                record = {
                    "task_id": sample.get("task_id", ""),
                    "question": sample.get("prompt", ""),
                    "solution": sample.get("code", ""),
                    "test_cases": sample.get("test_list", []),
                    "test_imports": sample.get("test_imports", ""),
                    "source_file": sample.get("source_file", "")
                }
                f.write(json.dumps(record) + "\n")

        print(f"Successfully saved {len(train_split)} samples to {output_path}")
        return True

    except Exception as e:
        print(f"Error downloading MBPP dataset: {e}")
        return False


def main():
    """Download and process all datasets"""
    print("\n" + "#"*60)
    print("# Starting Dataset Download and Processing")
    print("#"*60)

    results = {}

    # Download all datasets
    results['MATH'] = download_and_process_math()
    results['HotpotQA'] = download_and_process_hotpotqa()
    results['DROP'] = download_and_process_drop()
    results['MBPP'] = download_and_process_mbpp()

    # Print summary
    print("\n" + "#"*60)
    print("# Download Summary")
    print("#"*60)
    for dataset_name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"{dataset_name}: {status}")

    # Check output files
    print("\n" + "#"*60)
    print("# Output Files")
    print("#"*60)
    output_files = [
        "math_dataset.jsonl",
        "hotpotqa_dataset.jsonl",
        "drop_dataset.jsonl",
        "mbpp_dataset.jsonl"
    ]

    for output_file in output_files:
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            num_lines = sum(1 for _ in open(output_file))
            print(f"{output_file}: {num_lines} records, {file_size:,} bytes")
        else:
            print(f"{output_file}: NOT CREATED")


if __name__ == "__main__":
    main()
