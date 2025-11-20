#!/usr/bin/env python3
"""Retry downloading failed datasets - CommonsenseQA and MMLU"""

import json
import requests
from pathlib import Path
from datasets import load_dataset
import time

def download_commonsenseqa():
    print("Trying to download CommonsenseQA...")
    data_dir = Path("./data/raw/commonsenseqa")
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Try HuggingFace with different names
        for dataset_name in ["commonsense_qa", "tau/commonsense_qa", "allenai/commonsenseqa"]:
            try:
                print(f"Trying: {dataset_name}")
                dataset = load_dataset(dataset_name)

                # Save splits
                for split_name in dataset.keys():
                    split_data = dataset[split_name]
                    with open(data_dir / f"{split_name}.jsonl", "w") as f:
                        for item in split_data:
                            f.write(json.dumps(item) + "\n")
                    print(f"  - {split_name}: {len(split_data)} questions")

                print("✅ CommonsenseQA downloaded successfully")
                return True
            except Exception as e:
                print(f"  Failed: {e}")
                time.sleep(2)

        # Try direct download as fallback
        print("Trying direct download...")
        urls = {
            "train": "https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl",
            "dev": "https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl",
            "test": "https://s3.amazonaws.com/commensenseqa/test_rand_split_no_answers.jsonl"
        }

        for split, url in urls.items():
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(data_dir / f"{split}.jsonl", "w") as f:
                    f.write(response.text)
                print(f"  Downloaded {split} split")

        print("✅ CommonsenseQA downloaded via direct URLs")
        return True

    except Exception as e:
        print(f"❌ Failed to download CommonsenseQA: {e}")
        return False

def download_mmlu():
    print("\nTrying to download MMLU...")
    data_dir = Path("./data/raw/mmlu")
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Try different dataset names
        for dataset_name in ["cais/mmlu", "hendrycks/mmlu", "lukaemon/mmlu"]:
            try:
                print(f"Trying: {dataset_name}")
                # Try with 'all' config
                dataset = load_dataset(dataset_name, "all")

                # Save splits
                for split_name in dataset.keys():
                    split_data = dataset[split_name]
                    with open(data_dir / f"{split_name}.jsonl", "w") as f:
                        for item in split_data:
                            f.write(json.dumps(item) + "\n")
                    print(f"  - {split_name}: {len(split_data)} questions")

                print("✅ MMLU downloaded successfully")
                return True

            except Exception as e:
                print(f"  Failed: {e}")
                time.sleep(2)

        print("❌ Could not download MMLU - may need manual download")
        return False

    except Exception as e:
        print(f"❌ Failed to download MMLU: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Retrying failed dataset downloads")
    print("="*60)

    csqa_success = download_commonsenseqa()
    mmlu_success = download_mmlu()

    print("\n" + "="*60)
    print("Summary:")
    print(f"CommonsenseQA: {'✅ Success' if csqa_success else '❌ Failed'}")
    print(f"MMLU: {'✅ Success' if mmlu_success else '❌ Failed'}")
    print("="*60)
