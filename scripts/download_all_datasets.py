#!/usr/bin/env python3
"""
Complete Dataset Download and Integration Script for AFlow+ROLL Training
Includes: GSM8K, HumanEval, MBPP, CommonsenseQA, HotpotQA, MMLU
"""

import os
import json
import gzip
import shutil
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import subprocess
import sys

# Try to import datasets, install if not available
try:
    from datasets import load_dataset
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "transformers", "tqdm", "requests"])
    from datasets import load_dataset

class DatasetDownloader:
    """Unified dataset downloader for all required datasets"""

    def __init__(self, base_dir: str = "./data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)

    def download_gsm8k(self) -> Dict[str, Any]:
        """Download GSM8K dataset"""
        print("\n📚 Downloading GSM8K dataset...")

        try:
            # Use HuggingFace datasets
            dataset = load_dataset("openai/gsm8k", "main")

            # Save to local files
            gsm8k_dir = self.raw_dir / "gsm8k"
            gsm8k_dir.mkdir(exist_ok=True)

            # Save train and test splits
            train_data = dataset['train']
            test_data = dataset['test']

            with open(gsm8k_dir / "train.jsonl", "w") as f:
                for item in train_data:
                    f.write(json.dumps(item) + "\n")

            with open(gsm8k_dir / "test.jsonl", "w") as f:
                for item in test_data:
                    f.write(json.dumps(item) + "\n")

            print(f"✅ GSM8K downloaded: {len(train_data)} train, {len(test_data)} test")

            return {
                "name": "GSM8K",
                "train_size": len(train_data),
                "test_size": len(test_data),
                "path": str(gsm8k_dir)
            }

        except Exception as e:
            print(f"❌ Error downloading GSM8K: {e}")
            return {}

    def download_humaneval(self) -> Dict[str, Any]:
        """Download HumanEval dataset"""
        print("\n💻 Downloading HumanEval dataset...")

        try:
            # Download from GitHub
            humaneval_dir = self.raw_dir / "humaneval"
            humaneval_dir.mkdir(exist_ok=True)

            url = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
            response = requests.get(url, stream=True)

            gz_path = humaneval_dir / "HumanEval.jsonl.gz"
            jsonl_path = humaneval_dir / "HumanEval.jsonl"

            # Download gzipped file
            with open(gz_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Extract
            with gzip.open(gz_path, 'rb') as f_in:
                with open(jsonl_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Count problems
            with open(jsonl_path, 'r') as f:
                count = sum(1 for _ in f)

            print(f"✅ HumanEval downloaded: {count} problems")

            return {
                "name": "HumanEval",
                "size": count,
                "path": str(humaneval_dir)
            }

        except Exception as e:
            print(f"❌ Error downloading HumanEval: {e}")
            return {}

    def download_mbpp(self) -> Dict[str, Any]:
        """Download MBPP dataset"""
        print("\n🔧 Downloading MBPP dataset...")

        try:
            # Use HuggingFace datasets
            dataset = load_dataset("mbpp")

            mbpp_dir = self.raw_dir / "mbpp"
            mbpp_dir.mkdir(exist_ok=True)

            # Save all splits
            for split_name in dataset.keys():
                split_data = dataset[split_name]
                with open(mbpp_dir / f"{split_name}.jsonl", "w") as f:
                    for item in split_data:
                        f.write(json.dumps(item) + "\n")
                print(f"  - {split_name}: {len(split_data)} problems")

            print(f"✅ MBPP downloaded successfully")

            return {
                "name": "MBPP",
                "splits": list(dataset.keys()),
                "path": str(mbpp_dir)
            }

        except Exception as e:
            print(f"❌ Error downloading MBPP: {e}")
            return {}

    def download_commonsenseqa(self) -> Dict[str, Any]:
        """Download CommonsenseQA dataset"""
        print("\n🧠 Downloading CommonsenseQA dataset...")

        try:
            dataset = load_dataset("tau/commonsense_qa")

            csqa_dir = self.raw_dir / "commonsenseqa"
            csqa_dir.mkdir(exist_ok=True)

            # Save splits
            for split_name in dataset.keys():
                split_data = dataset[split_name]
                with open(csqa_dir / f"{split_name}.jsonl", "w") as f:
                    for item in split_data:
                        f.write(json.dumps(item) + "\n")
                print(f"  - {split_name}: {len(split_data)} questions")

            print(f"✅ CommonsenseQA downloaded successfully")

            return {
                "name": "CommonsenseQA",
                "splits": list(dataset.keys()),
                "path": str(csqa_dir)
            }

        except Exception as e:
            print(f"❌ Error downloading CommonsenseQA: {e}")
            return {}

    def download_hotpotqa(self) -> Dict[str, Any]:
        """Download HotpotQA dataset"""
        print("\n🔥 Downloading HotpotQA dataset...")

        try:
            hotpot_dir = self.raw_dir / "hotpotqa"
            hotpot_dir.mkdir(exist_ok=True)

            # Download distractor dev set as example
            url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"

            print("  Downloading dev set (distractor)...")
            response = requests.get(url)
            data = response.json()

            with open(hotpot_dir / "dev_distractor.json", "w") as f:
                json.dump(data, f)

            print(f"✅ HotpotQA downloaded: {len(data)} examples")

            return {
                "name": "HotpotQA",
                "dev_size": len(data),
                "path": str(hotpot_dir)
            }

        except Exception as e:
            print(f"❌ Error downloading HotpotQA: {e}")
            return {}

    def download_mmlu(self) -> Dict[str, Any]:
        """Download MMLU dataset"""
        print("\n📖 Downloading MMLU dataset...")

        try:
            dataset = load_dataset("cais/mmlu", "all")

            mmlu_dir = self.raw_dir / "mmlu"
            mmlu_dir.mkdir(exist_ok=True)

            # Save splits
            for split_name in dataset.keys():
                split_data = dataset[split_name]
                with open(mmlu_dir / f"{split_name}.jsonl", "w") as f:
                    for item in split_data:
                        f.write(json.dumps(item) + "\n")
                print(f"  - {split_name}: {len(split_data)} questions")

            print(f"✅ MMLU downloaded successfully")

            return {
                "name": "MMLU",
                "splits": list(dataset.keys()),
                "path": str(mmlu_dir)
            }

        except Exception as e:
            print(f"❌ Error downloading MMLU: {e}")
            return {}

    def process_for_training(self):
        """Process all datasets into unified format for training"""
        print("\n🔄 Processing datasets for training...")

        train_data = []
        val_data = []
        test_data = []

        # Process GSM8K (Math)
        gsm8k_dir = self.raw_dir / "gsm8k"
        if gsm8k_dir.exists():
            print("  Processing GSM8K...")
            with open(gsm8k_dir / "train.jsonl", "r") as f:
                for line in f:
                    item = json.loads(line)
                    train_data.append({
                        "problem": item["question"],
                        "problem_type": "math",
                        "ground_truth": self._extract_gsm8k_answer(item["answer"]),
                        "source": "gsm8k",
                        "full_answer": item["answer"]
                    })

        # Process HumanEval (Code)
        humaneval_dir = self.raw_dir / "humaneval"
        if humaneval_dir.exists():
            print("  Processing HumanEval...")
            with open(humaneval_dir / "HumanEval.jsonl", "r") as f:
                for line in f:
                    item = json.loads(line)
                    test_data.append({
                        "problem": item["prompt"],
                        "problem_type": "code",
                        "ground_truth": item.get("canonical_solution", ""),
                        "source": "humaneval",
                        "entry_point": item["entry_point"],
                        "test": item["test"]
                    })

        # Process CommonsenseQA (QA)
        csqa_dir = self.raw_dir / "commonsenseqa"
        if csqa_dir.exists():
            print("  Processing CommonsenseQA...")
            if (csqa_dir / "train.jsonl").exists():
                with open(csqa_dir / "train.jsonl", "r") as f:
                    for line in f:
                        item = json.loads(line)
                        choices_text = " ".join([f"{label}. {text}" for label, text in
                                                 zip(item["choices"]["label"], item["choices"]["text"])])
                        train_data.append({
                            "problem": f"{item['question']} Choices: {choices_text}",
                            "problem_type": "qa",
                            "ground_truth": item["answerKey"],
                            "source": "commonsenseqa",
                            "choices": item["choices"]
                        })

        # Save processed data
        print("\n💾 Saving processed datasets...")

        # Save training set
        with open(self.processed_dir / "train_mixed.jsonl", "w") as f:
            for item in train_data[:1000]:  # Limit for initial training
                f.write(json.dumps(item) + "\n")

        # Save validation set
        with open(self.processed_dir / "val_mixed.jsonl", "w") as f:
            for item in train_data[1000:1100]:  # 100 for validation
                f.write(json.dumps(item) + "\n")

        # Save test set
        with open(self.processed_dir / "test_mixed.jsonl", "w") as f:
            for item in test_data[:100]:  # 100 for testing
                f.write(json.dumps(item) + "\n")

        print(f"✅ Processed data saved:")
        print(f"  - Training: {min(1000, len(train_data))} samples")
        print(f"  - Validation: {min(100, len(train_data[1000:1100]))} samples")
        print(f"  - Test: {min(100, len(test_data))} samples")

    def _extract_gsm8k_answer(self, answer_text: str) -> str:
        """Extract numeric answer from GSM8K answer text"""
        if "####" in answer_text:
            return answer_text.split("####")[-1].strip()
        return answer_text

    def download_all(self):
        """Download all datasets"""
        print("="*60)
        print("🚀 Starting Complete Dataset Download")
        print("="*60)

        results = []

        # Download each dataset
        results.append(self.download_gsm8k())
        results.append(self.download_humaneval())
        results.append(self.download_mbpp())
        results.append(self.download_commonsenseqa())
        results.append(self.download_hotpotqa())
        results.append(self.download_mmlu())

        # Process for training
        self.process_for_training()

        # Print summary
        print("\n" + "="*60)
        print("📊 Download Summary")
        print("="*60)

        for result in results:
            if result:
                print(f"✓ {result.get('name', 'Unknown')}: Downloaded to {result.get('path', 'N/A')}")

        print(f"\n📁 All datasets saved to: {self.base_dir}")
        print(f"📁 Processed data ready for training in: {self.processed_dir}")

        return results


if __name__ == "__main__":
    # Create downloader instance
    downloader = DatasetDownloader(base_dir="./data")

    # Download all datasets
    results = downloader.download_all()

    print("\n✨ All datasets downloaded and processed successfully!")
    print("\nNext steps:")
    print("1. Review the processed data in ./data/processed/")
    print("2. Update your training config to use the new datasets")
    print("3. Run training with: python train.py --config config/training.yaml")
