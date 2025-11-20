#!/usr/bin/env python3
"""
Download MMLU dataset using mirrors and alternative methods
"""

import os
import json
import requests
import tarfile
import zipfile
from pathlib import Path
from datasets import load_dataset
import time

def try_huggingface_mirror():
    """Try downloading from HuggingFace using mirror endpoints"""
    print("Trying HuggingFace mirrors...")

    # Common mirror endpoints
    mirrors = [
        "https://hf-mirror.com",
        "https://huggingface.co",  # Original (may work with proxy)
    ]

    for mirror in mirrors:
        try:
            print(f"\nTrying mirror: {mirror}")
            os.environ["HF_ENDPOINT"] = mirror

            # Try different dataset names
            dataset_names = [
                ("cais/mmlu", "all"),
                ("hendrycks/mmlu", None),
                ("lukaemon/mmlu", None),
            ]

            for name, config in dataset_names:
                try:
                    print(f"  Attempting: {name}")
                    if config:
                        dataset = load_dataset(name, config, trust_remote_code=True)
                    else:
                        dataset = load_dataset(name, trust_remote_code=True)

                    print(f"✅ Successfully loaded from {name} via {mirror}")
                    return dataset
                except Exception as e:
                    print(f"    Failed: {str(e)[:100]}")
                    time.sleep(1)

        except Exception as e:
            print(f"Mirror {mirror} failed: {e}")

    return None

def download_from_github():
    """Download MMLU from GitHub repositories"""
    print("\nTrying GitHub download...")

    # Known GitHub URLs for MMLU
    github_urls = [
        # Hendrycks' original repository
        ("https://github.com/hendrycks/test/archive/master.zip", "zip"),
        # Alternative repositories
        ("https://github.com/hendrycks/test/raw/master/data.tar", "tar"),
    ]

    data_dir = Path("./data/raw/mmlu")
    data_dir.mkdir(parents=True, exist_ok=True)

    for url, file_type in github_urls:
        try:
            print(f"Downloading from: {url}")
            response = requests.get(url, stream=True, timeout=30)

            if response.status_code == 200:
                temp_file = data_dir / f"temp.{file_type}"

                # Download file
                with open(temp_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                print(f"Downloaded {file_type} file, extracting...")

                # Extract based on type
                if file_type == "zip":
                    with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                        zip_ref.extractall(data_dir)
                elif file_type == "tar":
                    with tarfile.open(temp_file, 'r') as tar_ref:
                        tar_ref.extractall(data_dir)

                temp_file.unlink()  # Remove temp file
                print("✅ Successfully extracted from GitHub")
                return True

        except Exception as e:
            print(f"  Failed: {e}")

    return False

def download_direct_files():
    """Try downloading individual MMLU files directly"""
    print("\nTrying direct file download...")

    data_dir = Path("./data/raw/mmlu")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Direct URLs for MMLU data (if available)
    base_urls = [
        "https://people.eecs.berkeley.edu/~hendrycks/data/",
        "https://raw.githubusercontent.com/hendrycks/test/master/",
    ]

    subjects = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology", "high_school_statistics",
        "high_school_us_history", "high_school_world_history", "human_aging",
        "human_sexuality", "international_law", "jurisprudence",
        "logical_fallacies", "machine_learning", "management", "marketing",
        "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
        "nutrition", "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy",
        "virology", "world_religions"
    ]

    success_count = 0
    for base_url in base_urls:
        print(f"\nTrying base URL: {base_url}")

        for split in ["dev", "test", "val"]:
            split_dir = data_dir / split
            split_dir.mkdir(exist_ok=True)

            for subject in subjects[:3]:  # Try first 3 subjects as test
                file_url = f"{base_url}{split}/{subject}_{split}.csv"

                try:
                    response = requests.get(file_url, timeout=5)
                    if response.status_code == 200:
                        file_path = split_dir / f"{subject}.csv"
                        with open(file_path, 'w') as f:
                            f.write(response.text)
                        success_count += 1
                        print(f"  ✓ Downloaded {subject}_{split}")
                    else:
                        print(f"  ✗ Failed {subject}_{split}: {response.status_code}")
                        break  # Try next base URL
                except Exception as e:
                    print(f"  ✗ Error downloading {subject}: {str(e)[:50]}")
                    break

            if success_count > 0:
                print(f"Downloaded {success_count} files")
                return True

    return False

def convert_to_jsonl(data_dir):
    """Convert downloaded data to JSONL format"""
    print("\nConverting to JSONL format...")

    # Find any CSV or JSON files and convert to JSONL
    data_dir = Path(data_dir)

    for csv_file in data_dir.glob("**/*.csv"):
        jsonl_file = csv_file.with_suffix('.jsonl')

        try:
            import csv
            with open(csv_file, 'r') as f_in, open(jsonl_file, 'w') as f_out:
                reader = csv.DictReader(f_in)
                for row in reader:
                    f_out.write(json.dumps(row) + '\n')
            print(f"  Converted {csv_file.name} to JSONL")
        except Exception as e:
            print(f"  Failed to convert {csv_file.name}: {e}")

def main():
    print("="*60)
    print("MMLU Dataset Download Script with Mirrors")
    print("="*60)

    # Try different methods in order
    dataset = try_huggingface_mirror()

    if dataset:
        print("\n✅ Successfully loaded MMLU from HuggingFace mirror!")

        # Save to local files
        data_dir = Path("./data/raw/mmlu")
        data_dir.mkdir(parents=True, exist_ok=True)

        for split_name in dataset.keys():
            split_data = dataset[split_name]
            with open(data_dir / f"{split_name}.jsonl", "w") as f:
                for item in split_data:
                    f.write(json.dumps(item) + "\n")
            print(f"  Saved {split_name}: {len(split_data)} samples")

        return True

    # Try GitHub download
    if download_from_github():
        print("\n✅ Downloaded MMLU from GitHub!")
        convert_to_jsonl("./data/raw/mmlu")
        return True

    # Try direct file download
    if download_direct_files():
        print("\n✅ Downloaded MMLU files directly!")
        convert_to_jsonl("./data/raw/mmlu")
        return True

    print("\n❌ Failed to download MMLU from all sources")
    print("\nAlternative solutions:")
    print("1. Use a VPN or proxy to access HuggingFace")
    print("2. Set environment variable: export HF_ENDPOINT=https://hf-mirror.com")
    print("3. Download manually from: https://github.com/hendrycks/test")
    print("4. Use ModelScope (Alibaba) mirror: https://modelscope.cn/datasets")

    return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
