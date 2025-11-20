# Dataset Download and Processing Solutions

Complete solutions for downloading and processing 4 datasets with corrected field access patterns.

## Overview

This package contains solutions for 4 commonly-used NLP datasets that have tricky field structures:

| Dataset | Issue | Solution |
|---------|-------|----------|
| **MATH** | Wrong dataset name (lighteval/MATH) | Use `EleutherAI/hendrycks_math` |
| **HotpotQA** | "string indices must be integers" (nested dicts) | Properly unpack nested structures |
| **DROP** | "string indices must be integers" (nested dict) | Access `answers_spans["spans"]` |
| **MBPP** | KeyError: 'text' (wrong field name) | Use `prompt` field, not `text` |

## Quick Start

### Option 1: Run All Fixes (Recommended)

```bash
python QUICK_FIX.py
```

This will download all 4 datasets and save them as JSONL files:
- `math_dataset.jsonl`
- `hotpotqa_dataset.jsonl`
- `drop_dataset.jsonl`
- `mbpp_dataset.jsonl`

### Option 2: Full Production Script

```bash
python download_datasets.py
```

More robust version with error handling, progress bars, and detailed logging.

### Option 3: Individual Examples

```bash
python dataset_examples.py
```

Shows minimal example code for accessing each dataset correctly.

## Files Included

### 1. `QUICK_FIX.py` (Simplest)
- Minimal, copy-paste ready code
- One function per dataset
- Outputs JSONL format
- Best for quick implementation

### 2. `download_datasets.py` (Most Robust)
- Production-ready implementation
- Error handling and retries
- Progress bars with tqdm
- Detailed logging and summary statistics
- Proper field extraction for nested structures

### 3. `dataset_examples.py` (Learning Resource)
- Minimal example code for each dataset
- Shows correct field access patterns
- Batch processing example
- Dataset structure diagnostic function
- Good for understanding the field layouts

### 4. `DATASET_FIXES.md` (Documentation)
- Detailed markdown documentation
- Complete field mappings
- Data structure diagrams
- Comprehensive troubleshooting guide

### 5. `DATASET_SUMMARY.txt` (Quick Reference)
- Summary table of all datasets
- Error diagnosis checklist
- Comparison of field structures
- Quick lookup format

## Dataset Details

### 1. MATH Dataset - Mathematics Problems

**Problem:** Error: "Dataset 'lighteval/MATH' doesn't exist on the Hub"

**Solution:**
```python
from datasets import load_dataset

dataset = load_dataset("EleutherAI/hendrycks_math")  # Correct name
sample = dataset["train"][0]

# Fields:
question = sample["problem"]       # The math problem
solution = sample["solution"]      # Worked solution
difficulty = sample["level"]       # Difficulty level
category = sample["type"]          # Problem category
```

**Key Points:**
- Correct dataset: `EleutherAI/hendrycks_math`
- Field names: `problem`, `solution`, `level`, `type`
- 12,500+ samples
- Includes worked solutions

---

### 2. HotpotQA Dataset - Multi-hop Question Answering

**Problem:** Error: "string indices must be integers" when accessing context

**Solution:**
```python
from datasets import load_dataset

dataset = load_dataset("hotpotqa/hotpot_qa", "distractor")
sample = dataset["train"][0]

# Top-level fields (direct access):
question = sample["question"]      # String
answer = sample["answer"]          # String

# Nested structure - supporting_facts (dict):
supporting = sample["supporting_facts"]
fact_titles = supporting["title"]  # List[str]
fact_sent_ids = supporting["sent_id"]  # List[int]

# Nested structure - context (dict with 2D array):
context = sample["context"]
context_text = " ".join([" ".join(s) for s in context["sentences"]])
```

**Key Points:**
- Correct dataset: `hotpotqa/hotpot_qa`
- Configuration: `"distractor"` or `"fullwiki"`
- Fields are nested dicts - must unpack carefully
- Context sentences are 2D arrays

---

### 3. DROP Dataset - Reading Comprehension

**Problem:** Error: "string indices must be integers" when accessing answers

**Solution:**
```python
from datasets import load_dataset

dataset = load_dataset("ucinlp/drop")
sample = dataset["train"][0]

# Top-level fields (direct access):
question = sample["question"]      # String
passage = sample["passage"]        # String

# Nested structure - answers_spans (dict):
answers_spans = sample["answers_spans"]
answer_text = answers_spans["spans"][0]   # First answer text
answer_type = answers_spans["types"][0]   # First answer type
```

**Key Points:**
- Correct dataset: `ucinlp/drop`
- `answers_spans` is a dict with `"spans"` and `"types"` keys
- Each answer has both text and type
- 77.4k training samples

---

### 4. MBPP Dataset - Programming Problems

**Problem:** Error: KeyError: 'text' when accessing problem description

**Solution:**
```python
from datasets import load_dataset

# IMPORTANT: Must use "sanitized" config, not "full"
dataset = load_dataset("mbpp", "sanitized")
sample = dataset["train"][0]

# Fields:
task_id = sample["task_id"]        # Integer
problem = sample["prompt"]         # Use "prompt" NOT "text"
solution = sample["code"]          # Sample solution
test_cases = sample["test_list"]   # Test assertions
test_imports = sample["test_imports"]  # Required imports
```

**Key Points:**
- Correct dataset: `mbpp` with config `"sanitized"`
- Field names differ by config: `"sanitized"` uses `"prompt"`, `"full"` uses `"text"`
- 974 training samples
- Includes test cases and required imports

---

## Common Errors and Solutions

### Error: "string indices must be integers"

**Cause:** Trying to access a nested dictionary as if it were a string.

**Example:**
```python
# WRONG - context is a dict, not a string
text = sample["context"].upper()  # ✗ This fails

# RIGHT - access nested field
text = sample["context"]["sentences"]  # ✓ Correct
```

**Solution:** Check the data structure first:
```python
print(type(sample["context"]))      # See if it's dict, list, str, etc.
print(sample["context"].keys())     # See what keys are available
```

### Error: "Dataset doesn't exist"

**Cause:** Wrong dataset name or namespace.

**Examples:**
```python
# WRONG
load_dataset("lighteval/MATH")      # ✗ Doesn't exist

# RIGHT
load_dataset("EleutherAI/hendrycks_math")  # ✓ Correct name
```

**Solution:** Check the full dataset identifier including organization.

### Error: KeyError on field name

**Cause:** Using wrong field name for a split/config.

**Example:**
```python
# WRONG - "text" doesn't exist in sanitized split
text = sample["text"]               # ✗ KeyError: 'text'

# RIGHT - use "prompt" for sanitized config
text = sample["prompt"]             # ✓ Correct field
```

**Solution:** Verify the correct split/config and field names.

## Field Access Patterns

### Simple Fields (Direct Access)

```python
# These are strings, integers, etc.
question = sample["question"]
answer = sample["answer"]
difficulty = sample["level"]
```

### Nested Dictionaries (Unpack Keys)

```python
# supporting_facts is a dict
supporting = sample["supporting_facts"]
titles = supporting["title"]        # Access the "title" key
sent_ids = supporting["sent_id"]    # Access the "sent_id" key
```

### Nested Lists (Index and Iterate)

```python
# context["sentences"] is a 2D list
context = sample["context"]
sentences = context["sentences"]    # Get the 2D array
flat_text = " ".join([" ".join(s) for s in sentences])
```

### Lists with Multiple Values (Index or Iterate)

```python
# answers_spans has lists of values
answers_spans = sample["answers_spans"]
first_answer = answers_spans["spans"][0]      # Get first element
all_answers = answers_spans["spans"]           # Get entire list

# Iterate over answers and types together
for ans_text, ans_type in zip(answers_spans["spans"], answers_spans["types"]):
    print(f"{ans_text} ({ans_type})")
```

## Output Format

All solutions output JSONL format (one JSON object per line):

```json
{"question": "...", "answer": "...", "field3": "..."}
{"question": "...", "answer": "...", "field3": "..."}
{"question": "...", "answer": "...", "field3": "..."}
```

Benefits:
- Easy to process line-by-line
- Can be read with pandas: `pd.read_json("file.jsonl", lines=True)`
- Can be loaded with: `jq -s '.' file.jsonl`
- Streaming-friendly format

## Processing Large Datasets

For datasets too large to fit in memory:

```python
from datasets import load_dataset
import json

dataset = load_dataset("dataset_name")
train_data = dataset["train"]
batch_size = 100

with open("output.jsonl", "w") as f:
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i+batch_size]
        for sample in batch:
            record = {
                # Extract fields
            }
            f.write(json.dumps(record) + "\n")
```

## Installation

Ensure you have the required packages:

```bash
pip install datasets transformers tqdm
```

## Performance Notes

Approximate download times and sizes:

| Dataset | Size | Download Time | Notes |
|---------|------|---------------|-------|
| MATH | ~10 MB | < 1 min | Fast, smallest |
| HotpotQA | ~350 MB | 2-5 min | Medium size |
| DROP | ~500 MB | 3-8 min | Largest |
| MBPP | ~5 MB | < 1 min | Smallest |

## Citation

If you use these datasets, please cite the original papers:

**MATH:**
```bibtex
@dataset{hendrycks_math,
  title={MATH Dataset},
  author={Hendrycks, Dan and Basart, Steven and Kadavath, Saurav and Mazeika, Mantas and Arpit, Devansh},
  year={2021}
}
```

**HotpotQA:**
```bibtex
@dataset{yang2018hotpotqa,
  title={HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  author={Yang, Zhilin and Qi, Peng and Zhang, Siyuan and Bengio, Yoshua and Cohen, William W and Salakhutdinov, Ruslan R and Manning, Christopher D},
  year={2018}
}
```

**DROP:**
```bibtex
@dataset{dua2019drop,
  title={DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs},
  author={Dua, Dheeru and Wang, Yada and Dasigi, Pradeep and Stanovsky, Gabriel and Singh, Sameer and Gardner, Matt A},
  year={2019}
}
```

**MBPP:**
```bibtex
@dataset{austin2021program,
  title={Program Synthesis with Large Language Models},
  author={Austin, Jacob and Odena, Augustus and Nye, Maxwell I and Bosma, Maarten and Michalewski, Henryk and Dohan, David and Yildirim, Aditya E and Jain, Akshay and Gu, Charles E and Harrison, Josh and others},
  year={2021}
}
```

## Troubleshooting

### Still getting errors?

1. **Update datasets package:**
   ```bash
   pip install --upgrade datasets
   ```

2. **Clear cache:**
   ```bash
   rm -rf ~/.cache/huggingface/datasets/
   ```

3. **Check internet connection:**
   - HuggingFace may be slow or unreachable

4. **Use diagnostic function:**
   ```python
   from dataset_examples import diagnose_dataset_structure
   diagnose_dataset_structure("EleutherAI/hendrycks_math")
   ```

5. **Check HuggingFace Hub:**
   - Visit https://huggingface.co/datasets to verify dataset exists

## Support

For issues with specific datasets, check:
- HuggingFace Dataset Hub: https://huggingface.co/datasets
- Dataset documentation in `DATASET_FIXES.md`
- Example code in `dataset_examples.py`
- Diagnostic function in `dataset_examples.py`

## License

These solutions are provided as-is. Please refer to individual dataset licenses:
- MATH: MIT
- HotpotQA: CC-BY-4.0
- DROP: Check original paper
- MBPP: Check original paper

---

**Last Updated:** 2024
**Status:** Production Ready
