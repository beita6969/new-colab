# Dataset Download and Processing Guide

This document provides the corrected methods for downloading and processing all four datasets.

---

## 1. MATH Dataset

### Problem
```
Error: "Dataset 'lighteval/MATH' doesn't exist on the Hub"
```

### Solution
The correct dataset name is **`EleutherAI/hendrycks_math`** (not `lighteval/MATH`)

### Correct Load Command
```python
from datasets import load_dataset

dataset = load_dataset("EleutherAI/hendrycks_math")
```

### Field Names
| Original Field | Mapped Field | Description |
|---|---|---|
| `problem` | question | Mathematical problem statement |
| `level` | difficulty | Difficulty level (e.g., "Level 5") |
| `type` | category | Problem category (algebra, geometry, etc.) |
| `solution` | answer | Worked solution with explanation |
| `split` | split | Data partition (train/test) |

### Example Code
```python
from datasets import load_dataset
import json

dataset = load_dataset("EleutherAI/hendrycks_math")
train_data = dataset["train"]

# Access a sample
sample = train_data[0]
print(sample["problem"])      # Mathematical problem
print(sample["solution"])     # Solution with steps
print(sample["level"])        # Difficulty level
```

### Key Details
- **Total Records**: 12,500+ samples
- **Splits**: train and test splits available
- **License**: MIT
- **Source**: Based on Hendrycks et al., NeurIPS 2021

---

## 2. HotpotQA Dataset

### Problem
```
Error: "string indices must be integers"
```

This occurs when trying to access nested dictionary fields incorrectly.

### Solution
Use correct dataset name: **`hotpotqa/hotpot_qa`** (with configuration)

### Correct Load Command
```python
from datasets import load_dataset

# Use the "distractor" configuration
dataset = load_dataset("hotpotqa/hotpot_qa", "distractor")
```

### Field Names and Structure

#### Top-level Fields
| Field | Type | Description |
|---|---|---|
| `id` | string | Unique identifier |
| `question` | string | The question text |
| `answer` | string | Expected answer |
| `type` | string | "comparison" or "bridge" |
| `level` | string | "easy", "medium", or "hard" |
| `supporting_facts` | dict | Document titles and sentence indices |
| `context` | dict | Context documents with sentences |

#### Nested Structure - `supporting_facts`
```python
supporting_facts = {
    "title": ["Document Title 1", "Document Title 2"],  # array of strings
    "sent_id": [0, 5]  # array of integers
}
```

#### Nested Structure - `context`
```python
context = {
    "title": ["Document Title 1", "Document Title 2"],  # array of strings
    "sentences": [
        ["Sentence 1 of doc 1", "Sentence 2 of doc 1"],  # 2D array
        ["Sentence 1 of doc 2"]
    ]
}
```

### Correct Access Code
```python
from datasets import load_dataset

dataset = load_dataset("hotpotqa/hotpot_qa", "distractor")
train_data = dataset["train"]

# Correctly access fields
sample = train_data[0]
question = sample["question"]          # Direct string access
answer = sample["answer"]              # Direct string access

# Access nested context
context = sample["context"]
titles = context["title"]              # List of document titles
sentences = context["sentences"]       # 2D list of sentences

# Flatten context text
context_text = " ".join([" ".join(sent_list) for sent_list in sentences])

# Access supporting facts
supporting = sample["supporting_facts"]
fact_titles = supporting["title"]      # Which documents support the answer
fact_sent_ids = supporting["sent_id"]  # Which sentences support the answer
```

### Dataset Configurations
- **distractor**: 97.9k rows with "train" and "validation" splits
- **fullwiki**: 105k rows with "train", "validation", and "test" splits

---

## 3. DROP Dataset

### Problem
```
Error: "string indices must be integers"
```

This occurs when accessing the nested `answers_spans` structure incorrectly.

### Solution
Use correct dataset name: **`ucinlp/drop`**

### Correct Load Command
```python
from datasets import load_dataset

dataset = load_dataset("ucinlp/drop")
```

### Field Names and Structure

#### Top-level Fields
| Field | Type | Description |
|---|---|---|
| `section_id` | string | Passage section identifier |
| `query_id` | string | Unique question identifier |
| `passage` | string | The text passage |
| `question` | string | The question requiring reasoning |
| `answers_spans` | dict | Answer information (nested) |

#### Nested Structure - `answers_spans`
```python
answers_spans = {
    "spans": ["answer text 1", "answer text 2"],  # Possible answer texts
    "types": ["span", "number"]                     # Answer types
}
```

### Correct Access Code
```python
from datasets import load_dataset

dataset = load_dataset("ucinlp/drop")
train_data = dataset["train"]

# Correctly access fields
sample = train_data[0]
question = sample["question"]          # Direct string access
passage = sample["passage"]            # Direct string access

# Access nested answers_spans
answers_spans = sample["answers_spans"]
answer_texts = answers_spans["spans"]  # List of answer texts
answer_types = answers_spans["types"]  # List of answer types

# Get first answer
primary_answer = answer_texts[0] if answer_texts else ""
primary_type = answer_types[0] if answer_types else ""

print(f"Question: {question}")
print(f"Passage: {passage}")
print(f"Answer: {primary_answer}")
print(f"Type: {primary_type}")
```

### Dataset Splits
- **train**: 77.4k rows
- **validation**: 9.54k rows

---

## 4. MBPP Dataset

### Problem
```
Error: KeyError: 'text'
```

The field name is `prompt` (not `text`) for the sanitized split.

### Solution
Use correct field names for the sanitized split

### Correct Load Command
```python
from datasets import load_dataset

# Load the sanitized split (not the full split)
dataset = load_dataset("mbpp", "sanitized")
```

### Field Names for Sanitized Split

| Field | Type | Description |
|---|---|---|
| `task_id` | integer | Unique problem identifier |
| `prompt` | string | Natural language problem description |
| `code` | string | Sample Python solution |
| `test_list` | list | Test cases as assertion strings |
| `test_imports` | string | Required imports for tests |
| `source_file` | string | Original source reference |

### Why the Error Occurs
- **Full split** uses field: `text`
- **Sanitized split** uses field: `prompt` (improved descriptions)

Both contain the problem description, but with different names.

### Correct Access Code
```python
from datasets import load_dataset

dataset = load_dataset("mbpp", "sanitized")
train_data = dataset["train"]

# Correctly access fields
sample = train_data[0]
task_id = sample["task_id"]            # Integer ID
problem_text = sample["prompt"]        # Use "prompt" NOT "text"
solution = sample["code"]              # Python solution code
tests = sample["test_list"]            # List of test assertions
imports = sample["test_imports"]       # Required imports

print(f"Task ID: {task_id}")
print(f"Problem: {problem_text}")
print(f"Solution:\n{solution}")
print(f"Tests: {tests}")
print(f"Imports: {imports}")
```

### Dataset Splits
- **train**: Main training data
- **test**: Test set

### Available Splits
- **sanitized**: Improved descriptions with test imports
- **full**: Original split with 'text' field instead of 'prompt'

---

## Complete Python Script

See `download_datasets.py` for a complete, production-ready script that:

1. Downloads all 4 datasets with correct parameters
2. Properly handles nested field structures
3. Exports data to JSONL format
4. Includes error handling and progress bars
5. Provides detailed logging and summary statistics

### Usage
```bash
python download_datasets.py
```

This creates four JSONL files:
- `math_dataset.jsonl`
- `hotpotqa_dataset.jsonl`
- `drop_dataset.jsonl`
- `mbpp_dataset.jsonl`

---

## Common Issues and Solutions

### Issue: "string indices must be integers"
**Cause**: Attempting to access a nested dictionary with array indexing or vice versa
**Solution**: Check the data structure first with `print(sample[field].keys())` or `print(type(sample[field]))`

### Issue: KeyError on field name
**Cause**: Using wrong field name for a dataset split
**Solution**: Verify the correct split name and field names using `print(dataset[split][0].keys())`

### Issue: Dataset doesn't exist
**Cause**: Incorrect dataset namespace or name
**Solution**: Check the full dataset identifier including organization (e.g., `EleutherAI/hendrycks_math`)

### Issue: Memory errors with large datasets
**Solution**: Process in batches instead of loading entire dataset:
```python
for i in range(0, len(dataset), 1000):
    batch = dataset[i:i+1000]
    # Process batch
```

---

## References

- **MATH Dataset**: [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math)
- **HotpotQA Dataset**: [hotpotqa/hotpot_qa](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
- **DROP Dataset**: [ucinlp/drop](https://huggingface.co/datasets/ucinlp/drop)
- **MBPP Dataset**: [mbpp](https://huggingface.co/datasets/mbpp)

---

## License Information

- **MATH**: MIT License (Hendrycks et al., NeurIPS 2021)
- **HotpotQA**: Creative Commons Attribution 4.0
- **DROP**: Not specified (check original paper)
- **MBPP**: Not specified (check original paper)
