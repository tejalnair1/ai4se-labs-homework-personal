# AI4SE Assignment — Code Representation Learning with Masked `if` Reconstruction

**Author:** Tejal Nair   

---

## Overview

This project implements a **complete end-to-end training pipeline** for learning representations of Python code using a **masked conditional reconstruction task**.  
The model learns to fill in missing `if` statements inside Python functions, enabling it to reason over control flow and logical structure.

The pipeline includes:

1. Repository cloning and corpus cleaning  
2. Instance generation for `(input, label)` pairs  
3. Tokenizer training  
4. Denoising pretraining  
5. Fine-tuning on masked `if` restoration  
6. Evaluation on both internal and provided test sets  

---

## Project Structure

```
Assign1/
├── artifacts/
│   └── hf_tokenizer_fast/              # Custom tokenizer (trained with train_tokenizer.py)
├── data/
│   ├── instance_corpus/                # Cleaned Python source files
│   ├── instances_built/
│   │   ├── generated-testset.csv       # Evaluation results on internal test
│   │   ├── maskable_if_counts.csv      # IF count statistics
│   │   ├── provided_test.csv           # Raw provided test file
│   │   ├── provided-testset.csv        # Evaluation results on provided test
│   │   ├── test.jsonl                  # Test split of instances
│   │   ├── train.jsonl                 # Train split
│   │   └── val.jsonl                   # Validation split
│   ├── repos_cloned/                   # Cloned GitHub repositories
│   ├── python_repos.csv                # Metadata of repositories
│   └── repos_list.txt                  # Input repo list for cloning
├── runs/
│   ├── pretrain_denoise_dev/           # Pretraining checkpoints
│   └── finetune_ifmask_mini/           # Fine-tuned model
├── scripts/
│   ├── build_instances_streaming.py    # Instance builder
│   ├── corpus_worker.py                # Cleaning and corpus processing
│   ├── count_maskable_ifs.py           # IF counting utility
│   ├── eval_both_fast.py               # Combined evaluation script
│   ├── pretrain_denoise.py             # Pretraining driver
│   ├── train_finetune_mini.py          # Fine-tuning driver
│   ├── train_tokenizer.py              # Tokenizer trainer
│   └── get_data.ipynb                  # Notebook for dataset inspection
├── readme.md                           # Project documentation (this file)
└── Assignment1.pdf                     # Final report write-up
```

---

##Environment Setup

Create a clean Python 3.11+ environment and install dependencies:

```bash
conda create -n ai4se python=3.11
conda activate ai4se

pip install torch transformers datasets tqdm sentencepiece accelerate astor
```

---

## Step 1. Clone Repositories

Provide your list of GitHub repository URLs in **`repos_list.txt`** .

Run:

```bash
python build_pretrain_corpus.py   --repo-list data/repos_list.txt   --clone-dir data/repos_cloned
```

This will clone each repository into `data/repos_cloned/`.

Example output:

```
[clone] repos in list: 120
[clone] cloned ok: 118 | failed: 2
```

---

## Step 2. Process and Clean the Corpus

Once repositories are cloned, run:

```bash
python scripts/corpus_worker.py   --no-clone   --clone-dir data/repos_cloned   --corpus-dir data/instance_corpus   --clean   --min-functions 1   --require-maskable-if   --mode process
```

This script filters out non-Python files, syntax-invalid code, and functions without any `if` statements.  
Each retained file is written to `data/instance_corpus/`.

---

## Step 3. Build Training Instances

After corpus cleaning, extract functions and build `(input, label)` pairs:

```bash
python scripts/build_instances_streaming.py   --corpus-dir data/instance_corpus   --out-dir data/instances_built
```

Example log:

```
===== Instance build summary =====
Files processed:       20,814
Instances (train):     176,226
Instances (val):       21,123
Instances (test):      21,262
Total instances:       218,611
Written JSONL → data/instances_built/
```

---

## Step 4. Train Tokenizer

Train a custom tokenizer on the cleaned corpus.

```bash
python scripts/train_tokenizer.py   --corpus data/instance_corpus   --save-dir artifacts/hf_tokenizer_fast   --vocab-size 32000
```

Output:
```
Saved tokenizer → artifacts/hf_tokenizer_fast
```

---

## Step 5. Pretraining

Pretrain the model using a denoising objective (span masking / infilling).

```bash
python scripts/pretrain_denoise.py   --tokenizer artifacts/hf_tokenizer_fast   --train data/instances_built/train.jsonl   --val data/instances_built/val.jsonl   --output runs/pretrain_denoise_dev   --epochs 1   --batch-size 4
```

Example output:
```
[pretrain] epoch=1 step=5124 loss=1.02
[pretrain] saved checkpoint → runs/pretrain_denoise_dev/final
```

---

## Step 6. Fine-Tuning on Masked `if` Reconstruction

Fine-tune the pretrained model on the masked `if` task:

```bash
python scripts/train_finetune_mini.py
```

If you need a smaller run for limited memory, restrict the training subset:

```bash
MAX_TRAIN_SAMPLES=5000 python scripts/train_finetune_mini.py
```

Example output:

```
[finetune-mini] saved → runs/finetune_ifmask_mini/final | used 20,000 train samples, 2000 steps
```

---

## Step 7. Evaluation

Run both evaluations together:

```bash
python scripts/eval_both_fast.py
```

Progress bars will appear for both test sets, and outputs will be written to:

- `data/instances_built/generated-testset.csv`
- `data/instances_built/provided-testset.csv`

Example console output:

```
[gen-test] wrote data/instances_built/generated-testset.csv | kept=4279 dropped=721 | accuracy=0.000
[prov] wrote data/instances_built/provided-testset.csv | kept=294 dropped=0 | accuracy=0.000
```

---

## Evaluation Summary

| Dataset | Kept | Dropped | Accuracy |
|----------|------|----------|-----------|
| Generated Test Set | 4,279 | 721 | **0.000** |
| Provided Test Set | 294 | 0 | **0.000** |

---


## Quick Command Summary

| Stage | Command | Output |
|-------|----------|--------|
| **Clone Repos** | `python build_pretrain_corpus.py --repo-list data/repos_list.txt --clone-dir data/repos_cloned` | Raw repositories |
| **Clean Corpus** | `python scripts/corpus_worker.py --no-clone --clone-dir data/repos_cloned --corpus-dir data/instance_corpus --clean --min-functions 1 --require-maskable-if --mode process` | Clean corpus |
| **Build Instances** | `python scripts/build_instances_streaming.py --corpus-dir data/instance_corpus --out-dir data/instances_built` | train/val/test JSONL |
| **Train Tokenizer** | `python scripts/train_tokenizer.py --corpus data/instance_corpus --save-dir artifacts/hf_tokenizer_fast --vocab-size 32000` | Custom tokenizer |
| **Pretrain Model** | `python scripts/pretrain_denoise.py --tokenizer artifacts/hf_tokenizer_fast --train data/instances_built/train.jsonl --val data/instances_built/val.jsonl --output runs/pretrain_denoise_dev` | Pretrained checkpoint |
| **Fine-Tune** | `python scripts/train_finetune_mini.py` | Fine-tuned checkpoint |
| **Evaluate** | `python scripts/eval_both_fast.py` | CSV results |

---
