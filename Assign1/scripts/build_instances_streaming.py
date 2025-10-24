#!/usr/bin/env python3
import ast, hashlib, json, warnings
from pathlib import Path
from tqdm import tqdm

SRC_DIR = Path("data/instance_corpus")
OUT_DIR = Path("data/instances_built"); OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = OUT_DIR / "train.jsonl"
VAL_PATH   = OUT_DIR / "val.jsonl"
TEST_PATH  = OUT_DIR / "test.jsonl"

MASK_TOKEN = "[IF_MASK]"
MAX_INSTANCES_PER_FILE = None   # set to an int to cap per-file
RNG_SALT = b"ifmask-v1"         # keep split deterministic across runs

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def instance_hash(inp: str, lbl: str) -> str:
    m = hashlib.sha1()
    m.update(RNG_SALT); m.update(b"\x00")
    m.update(inp.encode("utf-8")); m.update(b"\x00")
    m.update(lbl.encode("utf-8"))
    return m.hexdigest()

def split_bucket(h: str) -> str:
    #  80/10/10 split
    # Map last byte to bucket: 0-7: train (80%), 8: val (10%), 9: test (10%)
    last = int(h[-2:], 16) % 10
    if last <= 7:  # 0..7
        return "train"
    elif last == 8:
        return "val"
    else:
        return "test"

def mask_instances_for_function(fn_node, src_lines):
    f_start = fn_node.lineno - 1
    f_end   = fn_node.end_lineno
    func_lines = src_lines[f_start:f_end]

    def abs_to_rel(lno): return lno - 1 - f_start

    for n in ast.walk(fn_node):
        if isinstance(n, ast.If) and hasattr(n, "lineno"):
            hdr_abs_idx = n.lineno - 1
            hdr_rel_idx = abs_to_rel(n.lineno)
            header_line = src_lines[hdr_abs_idx]

            # header up to colon (if present) if (---): ---
            cpos = header_line.find(":")
            header_only = header_line[:cpos+1] if cpos != -1 else header_line.rstrip("\n")
            indent = header_only[:len(header_only) - len(header_only.lstrip())]

            masked = func_lines.copy()
            masked[hdr_rel_idx] = f"{indent}{MASK_TOKEN}\n"
            yield ("".join(masked), header_only.rstrip("\n"))

def process_file(p: Path):
    try:
        src = p.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(src)
    except Exception:
        return []

    lines = src.splitlines(keepends=True)
    inst = []
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and hasattr(n, "lineno")):
        for inp, lbl in mask_instances_for_function(fn, lines):
            inst.append((inp, lbl))
    if MAX_INSTANCES_PER_FILE and len(inst) > MAX_INSTANCES_PER_FILE:
        # downsample without randomness: take first N
        inst = inst[:MAX_INSTANCES_PER_FILE]
    return inst

def main():
    py_files = [p for p in SRC_DIR.iterdir() if p.suffix == ".py"]
    if not py_files:
        print(f"[warn] No .py files found under {SRC_DIR}")
        return

    fw_train = open(TRAIN_PATH, "w", encoding="utf-8")
    fw_val   = open(VAL_PATH,   "w", encoding="utf-8")
    fw_test  = open(TEST_PATH,  "w", encoding="utf-8")

    seen = set()
    n_train = n_val = n_test = 0
    n_dup = n_files = 0

    try:
        for p in tqdm(py_files, desc="Building instances", unit="file"):
            pairs = process_file(p)
            n_files += 1
            for inp, lbl in pairs:
                h = instance_hash(inp, lbl)
                if h in seen:
                    n_dup += 1
                    continue
                seen.add(h)
                rec = {"input": inp, "label": lbl}
                b = split_bucket(h)
                if b == "train":
                    fw_train.write(json.dumps(rec, ensure_ascii=False) + "\n"); n_train += 1
                elif b == "val":
                    fw_val.write(json.dumps(rec, ensure_ascii=False) + "\n"); n_val += 1
                else:
                    fw_test.write(json.dumps(rec, ensure_ascii=False) + "\n"); n_test += 1
    finally:
        fw_train.close(); fw_val.close(); fw_test.close()

    total = n_train + n_val + n_test
    print("\n===== Instance build summary =====")
    print(f"Files processed:       {n_files:,}")
    print(f"Instances (train):     {n_train:,}")
    print(f"Instances (val):       {n_val:,}")
    print(f"Instances (test):      {n_test:,}")
    print(f"Total instances:       {total:,}")
    print(f"Duplicates skipped:    {n_dup:,}")
    print(f"Written JSONL →\n  {TRAIN_PATH}\n  {VAL_PATH}\n  {TEST_PATH}")

if __name__ == "__main__":
    main()
