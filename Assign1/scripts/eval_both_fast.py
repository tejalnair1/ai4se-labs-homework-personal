import os, csv, re, math, ast
from pathlib import Path
from typing import List, Tuple

import torch
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, PreTrainedTokenizerFast
from tqdm import tqdm

CKPT_DIR       = os.environ.get("CKPT_DIR", "runs/finetune_ifmask_mini/final")
DATA_DIR       = os.environ.get("DATA_DIR", "data/instances_built")
TEST_JSONL     = os.environ.get("TEST_PATH", f"{DATA_DIR}/test.jsonl")
PROVIDED_CSV   = os.environ.get("PROVIDED_CSV", f"{DATA_DIR}/provided_test.csv")  # CodeSearchNet style
OUT_DIR        = os.environ.get("OUT_DIR", DATA_DIR)

GEN_OUT_NAME   = os.environ.get("GEN_OUT_NAME", "generated-testset.csv")
PROV_OUT_NAME  = os.environ.get("PROV_OUT_NAME", "provided-testset.csv")

# Speed
BATCH_SIZE     = int(os.environ.get("EVAL_BATCH_SIZE", "32"))
MAX_INPUT_LEN  = int(os.environ.get("EVAL_MAX_INPUT_LEN", "256"))
GEN_MAX_NEW    = int(os.environ.get("GEN_MAX_NEW", "32"))
KEEP_MASK_ONLY = bool(int(os.environ.get("KEEP_MASK_ONLY", "1")))
SAMPLE_N_GEN   = int(os.environ.get("SAMPLE_N_GEN", "0"))  # sample size for generated test
SAMPLE_N_PROV  = int(os.environ.get("SAMPLE_N_PROV", "0")) # sample size for provided test instances
MAX_FUNCS_PER_ROW = int(os.environ.get("MAX_FUNCS_PER_ROW", "1"))  # 1 = one instance per function/file

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MASK_TOKEN = "[IF_MASK]"

# Canonicalization
def normalize_quotes_ws(s: str) -> str:
    s = s.strip().replace("’", "'").replace("“", '"').replace("”", '"')
    return " ".join(s.split())

_paren_wrap = re.compile(r"^\(\s*(?P<body>.*)\s*\)$")
def strip_trailing_colon(s: str) -> str:
    return s[:-1].rstrip() if s.endswith(":") else s

def canonical_if_header(s: str) -> str:
    s = s.splitlines()[0] if "\n" in s else s
    s = s.split(":", 1)[0] + ":" if ":" in s else s
    s = normalize_quotes_ws(s).lstrip()
    if not s.startswith("if"):
        s = ("if " + s).strip()
    s = strip_trailing_colon(s)
    rest = s[2:].lstrip() if s.startswith("if") else s
    m = _paren_wrap.match(rest)
    if m:
        rest = normalize_quotes_ws(m.group("body"))
    return normalize_quotes_ws("if " + rest)

def write_csv(path: Path, rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Input provided to the model",
            "Whether the prediction is correct (true/false)",
            "Expected if condition",
            "Predicted if condition",
            "Prediction score (0-100)",
        ])
        w.writerows(rows)

def batchify_indices(n_items: int, batch_size: int):
    for i in range(0, n_items, batch_size):
        yield range(i, min(i + batch_size, n_items))

#  Provided CSV → instances
def build_instances_from_code(src: str) -> List[Tuple[str, str]]:
    """Return list of (masked_function_text, label_if_header) for one code blob."""
    try:
        tree = ast.parse(src)
    except Exception:
        return []
    lines = src.splitlines(keepends=True)
    out = []
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and hasattr(n, "lineno")):
        f_start, f_end = fn.lineno-1, fn.end_lineno
        func_lines = lines[f_start:f_end]
        def abs_to_rel(lno): return lno - 1 - f_start

        made = 0
        for n in ast.walk(fn):
            if isinstance(n, ast.If) and hasattr(n, "lineno"):
                hdr_abs = n.lineno - 1
                hdr_rel = abs_to_rel(n.lineno)
                if hdr_rel < 0 or hdr_rel >= len(func_lines):
                    continue
                header_line = lines[hdr_abs]
                colon = header_line.find(":")
                header_only = header_line[:colon+1] if colon != -1 else header_line.rstrip("\n")
                indent = header_only[:len(header_only) - len(header_only.lstrip())]
                masked = func_lines.copy()
                masked[hdr_rel] = f"{indent}{MASK_TOKEN}\n"
                out.append(("".join(masked), header_only.rstrip("\n")))
                made += 1
                if made >= MAX_FUNCS_PER_ROW:  # cap instances
                    break
    return out

def load_provided_codesearchnet(csv_path: str) -> Tuple[List[str], List[str]]:
    """CSV columns: id,code,code_tokens,docstring,docstring_tokens"""
    inputs, labels = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = {c.strip().lower() for c in (reader.fieldnames or [])}
        required = {"id","code","code_tokens","docstring","docstring_tokens"}
        if not required.issubset(cols):
            raise ValueError(f"Provided CSV missing expected columns; found: {cols}")
        for row in reader:
            code = row["code"]
            pairs = build_instances_from_code(code)
            for inp, lab in pairs:
                inputs.append(inp)
                labels.append(lab)
    return inputs, labels

# Eval
@torch.inference_mode()
def eval_split(model, tok, inputs: List[str], labels: List[str], title: str) -> Tuple[List[List[str]], float, int, int]:
    rows, correct = [], 0
    mask_id = tok.convert_tokens_to_ids(MASK_TOKEN)
    total = len(inputs)
    kept_total = 0
    n_batches = math.ceil(total / BATCH_SIZE)
    pbar = tqdm(total=n_batches, desc=f"Evaluating {title}", unit="batch")

    for chunk in batchify_indices(total, BATCH_SIZE):
        chunk = list(chunk)
        texts = [inputs[i] for i in chunk]
        enc = tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LEN,
            return_token_type_ids=False,
        )
        if "token_type_ids" in enc:
            enc.pop("token_type_ids")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        # only keep examples where [IF_MASK] survived the truncation
        if KEEP_MASK_ONLY:
            ids = enc["input_ids"]
            keep_mask = (ids == mask_id).any(dim=1)
            if keep_mask.sum().item() == 0:
                pbar.update(1)
                continue
            for k in list(enc.keys()):
                enc[k] = enc[k][keep_mask]
            kept_idx = [i for i, ok in zip(chunk, keep_mask.tolist()) if ok]
        else:
            kept_idx = chunk

        kept_total += len(kept_idx)
        seqs = model.generate(
            **enc,
            max_new_tokens=GEN_MAX_NEW,
            do_sample=False,
            num_beams=1,
        )
        preds = tok.batch_decode(seqs, skip_special_tokens=True)

        for out_i, src_i in enumerate(kept_idx):
            x = inputs[src_i]
            y = labels[src_i]
            p = preds[out_i]
            ok = (canonical_if_header(p) == canonical_if_header(y))
            correct += int(ok)
            rows.append([x, str(ok).lower(), y, p, "100.0"])

        pbar.update(1)

    pbar.close()
    acc = correct / max(1, kept_total)
    dropped = total - kept_total
    return rows, acc, kept_total, dropped

def main():
    # Load model/tokenizer once for both runs
    print(f"[load] model/tokenizer from {CKPT_DIR} on {DEVICE}")
    tok = PreTrainedTokenizerFast.from_pretrained(CKPT_DIR)
    model = T5ForConditionalGeneration.from_pretrained(CKPT_DIR).to(DEVICE)
    model.config.use_cache = True
    if DEVICE == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
            model.half()
        except Exception:
            pass

    # 1) Generated test.jsonl
    if Path(TEST_JSONL).exists():
        ds = load_dataset("json", data_files={"d": TEST_JSONL})["d"]
        inputs_gen = list(ds["input"])
        labels_gen = list(ds["label"])
        if SAMPLE_N_GEN and SAMPLE_N_GEN < len(inputs_gen):
            inputs_gen = inputs_gen[:SAMPLE_N_GEN]
            labels_gen = labels_gen[:SAMPLE_N_GEN]
        print(f"[gen-test] n={len(inputs_gen)} | batch={BATCH_SIZE} | max_in={MAX_INPUT_LEN} | gen_new={GEN_MAX_NEW}")
        rows, acc, kept, dropped = eval_split(model, tok, inputs_gen, labels_gen, "generated-testset")
        out_path = Path(OUT_DIR) / GEN_OUT_NAME
        write_csv(out_path, rows)
        print(f"[gen-test] wrote {out_path} | kept={kept} dropped={dropped} | accuracy={acc:.3f}")
    else:
        print(f"[gen-test] skip: {TEST_JSONL} not found")

    # 2) Provided CSV
    if Path(PROVIDED_CSV).exists():
        print(f"[prov] building instances from {PROVIDED_CSV} …")
        inputs_p, labels_p = load_provided_codesearchnet(PROVIDED_CSV)
        if SAMPLE_N_PROV and SAMPLE_N_PROV < len(inputs_p):
            inputs_p = inputs_p[:SAMPLE_N_PROV]
            labels_p = labels_p[:SAMPLE_N_PROV]
        print(f"[prov] n={len(inputs_p)} | batch={BATCH_SIZE} | max_in={MAX_INPUT_LEN} | gen_new={GEN_MAX_NEW}")
        rows_p, acc_p, kept_p, dropped_p = eval_split(model, tok, inputs_p, labels_p, "provided-testset")
        out_path_p = Path(OUT_DIR) / PROV_OUT_NAME
        write_csv(out_path_p, rows_p)
        print(f"[prov] wrote {out_path_p} | kept={kept_p} dropped={dropped_p} | accuracy={acc_p:.3f}")
    else:
        print(f"[prov] skip: {PROVIDED_CSV} not found")

if __name__ == "__main__":
    main()
