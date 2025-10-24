import argparse, random, warnings
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    T5Config, T5ForConditionalGeneration,
    PreTrainedTokenizerFast, DataCollatorForSeq2Seq,
    Trainer, TrainingArguments
)

warnings.filterwarnings("ignore", category=UserWarning)

TOKENIZER_DIR = "artifacts/hf_tokenizer_fast"
CORPUS_DIR    = Path("data/instance_corpus")
OUT_DIR       = "runs/pretrain_denoise"

MAX_INPUT_LEN = 512
MASK_FRACTION = 0.15
SEED          = 42
TRAIN_FRACTION = 0.9

# Corpus -> chunks
def iter_corpus_chunks(corpus_dir: Path, target_chars=4000):
    rng = random.Random(SEED)
    for p in corpus_dir.iterdir():
        if p.suffix != ".py":
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if not txt.strip():
            continue
        # line-preserving chunking
        buf, size = [], 0
        for line in txt.splitlines(keepends=True):
            buf.append(line)
            size += len(line)
            if size >= target_chars:
                yield {"text": "".join(buf)}
                buf, size = [], 0
        if buf:
            yield {"text": "".join(buf)}

# Masking on token IDs for denoising
def make_masker(tokenizer: PreTrainedTokenizerFast, mask_fraction=MASK_FRACTION, max_len=MAX_INPUT_LEN):
    mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    special_ids = {
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        mask_id,
    }

    def _mask(batch):
        enc = tokenizer(batch["text"], max_length=max_len, truncation=True)
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        labels = input_ids.copy()

        # choose maskable positions
        maskable = [i for i, tid in enumerate(input_ids) if tid not in special_ids]
        k = max(1, int(len(maskable) * mask_fraction))
        rng = random.Random(SEED + hash(tuple(input_ids)) % (10**6))  # deterministic per example
        rng.shuffle(maskable)
        for i in maskable[:k]:
            input_ids[i] = mask_id

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
        }
    return _mask

def build_t5_tiny_config(vocab_size: int) -> T5Config:
    return T5Config(
        vocab_size=vocab_size,
        d_model=256,
        d_ff=1024,
        num_layers=8,
        num_heads=8,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        feed_forward_proj="relu",
        decoder_start_token_id=0,
    )

def main():
    ap = argparse.ArgumentParser(description="Denoising pre-training on code (reconstruct masked tokens).")
    ap.add_argument("--tokenizer-dir", default=TOKENIZER_DIR)
    ap.add_argument("--corpus-dir", default=str(CORPUS_DIR))
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--init", choices=["from_scratch", "t5-small"], default="from_scratch",
                    help="Initialize model from scratch (tiny T5) or from t5-small checkpoint.")
    ap.add_argument("--max-input-len", type=int, default=MAX_INPUT_LEN)
    ap.add_argument("--mask-frac", type=float, default=MASK_FRACTION)
    ap.add_argument("--num-train-epochs", type=int, default=2)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--train-examples", type=int, default=200_000,
                    help="Cap the number of training chunks (for speed).")
    ap.add_argument("--eval-examples", type=int, default=10_000)
    ap.add_argument("--per-device-train-batch-size", type=int, default=4)
    ap.add_argument("--per-device-eval-batch-size", type=int, default=4)
    ap.add_argument("--grad-accum-steps", type=int, default=4)
    args = ap.parse_args()

    random.seed(SEED)

    # 1) tokenizer
    tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    tok.add_special_tokens({"additional_special_tokens": ["[IF_MASK]"]})

    # 2) build small dataset from code files
    gen = iter_corpus_chunks(Path(args.corpus_dir), target_chars=4000)
    exs = []
    for _, ex in zip(range(args.train_examples + args.eval_examples), gen):
        exs.append(ex)
    if not exs:
        print("[warn] no examples found—check corpus_dir")
        return

    n_train = min(args.train_examples, int(len(exs) * TRAIN_FRACTION))
    train_raw = exs[:n_train]
    eval_raw  = exs[n_train:n_train + args.eval_examples]

    ds_train = Dataset.from_list(train_raw)
    ds_eval  = Dataset.from_list(eval_raw)

    # 3) build/initialize the model
    if args.init == "from_scratch":
        cfg = build_t5_tiny_config(vocab_size=len(tok))
        model = T5ForConditionalGeneration(cfg)
        model.resize_token_embeddings(len(tok))
    else:
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        model.resize_token_embeddings(len(tok))

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("high")
            model = torch.compile(model)
        except Exception:
            pass

    # 4) tokenization + masking map  (single process)
    masker = make_masker(tok, mask_fraction=args.mask_frac, max_len=args.max_input_len)
    ds_train_tok = ds_train.map(
        masker,
        batched=False,
        num_proc=1,
        remove_columns=ds_train.column_names,
        load_from_cache_file=True,
    )
    ds_eval_tok  = ds_eval.map(
        masker,
        batched=False,
        num_proc=1,
        remove_columns=ds_eval.column_names,
        load_from_cache_file=True,
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model)

    # 5) training
    print(f"[pretrain] train examples: {len(ds_train_tok):,} | eval examples: {len(ds_eval_tok):,}")

    args_hf = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=0.03,
        weight_decay=0.01,
        lr_scheduler_type="cosine",

        eval_strategy="steps",
        eval_steps=8000,
        save_steps=8000,
        logging_steps=1000,

        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,

        fp16=torch.cuda.is_available(),

        dataloader_num_workers=0,
        dataloader_pin_memory=False,

        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args_hf,
        train_dataset=ds_train_tok,
        eval_dataset=ds_eval_tok,
        tokenizer=tok,
        data_collator=collator,
    )

    trainer.train()

    final_dir = Path(args.out_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))
    print(f"[pretrain] saved checkpoint -> {final_dir}")

if __name__ == "__main__":
    main()
