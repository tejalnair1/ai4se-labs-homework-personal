import os, torch, math
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    PreTrainedTokenizerFast,
    DataCollatorForSeq2Seq,
    Trainer,
    Seq2SeqTrainingArguments,
)

PRETRAIN_CKPT = os.environ.get("PRETRAIN_CKPT", "runs/pretrain_denoise_dev/final")
DATA_DIR      = os.environ.get("DATA_DIR", "data/instances_built")
OUT_DIR       = os.environ.get("OUT_DIR", "runs/finetune_ifmask_mini")

# Mini settings for memory
MAX_INPUT_LEN  = 128   # shorter sequences = much faster
MAX_LABEL_LEN  = 32
MAX_TRAIN_SAMPLES = int(os.environ.get("MAX_TRAIN_SAMPLES", "5000"))  # less memory
MAX_STEPS      = int(os.environ.get("MAX_STEPS", "2000"))              # ~2000 quick pass

def main():
    tok = PreTrainedTokenizerFast.from_pretrained(PRETRAIN_CKPT)
    tok.add_special_tokens({"additional_special_tokens": ["[IF_MASK]"]})

    model = T5ForConditionalGeneration.from_pretrained(PRETRAIN_CKPT)
    model.resize_token_embeddings(len(tok))
    model.config.use_cache = False

    ds = load_dataset(
        "json",
        data_files={
            "train": f"{DATA_DIR}/train.jsonl",
            "validation": f"{DATA_DIR}/val.jsonl",
        },
    )

    # Pre-filter by rough length using tokenizer's length estimation
    def keep_short(example):
        return (len(tok(example["input"]).input_ids) <= MAX_INPUT_LEN) and \
               (len(tok(example["label"]).input_ids) <= MAX_LABEL_LEN)

    # Shuffle, filter, subsample
    train = ds["train"].shuffle(seed=42).filter(keep_short)
    if MAX_TRAIN_SAMPLES and len(train) > MAX_TRAIN_SAMPLES:
        train = train.select(range(MAX_TRAIN_SAMPLES))

    # tiny val set for after training
    val = ds["validation"].shuffle(seed=42).filter(keep_short).select(range(min(2000, len(ds["validation"]))))

    def tok_fn(batch):
        enc = tok(batch["input"], max_length=MAX_INPUT_LEN, truncation=True)
        with tok.as_target_tokenizer():
            lab = tok(batch["label"], max_length=MAX_LABEL_LEN, truncation=True)
        enc["labels"] = lab["input_ids"]
        return enc

    train_tok = train.map(tok_fn, batched=True, remove_columns=train.column_names, num_proc=1, load_from_cache_file=True)
    val_tok   = val.map(tok_fn,   batched=True, remove_columns=val.column_names,   num_proc=1, load_from_cache_file=True)

    collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model)

    if torch.cuda.is_available():
        optim_name = "adamw_torch_fused"; use_fp16 = True
    else:
        optim_name = "adafactor";         use_fp16 = False

    args = Seq2SeqTrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=2,     
        gradient_accumulation_steps=2,  
        learning_rate=3e-4,
        warmup_ratio=0.03,
        weight_decay=0.0,
        lr_scheduler_type="cosine",

        # speed: cap steps, no eval during training, minimal logging/saving
        max_steps=MAX_STEPS,
        evaluation_strategy="no",
        save_strategy="no",
        logging_steps=1000000000,  # disables logging

        fp16=use_fp16,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        predict_with_generate=False,
        group_by_length=True,
        optim=optim_name,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=None, # no eval during training
        tokenizer=tok,
        data_collator=collator,
    )

    trainer.train()

    # Save
    save_dir = f"{OUT_DIR}/final"
    trainer.save_model(save_dir)
    tok.save_pretrained(save_dir)
    print(f"[finetune-mini] saved -> {save_dir} | used {len(train_tok):,} train samples, {MAX_STEPS} steps")

if __name__ == "__main__":
    main()
