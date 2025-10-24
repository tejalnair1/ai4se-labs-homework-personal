from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

CORPUS_DIR = Path("data/instance_corpus")  # use cleaned files
ART_DIR = Path("artifacts/hf_tokenizer_fast")
ART_DIR.mkdir(parents=True, exist_ok=True)

def iter_texts(dirpath: Path, batch_size=1000):
    buf = []
    for p in dirpath.iterdir():
        if p.suffix != ".py": 
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            buf.append(txt)
        except Exception:
            continue
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if buf:
        yield buf

def main():
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = ByteLevel()

    trainer = BpeTrainer(
        vocab_size=32000,
        min_frequency=2,
        initial_alphabet=ByteLevel.alphabet(),
        special_tokens=["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]","[IF_MASK]"]
    )

    tok.train_from_iterator(iter_texts(CORPUS_DIR), trainer=trainer)

    tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", tok.token_to_id("[CLS]")),
                        ("[SEP]", tok.token_to_id("[SEP]"))],
    )

    tok.save(str(ART_DIR / "code_tokenizer.json"))
    hf = PreTrainedTokenizerFast(
        tokenizer_file=str(ART_DIR / "code_tokenizer.json"),
        unk_token="[UNK]", pad_token="[PAD]",
        cls_token="[CLS]", sep_token="[SEP]",
        mask_token="[MASK]"
    )
    hf.add_special_tokens({"additional_special_tokens": ["[IF_MASK]"]})
    hf.save_pretrained(str(ART_DIR))
    print(f"Saved tokenizer -> {ART_DIR}")

if __name__ == "__main__":
    main()
