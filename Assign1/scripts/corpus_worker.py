#!/usr/bin/env python3
# corpus_worker.py
import argparse, io, os, re, sys, hashlib, tokenize, ast, shutil, multiprocessing as mp
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from typing import Tuple
from tqdm import tqdm

CLEAN = True
MAX_CHARS = 400_000
MAX_LINES = 15_000
MIN_LINES = 8
MIN_FUNCTIONS = 1
REQUIRE_MASKABLE_IF = False
NON_ASCII_SHARE_GT = 0.25

SKIP_PATH_PATTERNS = [
    r"/?\.git(/|$)", r"/?\.github(/|$)", r"/?\.venv(/|$)", r"/?env(/|$)", r"/?venv(/|$)",
    r"/?build(/|$)", r"/?dist(/|$)", r"/?site-packages(/|$)", r"/?third[_-]?party(/|$)",
    r"/?vendor(/|$)", r"/?__pycache__(/|$)", r"/?\.ipynb_checkpoints(/|$)", r"/?generated(/|$)",
]
SKIP_IF_PATH_CONTAINS = [
    "node_modules", "examples/generated", "egg-info", "migrations/auto", "docs/_build"
]

# Repo utils (optional cloning)

def to_owner_repo(line: str):
    s = line.strip()
    if not s:
        return None
    if s.startswith("http"):
        u = urlparse(s)
        parts = [p for p in u.path.split("/") if p]
        return "/".join(parts[:2]) if len(parts) >= 2 else None
    s = re.sub(r"^(?:github\.com/|com/|/)+", "", s)
    parts = s.split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else None

def clone_repo(owner_repo: str, clone_dir: Path) -> Path | None:
    try:
        from git import Repo  # requires GitPython
    except Exception:
        print("[error] GitPython not installed. Run: pip install gitpython", file=sys.stderr)
        return None

    dst = clone_dir / owner_repo.replace("/", "__")
    if dst.exists():
        shutil.rmtree(dst)
    url = f"https://github.com/{owner_repo}.git"
    try:
        Repo.clone_from(url, dst, depth=1, no_single_branch=True)
        return dst
    except Exception as e:
        print(f"[clone-fail] {owner_repo}: {e}", file=sys.stderr)
        return None

def maybe_clone_from_list(repos_list: Path, clone_dir: Path) -> list[Path]:
    clone_dir.mkdir(parents=True, exist_ok=True)
    if not repos_list or not repos_list.exists():
        return []
    repos = []
    with open(repos_list) as f:
        for raw in f:
            orp = to_owner_repo(raw)
            if orp and orp not in repos:
                repos.append(orp)
    print(f"[clone] repos in list: {len(repos)}")
    cloned = []
    for orp in repos:
        p = clone_repo(orp, clone_dir)
        if p:
            cloned.append(p)
    print(f"[clone] cloned ok: {len(cloned)}")
    return cloned

# Filtering helpers

def strip_comments_and_docstrings(source: str, clean: bool = CLEAN) -> str:
    if not clean:
        return source
    try:
        io_obj = io.StringIO(source)
        out = []
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            tok_type, tok_str, (sline, scol), (eline, ecol), _ = tok
            if tok_type == tokenize.COMMENT:
                continue
            if tok_type == tokenize.STRING and prev_toktype == tokenize.INDENT:
                # module/class/function docstring
                continue
            if sline > last_lineno:
                last_col = 0
            if scol > last_col:
                out.append(" " * (scol - last_col))
            out.append(tok_str)
            prev_toktype = tok_type
            last_lineno, last_col = eline, ecol
        return "".join(out)
    except Exception:
        return source

def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

def should_skip_path_str(s: str) -> bool:
    if any(re.search(rx, s) for rx in SKIP_PATH_PATTERNS):
        return True
    if any(w in s for w in SKIP_IF_PATH_CONTAINS):
        return True
    return False

def looks_minified_or_oneliner(text: str) -> bool:
    lines = text.splitlines()
    if len(lines) <= 1:
        return True
    if max((len(ln) for ln in lines), default=0) > 20_000:
        return True
    return False

def non_ascii_share(text: str) -> float:
    if not text:
        return 0.0
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return non_ascii / len(text)

def ast_function_if_stats(text: str):
  import warnings
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", SyntaxWarning)
    try:
        tree = ast.parse(text)
    except Exception:
        return 0, 0
    # If any SyntaxWarning captured, treat as 0/0 so it gets filtered out
    if any(isinstance(x.message, SyntaxWarning) for x in w):
        return 0, 0
    func_ct = 0
    if_ct = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_ct += 1
            if_ct += sum(isinstance(n, ast.If) for n in ast.walk(node))
    return func_ct, if_ct

# Worker (callable by pools)
def process_one_file(
    file_path_str: str,
    repo_dir_str: str,
    *,
    clean: bool = CLEAN,
    max_chars: int = MAX_CHARS,
    max_lines: int = MAX_LINES,
    min_lines: int = MIN_LINES,
    min_functions: int = MIN_FUNCTIONS,
    require_maskable_if: bool = REQUIRE_MASKABLE_IF,
    non_ascii_share_gt: float = NON_ASCII_SHARE_GT,
):
    """
    Returns (hash, out_name, cleaned) or None if filtered out.
    Pass strings (not Path) to keep pickling simple.
    """
    try:
        p = Path(file_path_str)
        s = f"/{p.as_posix()}/"
        if should_skip_path_str(s):
            return None

        raw = p.read_text(encoding="utf-8", errors="ignore")
        if not raw:
            return None
        if len(raw) > max_chars:
            return None

        line_count = raw.count("\n") + 1
        if line_count < min_lines or line_count > max_lines:
            return None

        if "\x00" in raw or non_ascii_share(raw) > non_ascii_share_gt:
            return None

        if looks_minified_or_oneliner(raw):
            return None

        cleaned = strip_comments_and_docstrings(raw, clean=clean)
        f_ct, if_ct = ast_function_if_stats(cleaned)
        if f_ct < min_functions:
            return None
        if require_maskable_if and if_ct < 1:
            return None

        repo_dir = Path(repo_dir_str)
        rel = p.relative_to(repo_dir)
        out_name = f"{repo_dir.name}__{rel.as_posix().replace('/', '__')}"
        h = sha1(cleaned)
        return (h, out_name, cleaned)
    except Exception:
        return None

# Main driver

def gather_candidates(cloned_dirs: list[Path]) -> list[tuple[str, str]]:
    candidates = []
    for repo_dir in cloned_dirs:
        for p in repo_dir.rglob("*.py"):
            # only cheap path-level skip here; heavy checks in worker
            s = f"/{p.as_posix()}/"
            if should_skip_path_str(s):
                continue
            candidates.append((str(p), str(repo_dir)))
    return candidates

def _starmap_process_one_file(args: Tuple[str, str, dict]):
    fp, rd, kw = args
    return process_one_file(fp, rd, **kw)
  
def run_process_pool(candidates, corpus_dir: Path, args):
    ctx = mp.get_context("spawn")
    max_workers = args.max_workers or max(1, (mp.cpu_count() or 4) - 1)
    chunksize = args.chunksize

    seen = set()
    kept = skipped = 0
    corpus_dir.mkdir(parents=True, exist_ok=True)

    kw = dict(
        clean=args.clean, max_chars=args.max_chars, max_lines=args.max_lines,
        min_lines=args.min_lines, min_functions=args.min_functions,
        require_maskable_if=args.require_maskable_if,
        non_ascii_share_gt=args.non_ascii_share_gt,
    )

    arg_iter = ((fp, rd, kw) for (fp, rd) in candidates)

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex, \
         tqdm(total=len(candidates), desc="Filtering", unit="file") as pbar:
        for res in ex.map(_starmap_process_one_file, arg_iter, chunksize=chunksize):
            if res is None:
                skipped += 1
            else:
                h, out_name, cleaned = res
                if h in seen:
                    skipped += 1
                else:
                    seen.add(h)
                    (corpus_dir / out_name).write_text(cleaned, encoding="utf-8")
                    kept += 1
            pbar.update(1)
            if pbar.n % 500 == 0:  # update postfix every 500
                pbar.set_postfix(kept=kept, skipped=skipped, uniq=len(seen), workers=max_workers)

    print(f"[processes] Kept: {kept:,} | Skipped: {skipped:,} | Unique: {len(seen):,} | Workers: {max_workers}")
def run_thread_pool(candidates, corpus_dir: Path, args):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    max_workers = args.max_workers or 32

    seen = set()
    kept = skipped = 0
    corpus_dir.mkdir(parents=True, exist_ok=True)

    kw = dict(
        clean=args.clean, max_chars=args.max_chars, max_lines=args.max_lines,
        min_lines=args.min_lines, min_functions=args.min_functions,
        require_maskable_if=args.require_maskable_if,
        non_ascii_share_gt=args.non_ascii_share_gt,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(process_one_file, fp, rd, **kw) for fp, rd in candidates]
        for fut in as_completed(futs):
            res = fut.result()
            if res is None:
                skipped += 1
                continue
            h, out_name, cleaned = res
            if h in seen:
                skipped += 1
                continue
            seen.add(h)
            (corpus_dir / out_name).write_text(cleaned, encoding="utf-8")
            kept += 1

    print(f"[threads] Kept: {kept:,} | Skipped: {skipped:,} | Unique: {len(seen):,} | Workers: {max_workers}")

def main():
    ap = argparse.ArgumentParser(description="Clone (optional), filter, clean, and dedupe Python files into a pretrain corpus.")
    ap.add_argument("--repos-list", type=Path, default=None, help="Text file with GitHub repo URLs or owner/repo lines.")
    ap.add_argument("--clone-dir", type=Path, required=True, help="Directory to place cloned repos (or where repos already exist).")
    ap.add_argument("--corpus-dir", type=Path, required=True, help="Output directory for cleaned corpus files.")
    ap.add_argument("--no-clone", action="store_true", help="Do not clone; just process existing repos under --clone-dir.")

    # parallelization
    ap.add_argument("--mode", choices=["process", "thread"], default="process", help="Parallelization backend.")
    ap.add_argument("--max-workers", type=int, default=None, help="Override worker count.")
    ap.add_argument("--chunksize", type=int, default=256, help="ProcessPool map chunksize.")

    # filters
    ap.add_argument("--clean", action="store_true", default=True, help="Strip comments/docstrings.")
    ap.add_argument("--no-clean", action="store_false", dest="clean", help="Do NOT strip comments/docstrings.")
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS)
    ap.add_argument("--max-lines", type=int, default=MAX_LINES)
    ap.add_argument("--min-lines", type=int, default=MIN_LINES)
    ap.add_argument("--min-functions", type=int, default=MIN_FUNCTIONS)
    ap.add_argument("--require-maskable-if", action="store_true", default=REQUIRE_MASKABLE_IF)
    ap.add_argument("--non-ascii-share-gt", type=float, default=NON_ASCII_SHARE_GT)

    args = ap.parse_args()

    clone_dir: Path = args.clone_dir
    corpus_dir: Path = args.corpus_dir

    # step 1: clone
    cloned_dirs: list[Path] = []
    if not args.no_clone and args.repos_list:
        cloned_dirs = maybe_clone_from_list(args.repos_list, clone_dir)
    else:
        # Use existing dirs inside clone_dir
        if not clone_dir.exists():
            print(f"[error] clone-dir does not exist: {clone_dir}", file=sys.stderr)
            sys.exit(2)
        cloned_dirs = [p for p in clone_dir.iterdir() if p.is_dir()]
        print(f"[scan] existing cloned dirs: {len(cloned_dirs)}")

    if not cloned_dirs:
        print("[warn] No repositories to process.", file=sys.stderr)
        sys.exit(0)

    # step 2: gather candidates
    candidates = gather_candidates(cloned_dirs)
    print(f"Candidate .py files to evaluate: {len(candidates):,}")

    # step 3: run pool
    if args.mode == "process":
        run_process_pool(candidates, corpus_dir, args)
    else:
        run_thread_pool(candidates, corpus_dir, args)

if __name__ == "__main__":
    main()
