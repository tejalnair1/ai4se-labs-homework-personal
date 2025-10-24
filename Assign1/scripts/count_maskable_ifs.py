import argparse, ast, warnings, multiprocessing as mp, csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm

# # Quiet noisy parse warnings from real-world code
# warnings.filterwarnings("ignore", category=SyntaxWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)

def count_in_file(path_str: str):
    """Return (file, ok, func_count, if_in_funcs_count). ok=False if parse failed."""
    p = Path(path_str)
    try:
        src = p.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(src)
    except Exception:
        return (p.name, False, 0, 0)

    func_ct = 0
    if_in_funcs = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_ct += 1
            if_in_funcs += sum(isinstance(n, ast.If) for n in ast.walk(node))
    return (p.name, True, func_ct, if_in_funcs)

def main():
    ap = argparse.ArgumentParser(description="Count maskable-if instances in a corpus of .py files.")
    ap.add_argument("--src-dir", type=Path, default=Path("data/instance_corpus"), help="Directory with cleaned .py files")
    ap.add_argument("--mode", choices=["process","thread"], default="process", help="Parallel backend")
    ap.add_argument("--max-workers", type=int, default=None, help="Override worker count")
    ap.add_argument("--csv-out", type=Path, default=Path("data/instances_built/maskable_if_counts.csv"))
    args = ap.parse_args()

    src = args.src_dir
    files = [str(p) for p in src.iterdir() if p.suffix == ".py"]
    files.sort()
    if not files:
        print(f"[warn] No .py files found under {src}")
        return

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)

    total_files = len(files)
    ok_files = 0
    fail_files = 0
    total_funcs = 0
    total_ifs = 0

    results = []

    if args.mode == "process":
        ctx = mp.get_context("spawn")
        max_workers = args.max_workers or max(1, (mp.cpu_count() or 4) - 1)
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex, \
             tqdm(total=total_files, desc="Counting", unit="file") as pbar:
            for file_name, ok, f_ct, if_ct in ex.map(count_in_file, files, chunksize=512):
                if ok:
                    ok_files += 1
                    total_funcs += f_ct
                    total_ifs += if_ct
                else:
                    fail_files += 1
                results.append((file_name, ok, f_ct, if_ct))
                pbar.update(1)
    else:
        max_workers = args.max_workers or 32
        with ThreadPoolExecutor(max_workers=max_workers) as ex, \
             tqdm(total=total_files, desc="Counting", unit="file") as pbar:
            futs = [ex.submit(count_in_file, f) for f in files]
            for fut in as_completed(futs):
                file_name, ok, f_ct, if_ct = fut.result()
                if ok:
                    ok_files += 1
                    total_funcs += f_ct
                    total_ifs += if_ct
                else:
                    fail_files += 1
                results.append((file_name, ok, f_ct, if_ct))
                pbar.update(1)

    # Write per-file breakdown
    with args.csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file","parsed_ok","function_count","if_in_functions_count"])
        for row in results:
            w.writerow(row)

    print("\n===== Summary =====")
    print(f"Files scanned:             {total_files:,}")
    print(f"Parsed OK:                 {ok_files:,}")
    print(f"Parse failed:              {fail_files:,}")
    print(f"Total functions:           {total_funcs:,}")
    print(f"Maskable ifs (upper bound):{total_ifs:,}")
    print(f"\nCSV written -> {args.csv_out}")

if __name__ == "__main__":
    main()
