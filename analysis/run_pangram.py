"""
Pangram AI-detection analysis on newspaper articles.
"""

import argparse
import json
import os
import time
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Run Pangram on DVHN articles")
    p.add_argument("--api-key", default=None)
    p.add_argument("--input", "-i",  default="classified_dvhn_articles_filtered_balanced.json")
    p.add_argument("--output", "-o",  default="pangram_results.jsonl")
    p.add_argument("--resume",    dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--limit", "-l",  type=int,   default=None)
    p.add_argument("--delay",  type=float, default=0.2)
    p.add_argument("--dry-run", action="store_true",
                   help="Validate everything without calling the API")
    p.add_argument("--test",   type=int,   default=None, metavar="N",
                   help="Send only N articles to the API (saves to test_output.jsonl)")
    return p.parse_args()


def build_text(article: dict) -> str:
    return (article.get("full_text") or "").strip()


def load_done_indices(output_path: str) -> set:
    done = set()
    if not Path(output_path).exists():
        return done
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(rec["article_index"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def extract_pangram_fields(result: dict) -> dict:
    windows = []
    for w in result.get("windows", []):
        windows.append({
            "label":               w.get("label"),
            "ai_assistance_score": w.get("ai_assistance_score"),
            "confidence":          w.get("confidence"),
        })
    return {
        "fraction_ai":          result.get("fraction_ai"),
        "fraction_ai_assisted": result.get("fraction_ai_assisted"),
        "fraction_human":       result.get("fraction_human"),
        "num_ai_segments":      result.get("num_ai_segments"),
        "windows":              windows,
    }


def write_summary(jsonl_path: str):
    summary_path = jsonl_path.replace(".jsonl", "_summary.json")
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Summary JSON saved → {summary_path}  ({len(records)} records)")


def process(client, articles: list, output_path: str, resume: bool, delay: float):
    if resume:
        done_indices = load_done_indices(output_path)
        print(f"  {len(done_indices)} articles already processed (skipping).")
        open_mode = "a"
    else:
        done_indices = set()
        open_mode    = "w"

    todo = [(i, art) for i, art in enumerate(articles) if i not in done_indices]
    print(f"Processing {len(todo)} articles → {output_path}\n")

    errors = 0
    with open(output_path, open_mode, encoding="utf-8") as out_f:
        for count, (idx, article) in enumerate(todo, start=1):
            text = build_text(article)

            if not text:
                print(f"[{count}/{len(todo)}] idx={idx} — SKIPPED (empty text)")
                continue

            try:
                result      = client.predict(text)
                pdata       = extract_pangram_fields(result)
                status = (
                    f"ai={pdata['fraction_ai']:.2f}  "
                    f"assisted={pdata['fraction_ai_assisted']:.2f}  "
                    f"human={pdata['fraction_human']:.2f}  "
                    f"segments={pdata['num_ai_segments']}"
                )
                print(f"[{count}/{len(todo)}] idx={idx} — {status}")

            except Exception as exc:
                print(f"[{count}/{len(todo)}] idx={idx} — ERROR: {exc}")
                pdata  = {"error": str(exc)}
                errors += 1

            record = {
                "article_index":   idx,
                "title":           article.get("title"),
                "date":            article.get("date"),
                "source":          article.get("source"),
                "predicted_topic": article.get("predicted_topic"),
                "text_length":     len(text),
                **pdata,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            if delay > 0:
                time.sleep(delay)

    total_done = len(done_indices) + len(todo)
    print(f"\nDone. {total_done} records written to {output_path}  ({errors} errors).")
    return errors


def main():
    args = parse_args()

    # ---- Load input ----
    print(f"Loading: {args.input}")
    with open(args.input, encoding="utf-8") as f:
        articles = json.load(f)
    print(f"  {len(articles)} articles loaded.")

    if args.limit:
        articles = articles[: args.limit]
        print(f"  --limit applied: using first {args.limit} articles.")

    # ---- Import Pangram ----
    try:
        from pangram import Pangram
    except ImportError:
        sys.exit(
            "\nERROR: 'pangram' package not found.\n"
            "Install it and re-run.\n"
        )

    api_key = args.api_key or os.environ.get("PANGRAM_API_KEY")
    client  = Pangram(api_key=api_key) if api_key else Pangram()

    # ---- Test mode: small sample, separate output file ----
    if args.test is not None:
        sample      = articles[: args.test]
        test_output = "test_output.jsonl"
        print(f"\nTEST MODE — sending {len(sample)} article(s) to Pangram → {test_output}\n")
        errors = process(client, sample, test_output, resume=False, delay=args.delay)
        write_summary(test_output)

        if errors == 0:
            print("\n✓ Test passed — all articles processed without errors.")
            print("  Run without --test to process all articles.")
        else:
            print(f"\n✗ Test finished with {errors} error(s). Check output above.")
        return

    # ---- Full run ----
    errors = process(client, articles, args.output, args.resume, args.delay)
    write_summary(args.output)


if __name__ == "__main__":
    main()
