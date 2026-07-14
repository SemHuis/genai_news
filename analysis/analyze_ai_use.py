#!/usr/bin/env python3
"""
Analyze AI-use across a folder of "combined articles" JSON files
(same schema as produced by merge_articles.py).

USAGE:
    python3 analyze_ai_use.py /path/to/folder [--outdir /path/to/outdir]

Every *.json / *.jsonl file in the folder is treated as one dataset, and
each analysis is produced per file (dataset name = filename without
extension). Only per-file results are produced (no combined "ALL" rows).
If you want a combined total across all files, concatenate the per-file
rows yourself and re-aggregate, or run this script on a folder containing
only the files you want combined.

Two definitions of "AI-use" are computed for every article:
  1) strict -> fraction_ai == 1  (the whole article is AI-written)
  2) broad  -> fraction_ai > 0 OR fraction_ai_assisted > 0
               (any AI involvement at all, fully AI or AI-assisted)

Every analysis row also reports `mean_ai_assistance_score`: the mean, over
the articles in that group, of each article's own ai_assistance_score
(itself the mean over that article's `windows[].ai_assistance_score`,
since one article can have multiple scored windows).

Output: CSV files written to --outdir (default: current directory):
  - ai_use_by_section_overall.csv
  - ai_use_by_section_quarter.csv
  - ai_use_by_author_overall.csv
  - ai_use_by_author_quarter.csv
  - ai_use_opinion_overall.csv
  - ai_use_opinion_quarter.csv
  - ai_use_by_topic.csv
  - ai_use_topic_share.csv
  - opinion_vs_non_opinion_ai_use.csv
  - text_features_by_ai_use.csv
  - text_features_stat_tests.csv

All of the above are produced twice: once for the full dataset (written
directly to --outdir), and once more restricted to articles from 2024
and 2025 only (written to a '2024_2025' subfolder of --outdir), so you
can see whether patterns look different in the most recent period. If
there are no 2024/2025 articles in the input, the second run is skipped.
"""

import argparse
import json
import os
import re
from datetime import datetime

import pandas as pd
from scipy.stats import mannwhitneyu

OPINION_SECTIONS = {
    "telegraaf": {"watuzegt"},
    "volkskrant": {"opinie en debat", "opinie", "opinie zaterdag"},
}

# An em-dash is either a real em-dash character, or a hyphen used as a
# stand-in for one, surrounded by spaces (" - "). A plain hyphen inside a
# word (e.g. "UAE-Team") or without surrounding spaces is NOT counted.
EM_DASH_REGEX = re.compile(r"—| - ")

# Quoted spans: text between a pair of straight double quotes. Curly/smart
# quotes weren't observed in this corpus, but are included for robustness.
QUOTE_REGEX = re.compile(r'"([^"]*)"|“([^”]*)”')

WORD_REGEX = re.compile(r"\w+", re.UNICODE)


def extract_text_features(text):
    """Return (word_count, em_dash_count, quote_count, quote_word_count)
    for a piece of full_text."""
    if not text:
        return 0, 0, 0, 0

    total_words = len(WORD_REGEX.findall(text))
    em_dash_count = len(EM_DASH_REGEX.findall(text))

    quote_count = 0
    quote_word_count = 0
    for m in QUOTE_REGEX.finditer(text):
        quoted_text = m.group(1) if m.group(1) is not None else m.group(2)
        quote_count += 1
        quote_word_count += len(WORD_REGEX.findall(quoted_text or ""))

    return total_words, em_dash_count, quote_count, quote_word_count


def to_quarter(date_str):
    """'30/09/2020' -> '2020-Q3'"""
    dt = datetime.strptime(date_str, "%d/%m/%Y")
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year}-Q{q}"


def quarter_to_numeric(quarter_str):
    """'2022-Q4' -> 2022.75, for chronological comparison."""
    year_str, q_str = quarter_str.split("-Q")
    return int(year_str) + (int(q_str) - 1) / 4


# Period split used for the topic analyses: quarters before 2023-Q1 vs. 2023-Q1 onward.
PERIOD_SPLIT_QUARTER = "2023-Q1"


def quarter_to_period(quarter_str):
    return ("before_" + PERIOD_SPLIT_QUARTER) if quarter_to_numeric(quarter_str) < quarter_to_numeric(PERIOD_SPLIT_QUARTER) \
        else (PERIOD_SPLIT_QUARTER + "_and_after")


def load_folder(folder_path):
    """Load every .json / .jsonl file in the folder as one dataset each."""
    articles_with_dataset = []

    files = sorted(
        f for f in os.listdir(folder_path)
        if f.lower().endswith(".json") or f.lower().endswith(".jsonl")
    )
    if not files:
        raise SystemExit(f"No .json or .jsonl files found in {folder_path}")

    for fname in files:
        path = os.path.join(folder_path, fname)
        dataset = os.path.splitext(fname)[0]

        if fname.lower().endswith(".jsonl"):
            records = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        else:
            with open(path, "r", encoding="utf-8") as f:
                records = json.load(f)
            if not isinstance(records, list):
                print(f"WARNING: {fname} does not contain a JSON list, skipping.")
                continue

        print(f"Loaded {len(records)} articles from {fname} (dataset='{dataset}')")
        for r in records:
            articles_with_dataset.append((r, dataset))

    return articles_with_dataset


def build_dataframe(articles_with_dataset):
    rows = []
    skipped_missing_fraction = 0
    skipped_bad_date = 0

    for a, dataset in articles_with_dataset:
        fraction_ai = a.get("fraction_ai")
        fraction_ai_assisted = a.get("fraction_ai_assisted")

        if fraction_ai is None or fraction_ai_assisted is None:
            skipped_missing_fraction += 1
            continue

        date_str = a.get("date")
        try:
            quarter = to_quarter(date_str)
        except (ValueError, TypeError):
            skipped_bad_date += 1
            continue

        windows = a.get("windows") or []
        scores = [
            w.get("ai_assistance_score") for w in windows
            if w.get("ai_assistance_score") is not None
        ]
        assistance_score = sum(scores) / len(scores) if scores else float("nan")

        full_text = a.get("full_text") or ""
        word_count, em_dash_count, quote_count, quote_word_count = extract_text_features(full_text)
        em_dash_per_1000_words = (em_dash_count / word_count * 1000) if word_count > 0 else float("nan")
        quote_word_fraction = (quote_word_count / word_count) if word_count > 0 else float("nan")

        rows.append({
            "dataset": dataset,
            "title": a.get("title"),
            "source": a.get("source"),
            "section": a.get("section") or "Geen Sectie",
            "topic": a.get("predicted_topic") or "Geen Topic",
            "date": date_str,
            "quarter": quarter,
            "author": a.get("author") or [],
            "fraction_ai": fraction_ai,
            "fraction_ai_assisted": fraction_ai_assisted,
            "ai_assistance_score": assistance_score,
            "is_ai_strict": fraction_ai == 1,
            "is_ai_broad": (fraction_ai > 0) or (fraction_ai_assisted > 0),
            "word_count": word_count,
            "em_dash_count": em_dash_count,
            "em_dash_per_1000_words": em_dash_per_1000_words,
            "quote_count": quote_count,
            "quote_word_fraction": quote_word_fraction,
        })

    if skipped_missing_fraction:
        print(f"Note: skipped {skipped_missing_fraction} article(s) with "
              f"missing fraction_ai/fraction_ai_assisted values.")
    if skipped_bad_date:
        print(f"Note: skipped {skipped_bad_date} article(s) with an "
              f"unparseable date.")

    df = pd.DataFrame(rows)
    return df


def summarize(df, group_cols):
    """
    Produce a long-form summary: dataset, group_cols..., definition,
    total_articles, ai_articles, fraction_ai_use, mean_ai_assistance_score
    """
    all_group_cols = ["dataset"] + group_cols
    results = []
    for definition, col in [("strict", "is_ai_strict"), ("broad", "is_ai_broad")]:
        grouped = df.groupby(all_group_cols).agg(
            total_articles=(col, "size"),
            ai_articles=(col, "sum"),
            mean_ai_assistance_score=("ai_assistance_score", "mean"),
        ).reset_index()
        grouped["definition"] = definition
        grouped["fraction_ai_use"] = grouped["ai_articles"] / grouped["total_articles"]
        results.append(grouped)

    out = pd.concat(results, ignore_index=True)
    ordered_cols = all_group_cols + [
        "definition", "total_articles", "ai_articles",
        "fraction_ai_use", "mean_ai_assistance_score",
    ]
    out = out[ordered_cols]
    return out.sort_values(all_group_cols + ["definition"]).reset_index(drop=True)


def explode_authors(df):
    """One row per (article, author) pair. Articles with no listed author
    are still included, tagged with the author 'NO AUTHOR', so they can be
    grouped together downstream instead of being silently dropped."""
    rows = []
    for _, row in df.iterrows():
        authors = row["author"] if row["author"] else ["NO AUTHOR"]
        for author in authors:
            rows.append({
                "dataset": row["dataset"],
                "author": author,
                "quarter": row["quarter"],
                "is_ai_strict": row["is_ai_strict"],
                "is_ai_broad": row["is_ai_broad"],
                "ai_assistance_score": row["ai_assistance_score"],
            })
    return pd.DataFrame(rows)


def with_all_newspapers_combined(df):
    """
    Return a copy of df with an extra block of rows tagged
    dataset='ALL newspapers', duplicating every row so that grouping by
    'dataset' afterwards yields both the per-newspaper breakdown and one
    combined total across every newspaper in the input. Used only where a
    combined overall figure is explicitly wanted (the topic analyses),
    not throughout the whole script.
    """
    return pd.concat([df, df.assign(dataset="ALL newspapers")], ignore_index=True)


def topic_share_of_ai_use(df, outdir):
    """
    For each (dataset, definition, period), what percentage of ALL
    articles flagged as AI-use belongs to each topic. This is a different
    denominator than ai_use_by_topic.csv: here the denominator is the
    total number of AI-use articles in that dataset/definition/period,
    not the total number of articles in that topic.
    """
    rows = []
    for definition, is_ai_col in [("strict", "is_ai_strict"), ("broad", "is_ai_broad")]:
        for (dataset, period), sub in df.groupby(["dataset", "period"]):
            ai_sub = sub[sub[is_ai_col]]
            total_ai_articles = len(ai_sub)
            if total_ai_articles == 0:
                continue
            topic_counts = ai_sub.groupby("topic").size().reset_index(name="ai_articles_in_topic")
            topic_counts["dataset"] = dataset
            topic_counts["period"] = period
            topic_counts["definition"] = definition
            topic_counts["total_ai_articles"] = total_ai_articles
            topic_counts["pct_of_ai_use_articles"] = topic_counts["ai_articles_in_topic"] / total_ai_articles
            rows.append(topic_counts)

    if not rows:
        out = pd.DataFrame(columns=["dataset", "period", "definition", "topic", "ai_articles_in_topic",
                                     "total_ai_articles", "pct_of_ai_use_articles"])
    else:
        out = pd.concat(rows, ignore_index=True)
        out = out[["dataset", "period", "definition", "topic", "ai_articles_in_topic",
                   "total_ai_articles", "pct_of_ai_use_articles"]]
        out = out.sort_values(["dataset", "period", "definition", "pct_of_ai_use_articles"],
                               ascending=[True, True, True, False])
        out = out.reset_index(drop=True)

    path = os.path.join(outdir, "ai_use_topic_share.csv")
    write_combined_and_per_period(out, path, outdir)


TEXT_FEATURE_METRICS = {
    "em_dash_count": "em_dash_count",
    "em_dash_per_1000_words": "em_dash_per_1000_words",
    "quote_count": "quote_count",
    "quote_word_fraction": "quote_word_fraction",
}


def stylometric_analysis(df, outdir):
    """
    Compare em-dash usage and quote usage between articles flagged as
    AI-use and articles not flagged, for both definitions (strict/broad),
    per newspaper and combined across all newspapers ("ALL newspapers").

    Two outputs:
      - text_features_by_ai_use.csv: descriptive stats (mean/median) per
        group.
      - text_features_stat_tests.csv: Mann-Whitney U test (two-sided)
        comparing the AI-use group against the non-AI-use group for each
        metric, since these counts/fractions are unlikely to be normally
        distributed.
    """
    desc_rows = []
    test_rows = []

    for definition, is_ai_col in [("strict", "is_ai_strict"), ("broad", "is_ai_broad")]:
        groups = list(df.groupby("dataset")) + [("ALL newspapers", df)]
        for dataset, sub in groups:
            for ai_flag, ai_sub in sub.groupby(is_ai_col):
                row = {
                    "dataset": dataset,
                    "definition": definition,
                    "ai_use_group": "AI-use" if ai_flag else "non-AI-use",
                    "num_articles": len(ai_sub),
                }
                for metric in TEXT_FEATURE_METRICS:
                    row[f"mean_{metric}"] = ai_sub[metric].mean()
                    row[f"median_{metric}"] = ai_sub[metric].median()
                desc_rows.append(row)

            ai_group = sub[sub[is_ai_col]]
            non_ai_group = sub[~sub[is_ai_col]]

            for metric in TEXT_FEATURE_METRICS:
                ai_vals = ai_group[metric].dropna()
                non_ai_vals = non_ai_group[metric].dropna()
                if len(ai_vals) < 2 or len(non_ai_vals) < 2:
                    u_stat, p_value = float("nan"), float("nan")
                else:
                    u_stat, p_value = mannwhitneyu(ai_vals, non_ai_vals, alternative="two-sided")
                test_rows.append({
                    "dataset": dataset,
                    "definition": definition,
                    "metric": metric,
                    "n_ai_use": len(ai_vals),
                    "n_non_ai_use": len(non_ai_vals),
                    "mean_ai_use": ai_vals.mean() if len(ai_vals) else float("nan"),
                    "mean_non_ai_use": non_ai_vals.mean() if len(non_ai_vals) else float("nan"),
                    "mannwhitney_u": u_stat,
                    "p_value": p_value,
                    "significant_at_0.05": (p_value < 0.05) if pd.notna(p_value) else False,
                })

    desc_df = pd.DataFrame(desc_rows).sort_values(["dataset", "definition", "ai_use_group"]).reset_index(drop=True)
    test_df = pd.DataFrame(test_rows).sort_values(["dataset", "definition", "metric"]).reset_index(drop=True)

    desc_path = os.path.join(outdir, "text_features_by_ai_use.csv")
    test_path = os.path.join(outdir, "text_features_stat_tests.csv")
    desc_df.to_csv(desc_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Wrote {desc_path} ({len(desc_df)} rows)")
    print(f"Wrote {test_path} ({len(test_df)} rows)")


def is_opinion_row(row):
    source = (row["source"] or "").strip().lower()
    section = (row["section"] or "").strip().lower()

    if source in {"de telegraaf", "telegraaf"} and section in OPINION_SECTIONS["telegraaf"]:
        return True
    if source in {"de volkskrant", "volkskrant"} and section in OPINION_SECTIONS["volkskrant"]:
        return True
    return False


def opinion_vs_non_opinion_analysis(df, outdir):
    """
    For de Telegraaf and de Volkskrant, restricted to 2023-Q1 onward,
    compare AI-use inside their opinion sections (WATUZEGT for de
    Telegraaf; Opinie / Opinie en Debat / Opinie zaterdag for de
    Volkskrant) against AI-use in their other, non-opinion sections.
    """
    source_norm = df["source"].fillna("").str.strip().str.lower()
    is_tgf = source_norm.isin({"de telegraaf", "telegraaf"})
    is_vk = source_norm.isin({"de volkskrant", "volkskrant"})

    sub = df[is_tgf | is_vk].copy()
    if sub.empty:
        print("Note: no de Telegraaf / de Volkskrant articles found; "
              "skipping opinion-vs-non-opinion analysis.")
        return

    sub["quarter_num"] = sub["quarter"].apply(quarter_to_numeric)
    sub = sub[sub["quarter_num"] >= quarter_to_numeric(PERIOD_SPLIT_QUARTER)]

    sub["newspaper"] = source_norm.loc[sub.index].apply(
        lambda s: "de Telegraaf" if s in {"de telegraaf", "telegraaf"} else "de Volkskrant"
    )
    sub["is_opinion"] = sub.apply(is_opinion_row, axis=1)
    sub["section_group"] = sub["is_opinion"].map({True: "opinion", False: "non-opinion"})

    rows = []
    for definition, col in [("strict", "is_ai_strict"), ("broad", "is_ai_broad")]:
        grouped = sub.groupby(["newspaper", "section_group"]).agg(
            total_articles=(col, "size"),
            ai_articles=(col, "sum"),
        ).reset_index()
        grouped["definition"] = definition
        grouped["fraction_ai_use"] = grouped["ai_articles"] / grouped["total_articles"]
        rows.append(grouped)

    out = pd.concat(rows, ignore_index=True)
    out = out[["newspaper", "section_group", "definition", "total_articles", "ai_articles", "fraction_ai_use"]]
    out = out.sort_values(["newspaper", "definition", "section_group"]).reset_index(drop=True)

    path = os.path.join(outdir, "opinion_vs_non_opinion_ai_use.csv")
    out.to_csv(path, index=False)
    print(f"Wrote {path} ({len(out)} rows) "
          f"[restricted to {PERIOD_SPLIT_QUARTER} onward]")


def write_combined_and_per_period(df, path_combined, outdir):
    """
    Write df (which must have a 'period' column) to path_combined as-is,
    and additionally write one CSV per distinct period value, named
    <basename>_<period>.csv, so the before/after split is unambiguous and
    doesn't rely on the reader filtering the 'period' column themselves.
    """
    df.to_csv(path_combined, index=False)
    print(f"Wrote {path_combined} ({len(df)} rows)")

    base, ext = os.path.splitext(os.path.basename(path_combined))
    for period, sub in df.groupby("period"):
        per_period_path = os.path.join(outdir, f"{base}_{period}{ext}")
        sub.to_csv(per_period_path, index=False)
        print(f"Wrote {per_period_path} ({len(sub)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Analyze AI-use across a folder of combined-article files.")
    parser.add_argument("folder", help="Folder containing the combined-article .json/.jsonl files")
    parser.add_argument("--outdir", default=".", help="Directory to write output CSVs to (default: current directory)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    articles_with_dataset = load_folder(args.folder)
    df = build_dataframe(articles_with_dataset)

    print("\n=== Running all analyses on the full dataset ===")
    run_all_analyses(df, args.outdir)

    df_recent = df[df["quarter"].str.startswith(("2024-", "2025-"))].copy()
    outdir_recent = os.path.join(args.outdir, "2024_2025")
    if df_recent.empty:
        print("\nNote: no articles found in 2024 or 2025; skipping the 2024-2025-only analyses.")
    else:
        print(f"\n=== Running all analyses again, restricted to 2024 and 2025 only "
              f"({len(df_recent)} articles) -> {outdir_recent} ===")
        run_all_analyses(df_recent, outdir_recent)


def run_all_analyses(df, outdir):
    os.makedirs(outdir, exist_ok=True)

    # ---------------------------------------------------------------
    # A) AI-use per section
    # ---------------------------------------------------------------
    section_overall = summarize(df, ["section"])
    path = os.path.join(outdir, "ai_use_by_section_overall.csv")
    section_overall.to_csv(path, index=False)
    print(f"Wrote {path} ({len(section_overall)} rows)")

    section_quarter = summarize(df, ["section", "quarter"])
    path = os.path.join(outdir, "ai_use_by_section_quarter.csv")
    section_quarter.to_csv(path, index=False)
    print(f"Wrote {path} ({len(section_quarter)} rows)")

    # ---------------------------------------------------------------
    # B) AI-use per author
    # ---------------------------------------------------------------
    author_df = explode_authors(df)
    n_no_author = int((df["author"].apply(len) == 0).sum())
    print(f"Note: {n_no_author} article(s) had no listed author; they are "
          f"included in the per-author analysis under the author 'NO AUTHOR'.")

    author_overall = summarize(author_df, ["author"])
    path = os.path.join(outdir, "ai_use_by_author_overall.csv")
    author_overall.to_csv(path, index=False)
    print(f"Wrote {path} ({len(author_overall)} rows)")

    author_quarter = summarize(author_df, ["author", "quarter"])
    path = os.path.join(outdir, "ai_use_by_author_quarter.csv")
    author_quarter.to_csv(path, index=False)
    print(f"Wrote {path} ({len(author_quarter)} rows)")

    # ---------------------------------------------------------------
    # C) AI-use in opinion sections (de Telegraaf / de Volkskrant)
    # ---------------------------------------------------------------
    opinion_mask = df.apply(is_opinion_row, axis=1)
    opinion_df = df[opinion_mask].copy()
    opinion_df["source"] = opinion_df["source"].fillna("Unknown")

    if opinion_df.empty:
        print("Note: no articles matched the de Telegraaf / de Volkskrant "
              "opinion sections; opinion CSVs will be empty.")

    opinion_overall = summarize(opinion_df, ["source", "section"])
    path = os.path.join(outdir, "ai_use_opinion_overall.csv")
    opinion_overall.to_csv(path, index=False)
    print(f"Wrote {path} ({len(opinion_overall)} rows)")

    opinion_quarter = summarize(opinion_df, ["source", "section", "quarter"])
    path = os.path.join(outdir, "ai_use_opinion_quarter.csv")
    opinion_quarter.to_csv(path, index=False)
    print(f"Wrote {path} ({len(opinion_quarter)} rows)")

    # ---------------------------------------------------------------
    # D) AI-use per topic
    # ---------------------------------------------------------------
    topic_df = with_all_newspapers_combined(df)
    topic_df["period"] = topic_df["quarter"].apply(quarter_to_period)

    topic_overall = summarize(topic_df, ["topic", "period"])
    path = os.path.join(outdir, "ai_use_by_topic.csv")
    write_combined_and_per_period(topic_overall, path, outdir)

    topic_share_of_ai_use(topic_df, outdir)

    # ---------------------------------------------------------------
    # F) Opinion vs. non-opinion AI-use (de Telegraaf / de Volkskrant,
    #    2023-Q1 onward)
    # ---------------------------------------------------------------
    opinion_vs_non_opinion_analysis(df, outdir)

    # ---------------------------------------------------------------
    # G) Em-dash and quote usage vs. AI-use
    # ---------------------------------------------------------------
    stylometric_analysis(df, outdir)


if __name__ == "__main__":
    main()
