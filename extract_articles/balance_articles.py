"""
Filter newspaper articles based on average monthly topic volume.
"""

import json
import os
import glob
import argparse
from collections import defaultdict
from datetime import datetime
import csv


def parse_year_month(date_str):
    """Parse 'DD/MM/YYYY' → 'YYYY-MM'"""
    try:
        dt = datetime.strptime(date_str.strip(), "%d/%m/%Y")
        return dt.strftime("%Y-%m")
    except ValueError:
        return None


def compute_topic_averages(articles):
    """Calculate the average amount of appearances of a topic per month"""
    topic_months = defaultdict(lambda: defaultdict(int))
    for article in articles:
        ym    = parse_year_month(article.get("date", ""))
        topic = article.get("predicted_topic", "").strip()
        if ym and topic:
            topic_months[topic][ym] += 1

    topic_averages = {}
    for topic, months in topic_months.items():
        counts = list(months.values())
        topic_averages[topic] = sum(counts) / len(counts)

    return topic_averages, topic_months


def get_topics_to_keep(topic_averages, threshold):
    return {topic for topic, avg in topic_averages.items() if avg >= threshold}


def filter_articles(articles, topics_to_keep):
    """Filter out articles that have a topic that isn't featured enough"""
    kept, dropped = [], []
    for article in articles:
        topic = article.get("predicted_topic", "").strip()
        ym    = parse_year_month(article.get("date", ""))
        if not ym:
            dropped.append(article)
        elif topic in topics_to_keep:
            kept.append(article)
        else:
            dropped.append(article)
    return kept, dropped


def main(argv=None):
    parser = argparse.ArgumentParser(description="Filter articles by average monthly topic volume.")
    parser.add_argument('-i', '--input',     default='.',           help='Input directory containing JSON files (default: current dir)')
    parser.add_argument('-o', '--output',    default=None,          help='Output directory for filtered JSON files (default: <input>/filtered)')
    parser.add_argument('-t', '--threshold', default=15,  type=int, help='Minimum average monthly article count to keep a topic (default: 15)')
    args = parser.parse_args(argv)

    if args.output is None:
        args.output = os.path.join(args.input, "balanced")

    os.makedirs(args.output, exist_ok=True)

    json_files    = glob.glob(os.path.join(args.input, "*.json"))
    all_summary   = []
    grand_kept    = 0
    grand_dropped = 0

    if not json_files:
        print(f"No JSON files found in '{args.input}'.")
        return

    for filepath in sorted(json_files):
        filename = os.path.basename(filepath)
        source   = os.path.splitext(filename)[0]

        print(f"\n── {source} ──")

        with open(filepath, "r", encoding="utf-8") as f:
            articles = json.load(f)

        if not isinstance(articles, list):
            print(f"  Skipping: expected a JSON list, got {type(articles)}")
            continue

        # Print overview of balancing
        print(f"  Total articles loaded : {len(articles):>6}")

        topic_averages, topic_months = compute_topic_averages(articles)
        topics_to_keep = get_topics_to_keep(topic_averages, args.threshold)

        print(f"  Topics found          : {len(topic_averages):>6}")
        print(f"  Topics kept (avg >= {args.threshold}): {len(topics_to_keep):>4}")
        print(f"  Topics dropped        : {len(topic_averages) - len(topics_to_keep):>4}")
        print()
        print(f"  {'Topic':<45} {'Avg/month':>10}  {'Decision'}")
        print(f"  {'-'*45} {'-'*10}  {'-'*8}")
        for topic, avg in sorted(topic_averages.items(), key=lambda x: -x[1]):
            decision = "KEEP" if topic in topics_to_keep else "drop"
            print(f"  {topic:<45} {avg:>10.1f}  {decision}")

        kept, dropped = filter_articles(articles, topics_to_keep)
        grand_kept    += len(kept)
        grand_dropped += len(dropped)
        pct = 100 * len(kept) / len(articles) if articles else 0
        print(f"\n  Articles kept         : {len(kept):>6}  ({pct:.1f}%)")
        print(f"  Articles dropped      : {len(dropped):>6}")

        out_path = os.path.join(args.output, f"{source}_balanced.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(kept, f, ensure_ascii=False, indent=2)
        print(f"  Written -> {out_path}")

        for topic, avg in topic_averages.items():
            n_months = len(topic_months[topic])
            total    = sum(topic_months[topic].values())
            all_summary.append({
                "source"         : source,
                "topic"          : topic,
                "months_present" : n_months,
                "total_articles" : total,
                "avg_per_month"  : round(avg, 2),
                "kept"           : "yes" if topic in topics_to_keep else "no",
            })

    summary_path = os.path.join(args.output, "topic_summary.csv")
    fieldnames   = ["source", "topic", "months_present", "total_articles", "avg_per_month", "kept"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_summary)

    print(f"\n── Grand total ──")
    print(f"  Articles kept   : {grand_kept}")
    print(f"  Articles dropped: {grand_dropped}")
    print(f"  Topic summary   -> {summary_path}")


if __name__ == "__main__":
    main()
