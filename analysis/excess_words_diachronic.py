#!/usr/bin/env python3
"""
Diachronic corpus shift analysis (overall, by newspaper group, by newspaper).

Significance testing uses Benjamini–Hochberg (BH) FDR correction applied
WITHIN each scope (overall / group / newspaper) across all 32 tests in that
scope (16 AI tests + 16 BL tests: 2 categories × 2 methods × 4 top-N values).

BH controls the False Discovery Rate — the expected proportion of false
positives among the tests called significant — rather than the Family-Wise
Error Rate of Bonferroni. It is less conservative and more appropriate when
exploring a set of related hypotheses.

BH-adjusted p-values are stored in ai_adj_p / bl_adj_p; significance flags
(ai_sig_bh / bl_sig_bh) mark adjusted p ≤ 0.05.

Outputs
-------
  output_shift/
    summary_all.csv
    overall/{cat}/words_*.csv  …
    groups/{group}/{cat}/…  …
    newspapers/{paper}/{cat}/…  …
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from scipy.stats import chi2 as chi2_dist


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
SCORED_CSV = Path("data/las_word_nl_scored.csv")
CORPUS_DIR = Path("stanza_parse/body")
OUTPUT_DIR = Path("output_shift")

PRE_START  = datetime(2021,  1,  1)
PRE_END    = datetime(2022, 11, 30)
POST_START = datetime(2024,  1,  1)
POST_END   = datetime(2025, 12, 31)
PRE_LABEL  = "2021-Nov2022"
POST_LABEL = "2024-2025"

TOP_N_VALUES = [20, 50, 100, 200]
MAX_N        = max(TOP_N_VALUES)

WORD_CATEGORIES: Dict[str, Set[str]] = {
    "content":  {"NOUN", "VERB", "ADJ", "ADV"},
    "function": {"ADP", "DET", "PRON", "SCONJ", "CCONJ", "AUX"},
}

NEWSPAPER_GROUPS: Dict[str, List[str]] = {
    "national": ["de Telegraaf", "de Volkskrant"],
    "regional": ["Dagblad van het Noorden", "Eindhovens Dagblad"],
    "local":    ["Steenwijker Courant", "Nieuwsblad Noordoost-Friesland"],
}
PAPER_TO_GROUP: Dict[str, str] = {
    paper: group
    for group, papers in NEWSPAPER_GROUPS.items()
    for paper in papers
}

EXCLUDE_UPOS     = {"PUNCT", "SYM", "X", "_SP"}
BASELINE_LPR_CAP = 0.10
MIN_CSV_COUNT    = 20

BH_ALPHA = 0.05   # FDR target; applied to BH-adjusted p-values


# ---------------------------------------------------------------------------
# STATISTICAL HELPERS
# ---------------------------------------------------------------------------
def parse_date(s: str) -> Optional[datetime]:
    try:
        return datetime.strptime(s, "%d/%m/%Y")
    except (ValueError, TypeError):
        return None


def classify(date: datetime) -> Optional[str]:
    if PRE_START <= date <= PRE_END:   return "pre"
    if POST_START <= date <= POST_END: return "post"
    return None


def ppm(count: int, total: int) -> float:
    return count / total * 1_000_000 if total else 0.0


def pct_change(pre: float, post: float) -> Optional[float]:
    return (post - pre) / pre * 100 if pre > 0 else None


def pearson_chi2(a: int, b: int, pre_n: int, post_n: int) -> Tuple[float, float]:
    """Pearson χ² on a 2×2 contingency table (word-set vs rest, pre vs post)."""
    c = pre_n  - a
    d = post_n - b
    N = a + b + c + d
    if N == 0 or (a+b) == 0 or (c+d) == 0 or (a+c) == 0 or (b+d) == 0:
        return 0.0, 1.0
    E_a = (a+b)*(a+c)/N;  E_b = (a+b)*(b+d)/N
    E_c = (c+d)*(a+c)/N;  E_d = (c+d)*(b+d)/N
    stat = sum((o-e)**2/e for o, e in [(a,E_a),(b,E_b),(c,E_c),(d,E_d)] if e > 0)
    return round(stat, 4), float(chi2_dist.sf(stat, df=1))


def bh_correction(
    p_values: List[float],
    alpha: float = BH_ALPHA,
) -> Tuple[List[bool], List[float]]:
    """
    Benjamini–Hochberg FDR correction.

    Returns
    -------
    rejected    : bool list — True if the test is significant at FDR = alpha
    adj_p       : float list — BH-adjusted p-values (compare directly to alpha)

    Adjusted p-value formula (Yekutieli & Benjamini, 1999):
        p̃_(i) = min_{j ≥ i} (m / j · p_(j))
    where tests are sorted by raw p-value ascending and m is the total count.
    """
    m = len(p_values)
    if m == 0:
        return [], []

    order   = sorted(range(m), key=lambda i: p_values[i])
    rank    = [0] * m
    for r, idx in enumerate(order, start=1):
        rank[idx] = r

    # Compute adjusted p-values working backwards from largest rank
    sorted_p  = [p_values[i] for i in order]
    adj_sorted = [0.0] * m
    adj_sorted[-1] = sorted_p[-1]
    for i in range(m - 2, -1, -1):
        adj_sorted[i] = min(adj_sorted[i + 1], sorted_p[i] * m / (i + 1))

    adj = [min(adj_sorted[rank[i] - 1], 1.0) for i in range(m)]
    rejected = [a <= alpha for a in adj]
    return rejected, adj


def sig_stars(rejected: bool, p_raw: float) -> str:
    """
    Display string combining BH decision and raw p-value.
    *** = BH-significant (FDR ≤ 0.05)
      * = nominally p < 0.05 but not BH-significant
        = not significant
    """
    if rejected:  return "***"
    if p_raw < 0.05: return "  *"
    return "   "


# ---------------------------------------------------------------------------
# PRINT WORDLISTS
# ---------------------------------------------------------------------------
def print_word_list(title: str, keys: List[str],
                    key_lookup: Dict[str, dict]) -> None:
    print(f"\n  ── {title}  (n={len(keys)})")
    if not keys:
        print("    (empty)")
        return
    print(f"  {'#':>4}  {'key':<32}  {'ai_G2':>9}  {'ai_LPR':>8}")
    print(f"  {'─'*4}  {'─'*32}  {'─'*9}  {'─'*8}")
    for i, key in enumerate(keys, 1):
        r = key_lookup.get(key, {})
        print(f"  {i:>4}  {key:<32}  "
              f"{r.get('G2',  float('nan')):>9.1f}  "
              f"{r.get('LPR', float('nan')):>+8.3f}")


# ---------------------------------------------------------------------------
# STATS PER WORD
# ---------------------------------------------------------------------------
def word_stats_df(
    keys: List[str], rank_col: str,
    pre_c: Dict[str, int], post_c: Dict[str, int],
    pre_n: int, post_n: int,
    key_lookup: Dict[str, dict],
) -> pd.DataFrame:
    """
    Build per-word DataFrame with corpus counts.
    p-values here are raw (uncorrected); BH correction is applied at
    the summary level across all conditions within a scope.
    """
    rows = []
    for rank, key in enumerate(keys, 1):
        a  = pre_c.get(key, 0)
        b  = post_c.get(key, 0)
        pp = ppm(a, pre_n)
        qp = ppm(b, post_n)
        pc = pct_change(pp, qp)
        chi2_v, p_v = pearson_chi2(a, b, pre_n, post_n)
        r  = key_lookup.get(key, {})
        rows.append({
            rank_col:                  rank,
            "key":                     key,
            "lemma":                   r.get("lemma", ""),
            "upos":                    r.get("upos",  ""),
            "ai_G2":                   r.get("G2",    None),
            "ai_LPR":                  r.get("LPR",   None),
            f"count_{PRE_LABEL}":      a,
            f"count_{POST_LABEL}":     b,
            f"ppm_{PRE_LABEL}":        round(pp, 3),
            f"ppm_{POST_LABEL}":       round(qp, 3),
            "pct_change":              round(pc, 2) if pc is not None else None,
            "chi2":                    chi2_v,
            "p_value_raw":             round(p_v, 8),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------------------------
def run_analysis(
    scope: str,
    pre_c: Dict[str, int], post_c: Dict[str, int],
    pre_n: int, post_n: int,
    cat_data: Dict[str, dict],
    out_dir: Path,
    key_lookup: Dict[str, dict],
) -> List[dict]:
    """
    Run (category × method × top-N) analysis for one scope.
    Returns raw summary rows; BH correction is applied externally after
    all scopes have been processed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for cat_name, data in cat_data.items():
        cat_dir = out_dir / cat_name
        cat_dir.mkdir(exist_ok=True)

        bl_cache: Dict[int, pd.DataFrame] = {}
        for N in TOP_N_VALUES:
            df_bl = word_stats_df(
                data["baseline"][:N], "rank_baseline",
                pre_c, post_c, pre_n, post_n, key_lookup,
            )
            df_bl.to_csv(cat_dir / f"baseline_top{N}.csv", index=False)
            bl_cache[N] = df_bl

        for method in ("G2", "LPR"):
            ranked = data["ai_by_g2"] if method == "G2" else data["ai_by_lpr"]

            for N in TOP_N_VALUES:
                ai_keys_n = ranked[:N]
                bl_keys_n = data["baseline"][:N]

                ai_pre  = sum(pre_c.get(k, 0)  for k in ai_keys_n)
                ai_post = sum(post_c.get(k, 0) for k in ai_keys_n)
                bl_pre  = sum(pre_c.get(k, 0)  for k in bl_keys_n)
                bl_post = sum(post_c.get(k, 0) for k in bl_keys_n)

                ai_pre_ppm  = ppm(ai_pre,  pre_n)
                ai_post_ppm = ppm(ai_post, post_n)
                bl_pre_ppm  = ppm(bl_pre,  pre_n)
                bl_post_ppm = ppm(bl_post, post_n)

                ai_pct = pct_change(ai_pre_ppm, ai_post_ppm)
                bl_pct = pct_change(bl_pre_ppm, bl_post_ppm)

                chi2_ai, p_ai = pearson_chi2(ai_pre,  ai_post,  pre_n, post_n)
                chi2_bl, p_bl = pearson_chi2(bl_pre,  bl_post,  pre_n, post_n)

                df_ai = word_stats_df(
                    ai_keys_n, f"rank_{method}",
                    pre_c, post_c, pre_n, post_n, key_lookup,
                )
                df_ai.to_csv(cat_dir / f"words_{method}_top{N}.csv", index=False)

                ai_pcts = df_ai["pct_change"].dropna()
                bl_pcts = bl_cache[N]["pct_change"].dropna()

                summary_rows.append({
                    "scope":              scope,
                    "category":           cat_name,
                    "method":             method,
                    "top_N":              N,
                    "n_tokens_pre":       pre_n,
                    "n_tokens_post":      post_n,
                    "n_ai_actual":        len(ai_keys_n),
                    "n_bl_actual":        len(bl_keys_n),
                    # AI aggregate
                    "ai_pre_ppm":         round(ai_pre_ppm,  3),
                    "ai_post_ppm":        round(ai_post_ppm, 3),
                    "ai_pct_change":      round(ai_pct,  2) if ai_pct  is not None else None,
                    "ai_mean_word_pct":   round(float(ai_pcts.mean()),   2) if len(ai_pcts) else None,
                    "ai_median_word_pct": round(float(ai_pcts.median()), 2) if len(ai_pcts) else None,
                    "ai_n_increasing":    int((ai_pcts > 0).sum()),
                    "ai_chi2":            chi2_ai,
                    "ai_p_raw":           round(p_ai, 8),
                    # BH fields filled in post-hoc (placeholders)
                    "ai_adj_p":           None,
                    "ai_sig_bh":          False,
                    # Baseline aggregate
                    "bl_pre_ppm":         round(bl_pre_ppm,  3),
                    "bl_post_ppm":        round(bl_post_ppm, 3),
                    "bl_pct_change":      round(bl_pct,  2) if bl_pct  is not None else None,
                    "bl_mean_word_pct":   round(float(bl_pcts.mean()),   2) if len(bl_pcts) else None,
                    "bl_median_word_pct": round(float(bl_pcts.median()), 2) if len(bl_pcts) else None,
                    "bl_n_increasing":    int((bl_pcts > 0).sum()),
                    "bl_chi2":            chi2_bl,
                    "bl_p_raw":           round(p_bl, 8),
                    "bl_adj_p":           None,
                    "bl_sig_bh":          False,
                    # Comparison
                    "ai_pct_exceeds_bl":  (ai_pct is not None and bl_pct is not None
                                           and ai_pct > bl_pct),
                    "bh_alpha":           BH_ALPHA,
                })

    return summary_rows


# ---------------------------------------------------------------------------
# BH FDR
# ---------------------------------------------------------------------------
def apply_bh_per_scope(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Benjamini–Hochberg correction within each scope.

    For each scope the family of tests consists of all 16 AI tests and all
    16 BL tests (= 32 total: 2 categories × 2 methods × 4 top-N × 2 sets).
    Mixing AI and BL tests in one family is conservative: a significant BL
    result 'uses up' some of the FDR budget, penalising AI tests slightly.

    Fills ai_adj_p, ai_sig_bh, bl_adj_p, bl_sig_bh in-place and returns df.
    """
    df = df.copy()
    for scope in df["scope"].unique():
        mask    = df["scope"] == scope
        indices = df.index[mask].tolist()

        ai_pvals = df.loc[indices, "ai_p_raw"].tolist()
        bl_pvals = df.loc[indices, "bl_p_raw"].tolist()
        combined = ai_pvals + bl_pvals

        rejected, adj = bh_correction(combined, alpha=BH_ALPHA)

        n = len(indices)
        for i, idx in enumerate(indices):
            df.at[idx, "ai_adj_p"]  = round(adj[i],     8)
            df.at[idx, "ai_sig_bh"] = bool(rejected[i])
            df.at[idx, "bl_adj_p"]  = round(adj[n + i], 8)
            df.at[idx, "bl_sig_bh"] = bool(rejected[n + i])

    return df


# ---------------------------------------------------------------------------
# PRINT SUMMARY
# ---------------------------------------------------------------------------
SEP  = "=" * 76
SEP2 = "─" * 76

def print_scope_summary(df: pd.DataFrame, scope: str) -> None:
    scope_df = df[df["scope"] == scope]
    if scope_df.empty:
        return

    pre_n  = scope_df.iloc[0]["n_tokens_pre"]
    post_n = scope_df.iloc[0]["n_tokens_post"]
    print(f"\n  Scope: {scope}  "
          f"(pre={pre_n:,} tokens  post={post_n:,} tokens)")

    for cat_name in WORD_CATEGORIES:
        for method in ("G2", "LPR"):
            sub = scope_df[
                (scope_df["category"] == cat_name) &
                (scope_df["method"]   == method)
            ]
            if sub.empty:
                continue
            print(f"\n    {cat_name} · {method}")
            print(f"    {'N':>4}  {'n_ai':>5}  "
                  f"{'AI Δ%':>7}  {'χ²':>8}  {'adj_p':>9}  {'sig':>3}  "
                  f"{'n↑':>4}  |  "
                  f"{'BL Δ%':>7}  {'χ²':>8}  {'adj_p':>9}  {'sig':>3}  "
                  f"{'n↑':>4}  {'AI>BL':>5}")
            print(f"    {'─'*4}  {'─'*5}  "
                  f"{'─'*7}  {'─'*8}  {'─'*9}  {'─'*3}  {'─'*4}  |  "
                  f"{'─'*7}  {'─'*8}  {'─'*9}  {'─'*3}  {'─'*4}  {'─'*5}")
            for _, r in sub.iterrows():
                ai_p = f"{r['ai_pct_change']:+.1f}%" if r["ai_pct_change"] is not None else "   N/A"
                bl_p = f"{r['bl_pct_change']:+.1f}%" if r["bl_pct_change"] is not None else "   N/A"
                s_ai = sig_stars(r["ai_sig_bh"], r["ai_p_raw"])
                s_bl = sig_stars(r["bl_sig_bh"], r["bl_p_raw"])
                gt   = "YES" if r["ai_pct_exceeds_bl"] else " no"
                print(f"    {int(r['top_N']):>4}  {int(r['n_ai_actual']):>5}  "
                      f"{ai_p:>7}  {r['ai_chi2']:>8.1f}  {r['ai_adj_p']:>9.4f}  {s_ai}  "
                      f"{int(r['ai_n_increasing']):>4}  |  "
                      f"{bl_p:>7}  {r['bl_chi2']:>8.1f}  {r['bl_adj_p']:>9.4f}  {s_bl}  "
                      f"{int(r['bl_n_increasing']):>4}  {gt:>5}")


# ---------------------------------------------------------------------------
# LOAD SCORED CSV
# ---------------------------------------------------------------------------
if not SCORED_CSV.exists():
    sys.exit(f"Scored CSV not found: {SCORED_CSV.resolve()}")

df_scored  = pd.read_csv(SCORED_CSV)
key_lookup = df_scored.set_index("key").to_dict("index")

for col in ("G2", "LPR", "c_H", "c_M", "upos"):
    if col not in df_scored.columns:
        sys.exit(f"Required column '{col}' missing. Run scoring step first.")

print(f"Loaded scored CSV: {len(df_scored):,} rows")

cat_data: Dict[str, dict] = {}
for cat_name, cat_upos in WORD_CATEGORIES.items():
    df_cat  = df_scored[
        df_scored["upos"].isin(cat_upos) &
        ((df_scored["c_H"] + df_scored["c_M"]) >= MIN_CSV_COUNT)
    ].copy()
    ai_pool = df_cat[df_cat["LPR"] > 0].copy()
    n_avail = len(ai_pool)

    ai_by_g2  = ai_pool.nlargest(min(MAX_N, n_avail), "G2")["key"].tolist()
    ai_by_lpr = ai_pool.nlargest(min(MAX_N, n_avail), "LPR")["key"].tolist()
    ai_union  = set(ai_by_g2) | set(ai_by_lpr)

    bl_pool = (
        df_cat[~df_cat["key"].isin(ai_union) &
               (df_cat["LPR"].abs() <= BASELINE_LPR_CAP)]
        .assign(abs_lpr=lambda x: x["LPR"].abs())
        .nsmallest(MAX_N, "abs_lpr")
    )
    baseline_keys = bl_pool["key"].tolist()

    cat_data[cat_name] = {
        "upos": cat_upos, "ai_by_g2": ai_by_g2,
        "ai_by_lpr": ai_by_lpr, "baseline": baseline_keys,
    }
    print(f"{cat_name}: {len(df_cat)} words, {n_avail} AI-overused, "
          f"{len(baseline_keys)} baseline")
    if len(baseline_keys) < MAX_N:
        print(f"  NOTE: only {len(baseline_keys)} baseline words available.")

all_keys: Set[str] = set()
for data in cat_data.values():
    all_keys |= set(data["ai_by_g2"]) | set(data["ai_by_lpr"]) | set(data["baseline"])
print(f"Unique keys to track: {len(all_keys):,}")


# ---------------------------------------------------------------------------
# PRINT ALL WORD LISTS
# ---------------------------------------------------------------------------
print(f"\n\n{SEP}")
print(f"  ALL WORD LISTS  (printed before corpus analysis begins)")
print(f"  Significance: BH FDR correction at α={BH_ALPHA}, applied within each scope")
print(SEP)

for cat_name, data in cat_data.items():
    print(f"\n{SEP2}")
    print(f"  CATEGORY: {cat_name.upper()}  "
          f"(UPOS: {', '.join(sorted(data['upos']))})")
    print(SEP2)
    for method in ("G2", "LPR"):
        ranked = data["ai_by_g2"] if method == "G2" else data["ai_by_lpr"]
        for N in TOP_N_VALUES:
            keys_n = ranked[:N]
            suffix = f"  [only {len(keys_n)} available]" if len(keys_n) < N else ""
            print_word_list(
                f"AI-overused · {cat_name} · {method} · top-{N}{suffix}",
                keys_n, key_lookup,
            )
    for N in TOP_N_VALUES:
        bl_n   = data["baseline"][:N]
        suffix = f"  [only {len(bl_n)} available]" if len(bl_n) < N else ""
        print_word_list(
            f"Baseline · {cat_name} · top-{N} (smallest |LPR|){suffix}",
            bl_n, key_lookup,
        )

print(f"\n{SEP}")
print(f"  END OF WORD LISTS — starting corpus pass")
print(SEP)


files = sorted(CORPUS_DIR.glob("*.jsonl"))
if not files:
    sys.exit(f"No .jsonl files found in {CORPUS_DIR.resolve()}")

print(f"\nFound {len(files)} corpus file(s)")
print(f"  PRE  ({PRE_LABEL}): {PRE_START.date()} → {PRE_END.date()}")
print(f"  POST ({POST_LABEL}): {POST_START.date()} → {POST_END.date()}")
print(f"  (Dec 2022 and 2023 skipped)\n")

ov_pre:  Dict[str, int] = defaultdict(int)
ov_post: Dict[str, int] = defaultdict(int)
ov_pre_n = ov_post_n = 0

pp_pre:   Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
pp_post:  Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
pp_pre_n:  Dict[str, int] = defaultdict(int)
pp_post_n: Dict[str, int] = defaultdict(int)

n_skip = 0
seen_papers: Set[str] = set()

for fpath in files:
    n_file = 0
    print(f"  {fpath.name} …", end=" ", flush=True)
    with open(fpath, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            date = parse_date(doc.get("date", ""))
            if date is None:
                continue
            period = classify(date)
            if period is None:
                n_skip += 1
                continue
            paper  = doc.get("source", fpath.stem)
            is_pre = (period == "pre")
            seen_papers.add(paper)
            for sentence in doc.get("parsed_sentences", []):
                for tok in sentence:
                    upos = tok.get("upos", "")
                    if upos in EXCLUDE_UPOS:
                        continue
                    lemma = tok.get("lemma", "").strip().lower()
                    if not lemma:
                        continue
                    key = f"{lemma}_{upos}"
                    if is_pre:
                        ov_pre_n += 1;  pp_pre_n[paper] += 1
                        if key in all_keys:
                            ov_pre[key]         += 1
                            pp_pre[paper][key]  += 1
                    else:
                        ov_post_n += 1; pp_post_n[paper] += 1
                        if key in all_keys:
                            ov_post[key]        += 1
                            pp_post[paper][key] += 1
            n_file += 1
    print(f"{n_file:,} articles")

print(f"\nOverall: pre={ov_pre_n:,} tokens  post={ov_post_n:,}  skipped={n_skip:,} articles")

unexpected = seen_papers - set(PAPER_TO_GROUP.keys())
if unexpected:
    print(f"WARNING: unexpected paper name(s): {unexpected}")

# Build group counts
grp_pre:   Dict[str, Dict[str, int]] = {}
grp_post:  Dict[str, Dict[str, int]] = {}
grp_pre_n:  Dict[str, int] = {}
grp_post_n: Dict[str, int] = {}
for grp, papers in NEWSPAPER_GROUPS.items():
    grp_pre[grp]   = defaultdict(int)
    grp_post[grp]  = defaultdict(int)
    grp_pre_n[grp]  = sum(pp_pre_n[p]  for p in papers)
    grp_post_n[grp] = sum(pp_post_n[p] for p in papers)
    for paper in papers:
        for k, v in pp_pre[paper].items():
            grp_pre[grp][k]  += v
        for k, v in pp_post[paper].items():
            grp_post[grp][k] += v
    print(f"  {grp}: pre={grp_pre_n[grp]:,}  post={grp_post_n[grp]:,}")

for paper in sorted(seen_papers):
    print(f"  {paper}: pre={pp_pre_n[paper]:,}  post={pp_post_n[paper]:,}")


# ---------------------------------------------------------------------------
# RUN ANALYSES
# ---------------------------------------------------------------------------
OUTPUT_DIR.mkdir(exist_ok=True)
all_raw: List[dict] = []

print(f"\n\nRunning analysis: overall …")
all_raw.extend(run_analysis(
    "overall", ov_pre, ov_post, ov_pre_n, ov_post_n,
    cat_data, OUTPUT_DIR / "overall", key_lookup,
))

for grp in NEWSPAPER_GROUPS:
    print(f"Running analysis: group '{grp}' …")
    all_raw.extend(run_analysis(
        f"group:{grp}", grp_pre[grp], grp_post[grp],
        grp_pre_n[grp], grp_post_n[grp],
        cat_data, OUTPUT_DIR / "groups" / grp, key_lookup,
    ))

for paper in sorted(seen_papers):
    safe = "".join(c if c.isalnum() or c in "- " else "_" for c in paper).strip()
    print(f"Running analysis: newspaper '{paper}' …")
    all_raw.extend(run_analysis(
        f"paper:{paper}", pp_pre[paper], pp_post[paper],
        pp_pre_n[paper], pp_post_n[paper],
        cat_data, OUTPUT_DIR / "newspapers" / safe, key_lookup,
    ))


# ---------------------------------------------------------------------------
# APPLY BH CORRECTION
# ---------------------------------------------------------------------------
print(f"\nApplying Benjamini–Hochberg correction (α={BH_ALPHA}) within each scope …")
df_all = apply_bh_per_scope(pd.DataFrame(all_raw))

n_scopes = df_all["scope"].nunique()
for scope in df_all["scope"].unique():
    sub = df_all[df_all["scope"] == scope]
    n_ai_sig = sub["ai_sig_bh"].sum()
    n_bl_sig = sub["bl_sig_bh"].sum()
    print(f"  {scope:<45}  AI sig: {n_ai_sig}/16  BL sig: {n_bl_sig}/16")

df_all.to_csv(OUTPUT_DIR / "summary_all.csv", index=False)
print(f"Saved {len(df_all)} rows → summary_all.csv")

# ---------------------------------------------------------------------------
# CONSOLE OUTPUT
# ---------------------------------------------------------------------------
print(f"\n\n{SEP}")
print(f"  RESULTS — {PRE_LABEL}  →  {POST_LABEL}")
print(f"  Benjamini–Hochberg FDR correction, α={BH_ALPHA}, within each scope")
print(f"  32 tests per scope: 16 AI + 16 BL (2 cats × 2 methods × 4 top-N)")
print(f"  *** BH-significant (adj_p ≤ {BH_ALPHA})   * p_raw < 0.05 (not BH-sig)")
print(f"  adj_p column shows BH-adjusted p-values")
print(SEP)

print(f"\n{SEP2}\n  OVERALL\n{SEP2}")
print_scope_summary(df_all, "overall")

print(f"\n{SEP2}\n  GROUPS\n{SEP2}")
for grp in NEWSPAPER_GROUPS:
    print_scope_summary(df_all, f"group:{grp}")

print(f"\n{SEP2}\n  NEWSPAPERS\n{SEP2}")
for paper in sorted(seen_papers):
    print_scope_summary(df_all, f"paper:{paper}")

# Quick cross-scope comparison: content · LPR · top-50
print(f"\n{SEP}")
print(f"  QUICK COMPARISON — content · LPR · top-50")
print(SEP)
subset = df_all[
    (df_all["category"] == "content") &
    (df_all["method"]   == "LPR")     &
    (df_all["top_N"]    == 50)
].copy()
print(f"\n  {'scope':<42}  {'pre_tok':>9}  {'AI Δ%':>7}  "
      f"{'adj_p':>9}  {'sig':>3}  {'BL Δ%':>7}  {'AI>BL':>5}")
print(f"  {'─'*42}  {'─'*9}  {'─'*7}  "
      f"{'─'*9}  {'─'*3}  {'─'*7}  {'─'*5}")
for _, r in subset.iterrows():
    ai_p = f"{r['ai_pct_change']:+.1f}%" if r["ai_pct_change"] is not None else "   N/A"
    bl_p = f"{r['bl_pct_change']:+.1f}%" if r["bl_pct_change"] is not None else "   N/A"
    gt   = "YES" if r["ai_pct_exceeds_bl"] else " no"
    print(f"  {r['scope']:<42}  {r['n_tokens_pre']:>9,}  {ai_p:>7}  "
          f"{r['ai_adj_p']:>9.4f}  {sig_stars(r['ai_sig_bh'], r['ai_p_raw'])}  "
          f"{bl_p:>7}  {gt:>5}")

print(f"\n{SEP}")
print(f"✓  Results in {OUTPUT_DIR.resolve()}/")
print(f"   summary_all.csv  — raw p-values + BH adj_p + sig_bh flags for all scopes")
print(f"   overall / groups / newspapers — per-word CSVs (raw p-values)")