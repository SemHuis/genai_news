"""
article_metrics.py
==================
Computes all stylometric and focal-word metrics for Dutch newspaper articles.
Designed to read pre-parsed JSONL output from stanza_parse.py.
Can also run in simplemma fallback mode (--simplemma) without pre-parsed input,
but POS-based and syntactic metrics will be NaN in that case.

Run once per outlet. Output filenames should follow the convention
{code}_metrics_results_body.csv, where code is one of:
  tgf   (Telegraaf)
  dvhn  (Dagblad van het Noorden)
  vk    (Volkskrant)
  ed    (Eindhovens Dagblad)
  stc   (Steenwijker Courant)
  nof   (Noordoost Friesland / nnofriesland)
analysis.R expects exactly these six filenames (see its INPUT_FILES config).

Metric families
---------------
LEXICAL DIVERSITY
  mtld               Measure of Textual Lexical Diversity (McCarthy & Jarvis 2010)
                     Computed on content-word lemmas (NOUN, VERB, ADJ, ADV).
                     NaN for articles with fewer than 50 content words.

LEXICAL SOPHISTICATION  (requires --freq_source plus the matching frequency file)
  prop_low_freq      Proportion of content tokens with Zipf < low_freq_threshold
  prop_high_freq     Proportion of content tokens with Zipf >= high_freq_threshold
  mean_zipf          Mean Zipf score across content tokens with known frequency

  Two frequency sources are supported; --freq_source picks which one is used:
    subtlex   Subtlex-NL (Keuleers et al. 2010) word-frequency norms, via --subtlex.
    sonar     A SoNaR-1-derived word-frequency list, via --sonar. SoNaR-1 itself
              ships as an annotated corpus tree (see README.pdf: DOC/COREF/NE/
              SRL/SPT/POS), not a ready-made frequency list, so --sonar expects a
              precomputed word (or lemma) + frequency table that you have already
              tallied from the POS/ directory's DCOI-tagged text (one word/lemma
              per line, e.g. via a one-off counting pass over POS/). If your file
              already has Zipf values, those are used directly; if it has raw
              counts, they are converted to Zipf using van Heuven et al. (2014):
              Zipf = log10(count-per-million-words) + 3, with the corpus size
              taken from --sonar_corpus_size if given, otherwise from the sum of
              all counts in the file.
  If --freq_source is omitted entirely, lexical sophistication metrics are NaN.

FUNCTION-WORD DISTRIBUTIONS
  freq_pronoun       PRON per 1000 words
  freq_aux           AUX per 1000 words
  freq_det           DET per 1000 words
  freq_cconj         CCONJ per 1000 words
  freq_adv           ADV per 1000 words
  freq_adp           ADP per 1000 words

SYNTACTIC COMPLEXITY
  mean_sent_len      Mean sentence length in tokens
  cv_sent_len        Coefficient of variation of sentence lengths (burstiness proxy)
  mean_dep_depth     Mean max dependency tree depth per sentence
  finite_verbs_per_sent  Mean finite verb count per sentence (clause density)
  mean_tunit_len     Mean T-unit length (main clause + subordinates; split at
                     coordinated root clauses)
  nom_rate           Proportion of NOUN tokens with Dutch nominalisation suffixes

FOCAL WORDS  (Juzek 2026 LPR word list required via --wordlist)
  Methodology follows Juzek (2026) Sec. 3.4-3.6: words are not selected by
  an arbitrary score threshold, but by RANK against a log prevalence ratio
  (LPR) computed upstream by the same pipeline diachronic_opm.py reads
  (las_word_{lang}.csv: key, lemma, upos, lpr_MH_guarded, lpr_guard_ok).
  --wordlist must be that file, not an older Delta_MH/relpct_MH-format CSV.

  For each N in --n_seeds (default 20, 50, 200 — Sec. 3.4-3.5's headline
  and embedding-robustness sizes):
    llm_style_word_ratio_top{N}        AI-seed (top-N by LPR) tokens per
                                        1000 words
    llm_style_word_count_top{N}        same, as a raw integer count (exact
                                        input for a corpus-level chi-square
                                        test; avoids re-deriving counts from
                                        a rounded per-1000-words rate)
    weighted_llm_ratio_top{N}_per1k    AI-seed tokens per 1000 words,
                                        weighted by each match's LPR value
    baseline_word_ratio_top{N}         matched baseline (near-zero-LPR,
                                        UPOS-matched to the AI seeds) tokens
                                        per 1000 words — Sec. 3.5's control
                                        condition, not an "anti-AI" extreme
    baseline_word_count_top{N}         raw integer count, baseline
    baseline_weighted_ratio_top{N}_per1k

  {category}_freq_per1k  One column per semantic category (built-in:
                         care_rigour_freq_per1k, emphasize_freq_per1k,
                         importance_freq_per1k — Juzek 2026 Sec. 3.6's three
                         diachronic concepts; or whatever your --categories
                         CSV defines). Validated against the top-
                         --category_pool_n AI pool (default 200) regardless
                         of which --n_seeds were requested. analysis.R
                         discovers these columns automatically and plots
                         each one separately.

                         CAVEAT: Dutch was not among the 10 languages with
                         2012-2024 WMT coverage Juzek (2026) used for the
                         diachronic concept analysis, so only 'emphasize'
                         has a paper-verified Dutch seed lemma
                         ('benadrukken', from the separate 34-language
                         qualitative table). 'care_rigour' and 'importance'
                         ship with unverified best-effort Dutch translations
                         — review/replace via --categories before treating
                         results as directly comparable to the paper's.

GENERAL
  total_words        Total token count
  total_sentences    Sentence count
  char_count         Character count of analysed text
  word_count_title   Word count of title field only (editorial influence proxy)

Usage
-----
  # Full run, SoNaR as the frequency source, default top-20/50/100/200:
  # python3 article_metrics.py --parsed_jsonl /home/semhuis/RUG/msc_thesis/analysis/stanza_parse/body/vk_parsed_body_only.jsonl --wordlist data/las_word_nl.csv --freq_source sonar --sonar data/sonar_newspapers_stanza_format.jsonl --output vk_metrics_results_body.csv

  # Only the headline top-20 analysis (skip 50/200):
  python article_metrics.py \\
      --articles articles.json \\
      --wordlist las_word_nl.csv \\
      --freq_source subtlex \\
      --subtlex subtlex_nl.csv \\
      --n_seeds 20 \\
      --output dvhn_metrics_results_body.csv

  # Full run, SoNaR-1-derived frequency list instead of Subtlex:
  python article_metrics.py \\
      --articles articles.json \\
      --wordlist las_word_nl.csv \\
      --freq_source sonar \\
      --sonar sonar1_word_freq.csv \\
      --output dvhn_metrics_results_body.csv

  # Without focal words:
  python article_metrics.py \\
      --articles articles.json \\
      --freq_source subtlex \\
      --subtlex subtlex_nl.csv \\
      --output dvhn_metrics_results_body.csv

  # Simplemma fallback (no POS/syntax metrics):
  python article_metrics.py \\
      --articles articles.json \\
      --simplemma \\
      --output dvhn_metrics_results_body.csv

  # Resume interrupted run:
  python article_metrics.py \\
      --articles articles.json \\
      --wordlist las_word_nl.csv \\
      --freq_source subtlex \\
      --subtlex subtlex_nl.csv \\
      --output dvhn_metrics_results_body.csv \\
      --checkpoint_every 500

  # HPC recommended settings:
  python article_metrics.py \\
      --articles articles.json \\
      --wordlist las_word_nl.csv \\
      --freq_source subtlex \\
      --subtlex subtlex_nl.csv \\
      --output dvhn_metrics_results_body.csv \\
      --text_scope body_only \\
      --batch_size 64 \\
      --checkpoint_every 1000
"""

import sys

if sys.version_info < (3, 8):
    sys.exit(
        "ERROR: Python 3.8+ required. Current: {}.{}.{}\n"
        "On HPC: module load Python/3.11.5-GCCcore-13.2.0".format(*sys.version_info[:3])
    )

import json
import csv
import argparse
import math
import os
import time
from collections import defaultdict, Counter

import pandas as pd
import simplemma

try:
    from lexicalrichness import LexicalRichness
    HAS_LEXRICH = True
except ImportError:
    HAS_LEXRICH = False
    print("[warning] lexicalrichness not installed — MTLD will be NaN.")
    print("[warning] pip install lexicalrichness")


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Dutch nominalisation suffixes — core productive suffixes only
# Validated against Dutch morphology (de Haas & Trommelen 1993)
# Removed: -age, -ance, -ence, -ment (loanword patterns, not core Dutch)
NOM_SUFFIXES = (
    'ing', 'heid', 'atie', 'sie', 'nis',
    'schap', 'iteit', 'tie',
)

# Content POS tags for MTLD and lexical sophistication
CONTENT_POS = {'NOUN', 'VERB', 'ADJ', 'ADV'}

# Function-word POS categories and output column names
FUNCTION_WORD_POS = {
    'PRON':  'freq_pronoun',
    'AUX':   'freq_aux',
    'DET':   'freq_det',
    'CCONJ': 'freq_cconj',
    'ADV':   'freq_adv',
    'ADP':   'freq_adp',
}

MTLD_MIN_CONTENT_WORDS = 50
MTLD_THRESHOLD = 0.72

DEFAULT_LOW_FREQ  = 3.0   # Zipf < 3.0 → low-frequency (< 1 per 1000 words)
DEFAULT_HIGH_FREQ = 4.0   # Zipf ≥ 5.0 → high-frequency (≥ 10 per million)

# Zipf score assigned to words with FREQlemma=0 in Subtlex, or to words
# not found in the frequency dictionary at runtime. A word appearing
# exactly once in Subtlex-NL's 43.8M-word corpus has
# Zipf = log10(1/43.8) + 3 ≈ 1.36, so 1.0 is a defensible floor: it
# sits just below "barely observed" and keeps all values on the 1–7 scale,
# so that unseen words count as low-frequency for prop_low_freq without
# producing −∞ values that would distort mean_zipf.
ZIPF_FLOOR_FOR_UNSEEN = 1.0


# Default top-N seed sizes for AI-overused / baseline word selection, per
# Juzek (2026) Sec. 3.4-3.5: the headline pre/post analysis uses top-20,
# the embedding-convergence robustness check additionally uses 50/200.
DEFAULT_N_SEEDS = [20, 50, 200]

# The semantic-category word sets (see BUILTIN_FOCAL_CATEGORIES below) are
# validated against the top-CATEGORY_POOL_N AI-overused pool specifically —
# Juzek (2026) Sec. 3.6/4.2 derive their three diachronic concepts from each
# language's top-200 LPR list, independent of whichever N(s) are requested
# for the main AI-vs-baseline ratio metrics.
CATEGORY_POOL_N = 200

# Focal word categories (built-in; overridable via --categories CSV).
# These three concepts — care/rigour, emphasize, importance — are exactly
# the three semantic concepts Juzek (2026) Figure 3 / Table 3 track in the
# 2012-2024 diachronic longitudinal analysis (Sec. 3.6), NOT the slightly
# different three-concept set in Table 2 (emphasize/importance/innovative)
# from the separate qualitative cross-lingual convergence analysis (Sec 4.2).
#
# IMPORTANT CAVEAT: Dutch was not among the 10 languages with 2012-2024 WMT
# coverage used in that diachronic analysis (Table 3 lists cs/de/en/es/fr/
# it/pt/ru/hi/zh only), so the paper provides no verified Dutch lemma list
# for 'care_rigour' or 'importance'. The only paper-confirmed Dutch lemma
# here is 'benadrukken' (emphasize/highlight, Table 2's qualitative table,
# which DOES cover all 34 languages including Dutch). Every other lemma
# below is an unverified best-effort translation of the concept and should
# be reviewed — ideally against this corpus's own top-200 Dutch LPR list —
# before being treated as equivalent to the paper's validated lists. Use
# --categories to supply a reviewed replacement.
BUILTIN_FOCAL_CATEGORIES = {
    'care_rigour': {
        # ADJ: careful, rigorous, thorough, precise, meticulous
        # UNVERIFIED for Dutch — paper has no Dutch entry for this concept.
        'grondig',
    },
    'emphasize': {
        # VERB: emphasize, stress, highlight
        # 'benadrukken' is paper-confirmed (Table 2, NL entry). The rest
        # are unverified extensions of the same concept.
        'benadrukken',
    },
    'importance': {
        # NOUN: importance, significance, priority, necessity
        # UNVERIFIED for Dutch — paper has no Dutch entry for this concept
        # either (not among the 20/34 languages in Table 2's importance row).
        'belang',
    },
}



# ---------------------------------------------------------------------------
# 1.  NLP BACKENDS
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 1.  SIMPLEMMA FALLBACK PARSER
# ---------------------------------------------------------------------------
# Used when --simplemma is passed and no pre-parsed JSONL is provided.
# Provides lemmatisation and basic sentence splitting only.
# POS tags and dependency parses are unavailable in this path.
# For full metrics, use stanza_parse.py to pre-parse your articles
# and pass the output JSONL via --parsed_jsonl.
# ---------------------------------------------------------------------------

class SimplemmaParser:
    """
    Lightweight fallback: simplemma lemmatisation + regex sentence splitting.
    Syntax and POS metrics will be NaN.
    Use stanza_parse.py for full metric coverage.
    """
    import re as _re
    _WORD_RE = _re.compile(
        r"[A-Za-z\xc0-\xd6\xd8-\xf6\xf8-\xff\u0100-\u024F]+"
        r"(?:[-'][A-Za-z\xc0-\xd6\xd8-\xf6\xf8-\xff\u0100-\u024F]+)*",
        _re.UNICODE,
    )
    _ABBREV = {'dr', 'mr', 'ir', 'drs', 'prof', 'vs', 'bijv', 'o.a', 'nl',
               'ca', 'zgn', 'mw', 'dhr', 'ed', 'enz', 'etc', 'fig', 'nr'}
    _SENT_RE = _re.compile(r'(?<=[.!?])\s+(?=[A-Z])', _re.UNICODE)

    def parse_batch(self, texts, batch_size=None):
        return [self._parse_one(t) if t and t.strip() else None for t in texts]

    def _parse_one(self, text):
        raw_sents = self._SENT_RE.split(text)
        sent_lengths = []
        for s in raw_sents:
            toks = self._WORD_RE.findall(s)
            if toks and toks[-1].lower().rstrip('.') in self._ABBREV:
                continue
            if toks:
                sent_lengths.append(len(toks))
        all_tokens = self._WORD_RE.findall(text.lower())
        lemmas = [simplemma.lemmatize(t, lang='nl') for t in all_tokens]
        return _PseudoDoc(all_tokens, lemmas, sent_lengths)

    @property
    def provides_syntax(self):
        return False

    @property
    def name(self):
        return "simplemma (lemma only — POS/syntax metrics unavailable)"


class _PseudoDoc:
    def __init__(self, tokens, lemmas, sent_lengths):
        self.tokens = tokens
        self.lemmas = lemmas
        self.sent_lengths = sent_lengths


# ---------------------------------------------------------------------------
# 2.  WORD-FREQUENCY LOADERS  (Subtlex-NL and SoNaR-1)
# ---------------------------------------------------------------------------
# Both loaders produce the same shape of output: a {lowercased word: zipf}
# dict. Which one is actually used is controlled by --freq_source; only one
# is loaded per run (see process_articles()).
# ---------------------------------------------------------------------------

def _parse_delimited_table(path, label):
    """
    Shared robust parser for tab/comma/semicolon-delimited frequency tables.
    Tries a manual line-by-line parser first (most robust against quoting
    issues in these files), then falls back to pandas. Used by both
    load_subtlex() and load_sonar() so the two loaders share one parsing
    strategy and only differ in which columns they look for afterwards.
    Returns a raw (string-typed) DataFrame; raises ValueError if nothing
    worked.
    """
    df = None

    # Strategy 1: manual parser - most robust against quoting issues
    for enc in ('utf-8', 'latin-1', 'cp1252', 'iso-8859-1'):
        try:
            with open(path, 'r', encoding=enc, errors='replace') as f:
                raw_lines = f.readlines()
            if not raw_lines:
                continue

            header = raw_lines[0].rstrip('\n\r')
            if '\t' in header:
                sep = '\t'
            elif ',' in header:
                sep = ','
            elif ';' in header:
                sep = ';'
            else:
                sep = '\t'

            cols = [c.strip().strip('"\'') for c in header.split(sep)]
            rows = []
            for line in raw_lines[1:]:
                line = line.rstrip('\n\r')
                if not line.strip():
                    continue
                cells = [c.strip().strip('"\'') for c in line.split(sep)]
                if len(cells) >= len(cols):
                    rows.append(cells[:len(cols)])

            if len(rows) < 100:
                continue

            df = pd.DataFrame(rows, columns=cols)
            print("[{}] Loaded {:,} rows  encoding={}  sep={!r}  "
                  "(manual parser)".format(label, len(df), enc, sep))
            break
        except Exception as e:
            print("[{}] Manual parse failed ({}): {}".format(label, enc, e))
            continue

    # Strategy 2: pandas fallback
    if df is None:
        for enc in ('utf-8', 'latin-1', 'cp1252', 'iso-8859-1'):
            for sep in ('\t', ',', ';'):
                try:
                    candidate = pd.read_csv(
                        path, sep=sep, encoding=enc,
                        quoting=3, on_bad_lines='skip',
                        engine='python', dtype=str,
                    )
                    if candidate.shape[1] >= 2 and candidate.shape[0] >= 100:
                        df = candidate
                        print("[{}] Loaded {:,} rows  encoding={}  sep={!r}  "
                              "(pandas)".format(label, len(df), enc, sep))
                        break
                except Exception:
                    continue
            if df is not None:
                break

    if df is None:
        raise ValueError(
            "Cannot parse {} file after all strategies failed.\n"
            "Path: {}\n"
            "Diagnose with: head -3 {}".format(label, path, path)
        )

    df.columns = [c.strip() for c in df.columns]
    return df


def load_subtlex(path, word_col=None, zipf_col=None):
    """
    Load Subtlex-NL frequency norms (Keuleers et al. 2010), building a
    {lowercased_lemma: Zipf_score} dict for use in _lex_sophistication().

    Column strategy (in priority order):
    1. FREQlemma — the sum of raw corpus counts across ALL inflected forms
       of the lemma a headword belongs to (e.g. the entry 'lopen' has
       FREQlemma = sum of 'lopen' + 'loopt' + 'liep' + 'gelopen' + …).
       This is the correct denominator when looking up by lemma (as Stanza
       provides), because it captures the full lexeme frequency, not just
       one word form. Converted to Zipf via van Heuven et al. (2014):
           Zipf = log10(FREQlemma / 43.8) + 3
       where 43.8 is the Subtlex-NL corpus size in millions of words.

    2. FREQcount — raw count of the specific word form. Used if FREQlemma
       is not present. Same Zipf formula (same 43.8M denominator).

    3. A pre-computed Zipf column ('Zipf', 'ZipfValue', …) — used as-is.

    4. Lg10WF — log10(FREQcount + 1). Used only as a last resort, with a
       warning that thresholds (--low_freq_threshold, --high_freq_threshold)
       need re-calibration since this is NOT on the Zipf 1–7 scale.

    Ultra-rare / unseen words:
       Words in the Subtlex file with FREQlemma=0 (or any frequency-source
       value of 0), and words looked up at runtime that are not in the file
       at all, are assigned Zipf = ZIPF_FLOOR = 1.0. Rationale: a word
       appearing exactly once in the 43.8M-word corpus has
       Zipf = log10(1/43.8) + 3 ≈ 1.36, so 1.0 sits just below "barely
       observed" and keeps all values on the meaningful part of the 1–7
       scale. This means unseen words count as low-frequency (below the
       default threshold of 3.0) without producing −∞ values that would
       distort mean_zipf. Note that _lex_sophistication() assigns the floor
       only for words it actually finds in the returned dict; words not in
       the dict at all are excluded from the mean but still counted in the
       low-frequency proportion. See ZIPF_FLOOR_FOR_UNSEEN in the constants
       section if you need to change the floor.

    Lookup at analysis time:
       _lex_sophistication() looks up w.lemma.lower() (Stanza lemma) against
       this dict. Dutch Stanza lemmas are citation forms (infinitive/singular/
       positive), which match Subtlex headword forms for the large majority of
       content words. Mismatches (Stanza lemma differs from the Subtlex
       headword) are simply treated as "not in dict" and excluded from
       mean_zipf (but see the UNSEEN handling in _lex_sophistication).
    """
    SUBTLEX_CORPUS_SIZE_M = 43.8   # Keuleers et al. 2010: 43.8 million tokens

    df = _parse_delimited_table(path, 'subtlex')

    if word_col is None:
        for c in ('Word', 'Spelling', 'word', 'spelling', 'WORD'):
            if c in df.columns:
                word_col = c
                break
    if word_col is None:
        raise ValueError("Cannot detect word column. Columns: {}. "
                         "Use --subtlex_word_col.".format(list(df.columns)))

    # Determine what we're actually computing Zipf from.
    # raw_col: a count column → convert via Zipf formula
    # pre_col: already a Zipf or log-frequency column → use directly (maybe warn)
    raw_col = None
    pre_col = None

    if zipf_col:
        # Caller explicitly specified one column — honour it
        if zipf_col in df.columns:
            pre_col = zipf_col
        else:
            raise ValueError("Specified --subtlex_zipf_col '{}' not in file. "
                             "Columns: {}".format(zipf_col, list(df.columns)))
    else:
        # Auto-detect: prefer FREQlemma → FREQcount → pre-scaled Zipf → Lg10WF
        if 'FREQlemma' in df.columns:
            raw_col = 'FREQlemma'
            print("[subtlex] Using 'FREQlemma' (aggregate lemma count) → converting "
                  "to Zipf = log10(FREQlemma / {}) + 3.".format(SUBTLEX_CORPUS_SIZE_M))
        elif 'FREQcount' in df.columns:
            raw_col = 'FREQcount'
            print("[subtlex] 'FREQlemma' not found; using 'FREQcount' (single-form "
                  "count) → converting to Zipf. Lemma-lookup accuracy may be "
                  "reduced for inflected words.")
        else:
            for c in ('Zipf', 'ZipfValue', 'zipf', 'ZIPF', 'Zipf_value'):
                if c in df.columns:
                    pre_col = c
                    print("[subtlex] Using pre-computed Zipf column '{}'.".format(c))
                    break
            if pre_col is None:
                for c in ('Lg10WF', 'lg10wf', 'Lg10CD'):
                    if c in df.columns:
                        pre_col = c
                        print("[subtlex] WARNING: using '{}' (log10 count, NOT Zipf). "
                              "This column is on a different scale (0–5) from the "
                              "default thresholds (low=3.0, high=5.0). Either pass "
                              "--low_freq_threshold 1.0 --high_freq_threshold 2.5 or "
                              "supply a file with FREQlemma / FREQcount.".format(c))
                        break
            if pre_col is None and raw_col is None:
                raise ValueError(
                    "Cannot find a usable frequency column. Columns: {}. "
                    "Expected: FREQlemma, FREQcount, Zipf, or Lg10WF.".format(
                        list(df.columns)))

    freq_dict = {}
    n_floored = 0
    n_failed  = 0

    for _, row in df.iterrows():
        w = str(row[word_col]).strip().lower()
        if not w or w in ('nan', 'none', ''):
            continue

        col = raw_col or pre_col
        raw = str(row[col]).strip().replace(',', '.').replace(' ', '')
        if raw in ('', 'nan', 'none', 'na', '-', '.'):
            n_failed += 1
            continue
        try:
            val = float(raw)
        except (ValueError, TypeError):
            n_failed += 1
            continue

        if raw_col:
            # Convert raw count to Zipf
            if val <= 0:
                zipf = ZIPF_FLOOR_FOR_UNSEEN
                n_floored += 1
            else:
                zipf = math.log10(val / SUBTLEX_CORPUS_SIZE_M) + 3
        else:
            zipf = val  # already Zipf or Lg10WF

        freq_dict[w] = zipf

    col_used = raw_col or pre_col
    print("[subtlex] {:,} entries loaded ({:,} floored to Zipf={}, "
          "{:,} skipped unparseable).".format(
              len(freq_dict), n_floored, ZIPF_FLOOR_FOR_UNSEEN, n_failed))
    print("[subtlex] Word col: '{}', freq col: '{}' → Zipf scale.".format(
          word_col, col_used))

    if freq_dict:
        vals = list(freq_dict.values())
        print("[subtlex] Zipf range: min={:.3f}, median={:.3f}, max={:.3f} "
              "(expected ~1–7 for a Zipf scale; ~0–5 for Lg10WF).".format(
                  min(vals), sorted(vals)[len(vals)//2], max(vals)))

    if not freq_dict:
        print("[subtlex] WARNING: freq_dict is empty — lexical sophistication will be NaN.")

    return freq_dict


def load_sonar(path, word_col=None, freq_col=None, corpus_size=None):
    """
    Load a SoNaR-1-derived word-frequency list as an alternative to Subtlex-NL.

    Supports two formats:
    1. JSONL — one JSON object per line with at least 'lemma' (or 'text') and
       'frequency' fields, as produced by a corpus frequency tally:
         {"text": "de", "lemma": "de", "upos": "DET", "frequency": 11205631}
       Lookup is by lemma (preferred) or text if lemma is absent.
    2. Delimited table (CSV/TSV) — any encoding, column names auto-detected.
       Column priority: word/lemma column auto-detected; frequency column
       converted from raw count to Zipf if not already a Zipf column.

    Zipf conversion: Zipf = log10(count / corpus_size_in_millions) + 3
    following van Heuven et al. (2014). corpus_size defaults to the sum of
    all counts in the file; pass --sonar_corpus_size to override.
    Words with count = 0 are assigned ZIPF_FLOOR_FOR_UNSEEN.
    """
    import json as _json

    # ── Auto-detect JSONL vs delimited ──────────────────────────────────────
    def _is_jsonl(p):
        try:
            with open(p, 'r', encoding='utf-8') as f:
                first = f.read(1).strip()
            return first == '{'
        except Exception:
            return False

    if _is_jsonl(path):
        # JSONL branch — stream line by line
        print("[sonar] Detected JSONL format — reading lemma/frequency fields.")
        raw_values = {}
        n_failed = 0
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = _json.loads(line)
                except _json.JSONDecodeError:
                    n_failed += 1
                    continue
                # Use lemma as lookup key (Stanza outputs lemmas); fall back to text
                w = str(obj.get('lemma') or obj.get('text') or '').strip().lower()
                if not w or w in ('nan', 'none', ''):
                    n_failed += 1
                    continue
                freq_raw = obj.get('frequency') or obj.get('freq') or obj.get('count')
                if freq_raw is None:
                    n_failed += 1
                    continue
                try:
                    raw_values[w] = float(freq_raw)
                except (ValueError, TypeError):
                    n_failed += 1

        total = corpus_size if corpus_size else sum(raw_values.values())
        if not total or total <= 0:
            raise ValueError(
                "Cannot compute corpus size for Zipf conversion. "
                "Pass --sonar_corpus_size explicitly."
            )
        print("[sonar] Converting JSONL counts to Zipf, corpus size "
              "= {:,.0f} tokens ({}).".format(
                  total, "from --sonar_corpus_size" if corpus_size
                  else "sum of counts in file"))

        freq_dict = {}
        n_floored = 0
        for w, count in raw_values.items():
            if count <= 0:
                freq_dict[w] = ZIPF_FLOOR_FOR_UNSEEN
                n_floored += 1
            else:
                freq_dict[w] = math.log10((count / total) * 1e6) + 3

        print("[sonar] {:,} entries loaded ({:,} floored to Zipf={}, "
              "{:,} skipped).".format(
                  len(freq_dict), n_floored, ZIPF_FLOOR_FOR_UNSEEN, n_failed))
        if freq_dict:
            vals = list(freq_dict.values())
            print("[sonar] Zipf range: min={:.3f}, median={:.3f}, max={:.3f}".format(
                min(vals), sorted(vals)[len(vals)//2], max(vals)))
        return freq_dict

    # ── Delimited table branch (CSV/TSV) ────────────────────────────────────
    df = _parse_delimited_table(path, 'sonar')

    if word_col is None:
        for c in ('Word', 'word', 'WORD', 'Lemma', 'lemma', 'LEMMA',
                  'Spelling', 'spelling'):
            if c in df.columns:
                word_col = c
                break
    if word_col is None:
        raise ValueError("Cannot detect word column. Columns: {}. "
                         "Use --sonar_word_col.".format(list(df.columns)))

    is_zipf = False
    if freq_col is None:
        for c in ('Zipf', 'ZipfValue', 'zipf', 'ZIPF', 'Zipf_value'):
            if c in df.columns:
                freq_col = c
                is_zipf = True
                print("[sonar] Using '{}' as a ready-made Zipf column.".format(c))
                break

    if freq_col is None:
        for c in ('freq', 'frequency', 'Frequency', 'FREQ', 'count', 'Count',
                  'COUNT', 'N', 'n', 'freq_count', 'Freq'):
            if c in df.columns:
                freq_col = c
                print("[sonar] Using '{}' as a raw frequency count column; "
                      "converting to Zipf.".format(c))
                break
    if freq_col is None:
        raise ValueError(
            "Cannot detect frequency column. Columns: {}. "
            "Use --sonar_freq_col.".format(list(df.columns)))

    raw_values = {}
    n_failed = 0
    for _, row in df.iterrows():
        w = str(row[word_col]).strip().lower()
        if not w or w in ('nan', 'none', ''):
            continue
        raw = str(row[freq_col]).strip().replace(',', '.').replace(' ', '')
        if raw in ('', 'nan', 'none', 'na', '-', '.'):
            n_failed += 1
            continue
        try:
            raw_values[w] = float(raw)
        except (ValueError, TypeError):
            n_failed += 1

    if is_zipf:
        freq_dict = raw_values
    else:
        total = corpus_size if corpus_size else sum(raw_values.values())
        if not total or total <= 0:
            raise ValueError(
                "Cannot compute corpus size for Zipf conversion. "
                "Pass --sonar_corpus_size explicitly.")
        print("[sonar] Converting raw counts to Zipf using corpus size "
              "= {:,.0f} tokens ({}).".format(
                  total, "from --sonar_corpus_size" if corpus_size
                  else "sum of counts in file"))
        freq_dict = {}
        n_floored = 0
        for w, count in raw_values.items():
            if count <= 0:
                freq_dict[w] = ZIPF_FLOOR_FOR_UNSEEN
                n_floored += 1
            else:
                freq_dict[w] = math.log10((count / total) * 1e6) + 3

    print("[sonar] {:,} entries loaded (skipped {:,} unparseable).".format(
        len(freq_dict), n_failed))
    print("[sonar] Word col: '{}', frequency col: '{}'".format(word_col, freq_col))
    if freq_dict:
        vals = list(freq_dict.values())
        print("[sonar] Zipf range: min={:.3f}, median={:.3f}, max={:.3f}".format(
            min(vals), sorted(vals)[len(vals)//2], max(vals)))
    if len(freq_dict) == 0:
        print("[sonar] WARNING: freq_dict is empty.")
    return freq_dict




# ---------------------------------------------------------------------------
# 3.  FOCAL WORD LIST LOADER
# ---------------------------------------------------------------------------

def load_focal_words(wordlist_path, n_seeds=None, categories_path=None,
                      category_pool_n=CATEGORY_POOL_N):
    """
    Load a Juzek (2026)-style LPR word list (las_word_{lang}.csv, as produced
    by the same pipeline diachronic_opm.py reads) and build top-N AI-seed /
    near-zero-LPR baseline lookup structures, plus the semantic-category
    lookup, following the paper's Sec. 3.4-3.6 methodology:

      - AI seeds: the top N content words (NOUN/VERB/ADJ/ADV) ranked by
        lpr_MH_guarded descending, restricted to lpr_guard_ok == 1 (the
        paper's count guard, c_M(w) >= 20 — sub-threshold items are already
        zeroed out in lpr_MH_guarded upstream, lpr_guard_ok flags whether an
        item passed). One seed set is built per N in `n_seeds`.
      - Baseline seeds: N content words with the SMALLEST |lpr_MH_guarded|
        (i.e. "near-zero LPR" — model and human prevalence as similar as
        possible), UPOS-matched to that N's AI-seed UPOS distribution, drawn
        from the same guarded pool, excluding the AI seeds themselves. This
        mirrors diachronic_opm.py's load_seeds_for_language(), but matches on
        |LPR| (as Sec. 3.5/5 of the paper explicitly does: "the headline
        baseline selected items with |LPR| ~ 0") rather than that script's
        |LAS| tie-break.
      - Categories: validated against the top-`category_pool_n` AI pool
        specifically (paper default 200, Sec. 3.6/4.2), independent of which
        N(s) were requested above — a category lemma only counts if it
        actually appears in this corpus's own top-200 AI-overused list, not
        merely because it's in the built-in/--categories candidate set.

    Required columns in wordlist_path: key, lemma, upos, lpr_MH_guarded,
    lpr_guard_ok. (Older las_word CSVs using Delta_MH/relpct_MH columns are
    NOT compatible with this schema — that was a column-naming mismatch in
    an earlier version of this script that never matched the real LPR
    pipeline output; see the CLI --help / module docstring for detail.)

    Returns (focal_lemmas, baseline_lemmas, category_lemmas):
      focal_lemmas / baseline_lemmas: {N: {lemma: [{key, upos, weight}, ...]}}
      category_lemmas: {category_name: set(lemma)}
    """
    if n_seeds is None:
        n_seeds = DEFAULT_N_SEEDS

    df = pd.read_csv(wordlist_path)

    base_required = {'key', 'lemma', 'upos'}
    missing_base = base_required - set(df.columns)
    if missing_base:
        raise ValueError(
            "Wordlist file '{}' is missing required column(s): {}.\n"
            "Columns found: {}.".format(
                wordlist_path, sorted(missing_base), list(df.columns)
            )
        )

    if {'lpr_MH_guarded', 'lpr_guard_ok'}.issubset(df.columns):
        pass  # already in the pre-guarded form
    elif {'LAS', 'c_M'}.issubset(df.columns):
        # Real-world las_word_{lang}.csv schema: no pre-guarded LPR columns,
        # but LAS (the AI-association ranking score diachronic_opm.py
        # already ranks/abs()-matches on) and c_M (the model-side count) are
        # present. Derive the guard exactly as Juzek (2026) Sec. 3.4 defines
        # it (c_M(w) >= 20; sub-threshold items get LPR=0).
        df['lpr_guard_ok'] = (df['c_M'] >= 20).astype(int)
        df['lpr_MH_guarded'] = df['LAS'].astype(float).where(
            df['lpr_guard_ok'] == 1, 0.0
        )
    else:
        raise ValueError(
            "Wordlist file '{}' has neither (lpr_MH_guarded, lpr_guard_ok) "
            "nor (LAS, c_M) columns, so the AI-association ranking can't be "
            "derived. Columns found: {}.".format(
                wordlist_path, list(df.columns)
            )
        )

    # Guarded content-word pool: passed the count guard, content UPOS only.
    pool = df[(df['lpr_guard_ok'] == 1) & df['upos'].isin(CONTENT_POS)].copy()
    pool['lpr_MH_guarded'] = pool['lpr_MH_guarded'].astype(float)
    pool['abs_lpr'] = pool['lpr_MH_guarded'].abs()

    def _select_baseline(ai_seeds, exclude_keys):
        """UPOS-matched, smallest |LPR|, excluding the AI seeds themselves."""
        upos_counts = Counter(ai_seeds['upos'])
        candidates = pool[~pool['key'].isin(exclude_keys)].sort_values('abs_lpr')
        parts = []
        for upos, count in upos_counts.items():
            parts.append(candidates[candidates['upos'] == upos].head(count))
        if not parts:
            return pool.iloc[0:0]
        return pd.concat(parts).reset_index(drop=True)

    def _to_lemma_dict(seed_df):
        out = defaultdict(list)
        for _, row in seed_df.iterrows():
            out[str(row['lemma']).lower()].append({
                'key':    row['key'],
                'upos':   row['upos'],
                'weight': float(row['lpr_MH_guarded']),
            })
        return dict(out)

    focal_lemmas = {}
    baseline_lemmas = {}
    for n in n_seeds:
        ai_seeds = pool.sort_values('lpr_MH_guarded', ascending=False).head(n)
        bl_seeds = _select_baseline(ai_seeds, set(ai_seeds['key']))
        focal_lemmas[n] = _to_lemma_dict(ai_seeds)
        baseline_lemmas[n] = _to_lemma_dict(bl_seeds)
        print("[focal] top-{}: {} AI-seed lemmas, {} baseline lemmas "
              "(UPOS-matched, |LPR| ~ 0).".format(
                  n, len(focal_lemmas[n]), len(baseline_lemmas[n])
              ))

    # Category pool: top-`category_pool_n` AI seeds, independent of n_seeds.
    category_ai_pool = pool.sort_values(
        'lpr_MH_guarded', ascending=False
    ).head(category_pool_n)
    category_pool_lemmas = set(category_ai_pool['lemma'].str.lower())

    if categories_path:
        cat_df = pd.read_csv(categories_path)
        raw_cats = defaultdict(set)
        for _, row in cat_df.iterrows():
            raw_cats[row['category']].add(str(row['lemma']).strip().lower())
        print("[focal] External category mapping loaded.")
    else:
        raw_cats = BUILTIN_FOCAL_CATEGORIES

    category_lemmas = {
        cat: lemmas & category_pool_lemmas
        for cat, lemmas in raw_cats.items()
    }

    print("[focal] Categories validated against top-{} AI pool:".format(
        category_pool_n
    ))
    for cat, lemmas in category_lemmas.items():
        if not lemmas:
            print("[focal]   {}: 0 lemmas matched — none of the candidate "
                  "words for this category appear in this corpus's own "
                  "top-{} AI-overused list. Check --categories or revisit "
                  "the built-in list for this language.".format(
                      cat, category_pool_n
                  ))
        else:
            print("[focal]   {}: {} lemmas matched: {}".format(
                cat, len(lemmas), sorted(lemmas)
            ))

    return focal_lemmas, baseline_lemmas, category_lemmas


# ---------------------------------------------------------------------------
# 4.  METRIC COMPUTATION
# ---------------------------------------------------------------------------

def compute_all_metrics(
    doc_or_pseudo,
    provides_syntax,
    freq_dict,
    low_thresh,
    high_thresh,
    focal_lemmas,
    baseline_lemmas,
    category_lemmas,
    article,
    text_scope,
):
    """Dispatcher — routes to pre-parsed (full metrics) or simplemma (limited) path."""
    # Character count and title word count are always available from raw article
    text_analysed = get_text(article, text_scope)
    char_count = len(text_analysed)
    title_text = article.get('title') or ''
    import re as _re
    word_count_title = len(_re.findall(r'\S+', title_text))

    base = {
        'char_count':       char_count,
        'word_count_title': word_count_title,
    }

    if doc_or_pseudo is None:
        m = empty_metrics(focal_lemmas, category_lemmas)
        m.update(base)
        return m

    if provides_syntax:
        m = compute_stanza_metrics(
            doc_or_pseudo, freq_dict, low_thresh, high_thresh,
            focal_lemmas, baseline_lemmas, category_lemmas
        )
    else:
        m = compute_simplemma_metrics(
            doc_or_pseudo, freq_dict, low_thresh, high_thresh,
            focal_lemmas, baseline_lemmas, category_lemmas
        )

    m.update(base)
    return m


def compute_stanza_metrics(doc, freq_dict, low_thresh, high_thresh,
                            focal_lemmas, baseline_lemmas, category_lemmas):
    all_words = [w for sent in doc.sentences for w in sent.words]
    total_words = len(all_words)
    total_sentences = len(doc.sentences)

    if total_words == 0:
        return empty_metrics(focal_lemmas, category_lemmas)

    # Content words for MTLD and lexical sophistication
    content_words = [
        w for w in all_words
        if w.upos in CONTENT_POS and w.lemma
    ]

    # --- MTLD ---
    mtld = _compute_mtld_from_lemmas(
        [w.lemma.lower() for w in content_words]
    )

    # --- Lexical sophistication ---
    lex = _lex_sophistication(content_words, freq_dict, low_thresh, high_thresh,
                               use_lemma=True)

    # --- Function words ---
    fw = _function_words(all_words, total_words)

    # --- Syntax ---
    syn = _syntax_stanza(doc, all_words, total_words, total_sentences)

    # --- Focal words (top-N AI seeds + matched baseline + categories) ---
    foc = _focal_words_stanza(all_words, total_words, focal_lemmas,
                               baseline_lemmas, category_lemmas)

    result = {
        'total_words':     total_words,
        'total_sentences': total_sentences,
        'mtld':            mtld,
    }
    result.update(lex)
    result.update(fw)
    result.update(syn)
    result.update(foc)
    return result


def compute_simplemma_metrics(pseudo, freq_dict, low_thresh, high_thresh,
                               focal_lemmas, baseline_lemmas, category_lemmas):
    tokens  = pseudo.tokens
    lemmas  = pseudo.lemmas
    total_words = len(tokens)

    if total_words == 0:
        return empty_metrics(focal_lemmas, category_lemmas)

    # MTLD on all lemmas (no POS filter available)
    mtld = _compute_mtld_from_lemmas(lemmas)

    # Lexical sophistication — lookup by surface form (no lemma available)
    # Same floor logic as _lex_sophistication(): OOV tokens count as
    # low-frequency for prop_low_freq but are excluded from mean_zipf.
    low_c = high_c = 0
    zipf_scores = []
    for tok in tokens:
        z = freq_dict.get(tok, ZIPF_FLOOR_FOR_UNSEEN)
        if z < low_thresh:
            low_c += 1
        if z >= high_thresh:
            high_c += 1
        if tok in freq_dict:
            zipf_scores.append(z)
    n = total_words
    lex = {
        'prop_low_freq':  round(low_c / n, 4) if n else None,
        'prop_high_freq': round(high_c / n, 4) if n else None,
        'mean_zipf':      round(sum(zipf_scores)/len(zipf_scores), 4) if zipf_scores else None,
    }

    # Sentence lengths from improved regex splitter
    sent_lengths = pseudo.sent_lengths
    syn = {
        'mean_sent_len':         _safe_mean(sent_lengths),
        'cv_sent_len':           _safe_cv(sent_lengths),
        'mean_dep_depth':        None,
        'finite_verbs_per_sent': None,
        'mean_tunit_len':        None,
        'nom_rate':              None,
    }
    if syn['mean_sent_len'] is not None:
        syn['mean_sent_len'] = round(syn['mean_sent_len'], 4)
    if syn['cv_sent_len'] is not None:
        syn['cv_sent_len'] = round(syn['cv_sent_len'], 4)

    # Function words — unavailable without POS
    fw = {col: None for col in FUNCTION_WORD_POS.values()}

    # Focal words — lemma-only matching (no UPOS filter)
    foc = _focal_words_simplemma(
        lemmas, total_words, focal_lemmas, baseline_lemmas, category_lemmas
    )

    result = {
        'total_words':     total_words,
        'total_sentences': len(sent_lengths),
        'mtld':            mtld,
    }
    result.update(lex)
    result.update(fw)
    result.update(syn)
    result.update(foc)
    return result


# ---------------------------------------------------------------------------
# 5.  COMPONENT FUNCTIONS
# ---------------------------------------------------------------------------

def _compute_mtld_from_lemmas(lemma_list):
    """
    Compute MTLD from a list of lemmas (content words only).
    Returns None if fewer than MTLD_MIN_CONTENT_WORDS lemmas.
    """
    if not HAS_LEXRICH or len(lemma_list) < MTLD_MIN_CONTENT_WORDS:
        return None
    try:
        lex = LexicalRichness(' '.join(lemma_list))
        if lex.words < MTLD_MIN_CONTENT_WORDS:
            return None
        return round(lex.mtld(threshold=MTLD_THRESHOLD), 4)
    except Exception:
        return None


def _lex_sophistication(content_words, freq_dict, low_thresh, high_thresh,
                         use_lemma=True):
    """
    Compute prop_low_freq, prop_high_freq, mean_zipf over content words.

    Lookup key: w.lemma.lower() (Stanza lemma, citation form) when
    use_lemma=True; w.text.lower() (surface form) otherwise.

    Words not found in freq_dict are assigned ZIPF_FLOOR_FOR_UNSEEN (=1.0)
    so they count as low-frequency for prop_low_freq — they are genuinely
    rare from the model's perspective — but are EXCLUDED from mean_zipf to
    avoid the floor value biasing the central tendency estimate. This is the
    conservative choice: prop_low_freq may be slightly inflated by genuine
    OOV words (proper nouns, neologisms, technical terms), but mean_zipf
    reflects only words with real frequency evidence.
    """
    if not freq_dict or not content_words:
        return {'prop_low_freq': None, 'prop_high_freq': None, 'mean_zipf': None}

    n = len(content_words)
    low_c = high_c = 0
    zipf_scores = []

    for w in content_words:
        key = (w.lemma.lower() if use_lemma else w.text.lower())
        z = freq_dict.get(key, ZIPF_FLOOR_FOR_UNSEEN)
        # prop_low/high: every content word contributes (OOV = ZIPF_FLOOR)
        if z < low_thresh:
            low_c += 1
        if z >= high_thresh:
            high_c += 1
        # mean_zipf: only words with real frequency evidence (not OOV floor)
        if key in freq_dict:
            zipf_scores.append(z)

    return {
        'prop_low_freq':  round(low_c / n, 4),
        'prop_high_freq': round(high_c / n, 4),
        'mean_zipf':      round(sum(zipf_scores)/len(zipf_scores), 4) if zipf_scores else None,
    }


def _function_words(all_words, total_words):
    if total_words == 0:
        return {col: None for col in FUNCTION_WORD_POS.values()}
    counts = defaultdict(int)
    for w in all_words:
        if w.upos in FUNCTION_WORD_POS:
            counts[w.upos] += 1
    per1k = 1000.0 / total_words
    return {
        col: round(counts[pos] * per1k, 4)
        for pos, col in FUNCTION_WORD_POS.items()
    }


def _syntax_stanza(doc, all_words, total_words, total_sentences):
    if total_sentences == 0:
        return _empty_syntax()

    sent_lengths   = []
    dep_depths     = []
    finite_counts  = []
    tunit_lens     = []

    for sent in doc.sentences:
        words = sent.words
        sent_lengths.append(len(words))

        # Dependency depth: max depth in this sentence's tree
        dep_depths.append(_max_dep_depth(words))

        # Finite verb count: VERB or AUX with VerbForm=Fin in feats
        # This is the clean clause density measure — no deprel mixing
        n_finite = sum(
            1 for w in words
            if w.feats and 'VerbForm=Fin' in w.feats
            and w.upos in ('VERB', 'AUX')
        )
        finite_counts.append(max(1, n_finite))

        # T-unit length: sentence split at coordinated root clauses
        # Root clause IDs
        root_ids = {w.id for w in words if w.deprel == 'root'}
        # Direct conj dependents of root = coordinated main clauses
        conj_of_root = {
            w.id for w in words
            if w.deprel == 'conj' and w.head in root_ids
        }
        n_tunits = max(1, len(root_ids) + len(conj_of_root))
        tunit_lens.append(len(words) / n_tunits)

    # Nominalisation rate: NOUN tokens with Dutch nom suffixes / all NOUNs
    nouns = [w for w in all_words if w.upos == 'NOUN' and w.lemma]
    nom_count = sum(1 for w in nouns if w.lemma.lower().endswith(NOM_SUFFIXES))
    nom_rate = round(nom_count / len(nouns), 4) if nouns else None

    return {
        'mean_sent_len':         _r(_safe_mean(sent_lengths)),
        'cv_sent_len':           _r(_safe_cv(sent_lengths)),
        'mean_dep_depth':        _r(_safe_mean(dep_depths)),
        'finite_verbs_per_sent': _r(_safe_mean(finite_counts)),
        'mean_tunit_len':        _r(_safe_mean(tunit_lens)),
        'nom_rate':              nom_rate,
    }


def _max_dep_depth(words):
    """Walk each word's head chain to root; return max depth in sentence."""
    id_to_word = {w.id: w for w in words}
    max_depth = 0
    for word in words:
        depth = 0
        current = word
        visited = set()
        while current.head != 0 and current.id not in visited:
            visited.add(current.id)
            parent = id_to_word.get(current.head)
            if parent is None:
                break
            current = parent
            depth += 1
        if depth > max_depth:
            max_depth = depth
    return max_depth


def _focal_words_stanza(all_words, total_words, focal_lemmas, baseline_lemmas,
                         category_lemmas):
    """
    Focal word matching using lemma+UPOS, following Juzek (2026)'s top-N
    AI-overused-word / near-zero-LPR-baseline design (Sec. 3.4-3.6).

    Produces, for each N in focal_lemmas/baseline_lemmas (default 20/50/200):
      llm_style_word_ratio_top{N}       per-1000-words rate, AI seeds
      llm_style_word_count_top{N}       raw count, AI seeds (for chi-square
                                         reconstruction downstream)
      weighted_llm_ratio_top{N}_per1k   per-1000-words rate weighted by LPR
      baseline_word_ratio_top{N}        per-1000-words rate, baseline seeds
      baseline_word_count_top{N}        raw count, baseline seeds
      baseline_weighted_ratio_top{N}_per1k

    plus one {category}_freq_per1k per semantic category (care_rigour,
    emphasize, importance by default — drawn from the top-200 AI pool
    independent of N; see load_focal_words()).
    """
    ns = sorted(focal_lemmas.keys())
    has_data = (
        any(len(focal_lemmas.get(n, {})) > 0 for n in ns) or
        any(len(baseline_lemmas.get(n, {})) > 0 for n in ns) or
        any(len(lemmas) > 0 for lemmas in category_lemmas.values())
    )
    if total_words == 0 or not ns or not has_data:
        return _empty_focal(focal_lemmas, category_lemmas)

    focal_counts      = {n: 0 for n in ns}
    focal_weighted     = {n: 0.0 for n in ns}
    baseline_counts    = {n: 0 for n in ns}
    baseline_weighted  = {n: 0.0 for n in ns}
    cat_counts = {cat: 0 for cat in category_lemmas}

    for w in all_words:
        lemma = w.lemma.lower() if w.lemma else ''
        if not lemma:
            continue

        for n in ns:
            entries = focal_lemmas[n].get(lemma)
            if entries:
                matching = [e for e in entries if e['upos'] == w.upos]
                if matching:
                    best = max(matching, key=lambda e: e['weight'])
                    focal_counts[n] += 1
                    focal_weighted[n] += best['weight']

            bl_entries = baseline_lemmas[n].get(lemma)
            if bl_entries:
                bl_matching = [e for e in bl_entries if e['upos'] == w.upos]
                if bl_matching:
                    bl_best = max(bl_matching, key=lambda e: e['weight'])
                    baseline_counts[n] += 1
                    baseline_weighted[n] += abs(bl_best['weight'])

        for cat, cat_lemmas in category_lemmas.items():
            if lemma in cat_lemmas:
                cat_counts[cat] += 1
                break

    per1k = 1000.0 / total_words
    result = {}
    for n in ns:
        result['llm_style_word_ratio_top{}'.format(n)] = round(focal_counts[n] * per1k, 4)
        result['llm_style_word_count_top{}'.format(n)] = focal_counts[n]
        result['weighted_llm_ratio_top{}_per1k'.format(n)] = round(focal_weighted[n] * per1k, 4)
        result['baseline_word_ratio_top{}'.format(n)] = round(baseline_counts[n] * per1k, 4)
        result['baseline_word_count_top{}'.format(n)] = baseline_counts[n]
        result['baseline_weighted_ratio_top{}_per1k'.format(n)] = round(baseline_weighted[n] * per1k, 4)
    for cat in category_lemmas:
        result['{}_freq_per1k'.format(cat)] = round(cat_counts[cat] * per1k, 4)
    return result


def _focal_words_simplemma(lemmas, total_words, focal_lemmas, baseline_lemmas,
                            category_lemmas):
    """Same as _focal_words_stanza but lemma-only matching (no UPOS available
    in the simplemma fallback path)."""
    ns = sorted(focal_lemmas.keys())
    has_data = (
        any(len(focal_lemmas.get(n, {})) > 0 for n in ns) or
        any(len(baseline_lemmas.get(n, {})) > 0 for n in ns) or
        any(len(cl) > 0 for cl in category_lemmas.values())
    )
    if total_words == 0 or not ns or not has_data:
        return _empty_focal(focal_lemmas, category_lemmas)

    focal_counts      = {n: 0 for n in ns}
    focal_weighted     = {n: 0.0 for n in ns}
    baseline_counts    = {n: 0 for n in ns}
    baseline_weighted  = {n: 0.0 for n in ns}
    cat_counts = {cat: 0 for cat in category_lemmas}

    for lemma in lemmas:
        for n in ns:
            entries = focal_lemmas[n].get(lemma)
            if entries:
                best = max(entries, key=lambda e: e['weight'])
                focal_counts[n] += 1
                focal_weighted[n] += best['weight']

            bl_entries = baseline_lemmas[n].get(lemma)
            if bl_entries:
                bl_best = max(bl_entries, key=lambda e: e['weight'])
                baseline_counts[n] += 1
                baseline_weighted[n] += abs(bl_best['weight'])

        for cat, cat_lemmas in category_lemmas.items():
            if lemma in cat_lemmas:
                cat_counts[cat] += 1
                break

    per1k = 1000.0 / total_words
    result = {}
    for n in ns:
        result['llm_style_word_ratio_top{}'.format(n)] = round(focal_counts[n] * per1k, 4)
        result['llm_style_word_count_top{}'.format(n)] = focal_counts[n]
        result['weighted_llm_ratio_top{}_per1k'.format(n)] = round(focal_weighted[n] * per1k, 4)
        result['baseline_word_ratio_top{}'.format(n)] = round(baseline_counts[n] * per1k, 4)
        result['baseline_word_count_top{}'.format(n)] = baseline_counts[n]
        result['baseline_weighted_ratio_top{}_per1k'.format(n)] = round(baseline_weighted[n] * per1k, 4)
    for cat in category_lemmas:
        result['{}_freq_per1k'.format(cat)] = round(cat_counts[cat] * per1k, 4)
    return result


# ---------------------------------------------------------------------------
# 6.  HELPERS
# ---------------------------------------------------------------------------

def _safe_mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def _safe_cv(values):
    """Coefficient of variation = std / mean. None if < 2 values."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    if mean == 0:
        return None
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance) / mean


def _r(v, digits=4):
    """Round if not None."""
    return round(v, digits) if v is not None else None


def _empty_syntax():
    return {
        'mean_sent_len':         None,
        'cv_sent_len':           None,
        'mean_dep_depth':        None,
        'finite_verbs_per_sent': None,
        'mean_tunit_len':        None,
        'nom_rate':              None,
    }


def _empty_focal(focal_lemmas, category_lemmas):
    ns = sorted(focal_lemmas.keys()) if focal_lemmas else DEFAULT_N_SEEDS
    result = {}
    for n in ns:
        result['llm_style_word_ratio_top{}'.format(n)] = None
        result['llm_style_word_count_top{}'.format(n)] = None
        result['weighted_llm_ratio_top{}_per1k'.format(n)] = None
        result['baseline_word_ratio_top{}'.format(n)] = None
        result['baseline_word_count_top{}'.format(n)] = None
        result['baseline_weighted_ratio_top{}_per1k'.format(n)] = None
    for cat in category_lemmas:
        result['{}_freq_per1k'.format(cat)] = None
    return result


def empty_metrics(focal_lemmas, category_lemmas):
    result = {
        'total_words':     0,
        'total_sentences': 0,
        'mtld':            None,
        'prop_low_freq':   None,
        'prop_high_freq':  None,
        'mean_zipf':       None,
    }
    for col in FUNCTION_WORD_POS.values():
        result[col] = None
    result.update(_empty_syntax())
    result.update(_empty_focal(focal_lemmas, category_lemmas))
    return result


TEXT_SCOPES = {
    'all':            ['title', 'highlight', 'full_text'],
    'body_only':      ['full_text'],
    'title_only':     ['title'],
    'highlight_only': ['highlight'],
}

def get_text(article, text_scope='all'):
    fields = TEXT_SCOPES.get(text_scope, TEXT_SCOPES['all'])
    parts = [article.get(f) or '' for f in fields]
    return ' '.join(p for p in parts if p.strip())


# ---------------------------------------------------------------------------
# 7.  CHECKPOINT SYSTEM
# ---------------------------------------------------------------------------

def load_checkpoint(output_path):
    """
    If output_path already exists, read it and return
    (set of already-processed (source, date, title) keys, list of existing rows).
    """
    if not os.path.exists(output_path):
        return set(), []

    existing_rows = []
    done_keys = set()
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows.append(row)
                done_keys.add((
                    row.get('source', ''),
                    row.get('date', ''),
                    row.get('title', ''),
                ))
        print("[checkpoint] Resuming — {:,} articles already processed.".format(
            len(existing_rows)
        ))
    except Exception as e:
        print("[checkpoint] Could not read existing output ({}). Starting fresh.".format(e))
        return set(), []

    return done_keys, existing_rows


def write_checkpoint(output_path, rows, fieldnames):
    """Write all rows to output CSV (overwrite)."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# 8.  MAIN PROCESSING LOOP
# ---------------------------------------------------------------------------

def load_parsed_jsonl(path, max_articles=None):
    """
    Load pre-parsed articles from a stanza_parse.py JSONL output file.
    Returns a list of dicts with metadata + parsed_sentences.
    """
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if max_articles and len(records) >= max_articles:
                break
    print("[parsed] Loaded {:,} pre-parsed articles from {}".format(
        len(records), path
    ))
    return records


class PreParsedDoc:
    """
    Wraps a parsed_sentences list from stanza_parse.py output
    into an object that looks like a Stanza Document to the metric functions.
    """
    def __init__(self, parsed_sentences):
        self.sentences = [
            PreParsedSentence(sent) for sent in (parsed_sentences or [])
        ]


class PreParsedWord:
    """Wraps a token dict to look like a Stanza Word."""
    __slots__ = ('id', 'text', 'lemma', 'upos', 'feats', 'deprel', 'head')

    def __init__(self, d):
        self.id     = d.get('id', 0)
        self.text   = d.get('text', '')
        self.lemma  = d.get('lemma', '')
        self.upos   = d.get('upos', '')
        self.feats  = d.get('feats', '')
        self.deprel = d.get('deprel', '')
        self.head   = d.get('head', 0)


class PreParsedSentence:
    """Wraps a list of token dicts to look like a Stanza Sentence."""
    def __init__(self, token_list):
        self.words = [PreParsedWord(t) for t in (token_list or [])]


def process_articles(
    articles_path,
    output_path,
    parsed_jsonl_path=None,
    wordlist_path=None,
    categories_path=None,
    n_seeds=None,
    category_pool_n=CATEGORY_POOL_N,
    freq_source=None,
    subtlex_path=None,
    subtlex_word_col=None,
    subtlex_zipf_col=None,
    sonar_path=None,
    sonar_word_col=None,
    sonar_freq_col=None,
    sonar_corpus_size=None,
    low_freq_threshold=DEFAULT_LOW_FREQ,
    high_freq_threshold=DEFAULT_HIGH_FREQ,
    text_scope='all',
    use_simplemma=False,
    batch_size=32,
    checkpoint_every=500,
    max_articles=None,
):
    t_start = time.time()

    # Determine parsing mode
    if parsed_jsonl_path:
        # Primary mode: read pre-parsed JSONL from stanza_parse.py
        # No NLP pipeline needed here
        parser = None
        provides_syntax = True
    else:
        # Fallback mode: simplemma only
        # For full metrics run stanza_parse.py first, then pass --parsed_jsonl
        parser = SimplemmaParser()
        provides_syntax = parser.provides_syntax
        if not use_simplemma:
            print("[parser] WARNING: no --parsed_jsonl provided.")
            print("[parser] Running simplemma fallback — POS/syntax metrics will be NaN.")
            print("[parser] For full metrics: run stanza_parse.py first, then use --parsed_jsonl.")

    # Load word-frequency norms — exactly one of Subtlex-NL or SoNaR-1,
    # chosen via --freq_source. Neither is loaded if freq_source is None.
    freq_dict = {}
    if freq_source == 'subtlex':
        if not subtlex_path:
            sys.exit("ERROR: --freq_source subtlex requires --subtlex PATH.")
        freq_dict = load_subtlex(subtlex_path, subtlex_word_col, subtlex_zipf_col)
    elif freq_source == 'sonar':
        if not sonar_path:
            sys.exit("ERROR: --freq_source sonar requires --sonar PATH.")
        freq_dict = load_sonar(sonar_path, sonar_word_col, sonar_freq_col,
                                sonar_corpus_size)
    else:
        print("[freq] No --freq_source selected — lexical sophistication "
              "metrics will be NaN.")
        print("[freq] Pass --freq_source subtlex --subtlex PATH, or "
              "--freq_source sonar --sonar PATH, to enable them.")

    # Load focal word list: top-N AI-overused seeds + matched near-zero-LPR
    # baseline seeds (Juzek 2026 Sec. 3.4-3.5), plus the three semantic
    # categories validated against the top-200 AI pool (Sec. 3.6).
    focal_lemmas = {}
    baseline_lemmas = {}
    category_lemmas = {}
    if wordlist_path:
        focal_lemmas, baseline_lemmas, category_lemmas = load_focal_words(
            wordlist_path, n_seeds, categories_path, category_pool_n
        )
    else:
        print("[focal] No wordlist provided — focal word metrics will be NaN.")
        # Use empty N/category sets so output columns are still consistent
        ns = n_seeds if n_seeds else DEFAULT_N_SEEDS
        focal_lemmas = {n: {} for n in ns}
        baseline_lemmas = {n: {} for n in ns}
        category_lemmas = {cat: set() for cat in BUILTIN_FOCAL_CATEGORIES}

    # Load articles or pre-parsed records
    if parsed_jsonl_path:
        # Pre-parsed mode: records already contain parsed_sentences
        records = load_parsed_jsonl(parsed_jsonl_path, max_articles)
        # Build article-like dicts from records for checkpoint keying
        articles = records
    else:
        with open(articles_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        if max_articles:
            articles = articles[:max_articles]

    # Resume from checkpoint
    done_keys, results = load_checkpoint(output_path)

    # Filter to unprocessed articles
    pending = []
    for a in articles:
        key = (
            a.get('source', ''),
            a.get('date', ''),
            a.get('title', ''),
        )
        if key not in done_keys:
            pending.append(a)

    total = len(articles)
    already_done = len(results)
    print("[articles] Total: {:,}  Already done: {:,}  Pending: {:,}".format(
        total, already_done, len(pending)
    ))
    print("[articles] Text scope: {}".format(
        text_scope
    ))
    if parsed_jsonl_path:
        print("[articles] Mode: pre-parsed JSONL (no Stanza needed)")
    elif parser:
        print("[articles] Parser: {}".format(parser.name))
    print("[articles] Checkpoint every {} articles.".format(checkpoint_every))

    if not pending:
        print("[articles] Nothing to do.")
        return results

    # Determine output fieldnames from first empty metrics call
    sample_metrics = empty_metrics(focal_lemmas, category_lemmas)
    sample_row = {
        'title': '', 'source': '', 'date': '', 'author': '',
        'section': '', 'predicted_topic': '', 'char_count': 0,
        'word_count_title': 0,
    }
    sample_row.update(sample_metrics)
    fieldnames = list(sample_row.keys())

    # If resuming, ensure fieldnames match existing file
    if results and set(fieldnames) != set(results[0].keys()):
        print("[checkpoint] WARNING: fieldnames differ from existing output. "
              "Consider deleting output and restarting.")

    if not parsed_jsonl_path:
        texts = [get_text(a, text_scope) for a in pending]

    for batch_start in range(0, len(pending), batch_size):
        batch_end = min(batch_start + batch_size, len(pending))
        batch_articles = pending[batch_start:batch_end]

        if parsed_jsonl_path:
            docs = [
                PreParsedDoc(a.get('parsed_sentences'))
                for a in batch_articles
            ]
        else:
            batch_texts = texts[batch_start:batch_end]
            docs = parser.parse_batch(batch_texts, batch_size=batch_size)

        for article, doc in zip(batch_articles, docs):
            metrics = compute_all_metrics(
                doc,
                provides_syntax,
                freq_dict,
                low_freq_threshold,
                high_freq_threshold,
                focal_lemmas,
                baseline_lemmas,
                category_lemmas,
                article,
                text_scope,
            )
            row = {
                'title':           article.get('title', ''),
                'source':          article.get('source', ''),
                'date':            article.get('date', ''),
                'author':          '; '.join(article.get('author') or []),
                'section':         article.get('section', ''),
                'predicted_topic': article.get('predicted_topic', ''),
            }
            row.update(metrics)
            results.append(row)

        # Progress + ETA
        n_done = already_done + batch_end
        elapsed = time.time() - t_start
        rate = batch_end / elapsed if elapsed > 0 else 0
        remaining = (len(pending) - batch_end) / rate if rate > 0 else 0
        print("  {:,}/{:,}  |  {:.1f} art/s  |  ETA {:.0f}m {:.0f}s".format(
            n_done, total, rate,
            remaining // 60, remaining % 60
        ))

        # Checkpoint
        if batch_end % checkpoint_every < batch_size or batch_end == len(pending):
            write_checkpoint(output_path, results, fieldnames)
            print("  [checkpoint] Saved {:,} rows to {}".format(
                len(results), output_path
            ))

    elapsed_total = time.time() - t_start
    print("\n[done] {:,} articles in {:.1f}s ({:.1f} art/s)".format(
        len(pending), elapsed_total, len(pending)/elapsed_total if elapsed_total > 0 else 0
    ))

    _print_summary(results, focal_lemmas, category_lemmas)
    return results


# ---------------------------------------------------------------------------
# 9.  SUMMARY
# ---------------------------------------------------------------------------

def _print_summary(results, focal_lemmas, category_lemmas):
    df = pd.DataFrame(results)

    ns = sorted(focal_lemmas.keys()) if focal_lemmas else DEFAULT_N_SEEDS
    focal_cols = []
    for n in ns:
        focal_cols += [
            'llm_style_word_ratio_top{}'.format(n),
            'weighted_llm_ratio_top{}_per1k'.format(n),
            'baseline_word_ratio_top{}'.format(n),
            'baseline_weighted_ratio_top{}_per1k'.format(n),
        ]

    # Coerce numeric columns
    numeric_cols = [
        'total_words', 'total_sentences', 'char_count',
        'mtld', 'prop_low_freq', 'prop_high_freq', 'mean_zipf',
        'mean_sent_len', 'cv_sent_len', 'mean_dep_depth',
        'finite_verbs_per_sent', 'mean_tunit_len', 'nom_rate',
    ] + focal_cols + list(FUNCTION_WORD_POS.values()) + [
        '{}_freq_per1k'.format(cat) for cat in category_lemmas
    ]

    sep = '=' * 65
    print('\n' + sep)
    print('SUMMARY STATISTICS  (n={:,})'.format(len(df)))
    print(sep)
    for col in numeric_cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(vals) > 0:
            print('  {:<30}  mean={:.4f}  n={:,}'.format(col, vals.mean(), len(vals)))
        else:
            print('  {:<30}  all NaN'.format(col))
    print(sep)


# ---------------------------------------------------------------------------
# 10.  CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Unified stylometric + focal-word metrics for Dutch articles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--articles',   required=False, default=None,
                   help='Path to articles JSON file. '
                        'Not required when --parsed_jsonl is provided.')
    p.add_argument('--parsed_jsonl', default=None,
                   help='Path to pre-parsed JSONL from stanza_parse.py. '
                        'When provided, Stanza is not loaded.')
    p.add_argument('--output',     default='article_metrics.csv')
    p.add_argument('--wordlist',   default=None,
                   help='las_word_{lang}.csv from the Juzek (2026) LPR pipeline '
                        '(key, lemma, upos, lpr_MH_guarded, lpr_guard_ok columns '
                        '— the same file diachronic_opm.py reads). If omitted, '
                        'focal metrics are NaN.')
    p.add_argument('--categories', default=None,
                   help='Optional CSV with lemma,category columns to override '
                        'the built-in care_rigour/emphasize/importance focal '
                        'categories (Juzek 2026 Sec. 3.6).')
    p.add_argument('--n_seeds', type=int, nargs='+', default=None,
                   help="Top-N AI-overused word counts to compute (and their "
                        "matched near-zero-LPR baselines), per Juzek (2026) "
                        "Sec. 3.4-3.5. Space-separated; default 20 50 200.")
    p.add_argument('--category_pool_n', type=int, default=CATEGORY_POOL_N,
                   help="Size of the top-N AI pool the three semantic "
                        "categories are validated against, independent of "
                        "--n_seeds (Sec. 3.6/4.2 use 200). Default 200.")
    p.add_argument('--freq_source', default=None, choices=['subtlex', 'sonar'],
                   help="Which word-frequency source to use for lexical "
                        "sophistication metrics. If omitted, inferred from "
                        "whichever of --subtlex/--sonar was given (error if "
                        "both were given); if neither was given, those "
                        "metrics are NaN.")
    p.add_argument('--subtlex',    default=None,
                   help='Subtlex-NL CSV (Keuleers et al. 2010). If omitted, lex sophistication is NaN.')
    p.add_argument('--subtlex_word_col', default=None)
    p.add_argument('--subtlex_zipf_col', default=None)
    p.add_argument('--sonar',      default=None,
                   help='Precomputed SoNaR-1 word/lemma-frequency table '
                        '(see README.pdf for the SoNaR-1 corpus this is '
                        'derived from). Alternative to --subtlex.')
    p.add_argument('--sonar_word_col', default=None)
    p.add_argument('--sonar_freq_col', default=None)
    p.add_argument('--sonar_corpus_size', type=float, default=None,
                   help='Total token count the --sonar frequency list was '
                        'tallied from, for raw-count-to-Zipf conversion. '
                        'Defaults to the sum of counts in the file itself.')
    p.add_argument('--low_freq_threshold',  type=float, default=DEFAULT_LOW_FREQ)
    p.add_argument('--high_freq_threshold', type=float, default=DEFAULT_HIGH_FREQ)
    p.add_argument('--text_scope', default='all',
                   choices=['all','body_only','title_only','highlight_only'],
                   help='Text fields to analyse (default: all)')
    p.add_argument('--simplemma',  action='store_true',
                   help='Use simplemma fallback. POS/syntax metrics will be NaN.')
    p.add_argument('--batch_size', type=int, default=32,
                   help='Articles per batch when using simplemma fallback (default 32).')
    p.add_argument('--checkpoint_every', type=int, default=500,
                   help='Write checkpoint every N articles (default 500). '
                        'Set lower for long runs on HPC.')
    p.add_argument('--max_articles', type=int, default=None)

    args = p.parse_args()

    # Infer --freq_source when not given explicitly, for backward
    # compatibility with scripts that only ever passed --subtlex.
    freq_source = args.freq_source
    if freq_source is None:
        if args.subtlex and args.sonar:
            sys.exit("ERROR: both --subtlex and --sonar were given; "
                      "specify --freq_source subtlex or --freq_source sonar "
                      "to say which one to actually use.")
        elif args.subtlex:
            freq_source = 'subtlex'
        elif args.sonar:
            freq_source = 'sonar'
        # else: stays None -> no frequency source, sophistication metrics NaN

    process_articles(
        articles_path=args.articles or args.parsed_jsonl,
        parsed_jsonl_path=args.parsed_jsonl,
        output_path=args.output,
        wordlist_path=args.wordlist,
        categories_path=args.categories,
        n_seeds=args.n_seeds,
        category_pool_n=args.category_pool_n,
        freq_source=freq_source,
        subtlex_path=args.subtlex,
        subtlex_word_col=args.subtlex_word_col,
        subtlex_zipf_col=args.subtlex_zipf_col,
        sonar_path=args.sonar,
        sonar_word_col=args.sonar_word_col,
        sonar_freq_col=args.sonar_freq_col,
        sonar_corpus_size=args.sonar_corpus_size,
        low_freq_threshold=args.low_freq_threshold,
        high_freq_threshold=args.high_freq_threshold,
        text_scope=args.text_scope,
        use_simplemma=args.simplemma,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        max_articles=args.max_articles,
    )


if __name__ == '__main__':
    main()
