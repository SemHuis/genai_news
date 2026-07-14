"""
Computes all stylometric and focal-word metrics for Dutch newspaper articles.
Designed to read pre-parsed JSONL output from stanza_parse.py.

Run once per outlet. Output filenames should follow the convention
{code}_metrics_results_body.csv, where code is one of:
  tgf   (Telegraaf)
  dvhn  (Dagblad van het Noorden)
  vk    (Volkskrant)
  ed    (Eindhovens Dagblad)
  stc   (Steenwijker Courant)
  nof   (Noordoost-Friesland)
analysis.R expects exactly these six filenames.

Metric families
---------------
LEXICAL DIVERSITY
  mtld               Measure of Textual Lexical Diversity (McCarthy & Jarvis 2010)
                     Computed on content-word lemmas (NOUN, VERB, ADJ, ADV).

LEXICAL SOPHISTICATION  (requires --freq_source plus the matching frequency file)
  prop_low_freq      Proportion of content tokens with Zipf < low_freq_threshold
  prop_high_freq     Proportion of content tokens with Zipf >= high_freq_threshold
  mean_zipf          Mean Zipf score across content tokens with known frequency

  Two frequency sources are supported; --freq_source picks which one is used:
    subtlex   Subtlex-NL (Keuleers et al. 2010) word-frequency norms, via --subtlex.
    sonar     A SoNaR-1-derived word-frequency list, via --sonar. SoNaR-1 itself
              ships as an annotated corpus tree, but then newspaper subcorpus
              is converted to Stanza POS tags. Counts are converted to Zipf 
              using van Heuven et al. (2014): Zipf = log10(count-per-million-words) + 3, 
              with the corpus size taken from the sum of all counts in the file.
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

  For each N in --n_seeds (default 20, 50, 100,200):
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
                         importance_freq_per1k — Juzek 2026 three
                         diachronic concepts; or whatever your --categories
                         CSV defines). Validated against the top-
                         --category_pool_n AI pool (default 200) regardless
                         of which --n_seeds were requested. analysis.R
                         discovers these columns automatically and plots
                         each one separately.

GENERAL
  total_words        Total token count
  total_sentences    Sentence count
  char_count         Character count of analysed text
  word_count_title   Word count of title field only (editorial influence proxy)

Usage
-----
  # Full run, SoNaR as the frequency source, default top-20/50/100/200:
  # python3 article_metrics.py \\
        --parsed_jsonl parsed_body_only.jsonl \\
        --wordlist data/las_word_nl.csv \\
        --freq_source sonar \\
        --sonar data/sonar_newspapers_stanza_format.jsonl \\
        --output metrics_results_body.csv
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

DEFAULT_LOW_FREQ  = 3.0   # Zipf < 3.0 → low-frequency 
DEFAULT_HIGH_FREQ = 4.0   # Zipf ≥ 4.0 → high-frequency

# Zipf score floor for unseen words
ZIPF_FLOOR_FOR_UNSEEN = 1.0

# Semantic category word sets for per-article frequency counting.
# Override via --categories CSV (lemma,category columns).
BUILTIN_FOCAL_CATEGORIES = {
    'care_rigour': {'grondig'},
    'emphasize':   {'benadrukken'},
    'importance':  {'belang'},
}



# ---------------------------------------------------------------------------
# 1.  NLP BACKENDS
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 2.  WORD-FREQUENCY LOADER  (SoNaR-1)
# ---------------------------------------------------------------------------
# Produces a {lowercased_lemma: zipf} dict for lexical sophistication.
# SoNaR accepts JSONL (one object per line with lemma+frequency) or
# delimited table formats (see load_sonar() docstring).
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

    # Auto-detect JSONL vs delimited
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

    # Delimited table branch (CSV/TSV)
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
# 3.  METRIC COMPUTATION 
# ---------------------------------------------------------------------------

def compute_all_metrics(
    doc,
    freq_dict,
    low_thresh,
    high_thresh,
    category_lemmas,
    article,
    text_scope,
):
    """Dispatcher — pre-parsed Stanza doc → full stylometric metrics."""
    text_analysed = get_text(article, text_scope)
    char_count = len(text_analysed)
    title_text = article.get('title') or ''
    import re as _re
    word_count_title = len(_re.findall(r'\S+', title_text))

    base = {
        'char_count':       char_count,
        'word_count_title': word_count_title,
    }

    if doc is None:
        m = empty_metrics(category_lemmas)
        m.update(base)
        return m

    m = compute_stanza_metrics(doc, freq_dict, low_thresh, high_thresh,
                                category_lemmas)
    m.update(base)
    return m


def compute_stanza_metrics(doc, freq_dict, low_thresh, high_thresh,
                            category_lemmas):
    all_words = [w for sent in doc.sentences for w in sent.words]
    total_words = len(all_words)
    total_sentences = len(doc.sentences)

    if total_words == 0:
        return empty_metrics(category_lemmas)

    content_words = [
        w for w in all_words
        if w.upos in CONTENT_POS and w.lemma
    ]

    mtld = _compute_mtld_from_lemmas([w.lemma.lower() for w in content_words])
    lex  = _lex_sophistication(content_words, freq_dict, low_thresh, high_thresh,
                                use_lemma=True)
    fw   = _function_words(all_words, total_words)
    syn  = _syntax_stanza(doc, all_words, total_words, total_sentences)
    cats = _count_categories(all_words, total_words, category_lemmas)

    result = {
        'total_words':     total_words,
        'total_sentences': total_sentences,
        'mtld':            mtld,
    }
    result.update(lex)
    result.update(fw)
    result.update(syn)
    result.update(cats)
    return result



# ---------------------------------------------------------------------------
# 4.  COMPONENT FUNCTIONS
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


def _count_categories(all_words, total_words, category_lemmas):
    """Count semantic-category word occurrences (lemma match) per 1k tokens."""
    if total_words == 0 or not category_lemmas:
        return {'{}_freq_per1k'.format(cat): None for cat in category_lemmas}
    cat_counts = {cat: 0 for cat in category_lemmas}
    for w in all_words:
        lemma = w.lemma.lower() if w.lemma else ''
        if not lemma:
            continue
        for cat, cat_lemmas in category_lemmas.items():
            if lemma in cat_lemmas:
                cat_counts[cat] += 1
                break
    per1k = 1000.0 / total_words
    return {'{}_freq_per1k'.format(cat): round(cat_counts[cat] * per1k, 4)
            for cat in category_lemmas}


# ---------------------------------------------------------------------------
# 5.  HELPERS
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


def _empty_categories(category_lemmas):
    return {'{}_freq_per1k'.format(cat): None for cat in category_lemmas}


def empty_metrics(category_lemmas):
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
    result.update(_empty_categories(category_lemmas))
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
    categories_path=None,
    sonar_path=None,
    sonar_word_col=None,
    sonar_freq_col=None,
    sonar_corpus_size=None,
    low_freq_threshold=DEFAULT_LOW_FREQ,
    high_freq_threshold=DEFAULT_HIGH_FREQ,
    text_scope='all',
    batch_size=32,
    checkpoint_every=500,
    max_articles=None,
):
    """
    Compute per-article stylometric metrics and semantic category frequencies.

    Requires pre-parsed JSONL (--parsed_jsonl) from stanza_parse.py.
    Word-frequency norms must be provided via --sonar (SoNaR-1 format).
    Semantic category frequencies (care_rigour, emphasize, importance) use
    the built-in word list or an override via --categories CSV.
    Focal/excess word analysis is handled separately by excess_words_diachronic.py.
    """
    t_start = time.time()

    if not parsed_jsonl_path:
        sys.exit("ERROR: --parsed_jsonl is required. "
                 "Run stanza_parse.py first, then pass its output here.")

    # Load SoNaR-1 word-frequency norms
    freq_dict = {}
    if sonar_path:
        freq_dict = load_sonar(sonar_path, sonar_word_col, sonar_freq_col,
                                sonar_corpus_size)
    else:
        print("[freq] No --sonar path provided — lexical sophistication metrics will be NaN.")

    # Load semantic category word sets
    if categories_path:
        import csv as _csv
        category_lemmas = defaultdict(set)
        with open(categories_path, newline='', encoding='utf-8') as _f:
            for row in _csv.DictReader(_f):
                category_lemmas[row['category']].add(row['lemma'].strip().lower())
        category_lemmas = dict(category_lemmas)
        print("[categories] Loaded from {}: {}".format(
            categories_path,
            {k: len(v) for k, v in category_lemmas.items()}))
    else:
        category_lemmas = {cat: set(lemmas)
                           for cat, lemmas in BUILTIN_FOCAL_CATEGORIES.items()}
        print("[categories] Using built-in categories: {}".format(
            {k: len(v) for k, v in category_lemmas.items()}))

    # Load pre-parsed records
    records = load_parsed_jsonl(parsed_jsonl_path, max_articles)
    articles = records

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
    print("[articles] Mode: pre-parsed JSONL | Checkpoint every {}.".format(checkpoint_every))

    if not pending:
        print("[articles] Nothing to do.")
        return results

    # Determine output fieldnames from first empty metrics call
    sample_metrics = empty_metrics(category_lemmas)
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

    for batch_start in range(0, len(pending), batch_size):
        batch_end = min(batch_start + batch_size, len(pending))
        batch_articles = pending[batch_start:batch_end]

        docs = [PreParsedDoc(a.get('parsed_sentences')) for a in batch_articles]

        for article, doc in zip(batch_articles, docs):
            metrics = compute_all_metrics(
                doc,
                freq_dict,
                low_freq_threshold,
                high_freq_threshold,
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

    _print_summary(results, category_lemmas)
    return results


def _print_summary(results, category_lemmas):
    df = pd.DataFrame(results)
    numeric_cols = [
        'total_words', 'total_sentences', 'char_count',
        'mtld', 'prop_low_freq', 'prop_high_freq', 'mean_zipf',
        'mean_sent_len', 'cv_sent_len', 'mean_dep_depth',
        'finite_verbs_per_sent', 'mean_tunit_len', 'nom_rate',
    ] + list(FUNCTION_WORD_POS.values()) + [
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


def main():
    p = argparse.ArgumentParser(
        description="Stylometric metrics for Dutch newspaper articles (SoNaR + Stanza).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--articles',       default=None,
                   help='Path to articles JSON (used only as fallback identifier). '
                        'Usually --parsed_jsonl is sufficient.')
    p.add_argument('--parsed_jsonl',   required=True,
                   help='Pre-parsed JSONL from stanza_parse.py (required).')
    p.add_argument('--output',         default='article_metrics.csv')
    p.add_argument('--categories',     default=None,
                   help='CSV with lemma,category columns to override built-in '
                        'care_rigour/emphasize/importance word sets.')
    p.add_argument('--sonar',          default=None,
                   help='SoNaR-1 word/lemma-frequency file (JSONL or delimited). '
                        'Required for lexical sophistication metrics.')
    p.add_argument('--sonar_word_col',      default=None)
    p.add_argument('--sonar_freq_col',      default=None)
    p.add_argument('--sonar_corpus_size',   type=float, default=None,
                   help='Total token count for raw-count→Zipf conversion. '
                        'Defaults to sum of counts in the file.')
    p.add_argument('--low_freq_threshold',  type=float, default=DEFAULT_LOW_FREQ)
    p.add_argument('--high_freq_threshold', type=float, default=DEFAULT_HIGH_FREQ)
    p.add_argument('--text_scope',     default='all',
                   choices=['all', 'body_only', 'title_only', 'highlight_only'])
    p.add_argument('--batch_size',     type=int, default=32)
    p.add_argument('--checkpoint_every', type=int, default=500)
    p.add_argument('--max_articles',   type=int, default=None)

    args = p.parse_args()

    process_articles(
        articles_path       = args.articles or args.parsed_jsonl,
        output_path         = args.output,
        parsed_jsonl_path   = args.parsed_jsonl,
        categories_path     = args.categories,
        sonar_path          = args.sonar,
        sonar_word_col      = args.sonar_word_col,
        sonar_freq_col      = args.sonar_freq_col,
        sonar_corpus_size   = args.sonar_corpus_size,
        low_freq_threshold  = args.low_freq_threshold,
        high_freq_threshold = args.high_freq_threshold,
        text_scope          = args.text_scope,
        batch_size          = args.batch_size,
        checkpoint_every    = args.checkpoint_every,
        max_articles        = args.max_articles,
    )


if __name__ == '__main__':
    main()