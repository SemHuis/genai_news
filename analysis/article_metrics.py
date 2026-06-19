"""
Computes all stylometric and focal-word metrics for Dutch newspaper articles.
Reads pre-parsed JSONL output from stanza_parse.py.
"""

import sys
import json
import csv
import argparse
import math
import os
import time
from collections import defaultdict

import pandas as pd
import simplemma

try:
    from lexicalrichness import LexicalRichness
    HAS_LEXRICH = True
except ImportError:
    HAS_LEXRICH = False
    print("[warning] lexicalrichness not installed — MTLD will be NaN.")
    print("[warning] pip install lexicalrichness")


# Dutch nominalisation suffixes — core productive suffixes only
# Validated against Dutch morphology (de Haas & Trommelen 1993)
# Removed: -age, -ance, -ence, -ment (loanword patterns, not core Dutch)
NOM_SUFFIXES = (
    'ing', 'heid', 'atie', 'sie', 'nis',
    'schap', 'iteit', 'isme', 'tie',
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

DEFAULT_LOW_FREQ  = 3.0
DEFAULT_HIGH_FREQ = 5.0

# Focal word categories (built-in; overridable via --categories CSV)
BUILTIN_FOCAL_CATEGORIES = {
    'emphasis': {
        'benadrukken', 'benadrukten', 'belichten', 'onderstrepen',
        'accentueren', 'verduidelijken', 'illustreren', 'markeren',
        'kenmerken', 'weerspiegelen', 'omvatten', 'tonen', 'wijzen',
        'aanduiden', 'uitlichten', 'aantonen',
    },
    'importance': {
        'belang', 'prioriteit', 'transparantie', 'consistentie', 'precisie',
        'veerkracht', 'inclusie', 'duurzaamheid', 'relevantie',
        'doelgerichtheid', 'vastberadenheid', 'eerlijkheid', 'teamgeest',
        'impact', 'waarde',
    },
    'innovation': {
        'innovatief', 'baanbrekend', 'geavanceerd', 'toonaangevend',
        'revolutionair', 'technologisch', 'modern', 'veelbelovend',
        'indrukwekkend', 'nauwlettend', 'duurzaam', 'robuust', 'dynamisch',
        'strategisch', 'effectief', 'efficiënt', 'significant', 'cruciaal',
    },
}


class _PseudoDoc:
    def __init__(self, tokens, lemmas, sent_lengths):
        self.tokens = tokens
        self.lemmas = lemmas
        self.sent_lengths = sent_lengths


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
    Load Subtlex-NL frequency norms.
    Uses manual line-by-line parsing to bypass pandas quoting issues,
    then falls back to pandas if needed.
    """
    df = _parse_delimited_table(path, 'subtlex')

    if word_col is None:
        for c in ('Word', 'Spelling', 'word', 'spelling', 'WORD'):
            if c in df.columns:
                word_col = c
                break
    if word_col is None:
        raise ValueError("Cannot detect word column. Columns: {}. "
                         "Use --subtlex_word_col.".format(list(df.columns)))

    if zipf_col is None:
        # Try Zipf-named columns first, then log-frequency equivalents.
        # Subtlex-NL (Keuleers et al. 2010) uses Lg10WF (log10 word frequency)
        # which is on a comparable scale to Zipf scores.
        for c in ('Zipf', 'ZipfValue', 'zipf', 'ZIPF', 'Zipf_value',
                  'Lg10WF', 'Lg10CD', 'SUBTLEXWF', 'lg10wf'):
            if c in df.columns:
                zipf_col = c
                print("[subtlex] Using '{}' as frequency column.".format(c))
                if c in ('Lg10WF', 'Lg10CD', 'lg10wf'):
                    print("[subtlex] NOTE: '{}' is log10 frequency, not Zipf.".format(c))
                    print("[subtlex] Thresholds --low_freq_threshold and "
                          "--high_freq_threshold should be adjusted accordingly.")
                    print("[subtlex] Typical Lg10WF range: 0 (rare) to 4+ (very common).")
                    print("[subtlex] Suggested: --low_freq_threshold 1.0 "
                          "--high_freq_threshold 2.5")
                break
    if zipf_col is None:
        raise ValueError(
            "Cannot detect frequency column. Columns: {}. "
            "Use --subtlex_zipf_col to specify which column to use.".format(
                list(df.columns)
            )
        )

    freq_dict = {}
    n_failed = 0
    for _, row in df.iterrows():
        w = str(row[word_col]).strip().lower()
        if not w or w in ('nan', 'none', ''):
            continue
        raw = str(row[zipf_col]).strip()
        # Handle Dutch locale decimal comma and common missing value markers
        raw = raw.replace(',', '.').replace(' ', '')
        if raw in ('', 'nan', 'none', 'na', '-', '.'):
            n_failed += 1
            continue
        try:
            freq_dict[w] = float(raw)
        except (ValueError, TypeError):
            n_failed += 1

    print("[subtlex] {:,} entries loaded (skipped {:,} unparseable values).".format(
        len(freq_dict), n_failed
    ))
    print("[subtlex] Word col: '{}', Zipf col: '{}'".format(word_col, zipf_col))

    # Diagnostic: show value range so user can verify thresholds are sensible
    if freq_dict:
        vals = list(freq_dict.values())
        print("[subtlex] {} value range: min={:.3f}, median={:.3f}, max={:.3f}".format(
            zipf_col,
            min(vals),
            sorted(vals)[len(vals)//2],
            max(vals),
        ))

    if len(freq_dict) == 0:
        print("[subtlex] WARNING: freq_dict is empty — lexical sophistication will be NaN.")
        print("[subtlex] Check that word_col and zipf_col point to the correct columns.")

    return freq_dict


def load_sonar(path, word_col=None, freq_col=None, corpus_size=None):
    """
    Load a SoNaR-1-derived word-frequency list as an alternative to Subtlex-NL.
    """
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
    else:
        is_zipf = freq_col.lower().startswith('zipf')

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
            "Use --sonar_freq_col to specify which column to use.".format(
                list(df.columns)
            )
        )

    # Parse raw values first (handles Dutch decimal commas and common NA markers)
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
        # Raw counts -> Zipf. corpus_size defaults to the sum of counts in
        # this file (assumes the file covers the whole reference corpus).
        total = corpus_size if corpus_size else sum(raw_values.values())
        if not total or total <= 0:
            raise ValueError(
                "Cannot compute corpus size for Zipf conversion (sum of "
                "counts is zero). Pass --sonar_corpus_size explicitly."
            )
        print("[sonar] Converting raw counts to Zipf using corpus size "
              "= {:,.0f} tokens ({}).".format(
                  total, "from --sonar_corpus_size" if corpus_size
                  else "sum of counts in file"))
        freq_dict = {}
        for w, count in raw_values.items():
            if count <= 0:
                continue
            per_million = (count / total) * 1e6
            freq_dict[w] = math.log10(per_million) + 3

    print("[sonar] {:,} entries loaded (skipped {:,} unparseable values).".format(
        len(freq_dict), n_failed
    ))
    print("[sonar] Word col: '{}', frequency col: '{}'".format(word_col, freq_col))

    if freq_dict:
        vals = list(freq_dict.values())
        print("[sonar] Zipf value range: min={:.3f}, median={:.3f}, max={:.3f}".format(
            min(vals), sorted(vals)[len(vals)//2], max(vals),
        ))

    if len(freq_dict) == 0:
        print("[sonar] WARNING: freq_dict is empty — lexical sophistication will be NaN.")
        print("[sonar] Check that word_col and freq_col point to the correct columns.")

    return freq_dict


def load_focal_words(wordlist_path, relpct_threshold, categories_path=None):
    """
    Load Juzek (2026) word list and build focal word lookup structures.
    """
    df = pd.read_csv(wordlist_path)
    focal_df = df[
        df['upos'].isin({'VERB', 'NOUN', 'ADJ', 'ADV'}) &
        (df['Delta_MH'] > 0) &
        (df['relpct_MH'] > relpct_threshold)
    ].copy()

    focal_lemmas = defaultdict(list)
    for _, row in focal_df.iterrows():
        focal_lemmas[str(row['lemma']).lower()].append({
            'key':    row['key'],
            'upos':   row['upos'],
            'weight': float(row['relpct_MH']),
            'delta':  float(row['Delta_MH']),
        })

    # Category mapping
    if categories_path:
        cat_df = pd.read_csv(categories_path)
        raw_cats = defaultdict(set)
        for _, row in cat_df.iterrows():
            raw_cats[row['category']].add(str(row['lemma']).strip().lower())
        category_lemmas = {
            cat: lemmas & set(focal_lemmas.keys())
            for cat, lemmas in raw_cats.items()
        }
        print("[focal] External category mapping loaded.")
    else:
        category_lemmas = {
            cat: lemmas & set(focal_lemmas.keys())
            for cat, lemmas in BUILTIN_FOCAL_CATEGORIES.items()
        }

    print("[focal] {:,} focal word entries "
          "(relpct_MH > {})".format(len(focal_df), relpct_threshold))
    for cat, lemmas in category_lemmas.items():
        print("[focal]   {}: {} lemmas".format(cat, len(lemmas)))

    return dict(focal_lemmas), category_lemmas


def compute_all_metrics(
    doc_or_pseudo,
    provides_syntax,
    freq_dict,
    low_thresh,
    high_thresh,
    focal_lemmas,
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
        m = empty_metrics(category_lemmas)
        m.update(base)
        return m

    if provides_syntax:
        m = compute_stanza_metrics(
            doc_or_pseudo, freq_dict, low_thresh, high_thresh,
            focal_lemmas, category_lemmas
        )
    else:
        m = compute_simplemma_metrics(
            doc_or_pseudo, freq_dict, low_thresh, high_thresh,
            focal_lemmas, category_lemmas
        )

    m.update(base)
    return m


def compute_stanza_metrics(doc, freq_dict, low_thresh, high_thresh,
                            focal_lemmas, category_lemmas):
    all_words = [w for sent in doc.sentences for w in sent.words]
    total_words = len(all_words)
    total_sentences = len(doc.sentences)

    if total_words == 0:
        return empty_metrics(category_lemmas)

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

    # --- Focal words ---
    foc = _focal_words_stanza(all_words, total_words, focal_lemmas, category_lemmas)

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
                               focal_lemmas, category_lemmas):
    tokens  = pseudo.tokens
    lemmas  = pseudo.lemmas
    total_words = len(tokens)

    if total_words == 0:
        return empty_metrics(category_lemmas)

    # MTLD on all lemmas (no POS filter available)
    mtld = _compute_mtld_from_lemmas(lemmas)

    # Lexical sophistication — lookup by surface form (no POS filter)
    low_c = high_c = 0
    zipf_scores = []
    for tok in tokens:
        if tok in freq_dict:
            z = freq_dict[tok]
            zipf_scores.append(z)
            if z < low_thresh:
                low_c += 1
            if z >= high_thresh:
                high_c += 1
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
        lemmas, total_words, focal_lemmas, category_lemmas
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


def _compute_mtld_from_lemmas(lemma_list):
    """
    Compute MTLD from a list of lemmas (content words only).
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
    if not freq_dict or not content_words:
        return {'prop_low_freq': None, 'prop_high_freq': None, 'mean_zipf': None}

    n = len(content_words)
    low_c = high_c = 0
    zipf_scores = []

    for w in content_words:
        key = (w.lemma.lower() if use_lemma else w.text.lower())
        if key in freq_dict:
            z = freq_dict[key]
            zipf_scores.append(z)
            if z < low_thresh:
                low_c += 1
            if z >= high_thresh:
                high_c += 1

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


def _focal_words_stanza(all_words, total_words, focal_lemmas, category_lemmas):
    """Focal word matching using lemma+UPOS — Juzek (2026) protocol."""
    if not focal_lemmas or total_words == 0:
        return _empty_focal(category_lemmas)

    focal_count = 0
    weighted_sum = 0.0
    cat_counts = {cat: 0 for cat in category_lemmas}

    for w in all_words:
        lemma = w.lemma.lower() if w.lemma else ''
        if lemma not in focal_lemmas:
            continue
        entries = focal_lemmas[lemma]
        # Exact lemma+UPOS match
        matching = [e for e in entries if e['upos'] == w.upos]
        if not matching:
            continue
        best = max(matching, key=lambda e: e['weight'])
        focal_count += 1
        weighted_sum += best['weight']
        for cat, cat_lemmas in category_lemmas.items():
            if lemma in cat_lemmas:
                cat_counts[cat] += 1
                break

    per1k = 1000.0 / total_words
    result = {
        'llm_style_word_ratio': round(focal_count * per1k, 4),
        'weighted_llm_ratio':   round(weighted_sum / total_words, 4),
    }
    for cat in category_lemmas:
        result['{}_freq_per1k'.format(cat)] = round(cat_counts[cat] * per1k, 4)
    return result


def _focal_words_simplemma(lemmas, total_words, focal_lemmas, category_lemmas):
    """Focal word matching on lemmas only (simplemma fallback)."""
    if not focal_lemmas or total_words == 0:
        return _empty_focal(category_lemmas)

    focal_count = 0
    weighted_sum = 0.0
    cat_counts = {cat: 0 for cat in category_lemmas}

    for lemma in lemmas:
        if lemma not in focal_lemmas:
            continue
        best = max(focal_lemmas[lemma], key=lambda e: e['weight'])
        focal_count += 1
        weighted_sum += best['weight']
        for cat, cat_lemmas in category_lemmas.items():
            if lemma in cat_lemmas:
                cat_counts[cat] += 1
                break

    per1k = 1000.0 / total_words
    result = {
        'llm_style_word_ratio': round(focal_count * per1k, 4),
        'weighted_llm_ratio':   round(weighted_sum / total_words, 4),
    }
    for cat in category_lemmas:
        result['{}_freq_per1k'.format(cat)] = round(cat_counts[cat] * per1k, 4)
    return result


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


def _empty_focal(category_lemmas):
    result = {
        'llm_style_word_ratio': None,
        'weighted_llm_ratio':   None,
    }
    for cat in category_lemmas:
        result['{}_freq_per1k'.format(cat)] = None
    return result


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
    result.update(_empty_focal(category_lemmas))
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
    relpct_threshold=500,
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

    # Load focal word list
    focal_lemmas = {}
    category_lemmas = {}
    if wordlist_path:
        focal_lemmas, category_lemmas = load_focal_words(
            wordlist_path, relpct_threshold, categories_path
        )
    else:
        print("[focal] No wordlist provided — focal word metrics will be NaN.")
        # Use empty category set so output columns are consistent
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

    # Coerce numeric columns
    numeric_cols = [
        'total_words', 'total_sentences', 'char_count',
        'mtld', 'prop_low_freq', 'prop_high_freq', 'mean_zipf',
        'mean_sent_len', 'cv_sent_len', 'mean_dep_depth',
        'finite_verbs_per_sent', 'mean_tunit_len', 'nom_rate',
        'llm_style_word_ratio', 'weighted_llm_ratio',
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
                   help='las_word_nl.csv (Juzek 2026). If omitted, focal metrics are NaN.')
    p.add_argument('--categories', default=None,
                   help='Optional CSV with lemma,category columns to override built-in focal categories.')
    p.add_argument('--relpct_threshold', type=float, default=500)
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
        relpct_threshold=args.relpct_threshold,
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
