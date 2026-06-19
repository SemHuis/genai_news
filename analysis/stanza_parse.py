"""
Runs the Stanza Dutch NLP pipeline over a corpus of articles and saves
the parsed output to disk. Metric computation is done separately in
article_metrics.py using the saved parses.
"""

import sys
import json
import argparse
import os
import time
import re


def build_pipeline(model_dir=None):
    try:
        import stanza
    except ImportError:
        sys.exit(
            "Stanza not installed.\n"
            "pip install stanza\n"
            "python -c \"import stanza; stanza.download('nl')\""
        )

    if model_dir:
        print("[stanza] Model directory: {}".format(model_dir))

    print("[stanza] Loading Dutch pipeline (tokenize, mwt, pos, lemma, depparse)...")
    pipeline_kwargs = dict(
        lang="nl",
        processors="tokenize,mwt,pos,lemma,depparse",
        download_method=None,
        verbose=False,
    )
    if model_dir:
        pipeline_kwargs['dir'] = model_dir

    nlp = stanza.Pipeline(**pipeline_kwargs)
    print("[stanza] Pipeline ready.")
    return nlp, stanza


def parse_batch(nlp, stanza, texts, batch_size=32):
    """
    Parse a list of texts in one Stanza batch call.
    Returns list of Stanza Document objects (None for empty texts).
    """
    results = [None] * len(texts)
    non_empty = [(i, t) for i, t in enumerate(texts) if t and t.strip()]

    if not non_empty:
        return results

    for start in range(0, len(non_empty), batch_size):
        chunk = non_empty[start: start + batch_size]
        idxs, txts = zip(*chunk)
        docs_in = [stanza.Document([], text=t) for t in txts]
        docs_out = nlp(docs_in)
        for idx, doc in zip(idxs, docs_out):
            results[idx] = doc

    return results


def doc_to_dict(doc):
    """
    Serialise a Stanza Document to a JSON-serialisable dict.
    Stores only the fields needed for metric computation.
    """
    if doc is None:
        return None

    sentences = []
    for sent in doc.sentences:
        tokens = []
        for word in sent.words:
            tokens.append({
                'id':     word.id,
                'text':   word.text,
                'lemma':  word.lemma,
                'upos':   word.upos,
                'feats':  word.feats,
                'deprel': word.deprel,
                'head':   word.head,
            })
        sentences.append(tokens)

    return sentences


_WORD_RE = re.compile(r'\S+')


# Valid text scope values and what they include
TEXT_SCOPES = {
    'all':       ['title', 'highlight', 'full_text'],
    'body_only': ['full_text'],
    'title_only':     ['title'],
    'highlight_only': ['highlight'],
}

def get_text(article, text_scope='all'):
    """
    Extract text from article fields based on scope.

    text_scope options:
      'all'            : title + highlight + full_text (default)
      'body_only'      : full_text only
      'title_only'     : title only
      'highlight_only' : highlight / lead paragraph only
    """
    fields = TEXT_SCOPES.get(text_scope, TEXT_SCOPES['all'])
    parts = [article.get(f) or '' for f in fields]
    return ' '.join(p for p in parts if p.strip())


def word_count(text):
    return len(_WORD_RE.findall(text)) if text else 0


def load_done_keys(output_path):
    """
    Read existing output file and return set of already-processed
    (source, date, title) keys.
    """
    done = set()
    if not os.path.exists(output_path):
        return done
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                done.add((
                    obj.get('source', ''),
                    obj.get('date', ''),
                    obj.get('title', ''),
                ))
        print("[checkpoint] Resuming — {:,} articles already parsed.".format(len(done)))
    except Exception as e:
        print("[checkpoint] Could not read existing output ({}). Starting fresh.".format(e))
        return set()
    return done


def write_record(f, article, parsed_sentences, text_scope):
    """Write one parsed article as a JSON line."""
    record = {
        'title':            article.get('title', ''),
        'source':           article.get('source', ''),
        'date':             article.get('date', ''),
        'author':           '; '.join(article.get('author') or []),
        'section':          article.get('section', ''),
        'predicted_topic':  article.get('predicted_topic', ''),
        'text_scope':       text_scope,
        'char_count':       len(get_text(article, text_scope)),
        'word_count_title': word_count(article.get('title') or ''),
        'parsed_sentences': parsed_sentences,
    }
    f.write(json.dumps(record, ensure_ascii=False) + '\n')


def process(
    articles_path,
    output_path,
    stanza_model_dir=None,
    text_scope='all',
    batch_size=32,
    checkpoint_every=500,
    max_articles=None,
):
    t_start = time.time()

    # Load Stanza
    nlp, stanza = build_pipeline(stanza_model_dir)

    # Load articles
    with open(articles_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    if max_articles:
        articles = articles[:max_articles]

    # Load checkpoint
    done_keys = load_done_keys(output_path)
    pending = [
        a for a in articles
        if (a.get('source', ''), a.get('date', ''), a.get('title', ''))
        not in done_keys
    ]

    total = len(articles)
    already_done = len(done_keys)
    print("[articles] Total: {:,}  Already parsed: {:,}  Pending: {:,}".format(
        total, already_done, len(pending)
    ))
    print("[articles] Text scope: {}".format(
        text_scope
    ))
    print("[articles] Batch size: {}".format(batch_size))
    print("[articles] Checkpoint every: {}".format(checkpoint_every))

    if not pending:
        print("[articles] Nothing to do.")
        return

    texts = [get_text(a, text_scope) for a in pending]
    n_written = 0

    # Open output in append mode so checkpoints accumulate
    with open(output_path, 'a', encoding='utf-8') as out_f:
        for batch_start in range(0, len(pending), batch_size):
            batch_end = min(batch_start + batch_size, len(pending))
            batch_texts    = texts[batch_start:batch_end]
            batch_articles = pending[batch_start:batch_end]

            docs = parse_batch(nlp, stanza, batch_texts, batch_size)

            for article, doc in zip(batch_articles, docs):
                parsed = doc_to_dict(doc)
                write_record(out_f, article, parsed, text_scope)
                n_written += 1

            # Flush to disk at checkpoint intervals
            if batch_end % checkpoint_every < batch_size or batch_end == len(pending):
                out_f.flush()
                os.fsync(out_f.fileno())

            # Progress + ETA
            n_done = already_done + batch_end
            elapsed = time.time() - t_start
            rate = batch_end / elapsed if elapsed > 0 else 0
            remaining = (len(pending) - batch_end) / rate if rate > 0 else 0
            print("  {:,}/{:,}  |  {:.1f} art/s  |  ETA {:.0f}m {:.0f}s  "
                  "[checkpoint every {}]".format(
                      n_done, total, rate,
                      remaining // 60, remaining % 60,
                      checkpoint_every,
                  ))

    elapsed_total = time.time() - t_start
    print("\n[done] Parsed {:,} articles in {:.1f}s ({:.1f} art/s)".format(
        n_written,
        elapsed_total,
        n_written / elapsed_total if elapsed_total > 0 else 0,
    ))
    print("[done] Output: {}".format(output_path))
    print("[done] Total lines in output: {:,}".format(already_done + n_written))


def main():
    p = argparse.ArgumentParser(
        description="Parse Dutch newspaper articles with Stanza and save output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--articles',   required=True,
                   help='Path to articles JSON file')
    p.add_argument('--output',     default='parsed_articles.jsonl',
                   help='Output JSONL file (default: parsed_articles.jsonl)')
    p.add_argument('--stanza_model_dir', default=None,
                   help='Stanza model cache directory. '
                        'On HPC use local scratch: /scratch/user/stanza_cache')
    p.add_argument('--text_scope', default='all',
                   choices=['all', 'body_only', 'title_only', 'highlight_only'],
                   help=(
                       'Which article fields to parse. '
                       'all = title + highlight + body (default); '
                       'body_only = full_text only; '
                       'title_only = title only; '
                       'highlight_only = lead paragraph only.'
                   ))
    p.add_argument('--batch_size', type=int, default=32,
                   help='Articles per Stanza batch (default 32). '
                        'Increase on HPC for speed.')
    p.add_argument('--checkpoint_every', type=int, default=500,
                   help='Flush to disk every N articles (default 500)')
    p.add_argument('--max_articles', type=int, default=None,
                   help='Process only first N articles (for testing)')
    args = p.parse_args()

    process(
        articles_path=args.articles,
        output_path=args.output,
        stanza_model_dir=args.stanza_model_dir,
        text_scope=args.text_scope,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        max_articles=args.max_articles,
    )


if __name__ == '__main__':
    main()
