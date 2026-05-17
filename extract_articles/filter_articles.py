import argparse
import json
import sys
import re
from datetime import datetime


def should_filter_article(article, min_words):
    """Determine if an article should be filtered out (excluded)."""
    title = article.get('title', '').strip()
    date = article.get('date', '').strip()
    full_text = article.get('full_text', '').strip()
    for name in article.get('author', ''):
        author = name.strip()
    source = article.get('source', '').strip()
    
    # Filter by word count (minimum)
    word_count = len(full_text.split()) if full_text else 0
    if word_count < min_words:
        return True, f"Too short ({word_count} words)"
    
    # Filter specific titles (exact match, case-insensitive)
    exact_titles_to_remove = [
        'Sudoku',
        'Scrypto',
        'Raatsel',
        'In Het Midden',
        'Colofon',
        'De Ommezwaai',
        'KILLER SUDOKU',
        'Correcties/aanvullingen',
        'jarig', 
        'Net op de wereld',
        'User Name: ='
    ]
    
    # Filter titles containing "op rapport" 
    title_lower = title.lower()
    if re.search(r'op\s+rapport', title_lower):
        return True, "Contains 'op rapport' in title"
    
    # Filter puzzles (look for "horizontaal:" AND "verticaal:" in full_text)
    if re.search(r'\bhorizontaal\s*:', full_text, re.IGNORECASE) and \
       re.search(r'\bverticaal\s*:', full_text, re.IGNORECASE):
        return True, "Contains puzzle clues (horizontaal/verticaal)"

    # Dates that are not in the format DD/MM/YYYY
    if not re.match(r'^\d{2}/\d{2}/\d{4}$', date):
        return True, "Date not in correct format"
    
    # Check if title is "No headline in original" and author is "Unknown Author"
    if title_lower == 'no headline in original' and author.lower() == 'unknown author':
        # Check if full text starts with x
        if full_text.lower().startswith('bbc 1\n\n') or \
            full_text.lower().startswith('npo 1\n\n') or \
            full_text.lower().startswith('npo1\n\n') or \
            full_text.lower().startswith('één\n\n') or \
            full_text.lower().startswith('losse prijs:') or \
            full_text.lower().startswith('reageren? ') or \
            full_text.lower().startswith('ook in deze rubriek?') or \
            full_text.lower().startswith('bekijk de volledige agenda') or \
            full_text.lower().startswith('dit recept wordt verzorgd door nancy') or \
            full_text.lower().startswith('top-10 kattennamen') or \
            full_text.lower().startswith('do ') or \
            full_text.lower().startswith('eredivisie') or \
            full_text.lower().startswith('eerste divisie') or \
            full_text.lower().startswith('derde divisie') or \
            full_text.lower().startswith('vierde divisie') or \
            full_text.lower().startswith('tweede divisie') or \
            full_text.lower().startswith('eerste klasse') or \
            full_text.lower().startswith('tweede klasse') or \
            full_text.lower().startswith('derde klasse') or \
            full_text.lower().startswith('ingrediënten') or \
            full_text.lower().startswith('je hebt nodig') or \
            full_text.lower().startswith('adres lübeckweg') or \
            full_text.lower().startswith('za ') or \
            full_text.lower().startswith('programma\n\n') or \
            full_text.lower().startswith('voetbal\n\n') or \
            full_text.lower().startswith('duitsland\n\n') or \
            full_text.lower().startswith('topscorers\n') or \
            full_text.lower().startswith('basketbal\n\n') or \
            full_text.lower().startswith('hockey\n\n') or \
            full_text.lower().startswith('reageren?\n\n') or\
            full_text.lower().startswith('1\n\n'):
            return True, "Starts with known patterns for local program listings, recipes, sports scores, or similar content"

        # Numbered enumeration like "1. item, 2. item, etc"
        if re.search(r'1\.\s+.*?[,\n\s]\s*2\.\s+.*?[,\n\s]\s*3\.', full_text):
            return True, "Numbered enumeration list"
            
        # Contains names with grades (e.g., "Vaessen 5,5")
        if re.search(r'\b[A-Z][a-z]+\s+\d+[,\.]\d+\b', full_text):
            return True, "Contains names with grades"
            
        # Football scores format (e.g., "Ajax-PSV 1-0" or "90+1. Korte 1-0")
        if re.search(r'\b\d+\s*[-–]\s*\d+\s*\(', full_text) or \
            re.search(r'\d+\+\d+\.\s+\w+\s+\d+\s*[-–]\s*\d+', full_text):
            return True, "Football match scores"
    
    # Article passes all filters
    return False, "Kept"


def filter_articles(input_file, output_file, source, min_words, verbose=False):
    """Filter articles from a JSON file based on multiple criteria."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
    except Exception as e:
        print(f"Error: Failed to read '{input_file}': {e}", file=sys.stderr)
        sys.exit(1)
    
    if not isinstance(articles, list):
        print(f"Error: JSON must be a list of articles, got {type(articles).__name__}", file=sys.stderr)
        sys.exit(1)
    
    total_count = len(articles)
    filtered_articles = []
    removed_articles = []
    
    for article in articles:
        if not isinstance(article, dict):
            print(f"Skipping non-dict article: {article}", file=sys.stderr)
            continue

        article['source'] = source
        should_remove, reason = should_filter_article(article, min_words=min_words)
        if should_remove:
            removed_articles.append({
                'title': article.get('title', 'Unknown Title'),
                'reason': reason
            })
        else:
            filtered_articles.append(article)

    # Write  articles to output
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_articles, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error: Failed to write to '{output_file}': {e}", file=sys.stderr)
        sys.exit(1)
    
    # Print statistics
    kept_count = len(filtered_articles)
    removed_count = len(removed_articles)

    print(f"\nArticle filtering complete:")
    print(f"  Total articles: {total_count}")
    print(f"  Kept: {kept_count}")
    print(f"  Removed: {removed_count}")
    print(f"  Retention rate: {100 * kept_count / total_count:.1f}%")
    print(f"\nOutput saved to: {output_file}")

    if verbose and removed_count > 0:
        # Count removals by reason
        reason_counts = {}
        for removed in removed_articles:
            reason = removed['reason']
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        print(f"\nRemoval breakdown:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

        print(f"\nSample of removed articles (first 15):")
        for i, removed in enumerate(removed_articles[:15]):
            print(f"  {i+1}. {removed['title'][:60]:60s} ({removed['reason']})")
        if removed_count > 15:
            print(f"  ... and {removed_count - 15} more")

    return total_count, kept_count, removed_articles


def main(argv=None):
    parser = argparse.ArgumentParser(description="Filter articles from a JSON file based on multiple criteria.")
    parser.add_argument('-i', '--input', required=True, help='Input JSON file with articles')
    parser.add_argument('-o', '--output', required=True, help='Output JSON file for filtered articles')
    parser.add_argument('-w', '--min-words', type=int, default=100, help='Minimum word count threshold (default: 100)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print detailed removal statistics')
    args = parser.parse_args(argv)
    
    filter_articles(args.input, args.output, source=args.source, min_words=args.min_words, verbose=args.verbose)


if __name__ == "__main__":
    main()
