import os
import re
import json
import argparse


def clean_text(text):
    """Removes tags and standardizes whitespace."""
    if not text:
        return ""
    text = re.sub(r'\\', '', text)
    return text.strip()


def clean_section(s):
    """Helper for cleaning section/category strings by removing common markers and trimming punctuation."""
    if not s:
        return "Unknown"
    s = clean_text(s)
    # remove common "page" markers like '; Blz. 12', '; Blz. 12-13', '; Blz. ..', etc.
    s = re.sub(r'(?i)[\s;]*Blz\.?[\s:]*[0-9\.\-–, ]*', '', s).strip()
    # trim leftover punctuation
    s = re.sub(r'^[\s;:,\-]+|[\s;:,\-]+$', '', s)
    return s if s else "Unknown"


def is_date_line(s):
    """Check if a line looks like a date."""
    return bool(re.search(r"\d{1,2}\s+[A-Za-z]+\s+\d{4}", s)) or bool(re.search(r"\d{4}", s))


def is_noise_line(s):
    """Check if a line contains metadata/noise keywords."""
    s_low = s.lower()
    noise_keywords = ['copyright', 'length', 'byline', 'highlight', 'body', 'volledige tekst', 'link naar pdf', 'load-date']
    return any(k in s_low for k in noise_keywords)


def token_overlap(a, b):
    """Compute token overlap between two strings as a proportion."""
    a_tokens = set(re.findall(r"\w+", (a or '').lower()))
    b_tokens = set(re.findall(r"\w+", (b or '').lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / float(min(len(a_tokens), len(b_tokens)))


def looks_like_paper_name(s, title, paper_keywords):
    """Check if a string looks like a newspaper/paper name."""
    if not s:
        return False
    s_low = s.lower()
    # Prefer short names that contain a paper keyword
    if any(pk in s_low for pk in paper_keywords):
        return True
    # Also accept short names (<=4 words) that are not too similar to the title
    if len(s.split()) <= 4 and token_overlap(s, title) < 0.4:
        return True
    return False


def parse_date(raw_text):
    """Parse a date string and return it as DD/MM/YYYY.
    """
    if not raw_text:
        return "Unknown Date"
    s = clean_text(raw_text)

    # Remove common weekday names
    s = re.sub(r"\b(maandag|dinsdag|woensdag|donderdag|vrijdag|zaterdag|zondag)\b", "", s, flags=re.I)

    # Map Dutch month names to month numbers
    months = {
        'januari':1, 'februari':2, 'maart':3, 'april':4, 'mei':5, 'juni':6,
        'juli':7, 'augustus':8, 'september':9, 'oktober':10, 'november':11, 'december':12
    }

    # Try patterns like '26 augustus 2024'
    m = re.search(r"(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]+)\s+(?P<year>\d{4})", s, flags=re.I)
    if m:
        day = int(m.group('day'))
        month_word = m.group('month').lower()
        year = m.group('year')
        month = months.get(month_word)
        if month:
            return f"{day:02d}/{month:02d}/{year}"

    # Try numeric formats like 26/08/2024 or 26-08-2024 or 26.08.2024
    m2 = re.search(r"(?P<day>\d{1,2})[\/\-\.](?P<month>\d{1,2})[\/\-\.](?P<year>\d{2,4})", s)
    if m2:
        day = int(m2.group('day'))
        month = int(m2.group('month'))
        year = m2.group('year')
        if len(year) == 2:
            # We assume 2000s for two-digit years
            year = '20' + year
        return f"{day:02d}/{month:02d}/{year}"
    return s


def extract_section(art):
    """Try to extract a section/category for the article.
      1. Look for explicit labels like 'SECTION:', 'SECTIE:' or 'Rubriek:'
      2. Fallback: check the first few lines for a short all-caps line
         that is likely a section header (e.g. 'BINNENLAND', 'SPORT').
    Returns cleaned section string or 'Unknown'.
    """
    if not art:
        return "Unknown"

    # Explicit labels
    for pat in [r'(?im)^\s*SECTION[:\-\s]+(.+)$', r'(?im)^\s*SECTIE[:\-\s]+(.+)$', r'(?im)^\s*Rubriek[:\-\s]+(.+)$']:
        m = re.search(pat, art)
        if m:
            return clean_section(m.group(1))

    # Alternative: Look at first lines for an uppercase short line
    lines = [l.strip() for l in art.splitlines() if l.strip()]
    for ln in lines[:8]:
        # Skip lines that are the title or obvious markers
        if re.search(r'\bNRC\b', ln, flags=re.I):
            continue
        if re.search(r'VOLLEDIGE TEKST', ln, flags=re.I):
            continue
        # Consider this a section if it's mostly uppercase letters and short
        if re.fullmatch(r"[A-Z0-9\-\'\.]+(?:\s+[A-Z0-9\-\'\.]+)*(?:\s*;\s*Blz\..*)?", ln):
            return clean_section(ln)

    return "Unknown"


def is_valid_name(text):
    """Simple heuristic checks to determine if a line of text is likely a person's (author's) name."""
    # Names less than 60 characters, don't end with a period, and have a good proportion of capitalized words.
    if len(text) > 60:
        return False
    if text.endswith('.'):
        return False
    words = text.split()
    if not words:
        return False
    cap_words = [w for w in words if w[0].isupper()]
    
    # At least 50% of words starts with a capital letter
    if len(cap_words) / len(words) < 0.5:
        return False

    return True


def parse_newspaper_batch(raw_text):
    """Parses raw text using multiple possible separators."""
    # Split by common separators (case-insensitive)
    # Handles "End of Document", "END OF DOCUMENT", or "Document 1 of 250"
    articles = re.split(r'(?i)End of Document|Document \d+ of \d+', raw_text)
    
    # Newspaper keywords for source detection
    paper_keywords = ['courant', 'dagblad', 'krant', 'nrc', 'telegraaf', 'parool', 'metro', 'nieuws', 'journaal', 'weekblad', 'blad']
    
    extracted = []
    for art in articles:
        art = art.strip()
        if not art:
            continue

        # Split into lines once for efficiency
        lines = [l.strip() for l in art.splitlines() if l.strip()]
        if not lines:
            continue

        # Heuristics: detect NRC-style blocks, but also accept other newspapers.
        is_nrc = bool(re.search(r"\bNRC\b", art, flags=re.I))

        # 1. Extract title
        title = "Unknown Title"
        if is_nrc:
            title_match = re.search(r'(?:^|\n)\s*(?:\d+\.\s*)?([^\n]+)\n\s*NRC', art, re.DOTALL | re.IGNORECASE)
            if title_match:
                title = clean_text(title_match.group(1))
        else:
            if lines:
                title = clean_text(lines[0])

        # 2. Extract source (paper/newspaper name)
        source = "Unknown Source"
        if len(lines) >= 2:
            candidate = lines[1]
            if not is_date_line(candidate) and not is_noise_line(candidate) and looks_like_paper_name(candidate, title, paper_keywords):
                source = clean_text(candidate)
            else:
                # Fallback: scan lines 2–6 for a reasonable paper name
                for ln in lines[1:6]:
                    if is_date_line(ln) or is_noise_line(ln):
                        continue
                    if looks_like_paper_name(ln, title, paper_keywords):
                        source = clean_text(ln)
                        break
                else:
                    # Final fallback: any short non-noise line with low title overlap
                    for ln in lines[1:6]:
                        if is_date_line(ln) or is_noise_line(ln):
                            continue
                        if token_overlap(ln, title) < 0.5 and len(ln) < 100:
                            source = clean_text(ln)
                            break

        # 3. Extract date
        date = "Unknown Date"
        if is_nrc:
            date_match = re.search(r'NRC\s*\n\s*([^\n]+)', art, flags=re.I)
            if date_match:
                date = parse_date(date_match.group(1))
        if date == "Unknown Date":
            date_guess = re.search(r"\d{1,2}\s+[A-Za-z]+\s+\d{4}", art, flags=re.I)
            if date_guess:
                date = parse_date(date_guess.group(0))

        # 4. Extract author
        author = "Unknown Author"
        author_match = re.search(r'Byline:\s*([^\n]+)', art, flags=re.I)
        if author_match:
            author = clean_text(author_match.group(1))
        else:
            match = re.search(r"^(.*?)\nLink naar PDF", art, re.MULTILINE)
            if match:
                potential_name = match.group(1).strip()
            if is_valid_name(potential_name):
                author = clean_text(potential_name)
        
        # Normalize author name: capitalize each word properly
        if author != "Unknown Author":
            words = author.split()
            normalized_words = []
            for i, word in enumerate(words):
                # Keep small words like 'van', 'de', 'den' lowercase unless at the start
                if i > 0 and word.lower() in ['van', 'de', 'den', 'der', 'te', 'ter', 'the']:
                    normalized_words.append(word.lower())
                else:
                    normalized_words.append(word.capitalize())
            author = ' '.join(normalized_words)

        # 5. Extract body: multiple fallbacks
        body = ""
        # a) Full-text marker
        m_body = re.search(r'VOLLEDIGE TEKST:(.*?)(?=Link naar PDF|Graphic|Load-Date|$)', art, re.DOTALL | re.IGNORECASE)
        if m_body:
            body = clean_text(m_body.group(1))
        else:
            # b) Body marker
            m_body2 = re.search(r'\bBody\b\s*\n(.*?)(?=Link naar PDF|Graphic|Load-Date|$)', art, re.DOTALL | re.IGNORECASE)
            if m_body2:
                body = clean_text(m_body2.group(1))
            else:
                # c) If NRC, take remainder after NRC/date line
                if is_nrc:
                    m_nrc_line = re.search(r'NRC\s*\n\s*[^\n]+', art, flags=re.I)
                    start_pos = m_nrc_line.end() if m_nrc_line else 0
                    remainder = art[start_pos:]
                    body = clean_text(re.split(r'(?i)Link naar PDF|Graphic|Load-Date', remainder)[0])
                else:
                    # d) Find first blank line after a short header and take remainder
                    body_candidate = ""
                    for i, ln in enumerate(lines[:60]):
                        if ln == "":
                            body_candidate = '\n'.join(lines[i+1:])
                            break
                    if body_candidate:
                        body = clean_text(re.split(r'(?i)Link naar PDF|Graphic|Load-Date', body_candidate)[0])
                    else:
                        # e) Last resort: take the whole block
                        body = clean_text(art)

        if body:
            extracted.append({
                "title": title,
                "date": date,
                "author": author,
                "source": source,
                "section": extract_section(art),
                "full_text": body
            })
    return extracted

def run_batch_processor(input_file, output_name):
    all_data = []

    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    except Exception as e:
        print(f"Failed to read {input_file}: {e}")
        return

    parsed = parse_newspaper_batch(raw_text)
    all_data.extend(parsed)
    print(f"Processed '{input_file}': found {len(parsed)} articles.")

    with open(output_name, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)
        # print 5 articles for verification
        print("\nSample extracted articles:")
        for art in all_data[:5]:  
            print(f"Title: {art['title']}")
            print(f"Date: {art['date']}")
            print(f"Author: {art['author']}")
            print(f"Source: {art['source']}")
            print(f"Section: {art['section']}")
            print(f"Full Text (first 200 chars): {art['full_text'][:200]}...")
            print("-" * 40)

    print(f"\nFINISH: Total articles extracted: {len(all_data)}")
    print(f"File created: {output_name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract articles and metadata from a plain text export of articles.")
    parser.add_argument('-i', '--input', default='all_articles.txt', help='Input text file (default: all_articles.txt)')
    parser.add_argument('-o', '--output', default='master_data.json', help='Output JSON file (default: master_data.json)')
    args = parser.parse_args(argv)
    run_batch_processor(args.input, args.output)


if __name__ == "__main__":
    main()