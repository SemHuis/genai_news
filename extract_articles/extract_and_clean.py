import re
import json
import os
import argparse


def clean_author(raw_author):
    if not raw_author:
        return []
    
    # Strip leading/trailing whitespace, periods, and slashes
    author_string = raw_author.strip().rstrip('./ ')

    # Prefix Removal (Words we strip from the very start of the whole block)
    prefixes = [
        'onze redacteur', 'onze verslaggever', 'onze verslaggevers', 
        'onze correspondente', 'onze correpsondente', 'by', 'door', 
        'tekst', 'redactie'
    ]
    prefix_pattern = r'^(' + '|'.join(prefixes) + r')\s*[:\-]?\s+'
    author_string = re.sub(prefix_pattern, '', author_string, flags=re.IGNORECASE)

    # Split: 'en', '&', or multiple spaces
    initial_list = re.split(r'\s+en\s+|\s*&\s*|\s*/\s*|\s*amp\s*|\s{2,}', author_string, flags=re.IGNORECASE)

    # Split for single spaces (e.g., MEREL VAN BEERS TOM TACKEN)
    intermediate_authors = []
    connectors = {'VAN', 'DE', 'DER', 'DEN', 'HET', 'IN', 'TE', "VAN'T", "VANDEN", "VANDE", "V.D.", "VD"}

    for entry in initial_list:
        words = entry.split()
        if len(words) >= 4:
            split_point = -1
            for i in range(2, len(words) - 1):
                curr, prev = words[i].upper(), words[i-1].upper()
                if prev == 'ERVEN' and curr == 'DORENS': continue
                if curr not in connectors and prev not in connectors:
                    split_point = i
                    break
            if split_point != -1:
                intermediate_authors.append(" ".join(words[:split_point]))
                intermediate_authors.append(" ".join(words[split_point:]))
                continue
        intermediate_authors.append(entry)

    # Deep Clean per author
    cleaned_list = []
    blacklist_keywords = ['illustratie', 'foto', 'foto\'s', 'beeld', 'verslaggeving', 'graphics', 'infographic', 
                        'onze correspondent', 'onze verslaggever', 'onze verslaggevers', 'onze redacteur',
                        'onze correspondente', 'onze correpsondente', 'redactie', 'tekst', 'door', 'by', 'getty',
                        'warschau', 'en foto', 'en foto\'s']
    
    for name in intermediate_authors:
        # Remove parentheses and their content, and clean up leading/trailing junk
        name = name.strip().lstrip('- ')
        name = re.sub(r'\(.*?\)', '', name) 
        
        # Handle keywords at the START (e.g., "FOTO JEFFREY" -> "JEFFREY")
        # The \s* handles cases with or without colons/spaces
        start_residue_pattern = r'^(' + '|'.join(blacklist_keywords) + r')s?[:\s\-]+'
        name = re.sub(start_residue_pattern, '', name, flags=re.IGNORECASE).strip()

        # Handle keywords in the MIDDLE (Truncate everything after)
        mid_residue_pattern = r'\s+(' + '|'.join(blacklist_keywords) + r')s?[:\s\-].*$'
        name = re.sub(mid_residue_pattern, '', name, flags=re.IGNORECASE).strip()
        name = name.split(',')[0].strip()

        # Catch standard emails and "JESPER @ED.NL"
        # If the string contains an '@', it's almost certainly not a name we want to keep
        if '@' in name:
            # Check if there is a name before the email in the same string
            name = re.sub(r'[\w\.-]+\s*@\s*[\w\.-]+.*$', '', name).strip()
        
        # Remove leading numbers (e.g., "12 Saskia" -> "Saskia")
        name = re.sub(r'^\d+\s+', '', name) 
        
        # Filter out colons and entries with too many words 
        if ':' in name or len(name.split()) > 5:
            continue
        
        # Filter out entries that are just keywords 
        if name.lower().rstrip('s') in blacklist_keywords:
            continue

        # Only keep names that are longer than 2 characters
        if name and len(name) > 2:
            cleaned_list.append(name)

    # Fix specific names
    if len(cleaned_list) >= 2:
        if cleaned_list[0].upper().endswith("VAN ERVEN") and cleaned_list[1].upper() == "DORENS":
            cleaned_list[0] = cleaned_list[0] + " " + cleaned_list.pop(1)
        if cleaned_list[0].upper() == "AMANDA BULTHUIS" and cleaned_list[1].upper() == "OERLEMANS":
            cleaned_list[0] = cleaned_list[0] + " " + cleaned_list.pop(1)
        if cleaned_list[0].upper() == "JOHN" and cleaned_list[1].upper() == "PAUL":
            cleaned_list[0] = cleaned_list[0] + " " + cleaned_list.pop(1)
        if cleaned_list[0].upper() == "WAFA AL" and cleaned_list[1].upper() == "ALI":
            cleaned_list[0] = cleaned_list[0] + " " + cleaned_list.pop(1)

    # Remove duplicates like ['JESPER DE VAAN', 'JESPER DE VAAN'])
    unique_list = []
    for n in cleaned_list:
        if n not in unique_list:
            unique_list.append(n)
            
    return unique_list


def format_date(raw_date):
    """Converts Dutch text dates to DD/MM/YYYY."""
    months_nl = {
        "januari": "01", "februari": "02", "maart": "03", "april": "04",
        "mei": "05", "juni": "06", "juli": "07", "augustus": "08",
        "september": "09", "oktober": "10", "november": "11", "december": "12"
    }
    date_match = re.search(r'(\d{1,2})\s+([a-z]+)\s+(\d{4})', raw_date, re.IGNORECASE)
    if date_match:
        day, month_name, year = date_match.groups()
        month_num = months_nl.get(month_name.lower(), "01")
        return f"{day.zfill(2)}/{month_num}/{year}"
    return raw_date


def clean_text_block(text):
    """General text cleanup (removing extra whitespace, etc.)"""
    if not text:
        return ""

    # Remove any "Bekijk de oorspronkelijke pagina:" and everything after it
    marker = "Bekijk de oorspronkelijke pagina:"
    if marker in text:
        text = text.split(marker)[0]

    # Remove "Link naar PDF" if it appears anywhere in the text
    text = text.replace("Link naar PDF", "")

    return text.strip()


def clean_article(article_dict):
    """
    The orchestrator for cleaning. 
    Pass a raw dictionary here and it returns a polished one.
    """
    article_dict["author"] = clean_author(article_dict["author"])
    article_dict["date"] = format_date(article_dict["date"])
    article_dict["full_text"] = clean_text_block(article_dict["full_text"])
    article_dict["highlight"] = clean_text_block(article_dict["highlight"])
    return article_dict


def parse_txt_to_list(input_file, source):
    """Reads the file and performs the initial regex extraction."""
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    chunks = content.split("End of Document")
    raw_articles = []

    for chunk in chunks:
        # Filter out numbered TOC listings
        clean_chunk = re.sub(r'^\d+\.\s+.*$', '', chunk, flags=re.MULTILINE).strip()
        
        if "Body" not in clean_chunk:
            continue

        lines = [l.strip() for l in clean_chunk.split('\n') if l.strip()]
        if len(lines) < 3: continue

        # Extract Raw Fields
        section_match = re.search(r'Section:\s*(.*?)(?:;|$)', clean_chunk)
        author_match = re.search(r'Byline:\s*(.*?)(?:\n|$)', clean_chunk)
        highlight_match = re.search(r'Highlight:\s*(.*?)(?=\n\s*Body|$)', clean_chunk, re.DOTALL)
        body_match = re.search(r'Body\s+(.*?)(?=Load-Date:|$)', clean_chunk, re.DOTALL)

        raw_articles.append({
            "title": lines[0],
            "source": source,
            "date": lines[2],
            "author": author_match.group(1) if author_match else "",
            "section": section_match.group(1) if section_match else "",
            "highlight": highlight_match.group(1) if highlight_match else "",
            "full_text": body_match.group(1) if body_match else ""
        })
    
    return raw_articles


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract and clean news articles.")
    parser.add_argument('-i', '--input', default='articles.txt', help='Input text file')
    parser.add_argument('-o', '--output', default=None, help='Output JSON file')
    parser.add_argument('-s', '--source', type=str, help='Source name to set for all articles)')
    args = parser.parse_args(argv)

    if args.output is None:
        file_base = os.path.splitext(args.input)[0]
        args.output = f"{file_base}.json"
    print(f"Reading from: {args.input}")

    # Execute the parsing and cleaning pipeline
    raw_data = parse_txt_to_list(args.input, args.source)
    print(f"Found {len(raw_data)} articles. Cleaning data...")
    final_data = [clean_article(art) for art in raw_data]
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
