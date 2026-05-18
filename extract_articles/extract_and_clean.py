import re
import json
import os
import argparse
import spacy

nlp = spacy.load("nl_core_news_lg")

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

def remove_gpe_entities(text):
    """Removes GPE entities from text using spaCy if they act as a dateline header."""
    words = text.split()
    if not words:
        return text
        
    # Clean only the first word, leave the second word exactly as-is
    preview_words = words[:2]
    preview_words[0] = preview_words[0].capitalize()
    text_preview = " ".join(preview_words)
    
    doc = nlp(text_preview)
    start_text_entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Find if a GPE exists in the preview
    gpe_entity = next((text for text, label in start_text_entities if label == "GPE"), None)
    
    if gpe_entity:
        # Use case-insensitive matching (.lower()) against the original text
        if text.lower().startswith(gpe_entity.lower()):
            # Slice out the length of the GPE from the original text string
            after_gpe = text[len(gpe_entity):].strip()
            # If the remaining text starts with an Uppercase letter
            if after_gpe and after_gpe[0].isupper():
                return after_gpe
    return text


def clean_text_block(text, is_full_text=False):
    """General text cleanup (removing extra whitespace, etc.)"""
    if not text:
        return ("", "") if is_full_text else ""

    # Make text a raw string to avoid escape character issues
    text = repr(text)[1:-1]
    
    # Add all text before to the highlight and remove it from the full_text
    highlight = ""
    if is_full_text:
        text_preview = " ".join(text.split()[:40])
        if "\\n\\n" in text_preview:
            parts = text.split("\\n\\n", 1)
            text = parts[1].strip()
            highlight = parts[0].strip()

    # Remove any "Bekijk de oorspronkelijke pagina:" and everything after it
    marker = "Bekijk de oorspronkelijke pagina:"
    if marker in text:
        text = text.split(marker)[0]

    # Remove "Link naar PDF" if it appears anywhere in the text
    text = text.replace("Link naar PDF", "")
    text = text.replace("\\n\\nPDF-bestand van dit document\\n\\n", "")
    text = text.replace("PDF-bestand van dit document\\n\\n", "")
    
    # Remove leading dashes with spaces (e.g., "- Amsterdam" -> "Amsterdam")
    text = re.sub(r'(?:^|\s)-\s+', ' ', text)
    highlight = re.sub(r'(?:^|\s)-\s+', ' ', highlight)
    
    # Remove trailing dashes with spaces (e.g., "Amsterdam - " -> "Amsterdam")
    text = re.sub(r'\s+-(?:\s|$)', ' ', text)
    highlight = re.sub(r'\s+-(?:\s|$)', ' ', highlight)

    # Collapse any leftover double spaces down to a single space
    text = re.sub(r'\s+', ' ', text).strip()
    highlight = re.sub(r'\s+', ' ', highlight).strip()

    # Remove leading dashes with spaces (e.g., "- Amsterdam" -> "Amsterdam")
    text = text.replace("- ", "")
    highlight = highlight.replace("- ", "")
    text = text.replace(" -", "")
    highlight = highlight.replace(" -", "")

    if text.split()[:1] == ["door"]:
        # Remove "door" and rest of text before \n\n if it appears at the very start of the text (common in Telegraaf)
        text = re.sub(r'^door\s+.*?\\n\\n', '', text, flags=re.IGNORECASE)

    # If the highlight is more than 7% of the full text, we assume it's not a highlight but rather an introductory paragraph and we move it to the full text instead
    # cannot divide by zero because if there is no text, there is also no highlight, so the condition will not be met and we will not move the highlight to the full text
    if len(text) != 0:
        if is_full_text and highlight and len(highlight) / len(text) > 0.07:
            text = highlight + " " + text
            highlight = "" 

    # If in the first 20 words of the text there is a "•" we remove everything before and including the "•" 
    if is_full_text:
        text_preview = " ".join(text.split()[:20])
        if "•" in text_preview:
            text = text.split("•", 1)[1].strip()

    text = remove_gpe_entities(text)
    highlight = remove_gpe_entities(highlight)
    
    if is_full_text:
        return text.strip(), highlight.strip()
    
    return text.strip()


def clean_article(article_dict):
    """
    The orchestrator for cleaning. 
    Pass a raw dictionary here and it returns a polished one.
    """
    article_dict["author"] = clean_author(article_dict["author"])
    article_dict["date"] = format_date(article_dict["date"])
    article_dict["full_text"], highlight = clean_text_block(article_dict["full_text"], is_full_text=True)
    article_dict["highlight"] = clean_text_block(article_dict["highlight"] + " " + highlight)
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
            "title": lines[0].split(";")[0].strip(),
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
