import re
import json
import os
import argparse
import spacy
import random

nlp = spacy.load("nl_core_news_lg")

def clean_author(raw_author):
    """Cleans the raw author string and returns a list of cleaned author names."""
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
            unique_list.append(n.upper())
            
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

    if "entity_ruler" in nlp.pipe_names:
        # Pak de bestaande component op
        ruler = nlp.get_pipe("entity_ruler")
    else:
        # Alleen toevoegen als hij nog niet bestaat
        ruler = nlp.add_pipe("entity_ruler", before="ner")

    patterns = [
    {"label": "GPE", "pattern": "Dhaka"},
    {"label": "GPE", "pattern": "Zandvoort"},
    {"label": "GPE", "pattern": "Apeldoorn"},
    {"label": "GPE", "pattern": "Ter Apel"},
    {"label": "GPE", "pattern": "Havelte"},
    {"label": "GPE", "pattern": "Kuinre"},
    {"label": "GPE", "pattern": "Sint Jansklooster"},
    {"label": "GPE", "pattern": "Scheerwolde"},
    {"label": "GPE", "pattern": "Steenwijk"},
    {"label": "GPE", "pattern": "Steenwjkerwold"},
    {"label": "GPE", "pattern": "Kalenberg"},
    {"label": "GPE", "pattern": "Tuk"},
    {"label": "GPE", "pattern": "Steenwijkerland"},
    {"label": "GPE", "pattern": "Belt-Schutsloot"},
    {"label": "GPE", "pattern": "Diever"},
    {"label": "GPE", "pattern": "Staphorst"},
    {"label": "GPE", "pattern": "Vledder"},
    {"label": "GPE", "pattern": "Zuidwolde"},
    {"label": "GPE", "pattern": "Willemsoord"},
    {"label": "GPE", "pattern": "Basse"},
    {"label": "GPE", "pattern": "Kallenkote"},
    {"label": "GPE", "pattern": "Dwingeloo"},
    {"label": "GPE", "pattern": "Feanwalden"},
    {"label": "GPE", "pattern": "Kollum"},
    {"label": "GPE", "pattern": "De Westereen"},
    {"label": "GPE", "pattern": "Burdaard"},
    {"label": "GPE", "pattern": "Ternaard"},
    {"label": "GPE", "pattern": "Buitenpost"},
    {"label": "GPE", "pattern": "Lauwersoog"},
    {"label": "GPE", "pattern": "Reitsum"},
    {"label": "GPE", "pattern": "Hantumhuzen"},
    {"label": "GPE", "pattern": "Surhuisterveen"},
    {"label": "GPE", "pattern": "Holwerd"},
    {"label": "GPE", "pattern": "Driezum"},
    {"label": "GPE", "pattern": "Paezens"},
    {"label": "GPE", "pattern": "Westergeast"},
    {"label": "GPE", "pattern": "Bartlehiem"},
    {"label": "GPE", "pattern": "Warfstermolen"},
    {"label": "GPE", "pattern": "Ingwierrum"},
    {"label": "GPE", "pattern": "Stadskanaal"},
    {"label": "GPE", "pattern": "Ingwierrum"}
    ]

    ruler.add_patterns(patterns)
        
    # Clean only the first word, leave the second word exactly as-is
    # Capitalize first letter of first words and lowercase the rest of first word
    preview_words = words[:10]
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
            # Remove all leading punctuation from after_gpe
            after_gpe = re.sub(r'^\W+', '', after_gpe)
            # If the remaining text starts with an Uppercase letter
            if after_gpe and after_gpe[0].isupper():
                return after_gpe
    return text 


def clean_cap_start(text: str) -> str:
    if not text:
        return text
        
    # Define our character sets (including Dutch/European diacritics)
    UPPER = r'A-ZÄËÏÖÜÁÉÍÓÚÀÈÌÒÙ'
    LOWER = r'a-zäëïöüáéíóúàèìòù'
    
    # Explicit separators (hyphens, dashes, colons)
    # Catches: "AMSTERDAM -", "DEN HAAG (ANP):"
    pattern_sep = re.compile(
        r'^\s*'
        r'(?:[' + UPPER + r']{2,}(?:[,\s/&\'-]+[' + UPPER + r'0-9]+)*)' 
        r'(?:\s*\([A-Za-z0-9\s]+\))?' 
        r'\s*[-—–:]\s+',
        flags=re.UNICODE
    )
    
    # Space-only separator
    # Catches: "AMSTERDAM In Amsterdam", "TOUR DE FRANCE 2024 De start"
    # The Lookahead at the end ensures the NEXT word starts with a Capital letter 
    # followed by a lowercase letter OR a word boundary (for single-letter words like "U")
    pattern_space = re.compile(
        r'^\s*'
        r'(?:[' + UPPER + r']{2,}(?:[,\s/&\'-]+[' + UPPER + r'0-9]+)*)'
        r'(?:\s*\([A-Za-z0-9\s]+\))?'
        r'\s+'
        r'(?=[' + UPPER + r'](?:[' + LOWER + r']|\b))',
        flags=re.UNICODE
    )

    # First try removing datelines with explicit punctuation
    cleaned = pattern_sep.sub('', text)
    
    # If no punctuation was found, try the space-only heuristic
    if cleaned == text:
        cleaned = pattern_space.sub('', cleaned)
        
    return cleaned.strip()


def normalize_quotes(text: str) -> str:
    # Convert non-quote multi-character
    text = text.replace(',,', '"')
    
    # Normalize all single and double smart quotes to ASCII ' and "
    safe_mapping = str.maketrans({
        '„': '"', '“': '"', '”': '"', '«': '"', '»': '"', '＂': '"',
        '‘': "'", '’': "'", '‚': "'", '‛': "'"
    })
    text = text.translate(safe_mapping)
    
    # Double single quotes to "
    text = text.replace("''", '"')
    
    # Convert double backticks ``
    text = text.replace("``", '"')
    
    return text


def clean_article_base(text):
    """Cleans newspaper layout artifacts from text while preserving punctuation,vcase sensitivity, 
    and function words essential for stylometry.
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove "Bekijk de oorspronkelijke pagina:..."and everything after it
    text = re.sub(
        r"Bekijk de oorspronkelijke pagina:\s*pagina\s*\d+.*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Remove "Link naar PDF" or "PDF-bestand van dit document" if it appears anywhere in the text
    text = text.replace("Link naar PDF", "")
    text = text.replace("\n\nPDF-bestand van dit document\n\n", "")
    text = text.replace("PDF-bestand van dit document\n\n", "")

    # If the text hits uppercase genre markers on a fresh line, drop everything below.
    review_footer_pattern = (
        r"\n\n(FICTIE|NON-FICTIE|KLASSIEK|LITERATUUR|FILM)\n\n.*$"
    )
    text = re.sub(
        review_footer_pattern, "", text, flags=re.IGNORECASE | re.DOTALL
    )

    # Tailored sub-check for short product info arrays at the very end of text
    # e.g., 'Hannibal; 896 pagina's; € 79,95.'
    text = re.sub(
        r"\n\n[^\n]+;\s*\d+\s*pagina\'s;\s*€.*$", "", text, flags=re.MULTILINE
    )

    # 3. Strip Graphic / Image Captions explicitly if they appear mid-text
    # Matches 'Graphic' followed by newlines and text up until the next double newline
    text = re.sub(r"Graphic\n\n.*?(?=\n\n|$)", "", text, flags=re.IGNORECASE)

    while True:
        # Match a line from the start (^) that contains no newlines,
        # is between 2 and 100 chars, does NOT end with [.!?], followed by \n\n
        match = re.match(r"^([^\n]{2,100})(?<![.!?])\n\n", text)
        if match:
            # Remove the matched meta-header line and the newlines
            text = text[match.end() :]
        else:
            # Stop the loop as soon as we hit the actual body text (which ends with a punctuation mark)
            break

    # Removes localized headers that act as list intros inside the text
    # Matches '3x [title]'
    text = re.sub(
        r"\d+x\s+[^\n]+(?=\n\n)", "", text, flags=re.IGNORECASE
    ) 
    # Also remove localized headers that are just a couple of words long and do not end with proper sentence punctuation, which are likely to be section headers or injected captions
    text = re.sub(r"\n\n([^\n]{2,45})(?<![.!?])\n\n", "\n\n", text)

    # Removes end-of-text snippets 
    text = re.sub(
        r"\n\n[^\n]+\n\n[^\n]+\n\n(?:â˜…|â˜†|[★☆★☆\d/])+\n\n.*$",
        "",
        text,
        flags=re.DOTALL,
    )

    # Cleans capitalized words from the start of the text
    text = clean_cap_start(text)

    # If the text starts with 'x/x' (e.g., "kollum/hallum") we remove that part
    text = re.sub(
    r"^[a-zA-ZÁÉÓÚÝŇáéóúýň]+/[a-zA-ZÁÉÓÚÝŇáéóúýň]+(?:\s*[-–—]\s*)?",
    "", text, flags=re.IGNORECASE)

    # Final whitespace normalization
    # Collapse multiple consecutive newlines into exactly two, and trim outer edges
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    text = normalize_quotes(text)

    return text


def clean_article_tgf(text):
    """Cleans the article text while preserving stylometric features, and also extracts a highlight if present."""

    # First apply the base cleaning 
    text = clean_article_base(text)

    # Strip Tabloid Section & Sub-Brand Stamps at the top
    # Matches common uppercase Telegraaf supplements (VROUW, VRIJ, STRIKTVRIJ, etc.) 
    # up to 35 characters long, followed by double newlines.
    brand_pattern = r"^(VROUW|VRIJ|STRIKTVRIJ|GLAMOUR|REIZEN|CULTUUR|SPORT|GELD)(?:\s+[A-Z\s]+)?\n\n"
    text = re.sub(brand_pattern, "", text, flags=re.IGNORECASE)

    # Strip Chronological Timeline Block Prefixes
    # Matches strings like "JAN/FEB/MRT." or "APR/MEI/JUN." at the start of a paragraph
    # including variations with up to 4 month groupings (e.g. JUL/AUG/SEP/OCT)
    timeline_pattern = r"(?:^[A-Z]{3}(?:/[A-Z]{3}){1,3}\.?\s+|\n\n[A-Z]{3}(?:/[A-Z]{3}){1,3}\.?\s+)"
    text = re.sub(timeline_pattern, "\n\n", text)

    # Strip CMS App/Video Marketing Calls-to-Action
    # Deletes standard boilerplate instructions regarding downloading the app or watching media
    text = re.sub(r"of\s+download\s+de\s+gratis\s+Telegraaf-app\s+voor\s+het\s+laatste\s+nieuws\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Bekijk\s+de\s+video\s+(?:bovenaan|onderaan)\s+deze\s+pagina\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Luister\s+hier\s+naar\s+de\s+podcast\.?", "", text, flags=re.IGNORECASE)

    # Removes introduction of author -> door Sharon Story
    text = re.sub(r'\bdoor\s+\w+\s+\w+', '', text, flags=re.IGNORECASE)

    # Normalize Tabloid Quote Introductions
    # Tabloid text often drops traditional punctuation for script-like colons 
    # e.g., 'Omtzigt snikt: „...' -> replaces the colon with a comma to stabilize standard syntax
    text = re.sub(r"(\w+)\s*:\s*(?=[„'\"\'“])", r"\1, ", text)

    # Remove leading dashes with spaces (e.g., "- Amsterdam" -> "Amsterdam")
    text = re.sub(r"^[ \t]*[-–—][-–— \t]*(?=\w)", "", text, flags=re.MULTILINE)
    # Remove trailing dashes with spaces (e.g., "Amsterdam - " -> "Amsterdam")
    text = re.sub(r'\s+-(?:\s|$)', ' ', text)
    # Collapse any leftover double spaces down to a single space
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove GPE entities if they appear in the first two words of the text
    text = remove_gpe_entities(text)

    # Re-normalize final layout padding strings
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_article_ed(text, author_list=None):
    """Cleans the article text while preserving stylometric features, and also extracts a highlight if present."""
    if not text or not isinstance(text, str):
        return "", ""

    # First apply the base cleaning 
    text = clean_article_base(text)

    # Strip Hyperlocal Dateline Preambles at the absolute start
    # Matches uppercase city names at the very beginning (e.g., "EINDHOVEN - " or "VALKENSWAARD, 12.00 uur - ")
    # Up to 30 characters long, ending with a space-hyphen-space or space-comma-space sequence.
    text = re.sub(r"^[A-ZÁÉÓÚÝŇ\s]{3,30}\s*[,-]\s*(?:\d{2}\.\d{2}\s*uur\s*-\s*)?", "", text)

    # Strip Embedded CMS Video/Photo Intermissions
    # Matches variations of "Lees verder onder de video/foto" sandwiched between newlines
    text = re.sub(r"\n\s*Lees\s+verder\s+onder\s+de\s+(?:video|foto)\s*\n", "\n", text, flags=re.IGNORECASE)
    # Removes lines that are likely to be injected section headers or video captions, which are short and do not end with proper sentence punctuation.
    text = re.sub(r"\n\n([^\n]{2,45})(?<![.!?])\n\n", "\n\n", text)

    # Strip Injected Author Sign-Off Footers
    # Pass the 'author' field from your JSON entry into this function.
    # If the text ends with the author's name on its own line, we cut it off.
    if author_list and isinstance(author_list, list):
        for author in author_list:
            if author:
                # Creates a regex that looks for the exact author name at the end of the text block
                # allowing for flexible trailing whitespaces.
                author_signature_pattern = r"\n\n" + re.escape(author) + r"\s*$"
                text = re.sub(author_signature_pattern, "", text, flags=re.IGNORECASE)

    # Re-normalize final layout padding strings
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_article_dvhn(text, author_list=None):
    if not text or not isinstance(text, str):
        return ""

    # First apply the base cleaning 
    text = clean_article_base(text)

    # Alternate check: If an author array is present, explicitly pop their name from the start
    if author_list and isinstance(author_list, list):
        for author in author_list:
            if author:
                # Strips name if it sits isolated on its own line at the top
                text = re.sub(r"^" + re.escape(author) + r"\s*\n\n", "", text, flags=re.IGNORECASE)

    # Remove GPE entities if they appear in the first two words of the text
    text = remove_gpe_entities(text)

    # If the text starts with lowercase and is shorter than 40 characters before \n\n then remove this part
    text = re.sub(r"^[a-zà-ÿ][^\n]{0,38}\n\n", "", text)

    # Re-normalize final layout padding strings
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_article_stc(text):
    if not text or not isinstance(text, str):
        return ""

    # First apply the base cleaning 
    text = clean_article_base(text)

    # Remove GPE entities if they appear in the first two words of the text
    text = remove_gpe_entities(text)

    # Re-normalize final layout padding strings
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_article_nof(text):
    if not text or not isinstance(text, str):
        return ""

    # First apply the base cleaning 
    text = clean_article_base(text)

    # Strip "Uit het Algemeen Nieuws- en Advertentieblad" and the rest of the dateline before \n\n
    text = re.sub(r"^Uit het\s+.*?\n\n", "", text, flags=re.IGNORECASE)

    # Remove GPE entities if they appear in the first two words of the text
    text = remove_gpe_entities(text)

    # Re-normalize final layout padding strings
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_article(article_dict, source=None):
    """ Cleans all fields of the article dictionary and returns the cleaned version. """
    article_dict["author"] = clean_author(article_dict["author"])
    article_dict["date"] = format_date(article_dict["date"])
    # Apply different cleaning rules specific to each source
    if source == "Eindhovens Dagblad":
        article_dict["full_text"] = clean_article_ed(article_dict["full_text"], author_list=article_dict["author"])
        article_dict["highlight"] = clean_article_ed(article_dict["highlight"], author_list=article_dict["author"])
    elif source == "de Telegraaf":
        article_dict["full_text"] = clean_article_tgf(article_dict["full_text"])
        article_dict["highlight"] = clean_article_tgf(article_dict["highlight"])
    elif source == "Dagblad van het Noorden":
        article_dict["full_text"] = clean_article_dvhn(article_dict["full_text"], author_list=article_dict["author"])
        article_dict["highlight"] = clean_article_dvhn(article_dict["highlight"], author_list=article_dict["author"])
    elif source == "Steenwijker Courant":
        article_dict["full_text"] = clean_article_stc(article_dict["full_text"])
        article_dict["highlight"] = clean_article_stc(article_dict["highlight"])
    elif source == "Nieuwsblad Noordoost-Friesland":
        article_dict["full_text"] = clean_article_nof(article_dict["full_text"])
        article_dict["highlight"] = clean_article_nof(article_dict["highlight"])
    else:
        article_dict["full_text"] = clean_article_base(article_dict["full_text"])
        article_dict["highlight"] = clean_article_base(article_dict["highlight"])
    return article_dict


def parse_txt_to_list(input_file, source):
    """Reads the file and performs the initial regex extraction."""
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    chunks = content.split("End of Document")
    # take 50 random chunks for testing purposes, we will remove this line for the final version
    # add a seed for reproducibility
    random.seed(42)
    chunks = random.sample(chunks, 300)
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
    final_data = [clean_article(art, source=args.source) for art in raw_data]
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
