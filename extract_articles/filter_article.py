import re
import json
import os
import argparse


def is_recipe(article_dict: dict) -> bool:
    """
    Checks if an article is a recipe based on common Dutch culinary keywords 
    and list formats. Returns True if it's likely a recipe.
    """
    full_text = article_dict.get('full_text', '').lower()
    title = article_dict.get('title', '').lower()
    
    # Title checks
    recipe_title_keywords = ['recept:', 'recept van de dag', 'koken:']
    if any(kw in title for kw in recipe_title_keywords):
        return True
        
    # Starting phrase check
    starts = ('ingrediënten', 'je hebt nodig', 'bereiding', 'nodig:')
    if full_text.startswith(starts):
        return True
        
    # Keyword Density Heuristic
    # If a text contains multiple cooking action verbs and measurements it's likely a recipe
    recipe_keywords = [
        'ingrediënten', 'bereidingswijze', 'bereidingstijd', 'kooktijd',
        'snijd', 'meng', 'voeg toe', 'verwarm de oven', 'eetlepel', 'theelepel',
        'bakplaat', 'koekenpan', 'scheutje', 'garneer', 'bereiding', 'strooi', 
        'verhit', 'braad', 'gram', 'ml '
    ]
    
    keyword_hits = sum(1 for kw in recipe_keywords if kw in full_text)
    
    # If we hit 3 or more distinct recipe-related words we remove it
    if keyword_hits >= 3:
        return True
        
    # Ingredient List Regex
    ingredient_pattern = r'(?m)^\s*\d+(?:[\.,]\d+)?\s*(?:gram|gr|ml|liter|el|tl|eetlepel|theelepel|stuks?|teentjes?|snufje)\s+[a-z]+'
    
    # If we find 3 or more ingredient lines it's likely a recipe
    if len(re.findall(ingredient_pattern, full_text)) >= 3:
        return True
        
    return False


def is_agenda(article_dict: dict) -> bool:
    """
    Controleert of een artikel een agenda of evenementenlijst is op basis van 
    tijdsblokken, losse tijden en typische agenda-woorden.
    """
    full_text = article_dict.get('full_text', '')
    text_lower = full_text.lower()
    title = article_dict.get('title', '').lower()
    
    # Title checks
    agenda_titles = ['agenda', 'uitagenda', 'evenementen', 'wat is er te doen', 'weekend', 'tips']
    if any(kw in title for kw in agenda_titles):
        return True

    # Look for time blocks (e.g. 11.00-17.00, 14:00 - 15:30)
    time_block_pattern = r'\b\d{1,2}[:.]\d{2}\s*-\s*\d{1,2}[:.]\d{2}\b'
    time_blocks = re.findall(time_block_pattern, full_text)
    
    # If there are 4 or more it is likely an agenda
    if len(time_blocks) >= 4:
        return True
        
    # Phrase and time check
    single_time_pattern = r'\b\d{1,2}[:.]\d{2}\b'
    single_times = re.findall(single_time_pattern, full_text)
    
    # Phrase check
    agenda_keywords = [
        'opg.',               
        'entree', 
        'excursie', 
        'stadswandeling', 
        'aanmelden via', 
        'organist; klassiek', 
        'o.l.v.'              
    ]
    
    keyword_hits = sum(1 for kw in agenda_keywords if kw in text_lower)
    
    # Minimum of 5 times and 2 phrases it's likely an agenda
    if len(single_times) >= 5 and keyword_hits >= 2:
        return True
        
    return False


def filter_article_base(article_dict: dict) -> bool:
    """
    Applies baseline filtering rules to all articles.
    Returns True if the article should be discarded (filtered out).
    """
    title = article_dict.get('title', '').lower()
    full_text = article_dict.get('full_text', '').lower()

    # Full text shorter than 50 words
    if len(full_text.split()) < 50:
        return True

    # Enumerations (photo article or puzzle?)
    lines = [line.strip() for line in full_text.split('\n') if line.strip()]
    if lines:
        enum_lines = sum(1 for line in lines if re.match(r'^[\d\-\*]', line))
        if len(lines) > 5 and (enum_lines / len(lines)) > 0.5:
            return True

    # Many types of puzzles
    puzzel_keywords = ['puzzel', 'sudoku', 'kruiswoord']
    if any(kw in title for kw in puzzel_keywords):
        return True
    if 'horizontaal:' in full_text and 'verticaal:' in full_text:
        return True

    # Colofon articles
    if 'colofon' in title or 'colofon' in full_text:
        return True

    # Recipes
    if is_recipe(article_dict):
        return True

    if is_agenda(article_dict):
        return True

    # Specific titles
    bad_titles = ['de ommezwaai', 'op rapport', 'net op de wereld', 'jarig']
    if any(bad in title for bad in bad_titles):
        return True

    return False


def filter_article_dvhn(article_dict: dict) -> bool:
    """Filters for Dagblad van het Noorden."""
    # Apply the base filter first
    if filter_article_base(article_dict):
        return True

    full_text = article_dict.get('full_text', '').lower()
    
    # No headline & no author & shorter than threshold (using 100 as placeholder)
    word_count = len(full_text.split())
    has_headline = bool(article_dict.get('headline_in_original'))
    has_author = bool(article_dict.get('author'))
    
    if not has_headline and not has_author and word_count < 100:
        return True

    # Specific starts
    starts = ('ingrediënten', 'je hebt nodig', 'za ', 'adres lübeckweg', 'reageren?\n\n')
    if full_text.startswith(starts):
        return True

    # Football/basketball/hockey table/topscorers/results/programma
    sports_keywords = ['uitslagen', 'programma', 'stand', 'topscorers']
    sports_contexts = ['voetbal', 'basketbal', 'hockey', 'sport']
    if any(kw in full_text for kw in sports_keywords) and any(sport in full_text for sport in sports_contexts):
        return True

    return False


def filter_article_tgf(article_dict: dict) -> bool:
    """Filters for de Telegraaf."""
    if filter_article_base(article_dict): 
        return True
    # Add Telegraaf specific rules here
    return False


def filter_article_ed(article_dict: dict) -> bool:
    """Filters for Eindhovens Dagblad."""
    if filter_article_base(article_dict): 
        return True

    full_text = article_dict.get('full_text', '').lower()

    # Specific starts
    starts = ('hoeveel verdien je')
    if full_text.startswith(starts):
        return True

    return False


def filter_article_stc(article_dict: dict) -> bool:
    """Filters for Steenwijker Courant."""
    if filter_article_base(article_dict): 
        return True

    full_text = article_dict.get('full_text', '').lower()
    
    # Starts with a date -> programme
    if re.match(r'^\d{1,2}\s+[a-z]{3}\.\s*\n\n', full_text):
        return True
    if re.match(r'^\d{1,2}\s+[a-z]{3}\.\s+t/m\s+\d{1,2}\s+[a-z]{3}\.\s*\n\n', full_text):
        return True

    # Specific starts
    starts = ('de namen van eindexamenkandidaten die geslaagd')
    if full_text.startswith(starts):
        return True


    return False


def filter_article_nof(article_dict: dict) -> bool:
    """Filters for Nieuwsblad Noordoost-Friesland."""
    if filter_article_base(article_dict): 
        return True

    full_text = article_dict.get('full_text', '').lower()

    # Specific starts
    starts = ('de resultaten van', 'de uitslagen van')
    if full_text.startswith(starts):
        return True

    return False


def is_junk(article_dict: dict, source: str = None) -> bool:
    """
    Central router that directs the article to the correct filter 
    based on its source publication.
    """
    if source == "Dagblad van het Noorden":
        return filter_article_dvhn(article_dict)
    elif source == "de Telegraaf":
        return filter_article_tgf(article_dict)
    elif source == "Eindhovens Dagblad":
        return filter_article_ed(article_dict)
    elif source == "Steenwijker Courant":
        return filter_article_stc(article_dict)
    elif source == "Nieuwsblad Noordoost-Friesland":
        return filter_article_nof(article_dict)
    else:
        # Fallback to base filter if source is unknown
        return filter_article_base(article_dict)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Filter out junk news articles.")
    parser.add_argument('-i', '--input', required=True, help='Input JSON file (from cleaning step)')
    parser.add_argument('-o', '--output', default=None, help='Output JSON file for kept articles')
    args = parser.parse_args(argv)

    if args.output is None:
        file_base = os.path.splitext(args.input)[0]
        args.output = f"{file_base}_filtered.json"
        
    print(f"Reading from: {args.input}")

    with open(args.input, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    print(f"Loaded {len(articles)} articles. Applying filters...")
    
    kept_articles = []
    discarded_count = 0
    
    for art in articles:
        # Assume the source is stored in the dictionary (e.g., art.get("source"))
        source = art.get("source")
        
        # If it is NOT junk, add it to the keep list
        if not is_junk(art, source=source):
            kept_articles.append(art)
        else:
            discarded_count += 1

    print(f"Filtered out {discarded_count} articles.")
    print(f"Keeping {len(kept_articles)} articles.")

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(kept_articles, f, ensure_ascii=False, indent=4)
        
    print(f"Filtered output saved to: {args.output}")


if __name__ == "__main__":
    main()
