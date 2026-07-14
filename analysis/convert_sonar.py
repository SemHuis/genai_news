import json

# Define the precise mapping from SoNaR/Alpino tags to standard Stanza UPOS tags
SONAR_TO_UPOS = {
    'LID': 'DET',    # Lidwoord -> Determiner
    'VZ': 'ADP',     # Voorzetsel -> Adposition
    'LET': 'PUNCT',  # Leesteken -> Punctuation
    'VG': 'CCONJ',   # Voegwoord -> Conjunction
    'WW': 'VERB',    # Werkwoord -> Verb
    'VNW': 'PRON',   # Voornaamwoord -> Pronoun
    'BW': 'ADV',     # Bijwoord -> Adverb
    'N': 'NOUN',     # Naamwoord -> Noun
    'ADJ': 'ADJ',    # Adjectief -> Adjective
    'TW': 'NUM',     # Telwoord -> Numeral
    'SPEC': 'X',     # Special / Foreign tokens
    'TSW': 'INTJ'    # Tussenwerpsel -> Interjection
}

def convert_sonar_to_stanza_format(input_tsv, output_jsonl):
    print(f"Reading SoNaR tokens from: {input_tsv}...")
    converted_count = 0

    with open(input_tsv, 'r', encoding='utf-8') as infile, \
         open(output_jsonl, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            # Split out the file columns (tab separated)
            parts = line.split('\t')
            if len(parts) < 2:
                continue

            # Split out word and tag string, e.g., "krant" and "N(soort,ev,...)"
            token_info = parts[0].strip().split()
            if len(token_info) < 2:
                continue

            word_text = token_info[0]
            full_xpos_tag = token_info[1]

            # Isolate the broad prefix (e.g., "N" or "WW")
            sonar_prefix = full_xpos_tag.split('(')[0]
            upos_tag = SONAR_TO_UPOS.get(sonar_prefix, 'X')

            # Grab the absolute baseline count from column 2
            try:
                frequency = int(parts[1].strip())
            except ValueError:
                continue # Safely ignores headers or corrupted lines

            # Construct a clean dictionary matching Stanza token layout
            # Storing 'text' and 'lemma' as identical to match your SoNaR file structure safely
            stanza_token = {
                "text": word_text,
                "lemma": word_text.lower(),
                "upos": upos_tag,
                "xpos": full_xpos_tag,
                "frequency": frequency
            }

            # Write to JSONL format
            outfile.write(json.dumps(stanza_token, ensure_ascii=False) + '\n')
            converted_count += 1

    print(f"Conversion complete! Processed {converted_count} tokens.")
    print(f"Saved layout to: {output_jsonl}")

if __name__ == "__main__":
    # Point these to your path files
    INPUT_SONAR_TSV = "data/lemmaposfreqlist.tsv"
    OUTPUT_STANZA_JSONL = "sonar_newspapers_stanza_format.jsonl"

    convert_sonar_to_stanza_format(INPUT_SONAR_TSV, OUTPUT_STANZA_JSONL)
