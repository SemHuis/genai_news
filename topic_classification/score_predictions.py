import ast
import json
import re
import csv
import sys
import argparse
from collections import Counter


def load_mapping(mapping_file):
    """Parses the IPTC JSON map to create a subtopic -> head_topic lookup."""
    with open(mapping_file, 'r') as f:
        iptc_map = json.load(f)

    subtopic_to_headtopic_name_map = {}
    for head_topic_entry, sub_topics_list in iptc_map.items():
        # Extract head topic name (e.g., 'crime, law and justice')
        head_topic_name_match = re.search(r'-\s*([^\t]+)', head_topic_entry)
        if head_topic_name_match:
            head_topic_name = head_topic_name_match.group(1).strip()
        else:
            continue

        for sub_topic_entry in sub_topics_list:
            # Extract subtopic name (e.g., 'law enforcement')
            sub_topic_name_match = re.search(r'-\s*([^\t]+)', sub_topic_entry)
            if sub_topic_name_match:
                sub_topic_name = sub_topic_name_match.group(1).strip()
                subtopic_to_headtopic_name_map[sub_topic_name] = head_topic_name
    return subtopic_to_headtopic_name_map

def map_subtopics_to_head_topics_weighted(subtopics, mapping):
    """
    Assigns a declining weight to subtopics based on their position.
    The 1st subtopic has more 'vote' than the 5th.
    """
    head_topic_scores = Counter()
    
    for index, subtopic in enumerate(subtopics):
        if subtopic in mapping:
            head_topic = mapping[subtopic]
            # Weight formula: 1 / (index + 1) -> 1st gets 1, 2nd gets 0.5, 3rd gets 0.33
            # OR use linear: weight = max(0.1, 1.0 - (index * 0.1))
            weight = 1.0 / (index + 1)
            head_topic_scores[head_topic] += weight
    
    if not head_topic_scores:
        return []

    # Get the topic with the highest score
    best_topic = max(head_topic_scores, key=head_topic_scores.get)
    return [best_topic]

def process_csv_for_subtopics(input_file, output_file, mapping_file):
    """Reads input CSV and writes results to a new CSV."""
    mapping = load_mapping(mapping_file)
    
    results = []
    with open(input_file, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['predicted_head_topic']
        
        for row in reader:
            try:
                # ast.literal_eval handles strings like "['sub1', 'sub2']"
                
                subtopics = ast.literal_eval(row['predicted_subtopics'])
                mapped = map_subtopics_to_head_topics_weighted(subtopics, mapping)
                row['predicted_head_topic'] = str(mapped)
            except Exception as e:
                row['predicted_head_topic'] = f"Error: {e}"
            results.append(row)
            
    with open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def main():
    parser = argparse.ArgumentParser(description="Score predicted topics against annotated topics.")
    parser.add_argument('-i', '--input', default='input/89_qwen_test.csv', help='Input CSV file with predicted (sub)topics (default: 89_qwen_test.csv)')
    parser.add_argument('-o', '--output', default='subtopics_results.csv', help='Output CSV file for results (default: results.csv)')
    parser.add_argument('-m', '--mapping', default='iptc_map.json', help='IPTC mapping JSON file (default: iptc_map.json)')
    parser.add_argument('-sub', '--subtopic', action='store_true', help='Activate if prediction file has subtopics (default: False)')
    args = parser.parse_args()
    
    if args.subtopic:
        process_csv_for_subtopics(args.input, args.output, args.mapping)
        print(f"Processing complete. Mapped subtopics saved to {args.output}")


if __name__ == "__main__":
    main()