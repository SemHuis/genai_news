import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import argparse


def get_articles_and_topics(root_directory):
    extracted_data = []
    for subdir, dirs, files in os.walk(root_directory):
        if 'dnaf.json' in files:
            file_path = os.path.join(subdir, 'dnaf.json')
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Extract the full text
                    token_string = data.get("doc", {}).get("token_string", "")
                    
                    # Navigate to the IPTC codes
                    iptc_groups = data.get("doc", {}).get("annotations", {}).get("iptc_codes", {}).get("certain", [])
                    filtered_codes = []
                    for group in iptc_groups:
                        for entry in group:
                            # code (e.g., ""medtop:11000000"")
                            code = entry[0]
                            # label (e.g., ""politics"")
                            label = entry[1]
                            
                            # Filter for for head-level topics (medtop:0 and medtop:1)
                            if code.startswith("medtop:0") or code.startswith("medtop:1"):
                                filtered_codes.append(label)
                    
                    # Add to data if there are filtered codes
                    if filtered_codes:
                        extracted_data.append({
                            "id": os.path.basename(subdir),
                            "iptc_filtered": (list(set(filtered_codes))),
                            "text": token_string
                        })
                    
            except (json.JSONDecodeError, FileNotFoundError):
                continue
    return extracted_data
    

def main():
    parser = argparse.ArgumentParser(description="Get articles and topics from JSON files and save to CSV.")
    # Make root_directory a required argument
    parser.add_argument('-r', '--root_directory', required=True, help='Root directory containing subfolders with dnaf.json files')
    parser.add_argument('-t', '--train_file', default='train.csv', help='Output CSV file for training data (default: train.csv)')
    parser.add_argument('-d', '--dev_file', default='dev.csv', help='Output CSV file for development data (default: dev.csv)')
    parser.add_argument('-s', '--split', type=float, default=0.4, help='Proportion of data to use for development set (default: 0.4)')
    args = parser.parse_args()

    extracted_data = get_articles_and_topics(args.root_directory)

    # Create DataFrame
    df = pd.DataFrame(extracted_data)
    n = len(df)
    print(f"Processed {n} folders with matching codes.")
    if n == 0:
        print("No data to split.")
    else:
        # Stratified split by iptc_filtered topics (first topic in list)
        train_df, dev_df = train_test_split(
            df,
            test_size=args.split,
            random_state=42,
            stratify=df['iptc_filtered'].apply(lambda x: x[0] if x else 'unknown')  
        )

        # Reset indices
        train_df = train_df.reset_index(drop=True)
        dev_df = dev_df.reset_index(drop=True)

        # Save to csv
        train_df.to_csv(args.train_file, sep=',', index=False, encoding='utf-8')
        dev_df.to_csv(args.dev_file, sep=',', index=False, encoding='utf-8')
        print(f"Saved {args.train_file} ({len(train_df)} articles) and {args.dev_file} ({len(dev_df)} articles) ")
        

if __name__ == "__main__":
    main()
