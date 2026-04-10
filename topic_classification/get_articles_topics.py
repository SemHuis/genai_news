import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

# add root directory as argument
if len(sys.argv) < 2:
    print("Usage: python get_text_topic.py <root_directory>")
    sys.exit(1)

root_directory = sys.argv[1]
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
                        # code (e.g., "medtop:11000000")
                        code = entry[0]
                        # label (e.g., "politics")
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
# Create DataFrame
df = pd.DataFrame(extracted_data)
n = len(df)
print(f"Processed {n} folders with matching codes.")
if n == 0:
    print("No data to split.")
else:
    # stratified split by iptc_filtered topics
    train_df, dev_df = train_test_split(
        df,
        test_size=0.4,
        random_state=42,
        stratify=df['iptc_filtered'].apply(lambda x: x[0] if x else 'unknown')  # use first topic as stratum
    )

    # reset indices
    train_df = train_df.reset_index(drop=True)
    dev_df = dev_df.reset_index(drop=True)

    print(train_df.head())
    print(f"Train: {len(train_df)} examples, Dev: {len(dev_df)} examples")

    # save to csv
    train_path = 'train.csv'
    dev_path = 'dev.csv'
    train_df.to_csv(train_path, sep=',', index=False, encoding='utf-8')
    dev_df.to_csv(dev_path, sep='\t', index=False, encoding='utf-8')
    print(f"Saved {train_path} and {dev_path}")
