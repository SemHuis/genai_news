import ast
import pandas as pd
import json
import re
import csv
import sys
import argparse
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False


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
                row['predicted_head_topic'] = str(''.join(mapped) if mapped else '')
            except Exception as e:
                row['predicted_head_topic'] = f"Error: {e}"
            results.append(row)
            
    with open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def check_at_least_one_correct(true_labels, predicted_labels):
    """Checks if at least one predicted label matches any of the true labels."""
    # Convert to sets for intersection checking
    true_set = set(true_labels)
    predicted_set = set(predicted_labels)
    return len(true_set.intersection(predicted_set)) > 0


def calculate_one_top_accuracy(predicted_file):
    """Compares predicted head topics to annotated head topics and calculates accuracy."""
    df = pd.read_csv(predicted_file)
    
    # Convert 'iptc_filtered' from string representation of list to actual list
    df['true_head_topics'] = df['iptc_filtered'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    # Create 'predicted_head_topics_list' for this cell's calculation
    df['predicted_head_topics_list'] = df['predicted_head_topic'].apply(lambda x: [x] if pd.notna(x) else [])

    # Apply the function to the DataFrame to get a boolean series
    df['at_least_one_correct'] = df.apply(
        lambda row: check_at_least_one_correct(row['true_head_topics'], row['predicted_head_topics_list']),
        axis=1
    )

    # Calculate the new accuracy
    new_accuracy = df['at_least_one_correct'].sum() / len(df)

    print(f"'At least one correct' Head Topic Accuracy: {new_accuracy}")


def calculate_f1_scores(predicted_file):
    """Calculates F1 scores for multi-label classification."""
    df = pd.read_csv(predicted_file)
    
    # Convert 'iptc_filtered' from string representation of list to actual list
    df['true_head_topics'] = df['iptc_filtered'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    # Create 'predicted_head_topics_list' for this cell's calculation
    df['predicted_head_topics_list'] = df['predicted_head_topic'].apply(lambda x: [x] if pd.notna(x) else [])

    mlb = MultiLabelBinarizer()
    y_true = mlb.fit_transform(df['true_head_topics'])
    y_pred = mlb.transform(df['predicted_head_topics_list'])

    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    print(f"Micro-averaged F1 score: {f1_micro}")
    print(f"Macro-averaged F1 score: {f1_macro}")


def confusion_matrix_all_topics(predicted_file, normalize=False, none_label="<NO_PRED>"):
    """Compute and return a confusion matrix for all head topics.
    Returns a pandas DataFrame with true topics as index and predicted topics as columns.
    """
    df = pd.read_csv(predicted_file)
    df['true_head_topics'] = df['iptc_filtered'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    # predicted is expected to be a single string label per row
    df['predicted_head_topic'] = df['predicted_head_topic'].fillna('').astype(str)

    y_true_expanded = []
    y_pred_expanded = []

    for _, row in df.iterrows():
        preds = row['predicted_head_topic'].strip()
        pred_label = preds if preds else none_label
        true_labels = row['true_head_topics'] if isinstance(row['true_head_topics'], (list, tuple)) else []
        if not true_labels:
            # If no true labels, count under a special missing-true label
            y_true_expanded.append('<NO_TRUE_LABEL>')
            y_pred_expanded.append(pred_label)
        else:
            for t in true_labels:
                y_true_expanded.append(t)
                y_pred_expanded.append(pred_label)

    # Build label set in sorted but stable order
    labels = sorted(list(set(y_true_expanded) | set(y_pred_expanded)))
    cm = confusion_matrix(y_true_expanded, y_pred_expanded, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    if normalize:
        # Normalize by true-label row sums to get per-class recall-like proportions
        with pd.option_context('mode.use_inf_as_na', True):
            row_sums = cm_df.sum(axis=1).replace(0, 1)
            cm_df = cm_df.div(row_sums, axis=0)

    return cm_df


def plot_confusion_matrix(cm_df, out_path=None, figsize=(12, 10), cmap='Blues', annot=True):
    """Plot and save a confusion matrix DataFrame as a heatmap PNG.
    If seaborn is available it will be used for a nicer plot; otherwise matplotlib.imshow is used.
    """
    plt.figure(figsize=figsize)
    if _HAS_SEABORN:
        sns.set(font_scale=0.8)
        ax = sns.heatmap(cm_df, annot=annot, fmt='.2f' if cm_df.dtypes.any() else 'd', cmap=cmap, linewidths=0.5)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    else:
        data = cm_df.values
        im = plt.imshow(data, interpolation='nearest', cmap=cmap)
        plt.colorbar(im)
        ticks = range(len(cm_df.columns))
        plt.xticks(ticks, cm_df.columns, rotation=90)
        plt.yticks(ticks, cm_df.index)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        if annot:
            # annotate with numbers
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    plt.text(j, i, f"{data[i, j]:.2f}" if data.dtype.kind == 'f' else f"{int(data[i,j])}",
                             ha='center', va='center', color='black', fontsize=6)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
        print(f"Saved confusion matrix figure to {out_path}")
    else:
        plt.show()
    plt.close()


def score_predictions(predicted_file):
    """Scores predictions using both accuracy and F1 metrics."""
    calculate_one_top_accuracy(predicted_file)
    calculate_f1_scores(predicted_file)


def main():
    parser = argparse.ArgumentParser(description="Score predicted topics against annotated topics.")
    parser.add_argument('-i', '--input', default='input/89_qwen_test.csv', help='Input CSV file with predicted (sub)topics (default: 89_qwen_test.csv)')
    parser.add_argument('-o', '--output', default='subtopics_results.csv', help='Output CSV file for results (default: results.csv)')
    parser.add_argument('-m', '--mapping', default='iptc_map.json', help='IPTC mapping JSON file (default: iptc_map.json)')
    parser.add_argument('-sub', '--subtopic', action='store_true', help='Activate if prediction file has subtopics (default: False)')
    parser.add_argument('--confusion', action='store_true', help='Print confusion matrix for all topics')
    parser.add_argument('--normalize-confusion', action='store_true', help='Normalize confusion matrix rows to proportions')
    parser.add_argument('--plot', action='store_true', help='Also generate a PNG heatmap of the confusion matrix')
    parser.add_argument('--plot-out', default='confusion_matrix.png', help='Output path for the confusion matrix PNG (default: confusion_matrix.png)')
    args = parser.parse_args()
    
    if args.subtopic:
        process_csv_for_subtopics(args.input, args.output, args.mapping)
        print(f"Processing complete. Mapped subtopics saved to {args.output}")
        score_predictions(args.output)
        if args.confusion:
            cm_df = confusion_matrix_all_topics(args.input, normalize=args.normalize_confusion)
            plot_confusion_matrix(cm_df, out_path=args.plot_out)
    else:
        score_predictions(args.input)
        if args.confusion:
            cm_df = confusion_matrix_all_topics(args.input, normalize=args.normalize_confusion)
            plot_confusion_matrix(cm_df, out_path=args.plot_out)

    
if __name__ == "__main__":
    main()