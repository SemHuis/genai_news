import os
import ast
import json
import time
import google.generativeai as genai
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer

# --- 1. SETUP ---
# Replace with your actual key from https://aistudio.google.com/
os.environ["GOOGLE_API_KEY"] = "AIzaSyCIJJfgfQ74jgqdLEe5qEq54wW0mmM-7ns"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel('gemini-3-flash-preview')

IPTC_TOPICS = [
    "arts, culture and entertainment", "crime, law and justice", "disaster and accident",
    "economy, business and finance", "education", "environment", "health",
    "human interest", "labour", "lifestyle and leisure", "politics",
    "religion and belief", "science and technology", "society", "sport",
    "conflict, war and peace", "weather"
]

SYSTEM_PROMPT = f"""
You are an expert news classifier. Use ONLY these IPTC categories: {IPTC_TOPICS}.
Return ONLY a valid Python list of strings. No explanations or extra text.
Include all topics that apply to the article.
"""

# --- 2. CORE FUNCTIONS ---

def classify_article(article):
    """Formats the JSON object and prompts Gemini."""
    title = article.get("title", "Geen titel")
    text = article.get("full_text", "")
    
    # Feeding both Title and Text significantly improves accuracy
    prompt_input = f"TITEL: {title}\n\nTEKST: {text}"
    
    try:
        response = model.generate_content(f"{SYSTEM_PROMPT}\n\n{prompt_input}")
        raw_output = response.text.strip().replace("```python", "").replace("```", "")
        prediction = ast.literal_eval(raw_output)
        
        # Ensurehallucinated labels are ignored
        return [p for p in prediction if p in IPTC_TOPICS]
    except Exception as e:
        print(f"Error classifying '{title[:30]}...': {e}")
        return []

def run_classification(input_json_path, output_json_path):
    # Load the JSON file
    print(f"Loading data from {input_json_path}...")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    y_true = []
    y_pred = []
    results = []

    print(f"Processing {len(articles)} articles. This will take roughly {len(articles) // 15} minutes.")

    for i, article in enumerate(articles):
        # 1. Get LLM Prediction
        prediction = classify_article(article)
        article["llm_prediction"] = prediction
        
        # 2. Get Ground Truth (if available in your data)
        # Assuming ground truth might be in 'iptc_filtered' as a list or string
        true_labels = article.get("iptc_filtered", [])
        if isinstance(true_labels, str):
            true_labels = ast.literal_eval(true_labels)
        
        y_true.append(true_labels)
        y_pred.append(prediction)
        results.append(article)

        print(f"[{i+1}/{len(articles)}] {article.get('title', '')[:50]}...")
        print(f"   -> {prediction}")

        # 3. Intermediate Save (Checkpoint) every 50 articles
        if (i + 1) % 50 == 0:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4)
            print(f"--- Checkpoint saved at article {i+1} ---")

        # Respect Rate Limits (Gemini Flash free tier is ~15 RPM)
        time.sleep(1.5)

    # 4. Final Save
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    # 5. Evaluation (only runs if you have ground truth labels)
    if any(y_true):
        run_evaluation(y_true, y_pred)

def run_evaluation(y_true, y_pred):
    print("\n" + "="*40)
    print("FINAL EVALUATION METRICS")
    print("="*40)
    
    mlb = MultiLabelBinarizer(classes=IPTC_TOPICS)
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)
    
    print(f"Exact Match Accuracy: {accuracy_score(y_true_bin, y_pred_bin):.4f}")
    print(f"Macro F1 Score: {f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0):.4f}")
    print("\nPer-Class Breakdown:")
    print(classification_report(y_true_bin, y_pred_bin, target_names=mlb.classes_, zero_division=0))

# --- 3. EXECUTION ---

if __name__ == "__main__":
    # Change these paths to your actual filenames
    INPUT_FILE = "nrc_small.json"
    OUTPUT_FILE = "nrc_results.json"
    
    if os.path.exists(INPUT_FILE):
        run_classification(INPUT_FILE, OUTPUT_FILE)
        print(f"\nProcessing complete. Results saved to {OUTPUT_FILE}")
    else:
        print(f"Error: {INPUT_FILE} not found.")