import torch
import pandas as pd
import time
import os
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# ====== CONFIG ======
BASE_MODEL = "Qwen/Qwen3.5-9B" 
INPUT_CSV = "train.csv"
OUTPUT_CSV = "qwen_train_few_desc.csv"
BATCH_SIZE = 8 

# Topic definitions to help the model understand the context of each ID
TOPIC_DESCRIPTIONS = {
    "arts, culture, entertainment and media": "Events and issues related to artistic expression, celebrities, movies, and the arts.",
    "crime, law and justice": "Illegal acts, police investigations, court cases, and the legal system.",
    "disaster, accident and emergency incident": "Natural disasters, transport accidents, and emergency services responses.",
    "economy, business and finance": "Market news, company performance, banking, and macroeconomics.",
    "education": "Schools, universities, teaching, and learning policies.",
    "environment": "Climate change, nature conservation, pollution, and ecology.",
    "health": "Medical research, diseases, public health, and wellness.",
    "human interest": "Emotional stories about people, animals, or quirky events.",
    "labour": "Employment, trade unions, strikes, and workplace issues.",
    "lifestyle and leisure": "Travel, hobbies, fashion, and personal habits.",
    "politics": "Government, elections, legislation, and political parties.",
    "religion and belief": "Faith, religious organizations, and spiritual practices.",
    "science and technology": "Inventions, scientific discoveries, and technical developments.",
    "society": "Social issues, demographics, welfare, and community groups.",
    "sport": "Competitive physical activities, athletes, and tournaments.",
    "conflict, war and peace": "Armed conflict, military operations, and peace negotiations.",
    "weather": "Meteorological conditions, forecasts, and extreme weather patterns."
}

TOPICS = list(TOPIC_DESCRIPTIONS.keys())
TOPIC_MAPPING = {i: topic for i, topic in enumerate(TOPICS)}

# Create a numbered list with descriptions for the prompt
TOPIC_LIST_STR = "\n".join([f"{i}: {topic} ({TOPIC_DESCRIPTIONS[topic]})" for i, topic in TOPIC_MAPPING.items()])

def load_model_and_tokenizer():
    print(f"[Loading] Model from {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    return model, tokenizer

def parse_model_output(response):
    match = re.search(r'\d+', response)
    if match:
        idx = int(match.group())
        if idx in TOPIC_MAPPING:
            return TOPIC_MAPPING[idx]
    
    clean_resp = response.lower()
    for topic in TOPICS:
        first_word = topic.split(',')[0].split(' ')[0] 
        if topic in clean_resp or first_word in clean_resp:
            return topic
    return None

def main():
    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
    else:
        df = pd.read_csv(INPUT_CSV)
        df['predicted_head_topic'] = None

    todo_indices = df[df['predicted_head_topic'].isna()].index.tolist()
    print(f"[Stats] Remaining: {len(todo_indices)} / {len(df)}")

    if not todo_indices:
        print("Everything is already processed.")
        return

    model, tokenizer = load_model_and_tokenizer()
    start_time = time.time()

    for i in range(0, len(todo_indices), BATCH_SIZE):
        batch_indices = todo_indices[i : i + BATCH_SIZE]
        batch_texts = df.loc[batch_indices, 'text'].fillna("No content").tolist()
        
        prompts = []
        for t in batch_texts:
            messages = [
                {
                    "role": "system", 
                    "content": (
                        f"You are an expert news classifier. Input language: Dutch.\n"
                        f"Categorize the text into exactly ONE topic by providing its ID number.\n\n"
                        f"TOPIC LIST:\n{TOPIC_LIST_STR}\n\n"
                        f"Response format: Return ONLY the number. NO THINKING."
                    )
                },
                # Few-Shot 1: Crime (ID 1)
                {"role": "user", "content": "Text to classify: Spaanse politie onderschept recordvangst cocaïne verstopt in lading bananen : Grootste ooit in Europa De Spaanse politie en beambten van de douane vonden onlangs maar liefst negen ton cocaïne in een lading bananen die uit Colombia werd geïmporteerd ."},
                {"role": "assistant", "content": "1"},
                
                # Few-Shot 2: Politics (ID 10)
                {"role": "user", "content": "Text to classify: Seehofer verliest van Merkel CSU-voorzitter Horst Seehofer was gisteravond van plan ontslag te nemen als partijvoorzitter en minister . Partijgenoten probeerden hem om te praten ."},
                {"role": "assistant", "content": "10"},
                
                # Few-Shot 3: Conflict (ID 15)
                {"role": "user", "content": "Text to classify: Dodelijke aanval op havenstad in Jemen hodeida De coalitie onder leiding van Saoedi-Arabië heeft gisteren een offensief geopend op de havenstad Hodeida in Jemen . Het is de grootste operatie in de nu al drie jaar aanslepende oorlog ."},
                {"role": "assistant", "content": "15"},

                # Target
                {"role": "user", "content": f"Text to classify: {t[:1200]}"}
            ]
            
            p = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False
            ) + "Result ID:" 
            prompts.append(p)
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2500).to("cuda")
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=5, 
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id, 
                eos_token_id=tokenizer.eos_token_id
            )
            
            responses = tokenizer.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            for idx, resp in enumerate(responses):
                prediction = parse_model_output(resp)
                df.at[batch_indices[idx], 'predicted_head_topic'] = prediction

        df.to_csv(OUTPUT_CSV, index=False)
        
        if (i // BATCH_SIZE) % 5 == 0:
            elapsed = time.time() - start_time
            print(f" Done {i+len(batch_indices)}/{len(todo_indices)} | Elapsed: {elapsed:.1f}s | Latest: {resp.strip()} -> {prediction}")

    print(f"[Success] Completed! Results in {OUTPUT_CSV}")

if __name__ == "__main__":
    main()