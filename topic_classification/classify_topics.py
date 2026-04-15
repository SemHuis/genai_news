#!/usr/bin/env python

"""
Classify Dutch news articles into 17 topics using a Qwen model."""

import torch
import pandas as pd
import time
import os
import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse


def create_arg_parser():
    p = argparse.ArgumentParser(description="Classify Dutch news articles using Qwen (Supports JSON/CSV).")
    
    # Model and I/O
    p.add_argument("--base_model", default="Qwen/Qwen3.5-9B", help="Model name or path.")
    p.add_argument("--input", required=True, help="Path to input file (.json or .csv).")
    p.add_argument("--output", help="Path to save results (defaults to 'classified_inputname').")
    
    # Data Structure
    p.add_argument("--text_col", default="full_text", help="Column name containing the article text.")
    p.add_argument("--title_col", default="title", help="Column name containing the article title.")
    p.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    
    # Text Processing Logic
    p.add_argument("--mode", choices=["lead", "truncate"], default="lead", 
                   help="'lead' extracts first 5 sentences; 'truncate' takes first N characters.")
    p.add_argument("--max_chars", type=int, default=1500, help="Max characters if mode is 'truncate'.")

    return p.parse_args()


def get_first_5_sentences(text):
    """Extract the first 5 sentences from the full text, or the first few if there are less than 5."""
    if not text or not isinstance(text, str):
        return ""
    # Split by sentence endings (.!?) or by newlines (common in news leads)
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    # Filter out empty strings and take the first 5
    clean_sentences = [s.strip() for s in sentences if s.strip()]
    return " ".join(clean_sentences[:5])


def load_model_and_tokenizer(base_model):
    """Load the specified model and tokenizer with appropriate settings."""
    print(f"[Loading] Model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    return model, tokenizer


def parse_model_output(response, id_to_label):
    """Extract the predicted category ID from the model's response and map it to the label."""
    match = re.search(r'\d+', response)
    if match:
        idx = int(match.group())
        if idx in id_to_label:
            return id_to_label[idx]
    return None


def main():
    args = create_arg_parser()

    topic_descriptions = {
        'disaster, accident and emergency incident - man-made or natural events resulting in injuries, death or damage, e.g., explosions, transport accidents, famine, drowning, natural disasters, emergency planning and response.': 0,
        'human interest - news about life and behavior of royalty and celebrities, news about obtaining awards, ceremonies (graduation, wedding, funeral, celebration of launching something), birthdays and anniversaries, and news about silly or stupid human errors.': 1,
        'politics - news about local, regional, national and international exercise of power, including news about election, fundamental rights, government, non-governmental organisations, political crises, non-violent international relations, public employees, government policies.': 2,
        'education - all aspects of furthering knowledge, formally or informally, including news about schools, curricula, grading, remote learning, teachers and students.': 3,
        'crime, law and justice - news about committed crime and illegal activities, the system of courts, law and law enforcement (e.g., judges, lawyers, trials, punishments of offenders).': 4,
        'economy, business and finance - news about companies, products and services, any kind of industries, national economy, international trading, banks, (crypto)currency, business and trade societies, economic trends and indicators (inflation, employment statistics, GDP, mortgages, ...), international economic institutions, utilities (electricity, heating, waste management, water supply).': 5,
        'conflict, war and peace - news about terrorism, wars, wars victims, cyber warfare, civil unrest (demonstrations, riots, rebellions), peace talks and other peace activities.': 6,
        'arts, culture, entertainment and media - news about cinema, dance, fashion, hairstyle, jewellery, festivals, literature, music, theatre, TV shows, painting, photography, woodworking, art exhibitions, libraries and museums, language, cultural heritage, news media, radio and television, social media, influencers, and disinformation.': 7,
        'labour - news about employment, employment legislation, employees and employers, commuting, parental leave, volunteering, wages, social security, labour market, retirement, unemployment, unions.': 8,
        'weather - news about weather forecasts, weather phenomena and weather warning.': 9,
        'religion and belief - news about religions, cults, religious conflicts, relations between religion and government, churches, religious holidays and festivals, religious leaders and rituals, and religious texts.': 10,
        'society - news about social interactions (e.g., networking), demographic analyses, population census, discrimination, efforts for inclusion and equity, emigration and immigration, communities of people and minorities (LGBTQ, older people, children, indigenous people, etc.), homelessness, poverty, societal problems (addictions, bullying), ethical issues (suicide, euthanasia, sexual behavior) and social services and charity, relationships (dating, divorce, marriage), family (family planning, adoption, abortion, contraception, pregnancy, parenting).': 11,
        'health - news about diseases, injuries, mental health problems, health treatments, diets, vaccines, drugs, government health care, hospitals, medical staff, health insurance.': 12,
        'environment - news about climate change, energy saving, sustainability, pollution, population growth, natural resources, forests, mountains, bodies of water, ecosystem, animals, flowers and plants.': 13,
        'lifestyle and leisure - news about hobbies, clubs and societies, games, lottery, enthusiasm about food or drinks, car/motorcycle lovers, public holidays, leisure venues (amusement parks, cafes, bars, restaurants, etc.), exercise and fitness, outdoor recreational activities (e.g., fishing, hunting), travel and tourism, mental well-being, parties, maintaining and decorating house and garden.': 14,
        'science and technology -  news about natural sciences and social sciences, mathematics, technology and engineering, scientific institutions, scientific research, scientific publications and innovation.': 15,
        'sport - news about sports that can be executed in competitions - basketball, football, swimming, athletics, chess, dog racing, diving, golf, gymnastics, martial arts, climbing, etc.; sport achievements, sport events, sport organisation, sport venues (stadiums, gymnasiums, ...), referees, coaches, sport clubs, drug use in sport.': 16
    }
    
    id_to_label = {v: k.split(" - ")[0] for k, v in topic_descriptions.items()}
    desc_block = "\n".join([f"ID {v}: {k}" for k, v in topic_descriptions.items()])
    
    # Load data
    if os.path.exists(args.output_json):
        df = pd.read_json(args.output_json)
    else:
        df = pd.read_json(args.input_json)
        df['predicted_topic'] = None

    todo_indices = df[df['predicted_topic'].isna()].index.tolist()
    print(f"[Stats] Total: {len(df)} | Remaining: {len(todo_indices)}")

    if not todo_indices:
        print("Processing already complete.")
        return

    model, tokenizer = load_model_and_tokenizer(args.base_model)

    # Batch processing
    for i in range(0, len(todo_indices), args.batch_size):
        batch_indices = todo_indices[i : i + args.batch_size]
        prompts = []

        for idx in batch_indices:
            title = df.at[idx, 'title']
            full_text = df.at[idx, 'full_text']
            
            # Combine Title + First 5 sentences
            lead_text = get_first_5_sentences(full_text)
            input_content = f"TITLE: {title}\nLEAD: {lead_text}"
            
            messages = [
                {
                    "role": "system", 
                    "content": f"You are a Dutch news topic classifier. Use these detailed definitions to find the BEST fit:\n\n"
                               f"{desc_block}\n\n"
                               f"INSTRUCTION: Respond ONLY with the ID number of the single best category. NO THINKING."
                },
                {"role": "user", "content": f"Text: {input_content}"}
            ]
            
            p = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ) + "Best Category ID:"
            prompts.append(p)

        # Generate answers
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda")
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=5, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
            )
            responses = tokenizer.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            for j, resp in enumerate(responses):
                prediction = parse_model_output(resp, id_to_label)
                df.at[batch_indices[j], 'predicted_topic'] = prediction

        # Save progress every batch
        df.to_json(args.output_json, orient='records', indent=4)
        
        if (i // args.batch_size) % 5 == 0:
            print(f"Progress: {i + len(batch_indices)}/{len(todo_indices)} | Last prediction: {prediction}")

    print(f"[Success] Results saved to {args.output_json}")

if __name__ == "__main__":
    main()