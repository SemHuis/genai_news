# Signs of Generative AI in Dutch Newspaper Articles

This repository contains the code used for my thesis project.
The goal of this thesis was to explore whether generative AI has changed Dutch newspaper reporting. 
Our method contains of two main pillars: AI detection and stylometry.

## Data

This section details the origin of our dataset and how it was processed before analysis.

### Data Extraction
The newspaper articles used in this thesis are available on [Nexis Uni](http://www.nexisuni.com/). We collected articles from six newspapers between December 2019 and December 2025.
We take a random sample of articles for each newspaper per month using `generate_sample_n.py`. The articles are now in a docx format. We convert them using `docx_to_txt.py`.
Next, we extract all fields (title, author, body_text, etc.) from the plain text and clean the fields `extract_and_clean.py`. Finally, we filter out short and other unusable articles `filter_article.py`

### Topic Classification and Balancing

Subsequently, we want to balance are data on topic. We perform a topic classification experiment to find an LLM and prompt combination for this task `eventdna_experiment`. 
The best combination, Qwen-3.5-9B with a fewshot prompt, is used to classify all filtered articles `classify_topics.py`. After balancing the articles on topic `balance_articles.py`, we remove topics with less than 15 articles per month. The amount of articles at each stage of this data pipeline is shown in the table below. Our final corpus consists of 66,295 newspaper articles. 

| Newspaper | Collected | Filtered | Balanced |
| :--- | ---: | ---: | ---: |
| De Volkskrant | 21,900 | 18,969 | 14,829 |
| De Telegraaf | 21,748 | 19,582 | 14,377 |
| Dagblad van het Noorden  | 18,244 | 11,126 | 6,076 |
| Eindhovens Dagblad | 18,250 | 16,041 | 12,085 |
| Steenwijker Courant | 15,165 | 12,565 | 8,166 |
| Nieuwsblad Noordoost-Friesland | 16,807 | 14,215 | 10,762 |
| **Total** | **112,114** | **102,208** | **66,295** |
---

## Method

Our analytical approach is built upon two pillars: detecting AI-generated articles using Pangram, and analyzing the style of the articles.

### 1. AI Detection
We analyze the corpus with Pangram v3, an AI detector with low false-positive and false-negative rates `run_pangram.py`. 
Furthermore, we analyze differences in AI use across topics, sections, and journalists `analyze_ai_use.py`.


### 2. Stylometric Analysis
Besides AI-detection, we analyze changes in writing style using linguistic features.
First, we parse all articles in our corpus using Stanza `stanza_parse.py`. 
Using sets of Dutch AI-excess words, we examine if these words are more prevalent after ChatGPT than before in our corpus `excess_words_diachronic.py`.
Furthermore, we measure syntactic features, lexical features, and stylistic homogenization `article_metrics.py`. All measures are shown in the table below.
We test changes in the metrics using interrupted time series (ITS) regression `analysis.R`.

| Category | Metric |
| :--- | :--- |
| **Syntactic Complexity** | Mean sentence length (words) |
| | Sentence-length variability (CV) |
| | Mean dependency-tree depth |
| | Finite verbs per sentence |
| | Mean T-unit length |
| | Nominalization rate |
| **Lexical Diversity** | MTLD |
| **Lexical Sophistication** | Low-frequency words (%) |
| | High-frequency words (%) |
| | Mean word frequency (Zipf) |
| **Function Words** | Pronouns (per 1k words) |
| | Auxiliary verbs (per 1k words) |
| | Determiners (per 1k words) |
| | Coordinating conjunctions (per 1k words) |
| | Adverbs (per 1k words) |
| | Adpositions (per 1k words) |
| **LLM-Excess Words** | Content Words |
| | Function Words |
| **Stylistic Homogenization** | Standard Deviation |
