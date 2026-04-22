from docx import Document
import os
import sys
import re

def normalize_text(text):
    """Normalize unusual line separators and multiple newlines.
    """
    if not text:
        return ""
    
    # Replace Unicode line/paragraph separators with standard newlines
    text = re.sub(r'[\u2028\u2029\u0085]', '\n', text)
    # Replace carriage returns + newline with just newline
    text = re.sub(r'\r\n', '\n', text)
    # Replace lone carriage returns with newlines
    text = re.sub(r'\r', '\n', text)
    # Replace 3+ consecutive newlines with exactly 2 (preserve paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip trailing/leading whitespace from each line but preserve structure
    lines = text.split('\n')
    lines = [line.rstrip() for line in lines]
    text = '\n'.join(lines)
    
    return text.strip()

def get_docx_text(path):
    """Extracts text from paragraphs AND tables to ensure nothing is missed."""
    doc = Document(path)
    content = []
    # Extract from paragraphs
    for para in doc.paragraphs:
        normalized = normalize_text(para.text)
        if normalized:  # only add non-empty paragraphs
            content.append(normalized)
    # Extract from tables
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                normalized = normalize_text(cell.text)
                if normalized:
                    content.append(normalized)
    # Join paragraphs with double newlines for readability
    return "\n\n".join(content)

def main():
    # Get articles from multiple docx files in a directory
    # Save all articles in a list
    if len(sys.argv) < 3:
        print("Usage: python docx_to_txt.py <directory> <output_file>")
        sys.exit(1)
    directory = sys.argv[1]
    all_articles = []
    for filename in os.listdir(directory):
        if filename.endswith(".DOCX"):
            file_path = os.path.join(directory, filename)
            text = get_docx_text(file_path)
            all_articles.append(text)
    
    # Save all articles in a txt file
    # Use second command-line argument as output filename if provided
    out_file = sys.argv[2] if len(sys.argv) > 2 else "all_articles.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        for article in all_articles:
            # Separate articles by two newlines
            f.write(article + "\n\n")
            
if __name__ == "__main__":    
    main()