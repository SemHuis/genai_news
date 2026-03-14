from docx import Document
import os
import sys

def get_docx_text(path):
    """Extracts text from paragraphs AND tables to ensure nothing is missed."""
    doc = Document(path)
    content = []
    # Extract from paragraphs
    for para in doc.paragraphs:
        content.append(para.text)
    # Extract from tables
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                # Add newline to keep structure
                content.append(cell.text + "\n")
    return "\n".join(content)

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