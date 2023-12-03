from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import urllib.request
from bs4 import BeautifulSoup

def custom_chunking(text, max_chunk_length):
    chunks = []
    current_chunk = ""

    paragraphs = text.split("\n")  # Split the text into paragraphs

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= max_chunk_length:
            current_chunk += paragraph + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

fetched_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/Goku')
article_read = fetched_data.read()
article_parsed = BeautifulSoup(article_read, 'html.parser')
paragraphs = article_parsed.find_all('p')

long_text = "\n".join([paragraph.text.strip() for paragraph in paragraphs])

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

chunk_size = 1000
text_chunks = custom_chunking(long_text, chunk_size)

summaries = []
for chunk in text_chunks:
    tokenized_chunk = tokenizer.encode(chunk, return_tensors='pt', max_length=1024, truncation=True)
    tokenized_chunk = tokenized_chunk.to(device)
    summary_ids = model.generate(tokenized_chunk, max_length=200, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summaries.append(summary)

final_summary = " ".join(summaries)
print(final_summary)