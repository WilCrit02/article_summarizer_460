# Importing necessary libraries
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import urllib.request
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize


def tokenize(text):
    return set(word_tokenize(text.lower()))

def calculate_precision_recall_fscore(reference_tokens, generated_tokens):
    """ Calculate precision, recall, and F1 score """
    common_tokens = reference_tokens.intersection(generated_tokens)
    
    precision = len(common_tokens) / len(generated_tokens) if generated_tokens else 0
    recall = len(common_tokens) / len(reference_tokens) if reference_tokens else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return precision, recall, f1_score

# Function to chunk text into smaller parts for processing
def custom_chunking(text, max_chunk_length):
    chunks = []  # List to store chunks
    current_chunk = ""  # Temporary string for the current chunk

    paragraphs = text.split("\n")  # Split the text into paragraphs

    # Loop through each paragraph
    for paragraph in paragraphs:
        # If adding the paragraph doesn't exceed max chunk length, add it
        if len(current_chunk + paragraph) <= max_chunk_length:
            current_chunk += paragraph + "\n"
        else:
            # If the current chunk is not empty, add it to the chunks list
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n"

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Fetching data from a URL
fetched_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/Machine_learning')
article_read = fetched_data.read()
article_parsed = BeautifulSoup(article_read, 'html.parser')
paragraphs = article_parsed.find_all('p')

given_summary = tokenize("Machine learning is a branch of artificial intelligence (AI) that focuses on the development of algorithms and statistical models that enable computers to perform tasks without explicit instructions. It involves the use of data and algorithms to imitate the way humans learn, gradually improving accuracy. The core idea of machine learning is to train predictive models on data, allowing systems to make decisions or predictions based on new data. These models are trained using a large set of data known as training data, which helps the system learn how to perform a specific task. There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning involves training models on labeled data, where the desired output is known. Unsupervised learning, on the other hand, deals with unlabeled data, allowing the model to identify patterns and relationships on its own. Reinforcement learning involves training models to make sequences of decisions by rewarding or penalizing them based on the actions they take. Machine learning has a wide range of applications, from email filtering and computer vision to predictive analytics in various fields like finance, healthcare, and marketing. Deep learning, a subset of machine learning, is particularly notable for its role in the advancement of areas such as natural language processing, image recognition, and autonomous vehicles. Challenges in machine learning include the risk of bias in training data, ethical concerns, and the need for large amounts of data to train effective models. Additionally, the interpretability of machine learning models can be limited, making it difficult to understand how certain decisions or predictions are made. As technology evolves, machine learning continues to play a critical role in advancing AI and offering innovative solutions across numerous industries. It represents a significant stride in creating systems that can learn and adapt from experience, much like human beings.")



# Joining the text of each paragraph into one long string
long_text = "\n".join([paragraph.text.strip() for paragraph in paragraphs])

# Initializing the BART tokenizer and model for summarization
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Set up the device (GPU or CPU) for running the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the chunk size for text processing
chunk_size = 1000
# Breaking the long text into manageable chunks
text_chunks = custom_chunking(long_text, chunk_size)

summaries = []  # List to store summaries of each chunk

# Processing each chunk of text
for chunk in text_chunks:
    # Tokenizing the chunk for the model
    tokenized_chunk = tokenizer.encode(chunk, return_tensors='pt', max_length=1024, truncation=True)
    tokenized_chunk = tokenized_chunk.to(device)

    # Generating a summary for the chunk
    summary_ids = model.generate(tokenized_chunk, max_length=500, min_length=5, length_penalty=20, num_beams=10, early_stopping=True)
    # Decoding the summary to text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summaries.append(summary)

# Joining all the summaries into a final text
final_summary = " ".join(summaries)
print(final_summary)
precision, recall, f1_score = calculate_precision_recall_fscore(given_summary, tokenize(final_summary))
print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")