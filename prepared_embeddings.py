import os
import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("aditeyabaral/sentencetransformer-xlm-roberta-base")


def get_embedding(main_body):
    sentences = [main_body]
    embeddings = model.encode(sentences)
    return embeddings


def process_html_files(input_folder):
    embeddings_list = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            with open(os.path.join(root, file), 'r', encoding='utf-8') as text_file:
                main_body_text = text_file.read()
            embeddings = get_embedding(main_body_text)
            embeddings_list.append(embeddings)
    return embeddings_list


input_folder_phishing = "ExtractedPhishing"
input_folder_legitimate = "ExtractedLegitimate"
output_folder = "Embedding"
benign_embedding = process_html_files(input_folder_legitimate)
phishing_embedding = process_html_files(input_folder_phishing)

output_file_path = os.path.join(output_folder, "embeddings/embeddings.pkl")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

with open(output_file_path, 'wb') as output_file:
    pickle.dump({"benign": benign_embedding, "phishing": phishing_embedding}, output_file,
                protocol=pickle.HIGHEST_PROTOCOL)
