import os
import shutil
import trafilatura
from bs4 import UnicodeDammit

file_id = 0


def extract_data(dataset_path, output_folder):
    global file_id
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            with open(dataset_path + "/" + file, "rb") as data:
                file_id += 1
                output_file_path = output_folder + "/" + str(file_id) + ".txt"
                raw_data = UnicodeDammit(data.read()).unicode_markup
                parsed_text = trafilatura.extract(raw_data)
                if parsed_text is None:
                    with open(output_file_path, 'w') as output_file:
                        output_file.write("EMPTY")
                    continue
                with open(output_file_path, 'w') as output_file:
                    output_file.write(parsed_text)


def preprocess_dataset(dataset_path, output_folder):
    i = 0
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through the dataset
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith("html.txt"):
                i += 1
                shutil.copy(os.path.join(root, file), output_folder + "/file" + str(i) + ".html")



dataset_path = "benign_25k"
output_folder = "Legitimate"
preprocess_dataset(dataset_path, output_folder)

dataset_path = "phish_sample_30k"
output_folder = "Phishing"

preprocess_dataset(dataset_path, output_folder)


dataset_path = "misleading"
output_folder = "Misleading"

preprocess_dataset(dataset_path, output_folder)

dataset_path = "Legitimate"
output_folder = "ExtractedLegitimate"
extract_data(dataset_path, output_folder)

dataset_path = "Misleading"
output_folder = "ExtractedLegitimate"
extract_data(dataset_path, output_folder)

file_id = 0
dataset_path = "Phishing"
output_folder = "ExtractedPhishing"
extract_data(dataset_path, output_folder)
