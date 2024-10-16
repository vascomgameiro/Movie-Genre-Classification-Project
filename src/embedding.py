import logging
import time

import pandas as pd
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from src.dataset import read_data

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Create formatter and add it to the handler
formatter = logging.Formatter("%(asctime)s - %(message)s")
handler.setFormatter(formatter)

# Add the handler to the logger
if not logger.hasHandlers():
    logger.addHandler(handler)

# Initialize the SentenceTransformer model and text splitter
model = SentenceTransformer("avsolatorio/GIST-small-Embedding-v0")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)


# Function to compute weighted average embeddings
def compute_weighted_avg_embedding(text, model):
    chunks = text_splitter.split_text(text)
    embeddings, lengths = [], []

    for chunk in chunks:
        chunk_embedding = model.encode(chunk, convert_to_tensor=True)
        embeddings.append(chunk_embedding)
        lengths.append(len(chunk))

    lengths = torch.tensor(lengths, dtype=torch.float32)
    weighted_sum = sum([embeddings[i] * lengths[i] for i in range(len(embeddings))])
    weighted_avg_embedding = weighted_sum / lengths.sum()

    return weighted_avg_embedding.cpu().numpy()


def process_and_save_in_chunks(df, text_column, model, chunk_size=300, output_path="data/processed/processed_embeddings.csv"):
    last_processed_idx = 0
    total_chunks = (len(df) + chunk_size - 1) // chunk_size  # Calculate total number of chunks

    try:
        for idx, chunk_id in enumerate(range(0, len(df), chunk_size)):
            start_time = time.time()  # Start time for processing the current chunk

            # Process the current chunk
            chunk = df.iloc[chunk_id : min(chunk_id + chunk_size, len(df))].copy()
            embeddings = chunk[text_column].apply(lambda x: compute_weighted_avg_embedding(x, model)).tolist()
            embeddings_df = pd.DataFrame(embeddings, columns=[f"embedding_{i+1}" for i in range(len(embeddings[0]))])

            write_mode = "w" if idx == 0 else "a"
            header = idx == 0
            embeddings_df.to_csv(output_path, mode=write_mode, header=header, index=False)

            last_processed_idx = chunk_id + len(chunk)
            duration = time.time() - start_time  # Duration for processing the chunk

            # Log the time taken and the progress
            logging.info(f"Processed chunk {idx + 1}/{total_chunks} in {duration:.2f} seconds.")
            logging.info(f"Progress: {((idx + 1) / total_chunks) * 100:.2f}%")

    except KeyboardInterrupt:
        logging.warning(f"Process interrupted. Last processed row index: {last_processed_idx}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

    finally:
        return last_processed_idx

def main():
    # Specify input and output paths
    input_path = "data/raw/train.txt"
    output_path = "data/processed/processed_embeddings.csv"
    columns = ["title", "from", "genre", "director", "description"]

    # Read the data
    df = read_data(input_path, columns)

    # Process the data in chunks of 300 rows and save it
    last_processed_row = process_and_save_in_chunks(df, "description", model, chunk_size=300, output_path=output_path)

    # Print the last processed row index
    print(f"Last processed row index: {last_processed_row}")


# Ensure that the script runs only when executed directly (not when imported as a module)
if __name__ == "__main__":
    main()
