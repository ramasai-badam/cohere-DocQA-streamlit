import os
import pandas as pd
from typing import Any, List
from io import StringIO,BytesIO
import random
import string
import numpy as np
import cohere
from numpy.linalg import norm
from PyPDF2 import PdfReader
import streamlit as st


co_client = cohere.Client(st.secrets["API_KEY"])
CHUNK_SIZE = 2000
OUTPUT_BASE_DIR = "./"


def get_random_string(length: int = 10):
    letters = string.ascii_letters
    result_str = "".join(random.choice(letters) for i in range(length))
    return result_str


def process_csv_file(st_file_object: Any, run_id: str = None):
    df = pd.read_csv(StringIO(st_file_object.getvalue().decode("utf-8")))
    run_id = get_random_string() if run_id is None else run_id

    output_path = os.path.join(OUTPUT_BASE_DIR, f" {run_id}.csv")

    return df, run_id, output_path, len(df)


def process_text_input(text: str, run_id: str = None):
    text = StringIO(text).read()
    chunks = [text[i: i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    df = pd.DataFrame.from_dict({"text": chunks})
    run_id = get_random_string() if run_id is None else run_id

    output_path = os.path.join(OUTPUT_BASE_DIR, f"{run_id}.csv")

    return df, run_id, output_path, len(df)

def read_pdf(pdffile) : 
    pdfReader = PdfReader(pdffile)
    count = pdfReader.numPages
    all_pages = ""
    for pagenum in range(count) :
        page = pdfReader.getPage(pagenum)
        all_pages += page.extract_text() 
    
    return all_pages



# def process_pdf_input(st_file_object: Any, run_id: str = None):
    
    # import fitz
    # doc = fitz.open(st_file_object)
    # text = ""
    # for page in doc:
    #     text+=page.get_text()

def process_pdf_input(train_file, run_id: str = None):

    text = read_pdf(train_file)
    text = StringIO(text).read()
    chunks = [text[i: i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    df = pd.DataFrame.from_dict({"text": chunks})
    run_id = get_random_string() if run_id is None else run_id
    output_path = os.path.join(OUTPUT_BASE_DIR, f"{run_id}.csv")

    return df, run_id, output_path, len(df)

def embed_stuff(list_of_texts):
    response = co_client.embed(model="small", texts=list_of_texts)
    return response.embeddings


def get_embeddings_from_df(df):
    return embed_stuff(list(df.text.values))


def top_n_neighbours_indices(
    prompt_embedding: np.ndarray, storage_embeddings: np.ndarray, n: int = 5
):
    if isinstance(storage_embeddings, list):
        storage_embeddings = np.array(storage_embeddings)
    if isinstance(prompt_embedding, list):
        storage_embeddings = np.array(prompt_embedding)
    similarity_matrix = (
        prompt_embedding
        @ storage_embeddings.T
        / np.outer(norm(prompt_embedding, axis=-1), norm(storage_embeddings, axis=-1))
    )
    num_neighbours = min(similarity_matrix.shape[1], n)
    indices = np.argsort(similarity_matrix, axis=-1)[:, -num_neighbours:]

    return indices


def select_prompts(list_of_texts, sorted_indices):
    return np.take_along_axis(np.array(list_of_texts)[:, None], sorted_indices, axis=0)


def get_augmented_prompts(prompt_embedding, storage_embeddings, storage_df) -> List:
    assert prompt_embedding.shape[0] == 1
    if isinstance(prompt_embedding, list):
        prompt_embedding = np.array(prompt_embedding)
    indices = top_n_neighbours_indices(prompt_embedding, storage_embeddings, n=5)
    similar_prompts = select_prompts(storage_df.text.values, indices)

    return similar_prompts[0]