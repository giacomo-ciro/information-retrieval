"""
BLEU Score Computation Module

This script calculates BLEU scores between training and development datasets
to identify the most similar responses. It uses parallel processing to improve
performance when computing scores across large datasets.
"""

import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def load_data(data_dir="./data"):
    """
    Load prompt and response datasets with optimized settings.
    
    Args:
        data_dir (str): Directory containing the CSV files
        
    Returns:
        tuple: Loaded dataframes (train_prompts, train_responses, dev_prompts, dev_responses)
    """
    # Load data more efficiently - usecols can be added if only specific columns are needed
    train_prompts = pd.read_csv(f"{data_dir}/train_prompts.csv")
    train_responses = pd.read_csv(f"{data_dir}/train_responses.csv")
    dev_prompts = pd.read_csv(f"{data_dir}/dev_prompts.csv")
    dev_responses = pd.read_csv(f"{data_dir}/dev_responses.csv")
    
    return train_prompts, train_responses, dev_prompts, dev_responses


def tokenize_responses(responses, column="model_response"):
    """
    Pre-tokenize responses by splitting text into words.
    
    Args:
        responses (pd.DataFrame): DataFrame containing the responses
        column (str): Column name containing the response text
        
    Returns:
        list: List of tokenized responses
    """
    return [str(resp).split() for resp in responses[column]]


def compute_bleu_batch(indices, train_tokens, dev_tokens):
    """
    Compute BLEU scores for a batch of training examples against all dev examples.
    
    Args:
        indices (range): Range of indices to process from train_tokens
        train_tokens (list): List of tokenized training responses
        dev_tokens (list): List of tokenized development responses
        
    Returns:
        np.ndarray: Matrix of BLEU scores for this batch (batch_size × len(dev_tokens))
    """
    # Initialize smoothing function (method3 = NIST geometric sequence smoothing)
    smooth_func = SmoothingFunction().method3
    result = np.zeros((len(indices), len(dev_tokens)))
    
    # Calculate BLEU scores for each pair of train and dev tokens
    for idx, i in enumerate(indices):
        for j in range(len(dev_tokens)):
            # Use bigram BLEU (weights=(0.5, 0.5, 0, 0)) as specified
            result[idx, j] = sentence_bleu(
                [dev_tokens[j]],  # Reference is from dev set
                train_tokens[i],  # Hypothesis is from train set
                weights=(0.5, 0.5, 0, 0),  # Equal weighting for unigrams and bigrams
                smoothing_function=smooth_func
            )
    
    return result


def compute_all_bleus_parallel(train_tokens, dev_tokens, n_jobs=None):
    """
    Compute all BLEU scores using parallel processing.
    
    Args:
        train_tokens (list): List of tokenized training responses
        dev_tokens (list): List of tokenized development responses
        n_jobs (int, optional): Number of parallel jobs. Defaults to CPU count.
        
    Returns:
        np.ndarray: Matrix of BLEU scores (n_train × n_dev)
    """
    # Use CPU count if n_jobs not specified, but limit to reasonable value
    if n_jobs is None:
        n_jobs = min(os.cpu_count(), 16)  # Prevent excessive parallelism
    
    n_train = len(train_tokens)
    n_dev = len(dev_tokens)
    bleus = np.zeros((n_train, n_dev))
    
    # Split work into batches for parallel processing
    batch_size = max(1, n_train // n_jobs)
    batches = [(range(i, min(i + batch_size, n_train))) for i in range(0, n_train, batch_size)]
    
    # Process batches in parallel with progress bar
    print(f"Computing BLEU scores using {n_jobs} workers...")
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        func = partial(compute_bleu_batch, train_tokens=train_tokens, dev_tokens=dev_tokens)
        results = list(tqdm(
            executor.map(func, batches), 
            total=len(batches), 
            desc="BLEU Computation"
        ))
    
    # Combine results from all batches
    for batch_idx, indices in enumerate(batches):
        start_idx = list(indices)[0]
        end_idx = start_idx + len(indices)
        bleus[start_idx:end_idx, :] = results[batch_idx]
    
    return bleus


def main():
    """Main execution function."""
    # Check for GPU availability (useful for potential future torch operations)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory if it doesn't exist
    os.makedirs("./save", exist_ok=True)
    
    # Load datasets
    _, train_responses, _, dev_responses = load_data()
    
    # Pre-tokenize responses (once instead of repeatedly in the loop)
    print("Tokenizing responses...")
    train_tokenized = tokenize_responses(train_responses)
    dev_tokenized = tokenize_responses(dev_responses)
    
    # Compute BLEU scores in parallel
    print(f"Computing BLEU scores between {len(train_tokenized)} train and {len(dev_tokenized)} dev responses...")
    bleus = compute_all_bleus_parallel(train_tokenized, dev_tokenized)
    
    # Transpose to match the expected dimensions (dev_size × train_size)
    # This makes each row correspond to a dev example and columns to train examples
    bleus = bleus.T  # Shape becomes (dev_responses.shape[0], train_responses.shape[0])
    np.save("./save/bleus.npy", bleus)
    print(f"Saved BLEU scores matrix with shape {bleus.shape} to ./save/bleus.npy")
    
    # Find the training responses that maximize the BLEU score for each dev response
    maximizers = np.argmax(bleus, axis=1)
    np.save("./save/maximizers.npy", maximizers)
    print(f"Saved {len(maximizers)} maximizer indices to ./save/maximizers.npy")


if __name__ == "__main__":
    main()