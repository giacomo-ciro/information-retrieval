import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import numpy as np
import torch
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import os
from tqdm import tqdm

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data more efficiently - specify dtypes if known
train_prompts = pd.read_csv("./data/train_prompts.csv")
train_responses = pd.read_csv("./data/train_responses.csv")
dev_prompts = pd.read_csv("./data/dev_prompts.csv")
dev_responses = pd.read_csv("./data/dev_responses.csv")

# Pre-tokenize responses (once instead of repeatedly in the loop)
train_tokenized = [str(resp).split() for resp in train_responses["model_response"]]
dev_tokenized = [str(resp).split() for resp in dev_responses["model_response"]]

# Define BLEU computation function for parallel processing
def compute_bleu_batch(indices, train_tokens, dev_tokens):
    smooth_func = SmoothingFunction().method3
    result = np.zeros((len(indices), len(dev_tokens)))
    
    for idx, i in enumerate(indices):
        for j in range(len(dev_tokens)):
            result[idx, j] = sentence_bleu(
                [dev_tokens[j]],
                train_tokens[i],
                weights=(0.5, 0.5, 0, 0),
                smoothing_function=smooth_func
            )
    
    return result

# Parallel BLEU score computation
def compute_all_bleus_parallel(train_tokens, dev_tokens, n_jobs=os.cpu_count()):
    n_train = len(train_tokens)
    n_dev = len(dev_tokens)
    bleus = np.zeros((n_train, n_dev))
    
    # Split work into batches
    batch_size = max(1, n_train // n_jobs)
    batches = [(range(i, min(i + batch_size, n_train))) for i in range(0, n_train, batch_size)]
    
    # Process batches in parallel with progress bar
    print(f"Computing BLEU scores using {n_jobs} workers...")
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        func = partial(compute_bleu_batch, train_tokens=train_tokens, dev_tokens=dev_tokens)
        results = list(tqdm(executor.map(func, batches), total=len(batches), desc="BLEU Computation"))
    
    # Combine results
    for batch_idx, indices in enumerate(batches):
        start_idx = list(indices)[0]
        bleus[start_idx:start_idx + len(indices), :] = results[batch_idx]
    
    return bleus

# Compute BLEU scores in parallel
print("Computing BLEU scores in parallel...")
bleus = compute_all_bleus_parallel(train_tokenized, dev_tokenized)

# Transpose to match the original code's dimensions if needed
bleus = bleus.T  # Now shape is (dev_prompts.shape[0], train_prompts.shape[0])
np.save("./save/bleus.npy", bleus)

# Find the responses in the train maximizing the bleu with the ones in the dev
maximizers = np.argmax(bleus, axis=1)
np.save("./save/maximizers.npy", maximizers)