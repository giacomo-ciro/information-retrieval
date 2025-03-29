"""
Text Similarity Model using Sentence Embeddings and BLEU Optimization

This script trains a transformation matrix Q that improves text similarity matching
between prompts and responses. It uses sentence embeddings to represent text and
optimizes a transformation matrix to maximize the BLEU score between predicted
and actual responses.

The main workflow:
1. Load prompt and response data
2. Generate or load text embeddings using SentenceTransformer
3. Train a transformation matrix to improve similarity matching
4. Evaluate performance using BLEU scores
"""

import os
import re

import numpy as np
import pandas as pd
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import cosine_similarity_matrix

# Configuration parameters
EPOCHS = 100
BATCH_SIZE = 128
EMBEDDING_MODEL = "all-mpnet-base-v2"

def _get_dynamic_embeddings(train_responses: pd.DataFrame, 
                           test_prompts: pd.DataFrame) -> tuple:
    """
    Generate or load pre-computed sentence embeddings for train and test data.
    
    Args:
        train_responses: DataFrame containing training responses with user_prompt column
        test_prompts: DataFrame containing test prompts with user_prompt column
        
    Returns:
        tuple: (train_embeddings, test_embeddings) as PyTorch tensors on the device
    """
    # Define file paths for cached embeddings
    train_embeddings_path = (
        f"./save/train_embeddings_"
        + re.sub("[^A-Za-z0-9]", "_", EMBEDDING_MODEL)
        + ".npy"
    )
    test_embeddings_path = (
        f"./save/test_embeddings_"
        + re.sub("[^A-Za-z0-9]", "_", EMBEDDING_MODEL)
        + ".npy"
    )

    # Load model if embeddings need to be computed
    if (not os.path.exists(train_embeddings_path) or
        not os.path.exists(test_embeddings_path)):
        print(f"Loading SentenceTransformer model: {EMBEDDING_MODEL}")
        model = SentenceTransformer(EMBEDDING_MODEL)

    # Generate or load train embeddings
    if not os.path.exists(train_embeddings_path):
        print("Computing train embeddings...")
        train_embeddings = model.encode(
            train_responses.user_prompt.to_list()
        )  # returns np.ndarray
        # Cache embeddings for future use
        np.save(train_embeddings_path, train_embeddings)
    else:
        print(f"Loading cached train embeddings from {train_embeddings_path}")
        train_embeddings = np.load(train_embeddings_path)

    # Generate or load test embeddings
    if not os.path.exists(test_embeddings_path):
        print("Computing test embeddings...")
        test_embeddings = model.encode(test_prompts.user_prompt.to_list())
        # Cache embeddings for future use
        np.save(test_embeddings_path, test_embeddings)
    else:
        print(f"Loading cached test embeddings from {test_embeddings_path}")
        test_embeddings = np.load(test_embeddings_path)

    # Convert to PyTorch tensors and move to the appropriate device
    return (torch.tensor(train_embeddings, device=device), 
            torch.tensor(test_embeddings, device=device))


class VectorDataset(Dataset):
    """
    PyTorch Dataset for handling embedding vectors.
    
    Attributes:
        X: Input embeddings
        Y: Target embeddings
    """
    def __init__(self, X, Y, device=None):
        """
        Initialize the dataset with input and target embeddings.
        
        Args:
            X: Input embeddings
            Y: Target embeddings
            device: Device to store tensors on (not used directly here as tensors should already be on device)
        """
        self.X = X
        self.Y = Y
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.X)
    
    def __getitem__(self, idx):
        """Return input and target embedding pair at specified index."""
        return self.X[idx], self.Y[idx]


def cosine_similarity(Q, X, Y):
    """
    Calculate the cosine similarity between X projected via Q and Y.
    
    Args:
        Q: Transformation matrix
        X: Input embeddings
        Y: Target embeddings
        
    Returns:
        torch.Tensor: Cosine similarity values
    """
    # Project X through the transformation matrix Q
    QX = torch.matmul(X, Q)

    # Calculate L2 norms
    QX_norm = torch.norm(QX, dim=1, keepdim=True)
    Y_norm = torch.norm(Y, dim=1, keepdim=True)

    # Calculate cosine similarity with epsilon for numerical stability
    return torch.sum(QX * Y, dim=1) / (QX_norm * Y_norm + 1e-8)


def cosine_loss(Q, X, Y):
    """
    Calculate loss based on negative cosine similarity (lower is better).
    
    Args:
        Q: Transformation matrix
        X: Input embeddings
        Y: Target embeddings
        
    Returns:
        torch.Tensor: Loss value (negative mean of cosine similarities)
    """
    sim = cosine_similarity(Q, X, Y)
    return -torch.mean(sim.squeeze())


def calculate_bleu(train_responses, dev_responses, train_embeddings, dev_embeddings, Q=None, description="Checking BLEU"):
    """
    Calculate BLEU score between predicted and actual responses.
    
    Args:
        train_responses: DataFrame of training responses
        dev_responses: DataFrame of development responses
        train_embeddings: Embeddings of training prompts 
        dev_embeddings: Embeddings of development prompts
        Q: Optional transformation matrix (if None, direct similarity is used)
        description: Description for progress bar
        
    Returns:
        float: Average BLEU score
    """
    smooth_func = SmoothingFunction().method3
    bleu = 0.0
    
    # Move tensors to CPU for numpy operations
    train_emb_cpu = train_embeddings.cpu().detach().numpy()
    dev_emb_cpu = dev_embeddings.cpu().detach().numpy()
    
    # Apply transformation if Q is provided
    if Q is not None:
        Q_cpu = Q.cpu().detach().numpy()
        train_emb_cpu = np.matmul(train_emb_cpu, Q_cpu)
    
    # Calculate similarity matrix
    sim = cosine_similarity_matrix(train_emb_cpu, dev_emb_cpu)
    
    # Calculate BLEU score for each development prompt
    for i in tqdm(range(sim.shape[1]), desc=description):
        most_similar_ix = np.argmax(sim[:, i])
        retrieved_response = train_responses.loc[most_similar_ix, "model_response"]
        true_response = dev_responses.loc[i, "model_response"]
        retrieved_response, true_response = map(str, [retrieved_response, true_response])

        bleu += sentence_bleu(
            [true_response.split()],
            retrieved_response.split(),
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=smooth_func
        )
    
    # Return average BLEU score
    return bleu / len(dev_responses)


def main():
    """Main execution function."""
    global device  # Used in multiple functions
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading datasets...")
    train_prompts = pd.read_csv("./data/train_prompts.csv")
    train_responses = pd.read_csv("./data/train_responses.csv")
    dev_prompts = pd.read_csv("./data/dev_prompts.csv")
    dev_responses = pd.read_csv("./data/dev_responses.csv")

    # Load pre-computed BLEU scores for optimal response matching
    print("Loading pre-computed BLEU scores...")
    bleus = np.load("./save/bleus.npy")
    
    # Validate BLEU array dimensions
    if (not bleus.shape[0] == dev_responses.shape[0] or
        not bleus.shape[1] == train_responses.shape[0]):
        raise AttributeError("Incompatible shape for BLEU array.")

    # Find best training responses for each dev prompt based on BLEU
    maximizers = np.argmax(bleus, axis=1)

    # Get sentence embeddings for prompts
    print("Getting sentence embeddings...")
    train_embeddings, dev_embeddings = _get_dynamic_embeddings(train_responses, dev_prompts)

    # Create dataset with dev embeddings and their best matching train embeddings
    print("Creating dataset...")
    dataset = VectorDataset(
        X=dev_embeddings,                    # The prompts we're finding responses for
        Y=train_embeddings[maximizers]       # The responses maximizing the BLEU score
    )

    # Split data into training and validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        shuffle=True, 
        batch_size=BATCH_SIZE, 
        pin_memory=device=='cpu'
    )
    val_loader = DataLoader(
        val_dataset, 
        shuffle=False, 
        batch_size=BATCH_SIZE, 
        pin_memory=device=='cpu'
    )

    # Initialize transformation matrix Q
    embedding_dim = train_embeddings.shape[1]
    Q = torch.nn.Parameter(
        torch.randn(embedding_dim, embedding_dim, 
                    dtype=torch.float32, device=device) * 0.01  # Small initialization for stability
    )

    # Setup optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW([Q], lr=0.01, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training parameters
    epochs = EPOCHS
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    # Baseline BLEU scores
    # 1. Without transformation
    print("\nBaseline performance evaluations:")
    val_bleu = calculate_bleu(
        train_responses, dev_responses, 
        train_embeddings, dev_embeddings,
        description="BLEU without matrix Q"
    )
    print(f"BLEU without matrix Q = {val_bleu:.6f}")

    # 2. Optimal BLEU (using pre-computed maximizers)
    bleu = 0.0
    smooth_func = SmoothingFunction().method3
    for i in tqdm(range(len(dev_responses)), desc="Checking optimal BLEU"):
        most_similar_ix = maximizers[i] 
        retrieved_response = train_responses.loc[most_similar_ix, "model_response"]
        true_response = dev_responses.loc[i, "model_response"]
        retrieved_response, true_response = map(str, [retrieved_response, true_response])

        bleu += sentence_bleu(
            [true_response.split()],
            retrieved_response.split(),
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=smooth_func
        )
    val_bleu = bleu / len(dev_responses)
    print(f"Optimal BLEU = {val_bleu:.6f}")

    # 3. Starting BLEU with random Q
    val_bleu = calculate_bleu(
        train_responses, dev_responses, 
        train_embeddings, dev_embeddings,
        Q=Q,
        description="BLEU with random Q"
    )
    print(f"Random Q BLEU = {val_bleu:.6f}")

    # --------------------------- TRAINING LOOP --------------------------------
    print("\nStarting training...")
    progress_bar = tqdm(range(epochs), desc="Training")
    for epoch in progress_bar:
        
        # Train
        Q.requires_grad = True
        train_loss = 0.0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            loss = cosine_loss(Q, batch_X, batch_Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        Q.requires_grad = False
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                loss = cosine_loss(Q, batch_X, batch_Y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_Q = Q.clone().detach()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Update progress bar with current losses
        progress_bar.set_postfix({"train_loss": f"{train_loss:.6f}", "val_loss": f"{val_loss:.6f}"})

    # Evaluate final BLEU score
    final_bleu = calculate_bleu(
        train_responses, dev_responses, 
        train_embeddings, dev_embeddings,
        Q=best_Q,
        description="Calculating final BLEU"
    )

    # Save the best model
    np.save("./save/Q.npy", best_Q.cpu().numpy())
    print(f"Final BLEU = {final_bleu:.6f}")


if __name__ == "__main__":
    main()