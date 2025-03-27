import pandas as pd
import sys
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset, DataLoader
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import os
from tqdm import tqdm
from utils import cosine_similarity_matrix
from sentence_transformers import SentenceTransformer
import re

EPOCHS = 100
BATCH_SIZE = 128
EMBEDDING_MODEL = "all-mpnet-base-v2"

def _get_dynamic_embeddings(
        train_responses: pd.DataFrame,
        test_prompts: pd.DataFrame
    )-> np.ndarray:
        
    # -------------- Get Dynamic Embeddings
    train_embeddings_path = (
        f"./save/"
        + "train_embeddings_"
        + re.sub("[^A-Za-z0-9]", "_", EMBEDDING_MODEL)
        + ".npy"
    )
    test_embeddings_path = (
        f"./save/"
        + "test_embeddings_"
        + re.sub("[^A-Za-z0-9]", "_", EMBEDDING_MODEL)
        + ".npy"
    )

    # Load model
    if (not os.path.exists(train_embeddings_path) or
        not os.path.exists(train_embeddings_path)):
        model = SentenceTransformer(EMBEDDING_MODEL)

    # Train
    if not os.path.exists(train_embeddings_path):
        train_embeddings = model.encode(
            train_responses.user_prompt.to_list()
        )  # returns np.ndarray
        np.save(train_embeddings_path, train_embeddings)
    else:
        train_embeddings = np.load(train_embeddings_path)

    # Test
    if not os.path.exists(test_embeddings_path):
        test_embeddings = model.encode(test_prompts.user_prompt.to_list())
        np.save(test_embeddings_path, test_embeddings)
    else:
        test_embeddings = np.load(test_embeddings_path)

    return torch.tensor(train_embeddings, device=device), torch.tensor(test_embeddings, device = device)

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data more efficiently - specify dtypes if known
train_prompts = pd.read_csv("./data/train_prompts.csv")
train_responses = pd.read_csv("./data/train_responses.csv")
dev_prompts = pd.read_csv("./data/dev_prompts.csv")
dev_responses = pd.read_csv("./data/dev_responses.csv")

# Load bleus and check validity
bleus = np.load("./save/bleus.npy")
if (not bleus.shape[0] == dev_responses.shape[0] or
    not bleus.shape[1] == train_responses.shape[0]):
    raise AttributeError("Incompatible shape for bleu array.")

maximizers = np.argmax(bleus, axis=1)       # responses in the train maximizing the bleu with the ones in the dev

# Get tf-idf embeddings (this part is already efficient)
# print("Computing TF-IDF embeddings...")
# vectorizer = TfidfVectorizer(
#     analyzer="char",
#     ngram_range=(3,3),
#     min_df=0.01,
#     max_df=0.75,
#     sublinear_tf=True,
# )
 
# train_prompts_list = train_responses.user_prompt.tolist()
# dev_prompts_list = dev_prompts.user_prompt.tolist()

# train_embeddings = vectorizer.fit_transform(train_prompts_list).toarray()
# dev_embeddings = vectorizer.transform(dev_prompts_list).toarray()

# train_embeddings  = (train_embeddings - train_embeddings.mean(axis = 1)[...,None]) / train_embeddings.std(axis = 1)[...,None]
# dev_embeddings  = (dev_embeddings - dev_embeddings.mean(axis = 1)[...,None]) / dev_embeddings.std(axis = 1)[...,None]

# Get dynamic embeddings

train_embeddings, dev_embeddings = _get_dynamic_embeddings(train_responses, dev_prompts)

# PyTorch Dataset implementation with GPU support
class VectorDataset(Dataset):
    def __init__(self, X, Y, device=device):
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

dataset = VectorDataset(
    X=dev_embeddings,                   # the ones wer are looking the response for
    Y=train_embeddings[maximizers]      # the response maximizing the bleu
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, pin_memory=device=='cpu')
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=128, pin_memory=device=='cpu')

# Similarity between X projected via Q and Y
def cosine_similarity(Q, X, Y):
    
    QX = torch.matmul(X, Q)

    QX_norm = torch.norm(QX, dim=1, keepdim=True)
    Y_norm = torch.norm(Y, dim=1, keepdim=True)

    return torch.sum(QX * Y, dim=1) / (QX_norm * Y_norm + 1e-8)

# Make cosine similarity a loss
def cosine_loss(Q, X, Y):
    sim = cosine_similarity(Q, X, Y)
    return -torch.mean(sim.squeeze())


# Initialize Q on the GPU
Q = torch.nn.Parameter(
    torch.randn(train_embeddings.shape[1], train_embeddings.shape[1], 
                dtype=torch.float32, device=device) * 0.01  # Smaller initialization for stability
)

# Use Adam optimizer for faster convergence
optimizer = torch.optim.AdamW([Q], lr=0.01, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

# Training Loop with early stopping
epochs = EPOCHS
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience = 10
patience_counter = 0
smooth_func = SmoothingFunction().method3

# Estimate BLEU without using Q
with torch.no_grad():
    bleu = 0.0
    sim = cosine_similarity_matrix(train_embeddings.cpu(), dev_embeddings.cpu().detach().numpy())
    for i in tqdm(range(sim.shape[1]), desc = "Checking Bleus"):
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
    val_bleu = bleu / len(dev_responses)
print(f"BLEU without matrix Q = {val_bleu:.6f}")

# Best possible BLEU
with torch.no_grad():
    bleu = 0.0
    for i in tqdm(range(sim.shape[1]), desc = "Checking Bleus"):
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

# Starting BLEU with random Q
with torch.no_grad():
    bleu = 0.0
    QX = np.matmul(train_embeddings.cpu().detach().numpy(), Q.cpu().detach().numpy())
    sim = cosine_similarity_matrix(QX, dev_embeddings.cpu().detach().numpy())
    for i in tqdm(range(sim.shape[1]), desc = "Checking Bleus"):
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
    
    val_bleu = bleu / len(dev_responses)
print(f"Random Q BLEU = {val_bleu:.6f}")

# --------------------------- TRAINING LOOP --------------------------------
print("Starting training...")
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
        bleu = 0
        for batch_X, batch_Y in tqdm(val_loader, desc=f"Val Epoch {epoch}", leave=False):
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
            progress_bar.set_description(f"Early stopping at epoch {epoch}")
            break
    
    # Update progress bar with current losses
    progress_bar.set_postfix({"train_loss": f"{train_loss:.6f}", "val_loss": f"{val_loss:.6f}"})


# Check BLEU on entire dataset
with torch.no_grad():
    bleu = 0.0
    QX = np.matmul(train_embeddings.cpu().detach().numpy(), Q.cpu().detach().numpy())
    sim = cosine_similarity_matrix(QX, dev_embeddings.cpu().detach().numpy())
    for i in tqdm(range(sim.shape[1]), desc = "Checking Bleus"):
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
    
    val_bleu = bleu / len(dev_responses)


# Save the best model
np.save("./save/Q.npy", best_Q.cpu().numpy())
print(f"Final BLEU = {val_bleu:.6f}")