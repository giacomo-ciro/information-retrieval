import logging
from typing import Optional

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

def compute_chunked_norms(
    matrix: np.ndarray, 
    chunk_size: int = 1000,
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """
    Compute matrix row norms in chunks to reduce memory usage.

    Parameters:
    -----------
    matrix : numpy.ndarray
        Input matrix to compute norms for
    chunk_size : int, optional
        Size of chunks to process (default: 1000)
    eps : float, optional
        Small epsilon to add to norms (default: 1e-4)
    logger : logging.Logger, optional
        Logger for tracking progress

    Returns:
    --------
    numpy.ndarray
        Column vector of norms with added epsilon
    """
    m = matrix.shape[0]
    norms = np.zeros((m, 1), dtype=np.float32)

    for i in range(0, m, chunk_size):
        if logger:
            logger.debug(f"Norm chunk {i // chunk_size:02} / {m // chunk_size}")
        
        chunk_end = min(i + chunk_size, m)
        chunk = matrix[i:chunk_end]
        
        # Compute norms for current chunk
        chunk_norms = np.linalg.norm(chunk, axis=1, keepdims=True)
        norms[i:chunk_end] = chunk_norms

    return norms

def cosine_similarity_matrix(
    A: np.ndarray,
    B: np.ndarray,
    chunk_size: int = 1000,
    eps: float = 1e-4,
    logger: logging.Logger = None,
) -> np.ndarray:
    """
    Compute cosine similarity between each pair of vectors in matrices A and B
    using a chunked approach to avoid memory issues.

    Parameters:
    A (numpy.ndarray): Matrix where each row is a vector
    B (numpy.ndarray): Matrix where each row is a vector
    chunk_size (int): Size of chunks to process at once
    eps (float): Small epsilon value to avoid division by zero

    Returns:
    numpy.ndarray: Matrix where element [i,j] is the cosine similarity
                   between A[i] and B[j]
    """

    if A.shape[1] != B.shape[1]:
        raise ValueError("Incompatible embedding dimensions.")
    
    if logger:
        logger.info(f"Embedding Dimensionality = {A.shape[1]}")
        tmp = (A.shape[0] * B.shape[0]) * 4 / 1e6
        logger.debug(f"Memory for Sim Matrix = {tmp:,} MB.")
        tmp = (A.shape[0] * A.shape[1]) * 4 / 1e6
        logger.debug(f"Memory for A norms = {tmp:,} MB")
        tmp = (B.shape[0] * B.shape[1]) * 4 / 1e6
        logger.debug(f"Memory for B norms = {tmp:,} MB")

    # Pre-compute norms for both matrices
    A_norms = compute_chunked_norms(A, chunk_size, logger)
    B_norms = compute_chunked_norms(B, chunk_size, logger)

    # Pre-normalize B since it's used in every chunk
    B_normalized = B / B_norms

    m, n = A.shape[0], B.shape[0]
    similarity_matrix = np.zeros((m, n), dtype=np.float32)  # Use float32 to save memory

    # Process A in chunks
    for i in range(0, m, chunk_size):
        if logger:
            logger.debug(f"Row chunk {i // chunk_size:02} / {m // chunk_size}")
        # Get the current chunk indices
        chunk_end = min(i + chunk_size, m)

        # Normalize only the current chunk of A
        A_chunk_normalized = A[i:chunk_end] / A_norms[i:chunk_end]

        # Process B in chunks for each A chunk
        for j in range(0, n, chunk_size):
            # Get the current chunk indices for B
            j_end = min(j + chunk_size, n)

            # Compute dot product for the current chunks
            similarity_matrix[i:chunk_end, j:j_end] = np.dot(
                A_chunk_normalized, B_normalized[j:j_end].T
            )

    if logger:
        logger.debug("Done!")

    return similarity_matrix


def evaluate_responses_retrieved(
    test_responses_retrieved: pd.DataFrame,
    test_responses: pd.DataFrame,
) -> float:
    
    if not isinstance(test_responses_retrieved, pd.DataFrame) or not isinstance(
        test_responses, pd.DataFrame
    ):
        raise ValueError("Expected pd.DataFrame as args.")

    if test_responses.shape[0] != test_responses_retrieved.shape[0]:
        raise ValueError("Args have different number of rows.")

    if (
        "conversation_id" not in test_responses.columns
        or "model_response" not in test_responses.columns
        or "conversation_id" not in test_responses_retrieved.columns
        or "model_response" not in test_responses_retrieved.columns
    ):
        raise AttributeError(
            "Expected both args to have [conversation_id, model_response] columns."
        )

    data = pd.DataFrame(
        {
            "conversation_id": test_responses_retrieved.conversation_id.tolist(),
            "model_response": test_responses.model_response.tolist(),
            "retrieved_response": test_responses_retrieved.model_response.tolist(),
            # 'user_prompt': test_responses_retrieved.conversation_prompt.tolist(), # optional, for debugging
        }
    ).astype(str)

    # Smoothing function
    smoothingfunction = SmoothingFunction()  # if you want to know more about smoothing functions: https://aclanthology.org/W14-3346.pdf

    # BLEU score calculation
    data["bleu_score"] = data.apply(
        lambda x: sentence_bleu(
            [x["model_response"].split()],
            x["retrieved_response"].split(),
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=smoothingfunction.method3,
        ),
        axis=1,
    )

    return data.bleu_score.mean()