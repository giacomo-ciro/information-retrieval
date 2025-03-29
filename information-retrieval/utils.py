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
    Compute L2 norms for each row in a matrix using a chunked approach to reduce memory usage.
    
    This function processes the matrix in chunks to avoid loading the entire matrix into memory
    at once, which helps when dealing with large matrices.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        Input matrix where each row is a vector to compute the norm for
    chunk_size : int, optional
        Size of chunks to process at once (default: 1000)
    logger : logging.Logger, optional
        Logger for tracking progress
        
    Returns:
    --------
    numpy.ndarray
        Column vector of L2 norms for each row in the input matrix
    """
    # Get number of rows in the matrix
    m = matrix.shape[0]
    # Initialize array to store the norms
    norms = np.zeros((m, 1), dtype=np.float32)

    # Process matrix in chunks
    for i in range(0, m, chunk_size):
        # Log progress if logger is provided
        if logger:
            logger.debug(f"Norm chunk {i // chunk_size:02} / {(m - 1) // chunk_size + 1}")
        
        # Calculate end index for current chunk
        chunk_end = min(i + chunk_size, m)
        # Extract current chunk
        chunk = matrix[i:chunk_end]
        
        # Compute L2 norms for current chunk (axis=1 computes norm for each row)
        # keepdims=True preserves the column dimension
        chunk_norms = np.linalg.norm(chunk, axis=1, keepdims=True)
        # Store the computed norms
        norms[i:chunk_end] = chunk_norms

    return norms


def cosine_similarity_matrix(
    A: np.ndarray,
    B: np.ndarray,
    chunk_size: int = 1000,
    eps: float = 1e-4,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Compute cosine similarity between each pair of vectors in matrices A and B
    using a chunked approach to avoid memory issues.
    
    Cosine similarity is computed as the dot product of normalized vectors.
    The function processes data in chunks to reduce memory requirements when
    dealing with large matrices.
    
    Parameters:
    -----------
    A : numpy.ndarray
        Matrix where each row is a vector
    B : numpy.ndarray
        Matrix where each row is a vector
    chunk_size : int, optional
        Size of chunks to process at once (default: 1000)
    eps : float, optional
        Small epsilon value added to norms to avoid division by zero (default: 1e-4)
        Note: This parameter is not currently used in the function
    logger : logging.Logger, optional
        Logger for tracking progress
        
    Returns:
    --------
    numpy.ndarray
        Matrix where element [i,j] is the cosine similarity between A[i] and B[j]
    """
    # Check that the matrices have compatible dimensions
    if A.shape[1] != B.shape[1]:
        raise ValueError("Incompatible embedding dimensions.")
    
    # Log memory usage statistics if logger is provided
    if logger:
        logger.info(f"Embedding Dimensionality = {A.shape[1]}")
        # Calculate approximate memory requirements in MB
        sim_matrix_memory = (A.shape[0] * B.shape[0]) * 4 / 1e6  # 4 bytes per float32
        logger.debug(f"Memory for Sim Matrix = {sim_matrix_memory:,.2f} MB")
        a_memory = (A.shape[0] * A.shape[1]) * 4 / 1e6
        logger.debug(f"Memory for A = {a_memory:,.2f} MB")
        b_memory = (B.shape[0] * B.shape[1]) * 4 / 1e6
        logger.debug(f"Memory for B = {b_memory:,.2f} MB")

    # Pre-compute norms for both matrices
    A_norms = compute_chunked_norms(A, chunk_size, logger)
    B_norms = compute_chunked_norms(B, chunk_size, logger)
    
    # Add small epsilon to avoid division by zero
    A_norms = np.maximum(A_norms, eps)
    B_norms = np.maximum(B_norms, eps)

    # Pre-normalize B since it's used in every chunk to save computation
    B_normalized = B / B_norms

    # Get dimensions
    m, n = A.shape[0], B.shape[0]
    
    # Initialize similarity matrix with zeros, using float32 to save memory
    similarity_matrix = np.zeros((m, n), dtype=np.float32)

    # Process A in chunks
    num_chunks = (m - 1) // chunk_size + 1
    for i in range(0, m, chunk_size):
        if logger:
            logger.debug(f"Row chunk {i // chunk_size:02} / {num_chunks}")
            
        # Calculate end index for current chunk of A
        chunk_end = min(i + chunk_size, m)

        # Normalize only the current chunk of A to save memory
        A_chunk_normalized = A[i:chunk_end] / A_norms[i:chunk_end]

        # Process B in chunks for each A chunk
        for j in range(0, n, chunk_size):
            # Calculate end index for current chunk of B
            j_end = min(j + chunk_size, n)

            # Compute dot product for the current chunks
            # This gives cosine similarity since vectors are normalized
            similarity_matrix[i:chunk_end, j:j_end] = np.dot(
                A_chunk_normalized, B_normalized[j:j_end].T
            )

    if logger:
        logger.debug("Similarity matrix computation completed!")

    return similarity_matrix


def evaluate_responses_retrieved(
    test_responses_retrieved: pd.DataFrame,
    test_responses: pd.DataFrame,
) -> float:
    """
    Evaluate the similarity between retrieved responses and ground truth responses 
    using BLEU score.
    
    This function compares each retrieved response with its corresponding ground truth
    response and calculates a BLEU score to measure their similarity.
    
    Parameters:
    -----------
    test_responses_retrieved : pd.DataFrame
        DataFrame containing retrieved responses, must have 'conversation_id' and 
        'model_response' columns
    test_responses : pd.DataFrame
        DataFrame containing ground truth responses, must have 'conversation_id' and
        'model_response' columns
        
    Returns:
    --------
    float
        Mean BLEU score across all response pairs
        
    Raises:
    -------
    ValueError
        If inputs are not DataFrames or have different number of rows
    AttributeError
        If required columns are missing from the DataFrames
    """
    # Validate input types
    if not isinstance(test_responses_retrieved, pd.DataFrame) or not isinstance(
        test_responses, pd.DataFrame
    ):
        raise ValueError("Expected pd.DataFrame as arguments.")

    # Check that DataFrames have the same number of rows
    if test_responses.shape[0] != test_responses_retrieved.shape[0]:
        raise ValueError("Arguments have different number of rows.")

    # Verify required columns exist in both DataFrames
    required_columns = ["conversation_id", "model_response"]
    for df, name in [(test_responses, "test_responses"), 
                     (test_responses_retrieved, "test_responses_retrieved")]:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise AttributeError(
                f"DataFrame {name} is missing required columns: {missing_cols}. "
                f"Expected columns: {required_columns}"
            )

    # Create a combined DataFrame for evaluation
    data = pd.DataFrame(
        {
            "conversation_id": test_responses_retrieved.conversation_id.tolist(),
            "model_response": test_responses.model_response.tolist(),
            "retrieved_response": test_responses_retrieved.model_response.tolist(),
            # 'user_prompt': test_responses_retrieved.conversation_prompt.tolist(), # optional, for debugging
        }
    ).astype(str)

    # Initialize smoothing function for BLEU calculation
    # Smoothing is necessary when reference or hypothesis sentences are short
    # https://aclanthology.org/W14-3346.pdf
    smoothingfunction = SmoothingFunction()

    # Calculate BLEU score for each pair of responses
    # Using equal weights for unigrams and bigrams (0.5, 0.5)
    # and 0 weight for trigrams and 4-grams (0, 0)
    data["bleu_score"] = data.apply(
        lambda x: sentence_bleu(
            [x["model_response"].split()],  # Reference (ground truth)
            x["retrieved_response"].split(),  # Hypothesis (retrieved)
            weights=(0.5, 0.5, 0, 0),  # Equal weight to unigrams and bigrams
            smoothing_function=smoothingfunction.method3,  # Method3 applies smoothing by adding 1 to counts
        ),
        axis=1,
    )

    # Return the mean BLEU score across all response pairs
    return data.bleu_score.mean()