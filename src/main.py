import logging
import argparse
import json
from pathlib import Path

import pandas as pd
from utils import evaluate_responses_retrieved
from retriever import Retriever


def load_config(config_path="config.json"):
    """
    Load configuration from JSON file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration parameters
    """
    with open(config_path, "r") as f:
        return json.load(f)


def setup_logging(debug_mode=False):
    """
    Configure logging based on debug flag.
    
    Args:
        debug_mode (bool): If True, set logging level to DEBUG, otherwise INFO
        
    Returns:
        logging.Logger: Configured logger
    """
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
    )
    return logging.getLogger(__name__)


def load_datasets(config, logger):
    """
    Load train, dev, and test datasets and configure them based on mode.
    
    In test mode, concatenates train and dev datasets for more training data.
    In evaluation mode, uses dev dataset as the test set.
    
    Args:
        config (dict): Configuration dictionary
        logger (logging.Logger): Logger for output
        
    Returns:
        tuple: (train_prompts, train_responses, test_prompts, test_responses)
    """
    # Load all datasets
    data_dir = Path("./data")
    train_prompts = pd.read_csv(data_dir / "train_prompts.csv")
    train_responses = pd.read_csv(data_dir / "train_responses.csv")
    dev_prompts = pd.read_csv(data_dir / "dev_prompts.csv")
    dev_responses = pd.read_csv(data_dir / "dev_responses.csv")
    test_prompts = pd.read_csv(data_dir / "test_prompts.csv")
    
    # Configure datasets based on mode
    if config["test"]:
        # In test mode, combine train and dev data for more training examples
        train_prompts = pd.concat(
            [train_prompts, dev_prompts], 
            axis=0
        ).reset_index(drop=True)  # drop=True to avoid duplicate index column
        
        train_responses = pd.concat(
            [train_responses, dev_responses], 
            axis=0
        ).reset_index(drop=True)
        
        test_responses = None  # Not available in test mode
    else:
        # In evaluation mode, use dev set as the test set
        test_prompts = dev_prompts
        test_responses = dev_responses
    
    # Log dataset sizes
    logger.info(f"Train prompts = {len(train_prompts):,}")
    logger.info(f"Test prompts = {len(test_prompts):,}")
    
    return train_prompts, train_responses, test_prompts, test_responses


def update_logs(config, bleu, log_path='logs.json'):
    """
    Update the logs file with the latest evaluation results.
    
    Args:
        config (dict): Configuration parameters
        bleu (float): BLEU score from evaluation
        log_path (str): Path to the logs file
    """
    try:
        with open(log_path, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Handle case when log file doesn't exist or is empty/corrupted
        data = []
    
    # Create new log entry
    new_data = {"bleu": bleu, **config}
    
    # Add to beginning of log list
    data.insert(0, new_data)
    
    # Save updated logs
    with open(log_path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    """
    Main function to run the retrieval system.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Response retrieval system")
    parser.add_argument("-D", '--debug', action='store_true', help="Enable debug logging")
    args = parser.parse_args()
    
    # Setup logging and load configuration
    logger = setup_logging(args.debug)
    config = load_config()
    
    # Load datasets
    train_prompts, train_responses, test_prompts, test_responses = load_datasets(config, logger)
    
    # Initialize retriever and retrieve responses
    retriever = Retriever(
        logger=logger,
        test=config["test"],
        # Note: The model "all-mpnet-base-v2" seems to be used by the Retriever class
    )
    
    test_responses_retrieved = retriever.retrieve(
        train_responses,
        test_prompts
    )
    
    # Process results based on mode
    if not config["test"]:
        # Evaluation mode: Calculate and log BLEU score
        bleu = evaluate_responses_retrieved(
            test_responses_retrieved,
            test_responses
        )
        logger.info(f"BLEU = {bleu}")
        update_logs(config, bleu)
    else:
        # Test mode: Save retrieved responses to file
        track = config["track"]
        output_path = f"./save_test/track_{track}_test.csv"
        logger.info(f"Saving retrieved test responses to {output_path}")
        
        # Ensure output directory exists
        Path("./save_test").mkdir(exist_ok=True)
        
        test_responses_retrieved.to_csv(
            output_path,
            columns=["conversation_id", "response_id"],
            index=False
        )


if __name__ == "__main__":
    main()