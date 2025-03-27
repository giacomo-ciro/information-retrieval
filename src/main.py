import logging
import argparse
import json

import pandas as pd
from utils import evaluate_responses_retrieved

from retriever import Retriever

# all-mpnet-base-v2

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-D", '--debug', action='store_true')
args = parser.parse_args()
if args.debug:
    level = logging.DEBUG
else:
    level = logging.INFO

# Get config
with open("config.json", "r") as f:
    config = json.load(f)

# Setup logging
logging.basicConfig(
    level=level,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

# Import data
train_prompts = pd.read_csv("./data/train_prompts.csv")
train_responses = pd.read_csv("./data/train_responses.csv")
dev_prompts = pd.read_csv("./data/dev_prompts.csv")
dev_responses = pd.read_csv("./data/dev_responses.csv")
test_prompts = pd.read_csv("./data/test_prompts.csv")

if config["test"]:
    train_prompts = pd.concat(
        [train_prompts, dev_prompts],
        axis = 0
    ).reset_index()
    train_responses = pd.concat(
        [train_responses, dev_responses],
        axis = 0
    ).reset_index()
    test_responses = None   # not available when generating test set
else:
    # train prompts unchanged
    test_prompts = dev_prompts
    test_responses = dev_responses
logger.info(f"Train prompts = {len(train_prompts):,}")
logger.info(f"Test prompts = {len(test_prompts):,}")

# Main Retrieval Logic
retriever = Retriever(
    logger=logger,
    test = config["test"],
)
test_responses_retrieved = retriever.retrieve(
    train_responses,
    test_prompts
)

# Evaluate retrieved responses

if not config["test"]:
    bleu = evaluate_responses_retrieved(
        test_responses_retrieved,
        test_responses
    )
    logging.info(f"BLEU = {bleu}")
    with open('logs.json', 'r+') as f:
        
        # Load previsou logs
        data = json.load(f)

        # update
        new_data = {"bleu":bleu}
        new_data.update(config)
        data.append(new_data)
        data.reverse()
        
        # Write to beginning of file and truncate the rest
        f.seek(0)
        json.dump(data, f, indent = 2)      # indent=4 to prettify
        f.truncate()
else:
    track = config["track"]
    PATH = f"./save_test/track_{track}_test.csv"
    logging.info(f"Saving retrieved test responses to {PATH}")

    test_responses_retrieved.to_csv(
        PATH,
        columns=["conversation_id", "response_id"],
        index=False
    )