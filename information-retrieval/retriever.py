import json
import os
import re
from logging import Logger
from multiprocessing import cpu_count

import fasttext
import numpy as np
import pandas as pd
from gensim.models import Doc2Vec, FastText
from gensim.models.doc2vec import TaggedDocument
from huggingface_hub import hf_hub_download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import cosine_similarity_matrix


class Retriever:
    """
    A retrieval system that finds the most similar training response for each test prompt.
    
    This class implements different retrieval strategies based on various embedding methods:
    - Track 1: TF-IDF based approaches (word, character, or both)
    - Track 2: Classic embedding models (Doc2Vec, FastText)
    - Track 3: Transformer-based embeddings or BM25
    
    Attributes:
        test (bool): Whether running in test mode
        logger (Logger): Logger instance for tracking progress
        config (dict): Configuration parameters loaded from config.json
    """
    
    def __init__(
            self,
            logger: Logger = None,
            test: bool = False
            ):
        """
        Initialize the Retriever class.
        
        Args:
            logger (Logger, optional): Logger for tracking progress
            test (bool, optional): Whether to run in test mode. Defaults to False.
        
        Raises:
            ValueError: If the track configuration is invalid
        """
        
        self.test = test
        self.logger = logger
        
        # Load configuration
        with open("config.json", "r") as f:
            self.config = json.load(f)
        
        # Validate track configuration
        if not isinstance(self.config["track"], int) or self.config["track"] not in [1, 2, 3]:
            raise ValueError("Argument 'track' must be one of 1, 2, 3.")

        # Create saving directory
        folder_path = "./save" if not test else "./save_test"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
    def _preprocess(
        self,
        train_responses: pd.DataFrame,
        test_prompts: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess the input data based on configuration.
        
        Args:
            train_responses (pd.DataFrame): Training responses dataset
            test_prompts (pd.DataFrame): Test prompts dataset
            
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Preprocessed train_responses and test_prompts
        """
        
        # Remove stopwords if configured
        if self.config["remove_stopwords"]:
            self.logger.info("Removing stopwords...")
            
            stop_words = set(stopwords.words('english'))
            
            def clean_sentence(sentence):
                """Remove stopwords from a sentence"""
                word_tokens = word_tokenize(str(sentence))
                filtered_words = [word for word in word_tokens if word.lower() not in stop_words]
                return ' '.join(filtered_words)
            
            train_responses["user_prompt"] = train_responses["user_prompt"].apply(clean_sentence)
            test_prompts["user_prompt"] = test_prompts["user_prompt"].apply(clean_sentence)
        
        return train_responses, test_prompts

    def retrieve(
        self,
        train_responses: pd.DataFrame,
        test_prompts: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Main retrieval method that routes to the appropriate track-specific method.
        
        Args:
            train_responses (pd.DataFrame): Training responses dataset with columns:
                - conversation_id: ID of the conversation
                - user_prompt: The prompt text
                - model_response: The response to return
            test_prompts (pd.DataFrame): Test prompts dataset with columns:
                - conversation_id: ID of the conversation
                - user_prompt: The prompt text to match
                
        Returns:
            pd.DataFrame: Retrieved responses with columns:
                - conversation_id: ID from test_prompts
                - response_id: ID of the retrieved response
                - conversation_prompt: Original prompt from test_prompts
                - response_prompt: Retrieved prompt from train_responses
                - model_response: Retrieved response
                
        Raises:
            ValueError: If arguments are invalid or results are incomplete
            AttributeError: If required columns are missing
        """
        # Validate input data types
        if not isinstance(train_responses, pd.DataFrame) or not isinstance(
            test_prompts, pd.DataFrame
        ):
            raise ValueError("Invalid arguments types.")

        # Validate required columns exist
        if (
            "conversation_id" not in train_responses.columns
            or "user_prompt" not in train_responses.columns
            or "model_response" not in train_responses.columns
            or "conversation_id" not in test_prompts.columns
            or "user_prompt" not in test_prompts.columns
        ):
            raise AttributeError("One of the args is missing a required attribute.")

        # Preprocess data
        train_responses, test_prompts = self._preprocess(
            train_responses,
            test_prompts
        )

        # Route to appropriate track method
        if self.config["track"] == 1:
            ans = self.retrieve_track_1(
                train_responses,
                test_prompts
            )
        elif self.config["track"] == 2:
            ans = self.retrieve_track_2(
                train_responses,
                test_prompts
            )
        elif self.config["track"] == 3:
            ans = self.retrieve_track_3(
                train_responses,
                test_prompts
            )
        
        # Validate output format
        if (
            "conversation_id" not in ans.columns
            or "response_id" not in ans.columns
            or "conversation_prompt" not in ans.columns
            or "response_prompt" not in ans.columns
            or "model_response" not in ans.columns
        ):
            raise ValueError("Missing required column in ans.")
        
        # Ensure all test prompts have retrieved responses
        if (
            ans.shape[0] != test_prompts.shape[0] or
            ans.isna().any().any()
        ):
            raise ValueError("Missing retrieved responses")
        
        return ans
    
    def retrieve_track_1(
        self,
        train_responses: pd.DataFrame,
        test_prompts: pd.DataFrame
    ) -> pd.DataFrame:
        """
        TF-IDF based retrieval (Track 1).
        
        Supports three TF-IDF strategies:
        - 'tf-idf-w': Word-based TF-IDF
        - 'tf-idf-c': Character-based TF-IDF
        - 'tf-idf-wc': Combined word and character TF-IDF
        
        Args:
            train_responses (pd.DataFrame): Training responses dataset
            test_prompts (pd.DataFrame): Test prompts dataset
            
        Returns:
            pd.DataFrame: Retrieved responses
            
        Raises:
            ValueError: If strategy is invalid for track 1
        """
        
        # Validate strategy for track 1
        if not re.match(r"tf-idf-[w|c|wc]", self.config["strategy"]):
            raise ValueError(f"Invalid strategy {self.config['strategy']} for track {self.config['track']}.")

        # Compute embeddings
        self.logger.info("Computing embeddings...")
        
        # Initialize empty arrays to store embeddings
        train_embeddings = np.empty((train_responses.shape[0], 0))
        test_embeddings = np.empty((test_prompts.shape[0], 0))

        # Add word-based TF-IDF features if enabled
        if "w" in self.config["strategy"]:
            ns = map(int, list(str(self.config["w_n_grams"])))
            for n in ns:
                vectorizer = TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(n, n),
                    min_df=self.config["min_df"],  # Use n-grams appearing in at least min_df documents
                    max_df=self.config["max_df"],  # Ignore terms appearing in more than max_df documents
                    stop_words="english",
                    sublinear_tf=True,  # Replace tf with 1 + log(tf)
                )
                # Fit vectorizer on training data and transform
                train_word_embeddings = vectorizer.fit_transform(
                    train_responses.user_prompt.to_list()
                ).toarray()
                
                # Transform test data using the same vectorizer
                test_word_embeddings = vectorizer.transform(
                    test_prompts.user_prompt.to_list()
                ).toarray()
                
                # Concatenate with existing embeddings
                train_embeddings = np.concatenate([train_embeddings, train_word_embeddings], axis=1)
                test_embeddings = np.concatenate([test_embeddings, test_word_embeddings], axis=1)
        
        # Add character-based TF-IDF features if enabled
        if "c" in self.config["strategy"]:
            ns = map(int, list(str(self.config["c_n_grams"])))
            for n in ns:
                vectorizer = TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(n, n),
                    min_df=self.config["min_df"],
                    max_df=self.config["max_df"],
                    stop_words="english",
                    sublinear_tf=True,  # Replace tf with 1 + log(tf)
                )
                # Fit vectorizer on training data and transform
                train_char_embeddings = vectorizer.fit_transform(
                    train_responses.user_prompt.to_list()
                ).toarray()
                
                # Transform test data using the same vectorizer
                test_char_embeddings = vectorizer.transform(
                    test_prompts.user_prompt.to_list()
                ).toarray()
                
                # Concatenate with existing embeddings
                train_embeddings = np.concatenate([train_embeddings, train_char_embeddings], axis=1)
                test_embeddings = np.concatenate([test_embeddings, test_char_embeddings], axis=1)
                
                self.logger.debug(vectorizer.get_feature_names_out())
        
        # Note: Dimensionality reduction with PCA is commented out
        # Uncomment if needed for high-dimensional TF-IDF vectors
        # pca = PCA(
        #     n_components=768,
        #     whiten=True,  # If true, uncorrelated outputs with unit component-wise variances
        # )
        # train_embeddings = pca.fit_transform(train_embeddings)
        # test_embeddings = pca.transform(test_embeddings)

        # Calculate similarity between embeddings
        self.logger.info("Computing similarity...")
        sim = cosine_similarity_matrix(
            train_embeddings,
            test_embeddings,
            logger=self.logger,
        )

        # Build output dataframe with retrieved responses
        ans = self._build_ans(
            train_responses=train_responses,
            test_prompts=test_prompts,
            sim=sim,
            bm25=False
        )

        return ans
        
    def _build_ans(
            self,
            train_responses: pd.DataFrame,
            test_prompts: pd.DataFrame,
            sim: np.ndarray = None,
            bm25: bool = False,
        ) -> pd.DataFrame:
        """
        Build the answer dataframe with retrieved responses.
        
        Args:
            train_responses (pd.DataFrame): Training responses dataset
            test_prompts (pd.DataFrame): Test prompts dataset
            sim (np.ndarray, optional): Similarity matrix between train and test
            bm25 (bool, optional): Whether to use BM25 for retrieval
            
        Returns:
            pd.DataFrame: DataFrame with retrieved responses
            
        Raises:
            ValueError: If neither sim nor bm25 is provided
        """
        # Initialize BM25 model if needed
        bm25_model = None
        if bm25:
            bm25_model = BM25Okapi(
                [doc.split(" ") for doc in train_responses.user_prompt.to_list()]
            )

        # Initialize output dataframe
        ans = pd.DataFrame(
            {
                "conversation_id": test_prompts.conversation_id,
                "response_id": "",
                "conversation_prompt": test_prompts.user_prompt,
                "response_prompt": "",  # The retrieved prompt
                "model_response": "",   # The retrieved response
            }
        )

        self.logger.info("Retrieving responses...")
        for i in range(ans.shape[0]):
            
            # Case 1: Use similarity matrix only
            if isinstance(sim, np.ndarray) and not bm25:
                most_similar_ix = np.argmax(sim[:, i])

            # Case 2: Use BM25 only
            elif not isinstance(sim, np.ndarray) and bm25_model:
                if i % 1000 == 0:
                    self.logger.debug(f"{i:05} / {ans.shape[0]}")
                tokenized_query = ans.loc[i, "conversation_prompt"].split(" ")
                scores = bm25_model.get_scores(tokenized_query)
                most_similar_ix = np.argmax(scores)

            # Case 3: Combine BM25 and similarity matrix (hybrid approach)
            elif isinstance(sim, np.ndarray) and bm25_model:
                # Get BM25 scores and find top-k candidates
                tokenized_query = ans.loc[i, "conversation_prompt"].split(" ")
                scores = bm25_model.get_scores(tokenized_query)
                top_k_ixs = np.argsort(scores)[-self.config["top_k_bm25"]:][::-1] 
                
                # Choose most similar from top BM25 candidates using similarity matrix
                most_similar_ix = top_k_ixs[np.argmax(sim[top_k_ixs, i])]
            
            else:
                raise ValueError("Either sim or bm25 must be not null.")

            # Set retrieved response information
            ans.loc[i, "response_id"] = train_responses.loc[
                most_similar_ix, "conversation_id"
            ]
            ans.loc[i, "response_prompt"] = train_responses.loc[
                most_similar_ix, "user_prompt"
            ]

            # In non-test mode, also include the model response
            if not self.test:
                ans.loc[i, "model_response"] = train_responses.loc[
                    most_similar_ix, "model_response"
                ]

        return ans

    def retrieve_track_2(
        self,
        train_responses: pd.DataFrame,
        test_prompts: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Embedding-based retrieval (Track 2) using Doc2Vec or FastText.
        
        Args:
            train_responses (pd.DataFrame): Training responses dataset
            test_prompts (pd.DataFrame): Test prompts dataset
            
        Returns:
            pd.DataFrame: Retrieved responses
            
        Raises:
            ValueError: If strategy is invalid for track 2
        """
        # Get embeddings based on selected strategy
        if self.config["strategy"] == "doc2vec":
            train_embeddings, test_embeddings = self._get_doc2vec_embeddings(
                train_responses,
                test_prompts
            )
        elif self.config["strategy"] == "fasttext":
            train_embeddings, test_embeddings = self._get_fasttext_embeddings(
                train_responses, 
                test_prompts
            )
        else:
            raise ValueError(f"Invalid strategy {self.config['strategy']} for track {self.config['track']}.")

        # Compute similarity between embeddings
        self.logger.info("Computing similarity...")
        sim = cosine_similarity_matrix(
            train_embeddings,
            test_embeddings,
            logger=self.logger,
        )

        # Build output dataframe with retrieved responses
        ans = self._build_ans(
            train_responses=train_responses,
            test_prompts=test_prompts,
            sim=sim,
            bm25=False
        )

        return ans

    def _get_doc2vec_embeddings(
        self,
        train_responses: pd.DataFrame,
        test_prompts: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get Doc2Vec embeddings for prompts.
        
        Args:
            train_responses (pd.DataFrame): Training responses dataset
            test_prompts (pd.DataFrame): Test prompts dataset
            
        Returns:
            tuple[np.ndarray, np.ndarray]: Train and test embeddings
        """
        # Create a list of TaggedDocument objects for Doc2Vec training
        corpus = [
            TaggedDocument(words=prompt.lower().split(), tags=[str(i)])
            for i, prompt in enumerate(train_responses.user_prompt.to_list())
        ]

        # Define model path based on configuration
        d2v_model_path = (
            f"./{'save' if not self.test else 'save_test'}/doc2vec"
            + "_"
            + str(self.config["vector_size"])
            + "_"
            + str(self.config["epochs"])
            + "_"
            + str(self.config["window"])
            + ".model"
        )

        # Load existing model or train a new one
        if os.path.exists(d2v_model_path):
            d2v_model = Doc2Vec.load(d2v_model_path)
        else:
            # Initialize model with configuration parameters
            d2v_model = Doc2Vec(
                vector_size=self.config["vector_size"],
                epochs=self.config["epochs"],
                window=self.config["window"],
                hs=0,                  # Use negative sampling
                sample=0.01,           # Downsampling threshold for frequent words
                negative=10,           # Number of negative samples
                min_count=10,          # Ignore words with frequency below this
                workers=cpu_count(),   # Use all available CPU cores
                dm=0,                  # Training algorithm: 0 = distributed bag of words
                dbow_words=0,          # Whether to train word vectors
                callbacks=(),          # Avoid logging
            )
            # Build vocabulary and train model
            d2v_model.build_vocab(corpus)
            d2v_model.train(
                corpus, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs
            )
            # Save model for future use
            d2v_model.save(d2v_model_path)

        # Get embeddings for train and test data
        train_embeddings = np.array(
            [
                d2v_model.dv[str(i)]
                for i in range(len(train_responses.user_prompt.to_list()))
            ]
        )
        test_embeddings = np.array(
            [
                d2v_model.infer_vector(prompt.lower().split())
                for prompt in test_prompts.user_prompt.to_list()
            ]
        )

        return train_embeddings, test_embeddings

    def _get_fasttext_embeddings(
        self,
        train_responses: pd.DataFrame,
        test_prompts: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get FastText embeddings for prompts.
        
        Args:
            train_responses (pd.DataFrame): Training responses dataset
            test_prompts (pd.DataFrame): Test prompts dataset
            
        Returns:
            tuple[np.ndarray, np.ndarray]: Train and test embeddings
        """
        # Use pretrained model or train a new one
        if self.config["fasttext_pretrained"]:
            # Load pretrained FastText model from Hugging Face
            model = fasttext.load_model(
                hf_hub_download(
                    repo_id="facebook/fasttext-en-vectors", filename="model.bin"
                )
            )

            def get_vector(word, model):
                """Get word vector from pretrained model"""
                return model.get_word_vector(word)
        
        else:  # Train a new FastText model
            # Tokenize sentences for training
            tokenized_sentences = [sentence.split() for sentence in train_responses.user_prompt.tolist()]
            
            # Initialize and train FastText model
            model = FastText(
                sentences=tokenized_sentences,
                vector_size=self.config["vector_size"],
                window=self.config["window"],
                min_count=self.config["min_tf"],
                workers=cpu_count(),
                epochs=self.config["epochs"],
                min_n=self.config["min_n"],        # Min length of char n-grams
                max_n=self.config["max_n"],        # Max length of char n-grams
                sg=1,                              # Use skipgram model
                word_ngrams=1,                     # Use n-grams
                negative=15,                       # Number of negative samples
                sample=1e-4                        # Threshold for downsampling
            )  

            def get_vector(word, model):
                """Get word vector from trained model"""
                return model.wv.get_vector(word)
        
        def get_sentence_embedding(sentence, model):
            """
            Compute a sentence embedding by averaging normalized word vectors.
            
            Args:
                sentence (str): Input sentence
                model: FastText model
                
            Returns:
                np.ndarray: Sentence embedding vector
            """
            words = sentence.lower().split()
            word_vectors = np.array(
                [get_vector(word, model) for word in words]
            )
            
            # Normalize word vectors and take the mean
            norms = np.linalg.norm(word_vectors, axis=1) + 1e-8  # Avoid division by zero
            word_vectors = word_vectors / norms[..., None]
            return np.mean(word_vectors, axis=0)

        # Get embeddings for train and test data
        train_embeddings = np.array(
            [
                get_sentence_embedding(i, model)
                for i in train_responses.user_prompt.to_list()
            ]
        )
        test_embeddings = np.array(
            [
                get_sentence_embedding(i, model)
                for i in test_prompts.user_prompt.to_list()
            ]
        )

        return train_embeddings, test_embeddings

    def retrieve_track_3(
        self,
        train_responses: pd.DataFrame,
        test_prompts: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Advanced retrieval (Track 3) using transformer models or BM25.
        
        Args:
            train_responses (pd.DataFrame): Training responses dataset
            test_prompts (pd.DataFrame): Test prompts dataset
            
        Returns:
            pd.DataFrame: Retrieved responses
            
        Raises:
            ValueError: If strategy is invalid for track 3
        """
        # Route to appropriate strategy implementation
        if self.config["strategy"] == "dynamic":
            return self._track_3_dynamic(
                train_responses,
                test_prompts
            )
        elif self.config["strategy"] == "bm25":
            return self._track_3_bm25(
                train_responses,
                test_prompts
            )
        else:
            raise ValueError(f"Invalid strategy {self.config['strategy']} for track {self.config['track']}.")           

    def _track_3_dynamic(
        self,
        train_responses: pd.DataFrame,
        test_prompts: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Implement dynamic (transformer-based) retrieval for track 3.
        
        Args:
            train_responses (pd.DataFrame): Training responses dataset
            test_prompts (pd.DataFrame): Test prompts dataset
            
        Returns:
            pd.DataFrame: Retrieved responses
        """
        # Get transformer-based embeddings
        train_embeddings, test_embeddings = self._get_dynamic_embeddings(
            train_responses, 
            test_prompts
        )

        # Compute similarity between embeddings
        self.logger.info("Computing similarity...")
        sim = cosine_similarity_matrix(
            train_embeddings,
            test_embeddings,
            logger=self.logger,
        )
        
        # Filter by language if configured
        if self.config["filter_language"]:
            # Apply language mask to zero out cross-language similarities
            sim *= self._get_languge_mask(
                train_responses,
                test_prompts
            )
        
        # Build output dataframe with retrieved responses
        ans = self._build_ans(
            train_responses=train_responses,
            test_prompts=test_prompts,
            sim=sim,
            bm25=self.config["top_k_bm25"]  # Use hybrid approach if top_k_bm25 > 0
        )

        return ans

    def _track_3_bm25(
        self,
        train_responses: pd.DataFrame,
        test_prompts: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Implement BM25-only retrieval for track 3.
        
        Args:
            train_responses (pd.DataFrame): Training responses dataset
            test_prompts (pd.DataFrame): Test prompts dataset
            
        Returns:
            pd.DataFrame: Retrieved responses
        """
        # Build output dataframe using only BM25 for retrieval
        ans = self._build_ans(
            train_responses=train_responses,
            test_prompts=test_prompts,
            sim=None,
            bm25=True
        )

        return ans

    def _get_dynamic_embeddings(
            self,
            train_responses: pd.DataFrame,
            test_prompts: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get transformer-based embeddings using SentenceTransformer models.
        
        Args:
            train_responses (pd.DataFrame): Training responses dataset
            test_prompts (pd.DataFrame): Test prompts dataset
            
        Returns:
            tuple[np.ndarray, np.ndarray]: Train and test embeddings
        """
        # Define file paths for cached embeddings
        train_embeddings_path = (
            f"./{'save' if not self.test else 'save_test'}/"
            + "train_embeddings_"
            + re.sub("[^A-Za-z0-9]", "_", self.config["embedding_model"])
            + ".npy"
        )
        test_embeddings_path = (
            f"./{'save' if not self.test else 'save_test'}/"
            + "test_embeddings_"
            + re.sub("[^A-Za-z0-9]", "_", self.config["embedding_model"])
            + ".npy"
        )

        # Load model if embeddings need to be computed
        if (not os.path.exists(train_embeddings_path) or
            not os.path.exists(test_embeddings_path)):
            model = SentenceTransformer(self.config["embedding_model"])

        # Compute or load train embeddings
        if not os.path.exists(train_embeddings_path):
            train_embeddings = model.encode(
                train_responses.user_prompt.to_list()
            )
            np.save(train_embeddings_path, train_embeddings)
        else:
            train_embeddings = np.load(train_embeddings_path)

        # Compute or load test embeddings
        if not os.path.exists(test_embeddings_path):
            test_embeddings = model.encode(test_prompts.user_prompt.to_list())
            np.save(test_embeddings_path, test_embeddings)
        else:
            test_embeddings = np.load(test_embeddings_path)

        return train_embeddings, test_embeddings
    
    def _get_languge_mask(
            self,
            train_responses: pd.DataFrame,
            test_prompts: pd.DataFrame,
    ) -> np.ndarray:
        """
        Create a binary mask where matches are only allowed for same-language pairs.
        
        Uses FastText language identification model to detect languages.
        
        Args:
            train_responses (pd.DataFrame): Training responses dataset
            test_prompts (pd.DataFrame): Test prompts dataset
            
        Returns:
            np.ndarray: Boolean mask of shape (n_train, n_test) where True indicates
                        same language between train and test samples
        """
        # Load language identification model
        model = fasttext.load_model(
            hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
        )
        
        # Detect language for training samples (1 for English, 0 otherwise)
        train_language = np.zeros(train_responses.shape[0])
        for i, prompt in enumerate(train_responses.user_prompt.tolist()):
            label, prob = model.predict(prompt)
            train_language[i] = label[0] == "__label__eng_Latn"
        
        # Detect language for test samples (1 for English, 0 otherwise)
        test_language = np.zeros(test_prompts.shape[0])
        for i, prompt in enumerate(test_prompts.user_prompt.tolist()):
            label, prob = model.predict(prompt)
            test_language[i] = label[0] == "__label__eng_Latn"
        
        # Reshape to allow broadcasting during multiplication
        train_language = train_language.reshape(-1, 1)
        test_language = test_language.reshape(1, -1)
        
        # Create mask where True indicates same language
        return train_language == test_language