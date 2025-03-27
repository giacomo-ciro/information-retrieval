import os
import re
import json
from logging import Logger
from multiprocessing import cpu_count

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import Doc2Vec, FastText
from gensim.models.doc2vec import TaggedDocument
import fasttext
from huggingface_hub import hf_hub_download

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from utils import cosine_similarity_matrix


class Retriever:
    def __init__(
            self,
            logger: Logger = None,
            test: bool=False
            ):
        
        self.test = test
        self.logger = logger
        with open("config.json", "r") as f:
            self.config = json.load(f)
        
        if not isinstance(self.config["track"], int) or self.config["track"] not in [1, 2, 3]:
            raise ValueError("Argument 'track' must be one of 1, 2, 3.")

        # Init saving folders
        folder_path = "./save" if not test else "./save_test"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
    def _preprocess(
        self,
        train_responses: pd.DataFrame,
        test_prompts: pd.DataFrame,
    ) -> pd.DataFrame:
        
        # Stop words
        if self.config["remove_stopwords"]:

            self.logger.info("Removing stopwords...")
            
            stop_words = set(stopwords.words('english'))
            
            def clean_sentence(sentence):
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
        # Check assumptions
        if not isinstance(train_responses, pd.DataFrame) or not isinstance(
            test_prompts, pd.DataFrame
        ):
            raise ValueError("Invalid arguments types.")

        if (
            "conversation_id" not in train_responses.columns
            or "user_prompt" not in train_responses.columns
            or "model_response" not in train_responses.columns
            or "conversation_id" not in test_prompts.columns
            or "user_prompt" not in test_prompts.columns
        ):
            raise AttributeError("One of the args in missing a required attributed.")

        # Preprocess
        train_responses, test_prompts = self._preprocess(
            train_responses,
            test_prompts
        )

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
        
        if (
            "conversation_id" not in ans.columns
            or "response_id" not in ans.columns
            or "conversation_prompt" not in ans.columns
            or "response_prompt" not in ans.columns
            or "model_response" not in ans.columns
        ):
            raise ValueError("Missing required column in ans.")
        
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
        
        if not re.match(r"tf-idf-[w|c|wc]", self.config["strategy"]):
            raise ValueError(f"Invalid strategy {self.config["strategy"]} for track {self.config["track"]}.")

        # -------- Compute Embeddings
        self.logger.info("Computing embeddings...")
        
        # To store the embeddings
        train_embeddings = np.empty(
            (train_responses.shape[0], 0)
        )
        test_embeddings = np.empty(
            (test_prompts.shape[0], 0)
        )

        # Estimate TF-IDF with words
        if "w" in  self.config["strategy"]:
            ns = map(int, list(str(self.config["w_n_grams"])))
            for n in ns:
                vectorizer = TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(n,n),
                    min_df=self.config["min_df"],       # use n-grams appearing in at least 1 document
                    max_df=self.config["max_df"],
                    stop_words="english",
                    sublinear_tf=True,  # replace tf with 1 + log(tf)
                )
                train_embeddings = np.concatenate(
                    [
                        train_embeddings,
                        vectorizer.fit_transform(
                            train_responses.user_prompt.to_list()
                        ).toarray()
                    ], axis = 1
                )
                test_embeddings = np.concatenate(
                    [
                        test_embeddings,
                        vectorizer.transform(
                            test_prompts.user_prompt.to_list()
                        ).toarray()
                    ], axis = 1
                )          
        
        # TF-IDF with subwords
        if "c" in  self.config["strategy"]:
            ns = map(int, list(str(self.config["c_n_grams"])))
            for n in ns:
                vectorizer = TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(n,n),
                    min_df=self.config["min_df"],           # with 0.01 bleu is 0.084
                    max_df=self.config["max_df"],
                    stop_words="english",
                    sublinear_tf=True,  # replace tf with 1 + log(tf)
                )
                train_embeddings = np.concatenate(
                    [
                        train_embeddings,
                        vectorizer.fit_transform(
                            train_responses.user_prompt.to_list()
                        ).toarray()
                    ], axis = 1
                )
                test_embeddings = np.concatenate(
                    [
                        test_embeddings,
                        vectorizer.transform(
                            test_prompts.user_prompt.to_list()
                        ).toarray()
                    ], axis = 1
                )  
                self.logger.debug(vectorizer.get_feature_names_out())
        # Reduce dimensionality
        # pca = PCA(
        #     n_components=768,
        #     whiten=True,        # if true, uncorrelated outputs with unit component-wise variances
        # )
        # train_embeddings = pca.fit_transform(train_embeddings)
        # test_embeddings = pca.transform(test_embeddings)

        # ------------ Retrieve most similar response
        self.logger.info("Computing similarity...")
        sim = cosine_similarity_matrix(
            train_embeddings,
            test_embeddings,
            logger=self.logger,
        )

        # Build output dataframe
        ans = self._build_ans(
            train_responses = train_responses,
            test_prompts = test_prompts,
            sim = sim,
            bm25 = False
        )

        return ans
        
    def _build_ans(
            self,
            train_responses: pd.DataFrame,
            test_prompts: pd.DataFrame,
            sim: np.ndarray = None,
            bm25: bool = False,
        )-> pd.DataFrame:

        # Initiliaze BM25
        if bm25:
            bm25 = BM25Okapi(
                [doc.split(" ") for doc in train_responses.user_prompt.to_list()]
            )

        ans = pd.DataFrame(
            {
                "conversation_id": test_prompts.conversation_id,
                "response_id": "",
                "conversation_prompt": test_prompts.user_prompt,
                "response_prompt": "",  # the retrieved one
                "model_response": "",  # what we are interested in
            }
        )

        self.logger.info("Retrieving responses...")
        for i in range(ans.shape[0]):
            
            # Sim only
            if isinstance(sim, np.ndarray) and not bm25:
                most_similar_ix = np.argmax(sim[:, i])

            # BM25 only
            elif not isinstance(sim, np.ndarray) and bm25:
                if i % 1000 == 0:
                    self.logger.debug(f"{i:05} / {ans.shape[0]}")
                tokenized_query = ans.loc[i, "conversation_prompt"].split(" ")
                scores = bm25.get_scores(tokenized_query)
                most_similar_ix = np.argmax(scores)

            # BM25 + Sim
            elif isinstance(sim, np.ndarray) and bm25:
                # Get BM25 Scores
                tokenized_query = ans.loc[i, "conversation_prompt"].split(" ")
                scores = bm25.get_scores(tokenized_query)
                top_k_ixs = np.argsort(scores)[-self.config["top_k_bm25"]:][::-1] 
                # Most similar from subset
                most_similar_ix = top_k_ixs[np.argmax(sim[top_k_ixs, i])]
            
            else:
                raise ValueError("Either sim or bm25 must be not null.")

            ans.loc[i, "response_id"] = train_responses.loc[
                most_similar_ix, "conversation_id"
            ]
            ans.loc[i, "response_prompt"] = train_responses.loc[
                most_similar_ix, "user_prompt"
            ]

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
        if self.config["strategy"] == "doc2vec":
            train_embeddings, test_embeddings = self._get_doc2vec_embeddings(
                train_responses,
                test_prompts
        )
        elif self.config["strategy"] == "fasttext":
            train_embeddings, test_embeddings = self._get_fasttext_embeddings(
                train_responses, test_prompts
            )
        else:
            raise ValueError(f"Invalid strategy {self.config["strategy"]} for track {self.config["track"]}.")

        

        # Compute similarity
        self.logger.info("Computing similarity...")
        sim = cosine_similarity_matrix(
            train_embeddings,
            test_embeddings,
            logger=self.logger,
        )

        # Build output dataframe
        ans = self._build_ans(
            train_responses = train_responses,
            test_prompts = test_prompts,
            sim = sim,
            bm25 = False
        )

        return ans

    def _get_doc2vec_embeddings(
        self,
        train_responses: pd.DataFrame,
        test_prompts: pd.DataFrame
    ) -> tuple[np.ndarray]:
        # create a list of TaggedDocument objects
        corpus = [
            TaggedDocument(words=prompt.lower().split(), tags=[str(i)])
            for i, prompt in enumerate(train_responses.user_prompt.to_list())
        ]

        # Check existence of model
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

        # train or load the model
        if os.path.exists(d2v_model_path):
            d2v_model = Doc2Vec.load(d2v_model_path)
        else:
            # initialize model
            d2v_model = Doc2Vec(
                vector_size=self.config["vector_size"],
                epochs=self.config["epochs"],
                window=self.config["window"],
                hs=0,
                sample=0.01,
                negative=10,
                min_count=10,
                workers=cpu_count(),
                dm=0,  # training algo, 1=distributed memory, 0 = distributed bag of words
                dbow_words=0,  # bool, train also word vecs
                callbacks=(),  # to avoid logging
            )
            d2v_model.build_vocab(corpus)
            d2v_model.train(
                corpus, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs
            )
            d2v_model.save(d2v_model_path)

        # Get embeddings
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
    ) -> tuple[np.ndarray]:
        # TODO: test on desktop (not enough RAM to load pretrained embeddings on laptopt)

        if self.config["fasttext_pretrained"]:
            
            model = fasttext.load_model(
                hf_hub_download(
                    repo_id="facebook/fasttext-en-vectors", filename="model.bin"
                )
            )

            def get_vector(word, model):
                return model.get_word_vector(word)
        
        else: # Train model

            tokenized_sentences = [sentence.split() for sentence in train_responses.user_prompt.tolist()]
            
            model = FastText(
                sentences=tokenized_sentences,
                vector_size=self.config["vector_size"],
                window=self.config["window"],
                min_count=self.config["min_tf"],
                workers=cpu_count(),
                epochs=self.config["epochs"],
                min_n=self.config["min_n"],        # Min length of char ngrams
                max_n=self.config["max_n"],         # Max length of char ngrams
                sg=1,                               # skipgram
                word_ngrams=1,
                negative=15,
                sample=1e-4
            )  

            def get_vector(word, model):
                return model.wv.get_vector(word)
        
        def get_sentence_embedding(sentence, model):
            words = sentence.lower().split()
            word_vectors = np.array(
                [get_vector(word, model) for word in words]
            )
            
            # Normalize and take the mean
            norms = np.linalg.norm(word_vectors, axis = 1) + 1e-8
            word_vectors = word_vectors / norms[...,None]
            return np.mean(word_vectors, axis=0)

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
        if self.config["strategy"] == "dynamic":
            return self._track_3_dynamic(
                train_responses,
                test_prompts
            )

        if self.config["strategy"] == "bm25":
            return self._track_3_bm25(
                train_responses,
                test_prompts
            )

        else:
            raise ValueError(f"Invalid strategy {self.config["strategy"]} for track {self.config["track"]}.")           

    def _track_3_dynamic(
        self,
        train_responses: pd.DataFrame,
        test_prompts: pd.DataFrame,
    ) -> tuple[np.ndarray]:
        
        train_embeddings, test_embeddings = self._get_dynamic_embeddings(
            train_responses, 
            test_prompts
        )

        # ------------ Retrieve most similar response
        self.logger.info("Computing similarity...")
        sim = cosine_similarity_matrix(
            train_embeddings,
            test_embeddings,
            logger=self.logger,
        )
        
        # Zero out non-same language similarities
        if self.config["filter_language"]:
            sim *= self._get_languge_mask(
                train_responses,
                test_prompts
            )
        
        # Build output dataframe
        ans = self._build_ans(
            train_responses = train_responses,
            test_prompts = test_prompts,
            sim = sim,
            bm25 = self.config["top_k_bm25"]
        )

        return ans

    def _track_3_bm25(
        self,
        train_responses: pd.DataFrame,
        test_prompts: pd.DataFrame,
    ) -> tuple[np.ndarray]:
        
        # Build output dataframe
        ans = self._build_ans(
            train_responses = train_responses,
            test_prompts = test_prompts,
            sim = False,
            bm25 = self.config["top_k_bm25"]
        )

        return ans

    def _get_dynamic_embeddings(
            self,
            train_responses: pd.DataFrame,
            test_prompts: pd.DataFrame
    )-> np.ndarray:
        
        # -------------- Get Dynamic Embeddings
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

        # Load model
        if (not os.path.exists(train_embeddings_path) or
            not os.path.exists(train_embeddings_path)):
            model = SentenceTransformer(self.config["embedding_model"])

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

        return train_embeddings, test_embeddings
    
    def _get_languge_mask(
            self,
            train_responses: pd.DataFrame,
            test_prompts: pd.DataFrame,
    )->np.ndarray:
        """
        Returns a boolean array where i-th entry is 1 if the i-th prompt is in english.
        """
        
        model = fasttext.load_model(
            hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
        )
        
        train_language = np.zeros(train_responses.shape[0])
        
        for i, prompt in enumerate(train_responses.user_prompt.tolist()):
            label, prob = model.predict(prompt)
            train_language[i] = label[0] == "__label__eng_Latn"
        
        test_language = np.zeros(test_prompts.shape[0])

        for i, prompt in enumerate(test_prompts.user_prompt.tolist()):
            label, prob = model.predict(prompt)
            test_language[i] = label[0] == "__label__eng_Latn"
        
        train_language = train_language.reshape(-1, 1)

        test_language = test_language.reshape(1, -1)

        return train_language == test_language
