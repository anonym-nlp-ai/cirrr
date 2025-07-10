from datetime import datetime
import os
import random
import json
from typing import List
from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer
from loguru import logger


class MedMCQASampler:
    def __init__(
        self,
        n_samples: int,
        token_threshold_min: int = 200,
        token_threshold_max: int = 500,
        output_dir: str = "./data/medmcqa",
    ):
        self.n_samples = n_samples
        self.token_threshold_min = token_threshold_min
        self.token_threshold_max = token_threshold_max
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
        self.dataset = None
        self.df = None
        self.output_dir = output_dir.rstrip("/")

    def load_dataset(self):
        logger.info("Loading MedMCQA dataset from Hugging Face...")
        self.dataset = load_dataset("openlifescienceai/medmcqa", split="train")
        self.df = self.dataset.to_pandas()
        logger.success(f"Dataset loaded with {len(self.df)} rows.")

    def compute_token_lengths(self, column_name: str):
        logger.info("Computing token lengths for all rows...")
        tokenized_sizes = []

        for idx, row in self.df.iterrows():
            text = row[column_name]
            if not isinstance(text, str):
                logger.warning(f"Skipping row {idx} due to invalid text: {text}")
                tokenized_sizes.append(0)
                continue

            encoded = self.tokenizer(
                text, padding=True, truncation=False, return_tensors="pt"
            )
            token_count = encoded.input_ids.size(1)
            tokenized_sizes.append(token_count)

        self.df["tokenized_size"] = tokenized_sizes

    def filter_and_sample(self):
        logger.info(f"Filtering rows with tokenized size > {self.token_threshold_min} and < {self.token_threshold_max}...")
        filtered_df = self.df[
            (self.df["tokenized_size"] > self.token_threshold_min)
            & (self.df["tokenized_size"] < self.token_threshold_max)
        ]
        logger.info(f"Found {len(filtered_df)} rows after filtering.")

        if len(filtered_df) < self.n_samples:
            logger.warning(
                f"Requested {self.n_samples} samples, but only {len(filtered_df)} available. Sampling all."
            )
            self.df = filtered_df.reset_index(drop=True)
        else:
            self.df = filtered_df.sample(n=self.n_samples, random_state=42).reset_index(
                drop=True
            )

        self.df["max_seq_len_exceeded"] = (
            self.df["tokenized_size"] > self.token_threshold_max
        )

    def export_to_json(self, output_file: str) -> str:
        logger.info(f"Exporting results to {output_file}...")
        result: List[dict] = []

        for docid, row in self.df.iterrows():
            result.append(
                {
                    "id": row["id"],
                    "docid": docid,
                    "doc": row["exp"],
                    "tokenized_size": row["tokenized_size"],
                    "max_seq_len_exceeded": row["max_seq_len_exceeded"],
                }
            )

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # create the output file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"{self.output_dir}/{output_file}_{timestamp}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.success(f"Exported {len(result)} items to {output_file}.")

        return file_path
