import os

import pandas as pd
import torch
import modal
import dotenv

from src.models.ModalConfig import ModalConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.models.utils.evaluation import evaluate_model


stub = ModalConfig.stub

# load for local training
dotenv.load_dotenv(".env")


@stub.function(
    image=ModalConfig.image,
    gpu=modal.gpu.T4(),
    mounts=[
        modal.Mount.from_local_dir(
            local_path=os.getenv("LOCAL_PROCESSED_DATA_DIR"),
            remote_path=os.getenv("MODAL_PROCESSED_DATA_DIR"),
        ),
        modal.Mount.from_local_dir(
            local_path=os.getenv("LOCAL_MODELS_DIR"),
            remote_path=os.getenv("MODAL_MODELS_DIR"),
        ),
    ],
    secrets=[
        modal.Secret.from_name("git-secret"),
        modal.Secret.from_dotenv(".env"),
    ],
    timeout=60*60*20,
)
def run_predict(model_name: str, val_data_path: str, tokenizer_name: str):
    models_save_dir = os.getenv("LOCAL_MODELS_DIR") if modal.is_local() \
        else os.getenv("MODAL_MODELS_DIR")
    processed_data_base_path = os.getenv("LOCAL_PROCESSED_DATA_DIR") if modal.is_local() \
        else os.getenv("MODAL_PROCESSED_DATA_DIR")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(models_save_dir + tokenizer_name)
    tokenizer.model_max_length = 400

    model = AutoModelForSequenceClassification.from_pretrained(models_save_dir + model_name).to(device)

    eval_results = evaluate_model(
        pd.read_csv(processed_data_base_path + val_data_path), model, tokenizer
    )
    print("eval_results:", eval_results)


@stub.local_entrypoint()
def main():
    run_predict.remote(
        model_name="bert-tiny-4.39m-nosql-e1-extended",
        tokenizer_name="bert-tiny-4.39m-nosql-e1-extended",
        val_data_path="01-06-2024-full-balanced-nosql.csv",
    )
