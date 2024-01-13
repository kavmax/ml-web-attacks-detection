import os

import pandas as pd
import torch
import modal
import dotenv

from src.models.ModalConfig import ModalConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline


stub = ModalConfig.stub

dotenv.load_dotenv(".env")


@stub.function(
    image=ModalConfig.image,
    gpu=ModalConfig.device,
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
        modal.Secret.from_dotenv("../../.env"),
    ],
)
def run_predict(model_name: str, text: str, tokenizer_name: str):
    models_save_dir = os.getenv("LOCAL_MODELS_DIR") if modal.is_local() \
        else os.getenv("MODAL_MODELS_DIR")
    processed_data_base_path = os.getenv("LOCAL_PROCESSED_DATA_DIR") if modal.is_local() \
        else os.getenv("MODAL_PROCESSED_DATA_DIR")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(models_save_dir + tokenizer_name)
    tokenizer.model_max_length = 512

    model = AutoModelForSequenceClassification.from_pretrained(models_save_dir + model_name).to(device)

    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
    print(classifier.predict(text))


@stub.local_entrypoint()
def main():
    run_predict.local(
        model_name="bert-tiny-4.39m-sql-v2",
        tokenizer_name="bert-tiny-4.39m-sql-v2",
        text=[
            """
POST /evidenceclean/ HTTP/1.1
User-Agent: Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8 (like Gecko)
Pragma: no-cache
Cache-control: no-cache
Accept: text/xml,application/xml,application/xhtml+xml,text/html;q=0.9,text/plain;q=0.8,image/png,*/*;q=0.5
Accept-Encoding: x-gzip, x-deflate, gzip, deflate
Accept-Charset: utf-8, utf-8;q=0.5, *;q=0.5
Accept-Language: en
Host: localhost:8080
Content-Type: application/x-www-form-urlencoded
Connection: close
Content-Length: 60

            """
        ]
    )
