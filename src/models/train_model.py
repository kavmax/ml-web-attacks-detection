import os

import torch
import modal
import dotenv
import pandas as pd

from src.models.ModalConfig import ModalConfig
from src.models.ModelDataset import ModelDataset
from src.models.utils.evaluation import evaluate_model
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification


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
        )
    ],
    secrets=[
        modal.Secret.from_name("git-secret"),
        modal.Secret.from_dotenv(".env"),
    ],
    network_file_systems={
        os.getenv("MODAL_MODELS_NFS_DIR"): modal.NetworkFileSystem.persisted(
            os.getenv("MODAL_SAVE_MODELS_TO_NFS_NAME")),
        os.getenv("MODAL_CACHE_DATA_DIR"): modal.NetworkFileSystem.persisted(
            os.getenv("MODAL_SAVE_CACHE_TO_NFS_NAME")),
    },
    timeout=60*60*20,
    memory=32000,
)
def run_train(
        wandb_run_name: str, model_name_to_save: str,
        train_filename: str, test_filename: str, val_filename: str,
        num_train_epochs: int,
        model_name: str = None, from_custom_trained_model_name: str = None,
):
    import wandb

    wandb.login(key=os.getenv("WANDB_KEY"))
    wandb.init(
        project=os.getenv("WAND_PROJECT_NAME"),
        name=wandb_run_name,
    )

    models_save_dir = os.getenv("LOCAL_MODELS_DIR") if modal.is_local() \
        else os.getenv("MODAL_MODELS_DIR")
    processed_data_base_path = os.getenv("LOCAL_PROCESSED_DATA_DIR") if modal.is_local() \
        else os.getenv("MODAL_PROCESSED_DATA_DIR")
    cache_data_base_path = os.getenv("LOCAL_CACHE_DATA_DIR") if modal.is_local() \
        else os.getenv("MODAL_CACHE_DATA_DIR")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("/", os.listdir("/"))
    print("/data/processed", os.listdir("/data/processed"))
    print(models_save_dir, os.listdir(models_save_dir))
    print(cache_data_base_path, os.listdir(cache_data_base_path))
    if not modal.is_local():
        print(os.getenv("MODAL_MODELS_NFS_DIR"), os.listdir(os.getenv("MODAL_MODELS_NFS_DIR")))

    if from_custom_trained_model_name:
        model_name = models_save_dir + from_custom_trained_model_name
        print("Using saved model from:", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, batched=True)
    tokenizer.model_max_length = 400

    print(tokenizer.is_fast)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    print("Loading data")
    test_dataset = ModelDataset(
        tokenizer=tokenizer,
        dataset_path=processed_data_base_path + test_filename,
        cache_load_from_path=cache_data_base_path + test_filename + ".pth",
        cache_save_to_path=cache_data_base_path + test_filename + ".pth",
        shuffle=True,
    )
    print("Loaded test data!")
    train_dataset = ModelDataset(
        tokenizer=tokenizer,
        dataset_path=processed_data_base_path + train_filename,
        cache_load_from_path=cache_data_base_path + train_filename + ".pth",
        cache_save_to_path=cache_data_base_path + train_filename + ".pth",
        shuffle=True,
    )
    print("Data loaded")

    training_args = TrainingArguments(
        output_dir=models_save_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        learning_rate=5e-5,
        weight_decay=0.03,
        logging_dir="/logs",
        logging_steps=5,
        report_to=["wandb"],  # enable logging to W&B
    )

    print("Start training (choose params)")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    print("Start training")
    trainer.train()
    model.eval()

    # save trained model
    tokenizer.save_pretrained(os.getenv("MODAL_MODELS_NFS_DIR") + model_name_to_save)
    model.save_pretrained(os.getenv("MODAL_MODELS_NFS_DIR") + model_name_to_save)

    evaluate_results = evaluate_model(pd.read_csv(processed_data_base_path + val_filename), model, tokenizer)
    print("evaluate_results:", evaluate_results)

    wandb.log(evaluate_results)
    wandb.finish()


@stub.local_entrypoint()
def main():
    epochs = [1]

    for epoch in epochs:
        model_name = f"bert-tiny-4.39m-nosql-e{epoch}-extended"

        run_train.remote(
            wandb_run_name=model_name,
            model_name_to_save=model_name,

            # model_name="mrm8488/bert-tiny-finetuned-sms-spam-detection",
            from_custom_trained_model_name="bert-tiny-4.39m-nosql-e50/",

            train_filename="01-10-2024-complete-full-balanced-nosql-train.csv",
            test_filename="01-10-2024-complete-full-balanced-nosql-test.csv",
            val_filename="01-10-2024-complete-full-balanced-nosql-val.csv",

            num_train_epochs=epoch,
        )

