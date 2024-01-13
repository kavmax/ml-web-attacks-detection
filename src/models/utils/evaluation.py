import evaluate
import pandas as pd

from datasets import Dataset as HuggingFaceDataset
from evaluate import evaluator


def evaluate_model(df_val: pd.DataFrame, model, tokenizer):
    try:
        print("hashes: ", hash(model), hash(tokenizer))
    except Exception as e:
        print(e)

    x_val, y_val = df_val["text"], df_val["benign"]

    evaluate_data = HuggingFaceDataset.from_dict({"texts": x_val, "labels": y_val})

    task_evaluator = evaluator("text-classification")
    eval_results = task_evaluator.compute(
        model_or_pipeline=model,
        tokenizer=tokenizer,
        data=evaluate_data,
        input_column="texts",
        label_column="labels",
        metric=evaluate.combine(["accuracy", "recall", "precision", "f1"]),
        label_mapping={"LABEL_0": 0, "LABEL_1": 1}
    )

    return eval_results
