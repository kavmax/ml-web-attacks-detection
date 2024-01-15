from typing import List

import evaluate
import pandas as pd
from sklearn.metrics import confusion_matrix

from datasets import Dataset as HuggingFaceDataset
from evaluate import evaluator
from transformers import pipeline


def calc_confusion_matrix(prediction_labels: List, true_labels: List):
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(
        y_true=true_labels,
        y_pred=prediction_labels
    )
    print(conf_matrix.ravel())
    print(conf_matrix)

    # Extract values from confusion matrix
    TN, FP, FN, TP = conf_matrix.ravel()

    # Calculate sensitivity (recall)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    accuracy = (TP + TN) / (TP + FN + FP + TN)
    f1 = (2 * precision * sensitivity) / (precision + sensitivity)
    balanced_accuracy = (sensitivity + specificity) / 2

    # Now, you have TN, FP, FN, TP, and sensitivity.
    print("True Negatives count:", TN)
    print("False Positives count:", FP)
    print("False Negatives count:", FN)
    print("True Positives count:", TP)
    print("------------------")
    print("Sensitivity (Recall):", sensitivity)
    print("Specificity:", specificity)
    print("Balanced Accuracy (mean of Sensitivity and Specificity):", balanced_accuracy)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("F1 score:", f1)


def evaluate_model_confusion(df_val: pd.DataFrame, model, tokenizer, device):
    x_val, y_val = df_val["text"], df_val["benign"]
    runtime_errors = 0

    classifier = pipeline(
        "text-classification", model=model, tokenizer=tokenizer, device=device)

    pred_labels = []
    y_val_included = []

    for idx, text in enumerate(x_val):
        try:
            classifier_res = classifier(text)

            pred_labels.append(
                0 if classifier_res[0]["label"] == "LABEL_0" else 1
            )
            y_val_included.append(y_val[idx])
            if idx % 20000 == 0:
                print("Runtime_errors:", runtime_errors, "idx:", idx)
        except RuntimeError:
            runtime_errors += 1

    calc_confusion_matrix(
        prediction_labels=pred_labels,
        true_labels=y_val_included,
    )

    print("Not included samples:", runtime_errors)


def evaluate_model(df_val: pd.DataFrame, model, tokenizer):
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
