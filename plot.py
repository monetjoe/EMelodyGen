import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from tqdm import tqdm
from config import *

plt.rcParams["font.sans-serif"] = "Times New Roman"


def most_common_elements(input_list: list):
    count = Counter(input_list)
    max_count = max(count.values())
    return [key for key, value in count.items() if value == max_count]


def plot_confusion_matrix(
    cm: np.ndarray,
    exp_name: str,
    labels_name=["Q1", "Q2", "Q3", "Q4"],
    fontsize: int = 18,
):
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # Normalized
    _, ax = plt.subplots()
    cax = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = plt.colorbar(cax, ax=ax)
    cbar.ax.tick_params(labelsize=fontsize)
    num_local = np.array(range(len(labels_name)))
    for i in range(len(labels_name)):
        for j in range(len(labels_name)):
            plt.text(
                j,
                i,
                format(cm[i, j], ".2f"),
                horizontalalignment="center",
                color="black" if cm[i, j] <= 0.5 else "white",
                fontsize=fontsize,
            )
    plt.xticks(num_local, labels_name, rotation=45, fontsize=fontsize)
    plt.yticks(num_local, labels_name, fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f"{EXPERIMENT_DIR}/mat-{exp_name}.jpg", bbox_inches="tight")
    plt.savefig(f"{EXPERIMENT_DIR}/mat-{exp_name}.pdf", bbox_inches="tight")
    plt.close()


def plots(exp_json: str, classes=["Q1", "Q2", "Q3", "Q4"]):
    exps = {}
    with open(exp_json, "r", encoding="utf-8") as file:
        data_dict = json.load(file)

    for key in tqdm(data_dict, desc="Loading experiment data..."):
        exp = key.split("/")[0]
        y_true = key.split("]")[0][-2:]
        y_pred = data_dict[key]
        if exp in exps:
            exps[exp][0].append(y_true)
            exps[exp][1].append(y_pred)

        else:
            exps[exp] = [[y_true], [y_pred]]

    for exp in tqdm(exps, desc="Plotting confusion matrix..."):
        y_trues, y_preds = exps[exp][0], exps[exp][1]
        report = classification_report(y_trues, y_preds, target_names=classes, digits=3)
        with open(f"{EXPERIMENT_DIR}/report-{exp}.log", "w", encoding="utf-8") as f:
            f.write(report)

        cm = confusion_matrix(y_trues, y_preds, normalize="all")
        plot_confusion_matrix(cm, exp)


def merge_data(exp_jsons: list):
    exps = {}
    for exp_json in tqdm(exp_jsons, desc="Loading jsons..."):
        with open(exp_json, "r", encoding="utf-8") as file:
            data_dict = json.load(file)

        for key in data_dict:
            if key in exps:
                exps[key].append(data_dict[key])

            else:
                exps[key] = [data_dict[key]]

    disputes = []
    keeps = {}
    for exp in tqdm(exps, desc="Searching most common elements..."):
        exps[exp] = most_common_elements(exps[exp])
        if len(exps[exp]) > 1:
            prompt = exp.split("]")[0][-2:]
            if prompt in exps[exp]:
                keeps[exp] = prompt

            else:
                disputes.append(f"{EXPERIMENT_DIR}/{exp}")
                print(f"{exp} : {exps[exp]}")

        else:
            keeps[exp] = exps[exp][0]

    print(len(disputes))
    with open(f"{EXPERIMENT_DIR}/survey_disputes.json", "w", encoding="utf-8") as file:
        json.dump(disputes, file, ensure_ascii=False, indent=4)

    with open(f"{EXPERIMENT_DIR}/survey_keeps.json", "w", encoding="utf-8") as file:
        json.dump(keeps, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # merge_data(
    #     [
    #         f"{EXPERIMENT_DIR}/survey_20240819_085035.json",
    #         f"{EXPERIMENT_DIR}/survey_20240819_121603.json",
    #         f"{EXPERIMENT_DIR}/survey_20240819_172916.json",
    #     ]
    # )
    plots("./exps/survey_keeps.json")
