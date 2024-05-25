import json
import numpy as np
import matplotlib.pyplot as plt
from utils import *

plt.rcParams["font.sans-serif"] = "Times New Roman"


def show_point(max_id, list):
    show_max = "(" + str(max_id + 1) + ", " + str(round(list[max_id], 2)) + ")"
    plt.annotate(
        show_max,
        xytext=(max_id + 1, list[max_id]),
        xy=(max_id + 1, list[max_id]),
        fontsize=6,
    )


def plot_loss(tra_acc_list, val_acc_list):
    x_acc = []
    for i in range(len(tra_acc_list)):
        x_acc.append(i + 1)

    x = np.array(x_acc)
    y1 = np.array(tra_acc_list)
    y2 = np.array(val_acc_list)
    min1 = np.argmin(y1)
    min2 = np.argmin(y2)

    plt.title("Loss of training and evaluation", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(x, y1, label="Training")
    plt.plot(x, y2, label="Evaluation")
    plt.plot(1 + min1, y1[min1], "r-o")
    plt.plot(1 + min2, y2[min2], "r-o")
    show_point(min1, y1)
    show_point(min2, y2)
    plt.legend()
    plt.xticks(range(1, len(tra_acc_list) + 1))
    plt.show()


def save_loss(tra_acc_list, val_acc_list, save_path="./output"):
    x_acc = []
    for i in range(len(tra_acc_list)):
        x_acc.append(i + 1)

    x = np.array(x_acc)
    y1 = np.array(tra_acc_list)
    y2 = np.array(val_acc_list)
    min1 = np.argmin(y1)
    min2 = np.argmin(y2)

    plt.title("Loss of training and evaluation", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(x, y1, label="Training")
    plt.plot(x, y2, label="Evaluation")
    plt.plot(1 + min1, y1[min1], "r-o")
    plt.plot(1 + min2, y2[min2], "r-o")
    show_point(min1, y1)
    show_point(min2, y2)
    plt.legend()
    plt.xticks(range(1, len(tra_acc_list) + 1))
    plt.savefig(save_path + "/loss.jpg", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    train_loss_list = []  # 用于存储第一个键的值
    eval_loss_list = []  # 用于存储第二个键的值

    # 打开并读取JSONL文件
    with open("./output/logs.jsonl", "r", encoding="utf-8") as file:
        for line in file:
            # 解析JSON对象
            obj = json.loads(line)
            # 根据键提取值，若键不存在则添加默认值None
            train_loss = float(obj.get("train_loss"))
            eval_loss = float(obj.get("eval_loss"))
            # 将值添加到对应的列表中
            train_loss_list.append(train_loss)
            eval_loss_list.append(eval_loss)

    save_loss(train_loss_list, eval_loss_list)
