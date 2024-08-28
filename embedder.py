import os
import json
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from modelscope.hub.api import HubApi
from modelscope.msdatasets import MsDataset
from sklearn.svm import LinearSVC
from tqdm import tqdm
from utils import APP_KEY, TEMP_DIR, OUTPUT_PATH


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(6, 128)  # 输入层到第一个隐藏层
        self.fc2 = nn.Linear(128, 64)  # 第一个隐藏层到第二个隐藏层
        self.fc3 = nn.Linear(64, 4)  # 第二个隐藏层到输出层
        self.dropout = nn.Dropout(0.5)  # Dropout层，丢弃比例为50%
        self.relu = nn.ReLU()  # ReLU激活函数

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def data():
    HubApi().login(APP_KEY)
    ds = MsDataset.load(
        "monetjoe/EMusicGen",
        subset_name="Analysis",
        split="train",
        cache_dir=f"{TEMP_DIR}/cache",
        trust_remote_code=True,
    )
    dataset = list(ds)
    p90 = int(len(dataset) * 0.9)
    trainset = dataset[:p90]
    testset = dataset[p90:]
    x_train, y_train, x_test, y_test = [], [], [], []
    for item in tqdm(trainset, desc="Loading trainset..."):
        x_train.append(
            [
                item["mode"],
                item["pitch"],
                item["range"],
                item["variation"],
                item["tempo"],
                item["volume"],
            ]
        )

        y_train.append(item["label"])

    for item in tqdm(testset, desc="Loading testset..."):
        x_test.append(
            [
                item["mode"],
                item["pitch"],
                item["range"],
                item["variation"],
                item["tempo"],
                item["volume"],
            ]
        )

        y_test.append(item["label"])

    return x_train, y_train, x_test, y_test


def svm(x_train, y_train, x_test, y_test):
    clf = LinearSVC(loss="hinge", random_state=42).fit(x_train, y_train)
    print(f"\nAcc: {clf.score(x_test, y_test) * 100.0}%")


def dnn(
    x_train,
    y_train,
    x_test,
    y_test,
    epochs=40,
    iter=10,
    learning_rate=0.001,
    bsz=1,
):
    wkspace = f"{OUTPUT_PATH}/embedder"
    os.makedirs(wkspace, exist_ok=True)
    if os.path.exists(f"{wkspace}/logs.jsonl"):
        os.remove(f"{wkspace}/logs.jsonl")

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    model = DNN()
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_test = x_test.cuda()
        y_test = y_test.cuda()
        model = model.cuda()
        criterion = criterion.cuda()

    # 创建 Dataset 对象
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    # 创建 DataLoader 对象
    train_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=bsz, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_eval_acc = 0.0
    for ep in range(epochs):
        model.train()
        loss_list = []
        running_loss = 0.0
        lr: float = optimizer.param_groups[0]["lr"]
        with tqdm(total=len(train_loader), unit="batch") as pbar:
            for i, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss: torch.Tensor = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % iter == iter - 1:
                    pbar.set_description(
                        f"Training: ep={ep + 1}/{epochs}, lr={lr}, loss={round(running_loss / iter, 4)}"
                    )
                    loss_list.append(running_loss / iter)

                running_loss = 0.0
                pbar.update(1)

        # 验证阶段
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f"Evaluating on testset..."):
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        eval_acc = correct / total
        print(f"Accuracy: {100 * eval_acc}%")
        with open(f"{wkspace}/logs.jsonl", "a", encoding="utf-8") as file:
            json_line = json.dumps(
                {
                    "epoch": ep + 1,
                    "loss": sum(loss_list) / len(loss_list),
                    "acc": eval_acc,
                }
            )
            file.write(json_line + "\n")

        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            torch.save(model.state_dict(), f"{wkspace}/weights.pth")
            print("Model saved.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    x_train, y_train, x_test, y_test = data()
    dnn(x_train, y_train, x_test, y_test)
    svm(x_train, y_train, x_test, y_test)
