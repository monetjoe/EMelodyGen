import os
import json
import time
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from modelscope.msdatasets import MsDataset
from modelscope import snapshot_download
from tqdm import tqdm
from transformers import GPT2Config, get_scheduler
from utils import Patchilizer, TunesFormer, PatchilizedData, DEVICE
from config import *


def init():
    random.seed(42)
    batch_size = min(torch.cuda.device_count(), 4)
    patchilizer = Patchilizer()
    patch_config = GPT2Config(
        num_hidden_layers=PATCH_NUM_LAYERS,
        max_length=PATCH_LENGTH,
        max_position_embeddings=PATCH_LENGTH,
        vocab_size=1,
    )
    char_config = GPT2Config(
        num_hidden_layers=CHAR_NUM_LAYERS,
        max_length=PATCH_SIZE,
        max_position_embeddings=PATCH_SIZE,
        vocab_size=128,
    )
    model: nn.Module = TunesFormer(patch_config, char_config, SHARE_WEIGHTS).to(DEVICE)
    # print parameter number
    print(
        f"Parameter Number: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    scaler = GradScaler()
    is_autocast = True
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    return batch_size, patchilizer, model, scaler, is_autocast, optimizer


def collate_batch(batch):
    input_patches = []

    for input_patch in batch:
        input_patches.append(input_patch.reshape(-1))

    input_patches = nn.utils.rnn.pad_sequence(
        input_patches, batch_first=True, padding_value=0
    )

    return input_patches.to(DEVICE)


def process_one_batch(batch, model):  # call model with a batch of input
    input_patches = batch
    loss: torch.Tensor = model(input_patches).loss
    return loss.mean()


def train_epoch(
    model: nn.Module,
    optimizer: optim.AdamW,
    lr_scheduler: optim.lr_scheduler.LambdaLR,
    is_autocast: bool,
    scaler: GradScaler,
    train_set: DataLoader,
):  # do one epoch for training
    tqdm_train_set = tqdm(train_set)
    total_train_loss = 0
    iter_idx = 1
    model.train()
    if hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()

    for batch in tqdm_train_set:
        try:
            if is_autocast:
                with autocast(device_type=DEVICE):
                    loss = process_one_batch(batch, model)

                if loss == None or torch.isnan(loss).item():
                    continue

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                loss = process_one_batch(batch, model)
                if loss == None or torch.isnan(loss).item():
                    continue

                loss.backward()
                optimizer.step()

        except RuntimeError as exception:
            if "memory" in str(exception):
                print(str(exception))
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()

                continue

            else:
                raise exception

        lr_scheduler.step()
        model.zero_grad(set_to_none=True)
        total_train_loss += loss.item()
        tqdm_train_set.set_postfix({"train_loss": total_train_loss / iter_idx})
        iter_idx += 1

    return total_train_loss / (iter_idx - 1)


def eval_epoch(model: nn.Module, eval_set):  # do one epoch for eval
    tqdm_eval_set = tqdm(eval_set)
    total_eval_loss = 0
    iter_idx = 1
    model.eval()

    # Evaluate data for one epoch
    for batch in tqdm_eval_set:
        with torch.no_grad():
            loss = process_one_batch(batch, model)
            if loss == None or torch.isnan(loss).item():
                continue

            total_eval_loss += loss.item()

        tqdm_eval_set.set_postfix({"eval_loss": total_eval_loss / iter_idx})
        iter_idx += 1

    return total_eval_loss / (iter_idx - 1)


if __name__ == "__main__":
    # load data
    dataset = MsDataset.load(
        f"monetjoe/{DATASET}",
        subset_name=SUBSET,
        cache_dir=TEMP_DIR,
    )
    trainset, evalset = [], []
    for song in dataset["train"]:
        trainset.append(
            {
                "control code": "A:" + song["label"] + "\n" + song["prompt"],
                "abc notation": song["data"],
            }
        )

    for song in dataset["test"]:
        evalset.append(
            {
                "control code": "A:" + song["label"] + "\n" + song["prompt"],
                "abc notation": song["data"],
            }
        )

    batch_size, patchilizer, model, scaler, is_autocast, optimizer = init()

    trainset = DataLoader(
        PatchilizedData(trainset, patchilizer),
        batch_size=batch_size,
        collate_fn=collate_batch,
        shuffle=True,
    )

    evalset = DataLoader(
        PatchilizedData(evalset, patchilizer),
        batch_size=batch_size,
        collate_fn=collate_batch,
        shuffle=True,
    )

    lr_scheduler: optim.lr_scheduler.LambdaLR = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=NUM_EPOCHS * len(trainset) // 10,
        num_training_steps=NUM_EPOCHS * len(trainset),
    )

    if LOAD_FROM_CHECKPOINT:
        tunesformer_weights_path = (
            snapshot_download("MuGeminorum/tunesformer", cache_dir=TEMP_DIR)
            + "/weights.pth"
        )
        checkpoint = torch.load(tunesformer_weights_path, weights_only=False)
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])

        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_sched"])
        pre_epoch = checkpoint["epoch"]
        best_epoch = checkpoint["best_epoch"]
        min_eval_loss = checkpoint["min_eval_loss"]
        print("Successfully Loaded Checkpoint from Epoch %d" % pre_epoch)

    else:
        pre_epoch = 0
        best_epoch = 0
        min_eval_loss = 100

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    for epoch in range(1, NUM_EPOCHS + 1 - pre_epoch):
        epoch += pre_epoch
        print(f"{'-' * 21}Epoch {str(epoch)}{'-' * 21}")
        train_loss = train_epoch(
            model,
            optimizer,
            lr_scheduler,
            is_autocast,
            scaler,
            trainset,
        )
        eval_loss = eval_epoch(model, evalset)
        with open(LOG_PATH, "a", encoding="utf-8") as jsonl_file:
            jsonl_file.write(
                json.dumps(
                    {
                        "epoch": int(epoch),
                        "train_loss": float(train_loss),
                        "eval_loss": float(eval_loss),
                        "time": f"{time.asctime(time.localtime(time.time()))}",
                    }
                )
                + "\n"
            )

        if eval_loss < min_eval_loss:
            best_epoch = epoch
            min_eval_loss = eval_loss
            torch.save(
                {
                    "model": (
                        model.module.state_dict()
                        if torch.cuda.device_count() > 1
                        else model.state_dict()
                    ),
                    "optimizer": optimizer.state_dict(),
                    "lr_sched": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "min_eval_loss": min_eval_loss,
                    "time_stamp": time.strftime(
                        "%a_%d_%b_%Y_%H_%M_%S", time.localtime()
                    ),
                },
                f"{OUTPUT_PATH}/weights.pth",
            )
            break

    print(f"Best Eval Epoch : {str(best_epoch)}\nMin Eval Loss : {str(min_eval_loss)}")
