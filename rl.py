import os
import torch
import random
from torch.optim import Adam
from transformers import GPT2Config
from modelscope.msdatasets import MsDataset
from utils import TunesFormer, Patchilizer, download, DEVICE
from generate import infer_abc
from config import *


class PPOTrainer:
    def __init__(
        self,
        model: TunesFormer,
        patchilizer: Patchilizer,
        lr=1e-5,
    ):
        self.model = model
        self.patchilizer = patchilizer
        self.optimizer = Adam(self.model.parameters(), lr=lr)

    def _rewards(self, generated_abc):
        # TODO: Placeholder - Reward computation logic
        rewards = [1.0] * len(generated_abc)
        return torch.tensor(rewards)

    def _str2tensor(self, input_str: str):
        # 将字符串转换成张量
        tensor = torch.tensor([float(char) for char in input_str])
        return tensor

    def train(self, prompts, epochs=500):
        for epoch in range(epochs):
            for i, prompt in enumerate(prompts):
                # Generate outputs
                with torch.no_grad():
                    generated_abc = infer_abc(prompt, self.patchilizer, self.model)

                # Compute rewards
                rewards = self._rewards(generated_abc)

                # Compute policy loss
                logits = self.model(prompt)
                log_probs = torch.log_softmax(logits, dim=-1)
                target_ids = self._str2tensor(generated_abc)[:, 1:].reshape(-1)
                log_probs = log_probs[:, :-1, :].reshape(-1, log_probs.size(-1))
                log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
                policy_loss = -(log_probs * rewards).mean()

                # Optimize model
                self.optimizer.zero_grad()
                policy_loss.backward()
                self.optimizer.step()

                print(
                    f"Epoch {epoch + 1}/{epochs}, Batch {i + 1}/{len(prompts)}, Loss: {policy_loss.item()}"
                )


def init_model():
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

    model = TunesFormer(patch_config, char_config, share_weights=SHARE_WEIGHTS)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Move model to GPU if available
    if not os.path.exists(WEIGHT_PATH):
        download()

    checkpoint = torch.load(WEIGHT_PATH)

    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint["model"])

    return model.to(DEVICE), Patchilizer()


if __name__ == "__main__":
    # Initialize TunesFormer model and tokenizer
    model, patchilizer = init_model()
    # Initialize PPO trainer for TunesFormer
    ppo_trainer = PPOTrainer(model, patchilizer)
    # load prompts from the dataset
    trainset = MsDataset.load(f"monetjoe/{DATASET}", split="train")
    evalset = MsDataset.load(f"monetjoe/{DATASET}", split="test")
    prompts = set("A:Q1\n", "A:Q2\n", "A:Q3\n", "A:Q4\n", "")
    for item in list(trainset) + list(evalset):
        prompts.add("A:" + item["label"] + "\n" + item["prompt"] + "\n")
        prompts.add(item["prompt"] + "\n")

    prompts = list(prompts)
    random.shuffle(prompts)
    # Train the model
    ppo_trainer.train(prompts)
