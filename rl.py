import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch import Tensor
from transformers import GPT2Config
from torch.distributions import Categorical
from modelscope.msdatasets import MsDataset
from modelscope.hub.api import HubApi
from modelscope import snapshot_download
from utils import TunesFormer, Patchilizer, DEVICE, APP_KEY
from generate import infer_abc
from config import *


class MusicGenEnv:
    def __init__(self, subset: str):
        self.prompts = self.prepare_prompts()
        self.current_index = 0
        self.subset = subset

    def prepare_prompts(self):
        HubApi().login(APP_KEY)
        ds = MsDataset.load(
            f"monetjoe/{DATASET}",
            subset_name=self.subset,
            cache_dir=TEMP_DIR,
            trust_remote_code=True,
        )
        dataset = list(ds["train"]) + list(ds["test"])
        prompt_set = set("A:Q1\n", "A:Q2\n", "A:Q3\n", "A:Q4\n", "")
        for item in dataset:
            prompt_set.add(f"A:{item['label']}\n{item['prompt']}\n")
            prompt_set.add(f"{item['prompt']}\n")

        return list(prompt_set)

    def reward_fn(self, action: str):  # action = generated_abc
        # 定义你的 reward 计算函数
        print(action)
        return np.random.rand()  # 示例：随机奖励

    def reset(self):
        self.current_index = 0
        return self.prompts[self.current_index]

    def step(self, action: str):
        reward = self.reward_fn(action)
        self.current_index += 1
        done = self.current_index >= len(self.prompts)
        next_prompt = self.prompts[self.current_index] if not done else None
        return next_prompt, reward, done


class PPOTrainer:
    def __init__(
        self,
        env: MusicGenEnv,
        vf_coef=0.5,
        lamda_kl=0.5,
        clip_param=0.2,
        lr=1e-5,
    ):
        self.env = env
        self.patchilizer = Patchilizer()
        self.init_model = self.load_model().eval()
        self.tuned_model = self.load_model(f"{OUTPUT_PATH}/weights.pth")
        self.vf_coef = vf_coef
        self.lamda_kl = lamda_kl
        self.clip_param = clip_param
        self.optimizer = optim.Adam(self.tuned_model.parameters(), lr=lr)
        self.mse = nn.MSELoss()

    def load_model(
        self,
        weights_path=snapshot_download("MuGeminorum/tunesformer", cache_dir=TEMP_DIR)
        + "/weights.pth",
    ):
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

        checkpoint = torch.load(weights_path, weights_only=False)
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])

        return model.to(DEVICE)

    def compute_kl(self, old_logits, new_logits):
        old_dist = Categorical(logits=old_logits)
        new_dist = Categorical(logits=new_logits)
        kl: Tensor = torch.distributions.kl_divergence(old_dist, new_dist)
        return kl.mean()

    def update(self, states, actions, rewards, old_logits, old_values: Tensor):
        self.tuned_model.train()

        new_logits: Tensor = self.tuned_model(**states).logits
        new_dist = Categorical(logits=new_logits)
        old_dist = Categorical(logits=old_logits)

        ratios = torch.exp(new_dist.log_prob(actions) - old_dist.log_prob(actions))
        advantages = rewards - old_values.detach()

        surr1: Tensor = ratios * advantages
        surr2: Tensor = (
            torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * advantages
        )
        policy_loss = -torch.min(surr1, surr2).mean()

        new_values = new_logits.mean(dim=-1)  # 假设新值是logits的均值
        value_loss = self.mse(new_values, rewards)

        kl_div = self.compute_kl(old_logits, new_logits)
        loss: torch.Tensor = (
            policy_loss + value_loss * self.vf_coef + kl_div * self.lamda_kl
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, num_epochs=100):
        for _ in tqdm(range(num_epochs), desc="Training PPO..."):
            states, actions, rewards, old_logits, old_values = [], [], [], [], []
            state = self.env.reset()
            done = False

            while not done:
                with torch.no_grad():
                    action, old_logit = infer_abc(
                        prompt=state,
                        patchilizer=self.patchilizer,
                        model=self.tuned_model,
                    )
                    old_value, _ = infer_abc(
                        prompt=state,
                        patchilizer=self.patchilizer,
                        model=self.init_model,
                    )

                next_state, reward, done = self.env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                old_logits.append(old_logit)
                old_values.append(old_value)

                state = next_state

            states = {key: torch.cat([s[key] for s in states]) for key in states[0]}
            actions = torch.cat(actions)
            rewards = Tensor(rewards, dtype=torch.float32)
            old_logits = torch.cat(old_logits)
            old_values = torch.cat(old_values)

            self.update(states, actions, rewards, old_logits, old_values)


if __name__ == "__main__":
    env = MusicGenEnv("VGMIDI")
    ppo_trainer = PPOTrainer(env)
    ppo_trainer.train(num_epochs=100)
