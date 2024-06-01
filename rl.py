import os
import torch
import numpy as np
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from utils import TunesFormer, Patchilizer, download, DEVICE
from config import *


class RewardFunction:
    def __init__(self, reference_text, model_name_or_path):
        self.reference_tokens = GPT2Tokenizer.from_pretrained(
            model_name_or_path
        ).encode(reference_text, return_tensors="pt")
        self.model_name_or_path = model_name_or_path
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

    def __call__(self, generated_text):
        generated_tokens = GPT2Tokenizer.from_pretrained(
            self.model_name_or_path
        ).encode(generated_text, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(input_ids=generated_tokens, labels=generated_tokens)
            loss = outputs[0]

        perplexity = torch.exp(loss)
        reward = -perplexity.item()
        return reward


class AbcGenEnv:
    def __init__(self, model_name_or_path: str, max_length: int):
        patch_config = GPT2Config(
            num_hidden_layers=PATCH_NUM_LAYERS,
            max_length=max_length,
            max_position_embeddings=max_length,
            vocab_size=1,
        )
        char_config = GPT2Config(
            num_hidden_layers=CHAR_NUM_LAYERS,
            max_length=max_length,
            max_position_embeddings=max_length,
            vocab_size=128,
        )
        self.patchilizer = Patchilizer()
        model = TunesFormer(patch_config, char_config, model_name_or_path)
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

        self.model = model.to(DEVICE)
        self.max_length = max_length
        self.current_patch = ""
        self.current_length = 0
        self.reward_fn = RewardFunction(
            reference_text="The quick brown fox jumps over the lazy dog",
            model_name_or_path="gpt2",
        )

    def reset(self):
        self.current_patch = ""
        self.current_length = 0
        return ""

    def step(self, action):
        action_token = self.patchilizer.decode([action])
        self.current_patch += action_token

        obs = self.current_patch[-self.max_length :]
        obs = obs if len(obs) > 0 else " "  # Ensure the observation is never empty

        reward = self.reward_fn(self.current_patch)
        self.current_length += 1
        done = self.current_length >= self.max_length

        return obs, reward, done, {}


class Tunedformer(nn.Module):
    def __init__(self, model_name_or_path, num_actions=512):
        super(Tunedformer, self).__init__()
        patch_config = GPT2Config(
            num_hidden_layers=PATCH_NUM_LAYERS,
            max_length=PATCH_SIZE,
            max_position_embeddings=PATCH_SIZE,
            vocab_size=1,
        )
        char_config = GPT2Config(
            num_hidden_layers=CHAR_NUM_LAYERS,
            max_length=PATCH_SIZE,
            max_position_embeddings=PATCH_SIZE,
            vocab_size=128,
        )
        self.config = char_config  # TODO: patch_config or char_config ?
        self.patchilizer = Patchilizer()
        model = TunesFormer(patch_config, char_config, model_name_or_path)
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

        self.tunesformer = model.to(DEVICE)
        self.action_layer = nn.Linear(self.config.n_embd, num_actions)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        tunesformer_outputs = self.tunesformer(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden_states = tunesformer_outputs[0]
        # Calculate attention weights using the additional component
        action = self.action_layer(hidden_states)
        action_probs = self.softmax(action)

        outputs = (action_probs,) + tunesformer_outputs[1:]

        if labels is not None:
            # Calculate the loss with the labels provided
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                outputs[0].view(-1, self.config.vocab_size), labels.view(-1)
            )
            outputs = (loss,) + outputs

        return outputs


class RolloutStorage:
    def __init__(self, max_length):
        self.observations = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.dones = []
        self.max_length = max_length

    def store(self, obs, action, action_probs, reward, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.action_probs.append(action_probs)
        self.rewards.append(reward)
        self.dones.append(done)

    def store_last_observation(self, value):
        self.observations.append(value)

    def clear(self):
        self.observations = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.dones = []

    def batch_by_indices(self, indices):
        obs_batch = [self.observations[i] for i in indices]
        action_batch = [self.actions[i] for i in indices]
        action_prob_batch = [self.action_probs[i] for i in indices]
        advantage_batch = [self.advantages[i] for i in indices]
        return_batch = [self.returns[i] for i in indices]
        return obs_batch, action_batch, action_prob_batch, advantage_batch, return_batch

    def __len__(self):
        return len(self.actions)


class PPOTrainer:
    def __init__(
        self,
        env: AbcGenEnv,
        model: nn.Module,
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-5,
        gamma=0.99,
        clip_param=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        num_epochs=10,
        batch_size=64,
    ):
        self.env = env
        self.model = model
        self.gamma = gamma
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.num_epochs = num_epochs  # Add this line
        self.batch_size = batch_size

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=betas, eps=eps
        )

    def train(self, num_steps):
        storage = RolloutStorage(self.env.max_length)
        obs = self.env.reset()
        done = True  # Add this line to initialize the 'done' variable

        for _ in range(num_steps):
            if done:
                obs = self.env.reset()

            for _ in range(self.env.max_length):
                obs_tensor = torch.tensor(
                    self.env.patchilizer.encode(obs), dtype=torch.long
                ).unsqueeze(0)
                if obs_tensor.shape[-1] == 0:
                    continue

                action_probs_tensor, value_tensor = self.model(obs_tensor)
                action_probs = action_probs_tensor.squeeze(0).detach().numpy()
                action = np.random.choice(len(action_probs), p=action_probs)

                next_obs, reward, done, _ = self.env.step(action)
                storage.store(obs, action, action_probs, reward, done)
                obs = next_obs

                if done:
                    break

            if not done:
                obs_tensor = torch.tensor(
                    self.env.patchilizer.encode(obs), dtype=torch.long
                ).unsqueeze(0)
                _, value_tensor = self.model(obs_tensor)
                storage.store_last_observation(value_tensor)
            else:
                storage.store_last_observation(torch.tensor(0.0))

            for _ in range(self.num_epochs):
                indices = np.arange(len(storage))
                np.random.shuffle(indices)

                for batch_start in range(0, len(storage), self.batch_size):
                    batch_indices = indices[batch_start : batch_start + self.batch_size]
                    (
                        obs_batch,
                        action_batch,
                        action_prob_batch,
                        advantage_batch,
                        return_batch,
                    ) = storage.batch_by_indices(batch_indices)

                    self.update(
                        obs_batch,
                        action_batch,
                        action_prob_batch,
                        advantage_batch,
                        return_batch,
                    )

            storage.clear()


if __name__ == "__main__":
    env = AbcGenEnv(SHARE_WEIGHTS, PATCH_SIZE)
    model = Tunedformer(
        model_name_or_path="./output/tuned_weights.pth", num_actions=512
    )
    trainer = PPOTrainer(
        env,
        model,
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-5,
        gamma=0.99,
        clip_param=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
    )

    num_steps = 10000
    trainer.train(num_steps)

    # Save the trained model
    torch.save(model.state_dict(), "./output/tuned_weights.pth")
