import torch
import numpy as np
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config


class TextGenerationEnvironment:
    def __init__(self, model_name_or_path, max_length=20):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        self.max_length = max_length
        self.current_text = ""
        self.current_length = 0

    # def generate_text(self, input_text, max_length=None):
    #     if max_length is None:
    #         max_length = self.max_length

    #     input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
    #     output = self.model.generate(input_ids=input_ids, max_length=max_length)
    #     generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
    #     return generated_text

    # def get_tokenizer(self):
    #     return self.tokenizer

    def reset(self):
        self.current_text = ""
        self.current_length = 0
        return ""

    def step(self, action):
        action_token = self.tokenizer.decode([action])
        self.current_text += action_token

        obs = self.current_text[-self.max_length :]
        obs = obs if len(obs) > 0 else " "  # Ensure the observation is never empty

        reward = self.reward_fn(self.current_text)
        self.current_length += 1
        done = self.current_length >= self.max_length

        return obs, reward, done, {}


class ModifiedGPT(nn.Module):
    def __init__(self, model_name_or_path, num_actions=512):
        super(ModifiedGPT, self).__init__()
        config = GPT2Config.from_pretrained(model_name_or_path)
        self.gpt = GPT2LMHeadModel.from_pretrained(model_name_or_path, config=config)
        self.action_layer = nn.Linear(config.n_embd, num_actions)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        gpt_outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = gpt_outputs[0]
        # Calculate attention weights using the additional component
        action = self.action_layer(hidden_states)
        action_probs = self.softmax(action)

        outputs = (action_probs,) + gpt_outputs[1:]

        if labels is not None:
            # Calculate the loss with the labels provided
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                outputs[0].view(-1, self.config.vocab_size), labels.view(-1)
            )
            outputs = (loss,) + outputs

        return outputs


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

    # def compute_returns(self, gamma):
    #     returns = []
    #     R = 0
    #     for r, done in zip(reversed(self.rewards), reversed(self.dones)):
    #         if done:
    #             R = 0
    #         R = r + gamma * R
    #         returns.insert(0, R)

    #     return returns

    # def compute_advantages(self, returns):
    #     advantages = []
    #     for action_prob, return_ in zip(self.action_probs, returns):
    #         advantages.append(return_ - action_prob)

    #     return advantages

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
        env,
        model,
        reward_fn,
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
        self.reward_fn = reward_fn
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
                    self.env.tokenizer.encode(obs), dtype=torch.long
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
                    self.env.tokenizer.encode(obs), dtype=torch.long
                ).unsqueeze(0)
                _, value_tensor = self.model(obs_tensor)
                storage.store_last_observation(value_tensor)
            else:
                storage.store_last_observation(torch.tensor(0.0))

            # returns = storage.compute_returns(self.gamma)
            # advantages = storage.compute_advantages(returns)

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
    env = TextGenerationEnvironment(model_name_or_path="gpt2", max_length=20)
    model = ModifiedGPT(model_name_or_path="gpt2", num_actions=512)
    reward_fn = RewardFunction(
        reference_text="The quick brown fox jumps over the lazy dog",
        model_name_or_path="gpt2",
    )
    trainer = PPOTrainer(
        env,
        model,
        reward_fn,
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
    torch.save(model.state_dict(), "./output/modified_gpt_model.pth")
