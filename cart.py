import gymnasium as gym
from dataclasses import dataclass
from nn import CartPoleNet
import torch
from torch.nn.functional import softmax, mse_loss
import random
import matplotlib.pyplot as plt


BUFFER_SIZE = 2048
BATCH_SIZE = 64
EPOCHS = 4
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 3e-4  # Reduced learning rate
TIME_STEPS = BUFFER_SIZE * 100
VERBOSE = False
SHOW_GRAPHS = True
REWARD_THRESHOLD = 475
ENTROPY_COEF = 0.01  # Added entropy coefficient
RENDER = True


@dataclass
class Transition:
    state: any
    action: any
    log_prob: float
    reward: float
    done: bool
    value: torch.Tensor


def compute_gae(transitions, gamma=GAMMA, lam=LAMBDA):
    advantages = []
    returns = []
    gae = 0
    next_value = 0

    for i in reversed(range(len(transitions))):
        t = transitions[i]
        # For terminal states, next_value should be 0
        if t.done:
            next_value = 0

        delta = t.reward + (gamma * next_value * (1 - t.done)) - t.value
        gae = delta + gamma * lam * (1 - t.done) * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + t.value)

        next_value = t.value

    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)
    return advantages, returns


def ppo_update(
    network,
    optimizer,
    transitions,
    advantages,
    returns,
    epochs,
    batch_size,
    clip_eps,
    entropy_coef,
):
    states = torch.stack([t.state for t in transitions])
    actions = torch.tensor([t.action for t in transitions])
    old_log_probs = torch.tensor([t.log_prob for t in transitions])
    returns = returns.detach()

    # Normalize advantages across the entire buffer
    advantages = advantages.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    num_batches = 0

    for _ in range(epochs):
        idxs = list(range(len(transitions)))
        random.shuffle(idxs)

        for start in range(0, len(transitions), batch_size):
            end = start + batch_size
            batch_idx = idxs[start:end]

            batch_states = states[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_returns = returns[batch_idx]

            logits, values = network(batch_states)
            probs = softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = mse_loss(values.squeeze(-1), batch_returns)
            entropy_loss = -entropy  # Negative because we want to maximize entropy

            loss = policy_loss + 0.5 * value_loss + entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            num_batches += 1

    avg_policy_loss = total_policy_loss / num_batches
    avg_value_loss = total_value_loss / num_batches
    avg_entropy = total_entropy / num_batches
    return avg_policy_loss, avg_value_loss, avg_entropy


def plot_rewards(rewards, threshold):
    plt.figure(figsize=(12, 8))
    plt.plot(rewards, label="Training Reward")
    plt.xlabel("Episode", fontsize=20)
    plt.ylabel("Training Reward", fontsize=20)
    plt.hlines(threshold, 0, len(rewards), color="y", label="Threshold")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


def plot_losses(policy_losses, value_losses):
    plt.figure(figsize=(12, 8))
    plt.plot(value_losses, label="Value Losses")
    plt.plot(policy_losses, label="Policy Losses")
    plt.xlabel("Update Step", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


def main():
    env = (
        gym.make("CartPole-v1", render_mode="human")
        if RENDER
        else gym.make("CartPole-v1")
    )
    network = CartPoleNet()
    optimizer = torch.optim.Adam(network.parameters(), lr=LR)

    buffer = []
    rewards = []
    policy_losses = []
    value_losses = []
    entropies = []

    observation, _ = env.reset()
    done = False
    episode_reward = 0

    print("Starting PPO training...")
    print(f"Total iterations: {TIME_STEPS // BUFFER_SIZE}")

    for iteration in range(TIME_STEPS // BUFFER_SIZE):
        buffer.clear()
        timestep = 0
        while len(buffer) < BUFFER_SIZE:
            if done:
                rewards.append(episode_reward)

                # Print progress every 10 episodes
                if len(rewards) % 10 == 0:
                    avg_reward = sum(rewards[-10:]) / 10
                    print(
                        f"Episode {len(rewards)}, Avg Reward (last 10): {avg_reward:.2f}"
                    )

                observation, _ = env.reset()
                episode_reward = 0
                done = False

            policy_logits, value = network(
                torch.tensor(observation, dtype=torch.float32)
            )
            probs = torch.softmax(policy_logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_observation, reward, terminated, truncated, info = env.step(
                action.item()
            )
            done = terminated or truncated
            episode_reward += reward

            buffer.append(
                Transition(
                    state=torch.tensor(observation, dtype=torch.float32),
                    action=action.item(),
                    log_prob=log_prob.detach(),
                    reward=reward,
                    done=done,
                    value=value.detach(),
                )
            )

            # Log timestep inf
            if VERBOSE:
                print("------------------")
                print(f"[Timestep {timestep}]")
                print(f"  Reward: {reward}")
                print(f"  Done: {done}")
                print(f"  Log Prob: {log_prob.item():.4f}")
                print(f"  Value Estimate: {value.item():.4f}")
                print(f"  Episode Reward So Far: {episode_reward:.2f}\n")
                print("------------------")

            observation = next_observation
            timestep += 1

        advantages, returns = compute_gae(buffer)
        policy_loss, value_loss, entropy = ppo_update(
            network,
            optimizer,
            buffer,
            advantages,
            returns,
            EPOCHS,
            BATCH_SIZE,
            CLIP_EPS,
            ENTROPY_COEF,
        )

        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        entropies.append(entropy)

        # Print training stats every 5 iterations
        if iteration % 5 == 0:
            print(
                f"Iteration {iteration}, Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Entropy: {entropy:.4f}"
            )

    print(f"Training completed. Total episodes: {len(rewards)}")
    if len(rewards) >= 10:
        print(f"Final average reward (last 10): {sum(rewards[-10:]) / 10:.2f}")

    if SHOW_GRAPHS:
        plot_rewards(rewards, REWARD_THRESHOLD)
        plot_losses(policy_losses, value_losses)


if __name__ == "__main__":
    main()
