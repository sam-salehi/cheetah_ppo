import gymnasium as gym
from cart import compute_gae, Transition, plot_rewards, plot_losses
from torch.nn.functional import mse_loss
import random
from nn import CheetahNet
import torch


BUFFER_SIZE = 2048
BATCH_SIZE = 64
EPOCHS = 4
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 3e-4  # Reduced learning rate
TIME_STEPS = BUFFER_SIZE * 500
VERBOSE = False
SHOW_GRAPHS = True
REWARD_THRESHOLD = 800
ENTROPY_COEF = 0.01  # Added entropy coefficient
SAVE_MODEL = True
TEST_COUNT = 5
SAVE_PATH = "models/cheet.pth"


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
    actions = torch.stack([t.action for t in transitions])
    old_log_probs = torch.tensor([t.log_prob for t in transitions])
    returns = returns.detach()

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

            mean, log_std, values = network(batch_states)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)

            # Compute new log probs (sum over action dims)
            new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
            entropy = (
                dist.entropy().sum(dim=-1).mean()
            )  # sum over dims, mean over batch

            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = mse_loss(values.squeeze(-1), batch_returns)
            entropy_loss = -entropy

            loss = policy_loss + 0.5 * value_loss + entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            num_batches += 1

    return (
        total_policy_loss / num_batches,
        total_value_loss / num_batches,
        total_entropy / num_batches,
    )


def take_action(network, obs, env):
    mean, log_std, val = network(torch.tensor(obs, dtype=torch.float32))
    std = torch.exp(log_std)

    dist = torch.distributions.Normal(mean, std)
    raw_action = dist.rsample()
    action = torch.tanh(raw_action)

    log_prob = dist.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-6)
    log_prob = log_prob.sum(dim=-1)
    next_observation, reward, terminated, truncated, info = env.step(
        action.detach().numpy()
    )
    return Transition(
        state=torch.tensor(obs, dtype=torch.float32),
        action=action.detach(),
        log_prob=log_prob.detach(),
        reward=reward,
        done=terminated or truncated,
        value=val.detach(),
    ), next_observation


def train():
    env = gym.make("HalfCheetah-v5")
    network = CheetahNet()
    optimizer = torch.optim.Adam(network.parameters(), lr=LR)

    buffer = []
    rewards = []
    policy_losses = []
    value_losses = []
    entropies = []

    observation, _ = env.reset()
    done = False
    episode_reward = 0

    print("Starting PPO Training ...")
    print(f"Totla iterations: {TIME_STEPS // BUFFER_SIZE}")

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

            transition, next_observation = take_action(network, observation, env)
            episode_reward += transition.reward
            done = transition.done
            buffer.append(transition)

            # Log timestep inf
            if VERBOSE:
                print("------------------")
                print(f"[Timestep {timestep}]")
                print(f"  Reward: {transition.reward}")
                print(f"  Done: {done}")
                print(f"  Log Prob: {transition.log_prob.item():.4f}")
                print(f"  Value Estimate: {transition.value.item():.4f}")
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

    for i in range(TEST_COUNT):
        env.close()
        env = gym.make("HalfCheetah-v5", render_mode="human")
        observation, _ = env.reset()
        done = False
        while not done:
            trans, observation = take_action(network, observation, env)
            done = trans.done

    if SAVE_MODEL:
        torch.save(network.state_dict(), SAVE_PATH)


def play(model_path, num_games):
    env = gym.make("HalfCheetah-v5", render_mode="human")
    network = CheetahNet()
    network.load_state_dict(torch.load(model_path))

    for i in range(num_games):
        env.close()
        env = gym.make("HalfCheetah-v5", render_mode="human")
        observation, _ = env.reset()
        done = False
        while not done:
            trans, observation = take_action(network, observation, env)
            done = trans.done


if __name__ == "__main__":
    train()
