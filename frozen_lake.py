import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import time

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class FrozenLakeEnvironment:
    def __init__(self, size=4, hole_probability=0.1, slip_probability=0.1):
        self.size = size
        self.slip_probability = slip_probability
        self.reset()

    def generate_map(self, hole_probability=0.1):
        grid = np.zeros((self.size, self.size), dtype=int)
        grid[0, 0] = 3
        grid[self.size - 1, self.size - 1] = 2

        for i in range(self.size):
            for j in range(self.size):
                if grid[i, j] == 0 and random.random() < hole_probability:
                    grid[i, j] = 1

        return grid

    def reset(self):
        self.grid = self.generate_map()
        self.agent_pos = [0, 0]
        self.done = False
        return self.get_state()

    def get_state(self):
        state = np.zeros(self.size * self.size + 8)

        pos_index = self.agent_pos[0] * self.size + self.agent_pos[1]
        state[pos_index] = 1

        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for i, (dx, dy) in enumerate(directions):
            new_x, new_y = self.agent_pos[0] + dx, self.agent_pos[1] + dy
            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                cell_type = self.grid[new_x, new_y]
                if cell_type == 1:
                    state[self.size * self.size + i] = -1
                elif cell_type == 2:
                    state[self.size * self.size + i] = 1

        return state

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True, {}

        if random.random() < self.slip_probability:
            action = random.randint(0, 3)

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dx, dy = moves[action]

        new_x = max(0, min(self.size - 1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.size - 1, self.agent_pos[1] + dy))

        self.agent_pos = [new_x, new_y]

        cell_type = self.grid[new_x, new_y]

        if cell_type == 1:
            reward = -100
            self.done = True
        elif cell_type == 2:
            reward = 100
            self.done = True
        else:
            goal_distance = abs(new_x - (self.size - 1)) + abs(new_y - (self.size - 1))
            reward = -1 - goal_distance * 0.1

        return self.get_state(), reward, self.done, {}

    def render(self):
        symbols = {0: '🧊', 1: '🕳️ ', 2: '🎯', 3: '🧊'}

        print("\n" + "=" * 30)
        for i in range(self.size):
            row = ""
            for j in range(self.size):
                if [i, j] == self.agent_pos:
                    row += "🤖"
                else:
                    row += symbols[self.grid[i, j]]
                row += " "
            print(row)
        print("=" * 30)


class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        self.buffer.append(Transition(state, action, next_state, reward, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=32):

        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.q_network = DQNNetwork(state_size, action_size)
        self.target_network = DQNNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.memory = ReplayBuffer(buffer_size)

        self.update_target_frequency = 100
        self.steps = 0

        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def store_experience(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(batch.state)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1)
        next_state_batch = torch.FloatTensor(batch.next_state)
        reward_batch = torch.FloatTensor(batch.reward)
        done_batch = torch.BoolTensor(batch.done)

        current_q_values = self.q_network(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (self.gamma * next_q_values * (~done_batch))

        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.update_target_frequency == 0:
            self.update_target_network()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()


def train_agent(episodes=2000, render_frequency=200):
    env = FrozenLakeEnvironment(size=6, hole_probability=0.15)
    state_size = env.size * env.size + 8
    action_size = 4

    agent = DQNAgent(state_size, action_size)

    scores = []
    losses = []
    success_rate = []
    recent_scores = deque(maxlen=100)

    print("🚀 Začínám trénink DQN agenta na Frozen Lake!")
    print(f"Prostředí: {env.size}x{env.size}, Epizody: {episodes}")
    print("=" * 50)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps_in_episode = 0
        episode_losses = []

        while not env.done and steps_in_episode < 200:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.store_experience(state, action, next_state, reward, done)

            loss = agent.learn()
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            total_reward += reward
            steps_in_episode += 1

        scores.append(total_reward)
        recent_scores.append(total_reward)
        if episode_losses:
            losses.append(np.mean(episode_losses))

        if len(recent_scores) >= 100:
            recent_successes = sum(1 for score in recent_scores if score > 50)
            success_rate.append(recent_successes)

        if episode % render_frequency == 0:
            avg_score = np.mean(recent_scores) if recent_scores else 0
            current_success_rate = success_rate[-1] if success_rate else 0

            print(f"\n📊 Epizoda {episode}")
            print(f"   Průměrné skóre (100 ep.): {avg_score:.2f}")
            print(f"   Úspěšnost: {current_success_rate}%")
            print(f"   Epsilon: {agent.epsilon:.3f}")
            print(f"   Kroky v epizodě: {steps_in_episode}")

            if episode % (render_frequency * 2) == 0:
                print(f"\n🎮 Ukázka hry - Epizoda {episode}")
                demo_episode(env, agent, render=True)

    print("\n🎉 Trénink dokončen!")

    print("\n" + "=" * 50)
    print("📈 FINÁLNÍ EVALUACE")
    print("=" * 50)

    success_count = 0
    eval_episodes = 100

    for _ in range(eval_episodes):
        state = env.reset()
        while not env.done:
            action = agent.get_action(state)
            state, reward, done, _ = env.step(action)
            if done and reward > 50:
                success_count += 1
                break

    final_success_rate = success_count / eval_episodes * 100
    print(f"Finální úspěšnost: {final_success_rate:.1f}% ({success_count}/{eval_episodes})")

    plot_training_results(scores, losses, success_rate)

    return agent, env


def demo_episode(env, agent, render=True):
    state = env.reset()
    total_reward = 0
    step_count = 0

    if render:
        print("Začátek hry:")
        env.render()
        time.sleep(1)

    while not env.done and step_count < 50:
        old_epsilon = agent.epsilon
        agent.epsilon = 0
        action = agent.get_action(state)
        agent.epsilon = old_epsilon

        state, reward, done, _ = env.step(action)
        total_reward += reward
        step_count += 1

        if render:
            action_names = ["⬆️ ", "⬇️ ", "⬅️ ", "➡️ "]
            print(f"\nKrok {step_count}: {action_names[action]} (reward: {reward:.1f})")
            env.render()
            time.sleep(0.5)

    if render:
        if total_reward > 50:
            print("🎉 ÚSPĚCH! Agent dosáhl cíle!")
        else:
            print("💥 Agent spadl do díry nebo nestihl dosáhnout cíle.")
        print(f"Celková odměna: {total_reward:.1f}")

    return total_reward


def plot_training_results(scores, losses, success_rate):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(scores, alpha=0.6)
    if len(scores) > 100:
        moving_avg = [np.mean(scores[i:i + 100]) for i in range(len(scores) - 100)]
        plt.plot(range(100, len(scores)), moving_avg, 'r-', linewidth=2, label='Klouzavý průměr (100)')
        plt.legend()
    plt.title('Skóre během tréninku')
    plt.xlabel('Epizoda')
    plt.ylabel('Celková odměna')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    if losses:
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epizoda')
        plt.ylabel('MSE Loss')
        plt.grid(True)

    plt.subplot(1, 3, 3)
    if success_rate:
        plt.plot(success_rate)
        plt.title('Úspěšnost (posledních 100 epizod)')
        plt.xlabel('Epizoda')
        plt.ylabel('Úspěšnost (%)')
        plt.grid(True)
        plt.ylim(0, 100)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    agent, env = train_agent(episodes=1500, render_frequency=150)

    print("\n" + "=" * 50)
    print("🎮 UKÁZKOVÉ HRY")
    print("=" * 50)

    for i in range(3):
        print(f"\n--- Hra {i + 1} ---")
        demo_episode(env, agent, render=True)
        print("\n" + "-" * 30)