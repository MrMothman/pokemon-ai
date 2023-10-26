# train.py
from pokemonEnvironment import PokemonEnvironment
from dqnAgent import DQNAgent

env = PokemonEnvironment("ROM/pc.gbc")
agent = DQNAgent(env.observation_space[0], env.action_size)

epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
num_episodes = 1000
batch_size = 32

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.select_action(state, epsilon)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if done:
            break

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
