import gymnasium as gym
import torch as T
import time
from pong.ppo_torch import Agent



def test_trained_agent():
    env = gym.make('CartPole-v1', render_mode='human')
    agent = Agent(n_actions=env.action_space.n, batch_size=5,
                  alpha=0.0003, n_epochs=4,
                  input_dims=env.observation_space.shape)
    
    agent.load_models()
    agent.actor.eval()
    n_test_games = 3 

    for i in range(n_test_games):
        observation, _ = env.reset()
        done = False
        score = 0

        print(f"--- Inizio Partita {i+1} ---")

        while not done:
            time.sleep(0.015)
            state = T.tensor([observation], dtype = T.float).to(agent.actor.device)
            dist = agent.actor(state)

            action = T.argmax(dist.probs).item()
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward

        print(f"Partita {i+1} finita. Punteggio: {score}")

    env.close()

if __name__ == '__main__':
    test_trained_agent()