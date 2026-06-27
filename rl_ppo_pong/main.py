import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py

from ppo_torch import Agent
from utils import plot_learning_curve
import os

gym.register_envs(ale_py)

if __name__ == '__main__':

    env = gym.make('ALE/Pong-v5', frameskip=1)

    # Riduzione a 84x84 in bianco e nero e saltiamo i frame 
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, 
                             frame_skip=4, scale_obs=False)
    
    env = FrameStackObservation(env, 4)

    N = 2048        
    batch_size = 64
    n_epochs = 10
    alpha = 0.00025  
    
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape)
    n_games = 10

    os.makedirs('plots', exist_ok=True)
    figure_file = 'plots/pong.png'

    best_score = -np.inf
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation, info = env.reset()
        done = False
        score = 0
        
        while not done:
            action, prob, val = agent.choose_action(observation)
            
            # CORREZIONE 2: Spacchettiamo i 5 valori delle nuove API di gymnasium
            observation_, reward, terminated, truncated, info = env.step(action)
            
            # Unifichiamo terminated (fine gioco) e truncated (fine tempo) nella variabile done
            done = terminated or truncated
            
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
                
            observation = observation_
            
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
                
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)