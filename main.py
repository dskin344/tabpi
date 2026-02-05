from imitation_datasets.dataset import BaselineDataset
from tqdm import tqdm
from rich import print
from torch.utils.data import DataLoader


from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from gymnasium import Env
import imageio

import gymnasium as gym
import numpy as np



def main():

    ds = BaselineDataset(f"NathanGavenski/CartPole-v1", source="huggingface", n_episodes=100)
    dload = DataLoader(ds, batch_size=32, shuffle=True)

    ds_eval = BaselineDataset(f"NathanGavenski/CartPole-v1", source="huggingface", n_episodes=100, split="eval")

    print(ds)
    print(len(ds))
    print(ds.average_reward, ds.states.shape)

    # state, action, next_state = ds[0]
    # print(state.shape, action.shape, next_state.shape)
    print(ds.states.shape, ds.actions.shape)
    print(type(ds.states))

    print(ds.actions)

    regressor = TabPFNRegressor()  # Uses TabPFN-2.5 weights, trained on synthetic data only.
    n_train, n_eval = 5000, 1000
    state_t, action_t = ds.states[0:n_train], ds.actions[0:n_train]
    regressor.fit(state_t, action_t)
    print('done fitting')
    # quit()

    # Predict on the test set
    state_e, action_e= ds_eval.states[0:n_eval], ds_eval.actions[0:n_eval], 
    print('eval shapes', state_e.shape, action_e.shape)
    yh = regressor.predict(state_e)

    # Evaluate the model
    mse = mean_squared_error(action_e, yh)
    r2 = r2_score(action_e, yh)

    print("Mean Squared Error (MSE):", mse)
    print("R² Score:", r2)

    #
    #
    #

    policy = lambda obs: regressor.predict(obs.reshape(1, -1))[0]
    env = gym.make("CartPole-v1", render_mode="rgb_array")  # drop render_mode if headless
    obs, info = env.reset(seed=0)

    done = False
    totals = []
    rewards = []
    frames = []

    for i in tqdm(range(10)):
        # while not done:
        for _ in tqdm(range(500)):
            frame = env.render()
            frames.append(frame)

            # --- call your policy ---
            action = policy(obs)          # must be 0 or 1
            print("predicted action:", action)
            # action = int(action) 
            action = (action >= 0.5).astype(int)
            print("converted action:", action)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            print(f'terminated: {terminated}, truncated: {truncated}, done: {done}, reward: {reward}')
            if done:
                break

        print(f'resetting env after episode {i} with total reward {sum(rewards)}')
        totals.append(sum(rewards))
        rewards = []
        obs, info = env.reset()

    env.close()
    print("Average total reward over 10 episodes:", np.mean(totals))
    imageio.mimsave('cartpole.mp4', frames, fps=50)
    print("Episode(s) saved to video cartpole.mp4")

if __name__ == "__main__":
    main()
