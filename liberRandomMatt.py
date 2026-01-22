import os
import numpy as np
import imageio
import time
import libero.libero.envs

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def main():
    task_suite = benchmark.get_benchmark("libero_spatial")()
    num_tasks = task_suite.get_num_tasks()
    task_names = task_suite.get_task_names()
    frames = []

    for t in range(5):

        task = task_suite.get_task(t)

        # Construct BDDL file path manually
        bddl_file_path = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )

        # Create environment arguments dictionary
        env_args = {
            "bddl_file_name": bddl_file_path,
            "camera_heights": 720,  # HD resolution
            "camera_widths": 1280,
            "camera_names": "galleryview",
        }

        # Create environment
        env = OffScreenRenderEnv(**env_args)
        action = np.zeros(7)
        amp = 0.9

        obs = env.reset()

        done , step= False, 0
        maxtime = 300
        while not done and step < maxtime:
            step += 1

            print(f"Using task: {task_names[t]}")

            #action = env.action_space.sample() # should sample random action

            
            delta = np.array([0.8, -0.6, 0.7, -0.5, 0.6, -0.4, 0.0]) * np.sign(
                np.sin(step * 0.1)
            )
            action = np.clip(delta, -1.0, 1)

            action = amp * np.array(
                [
                    np.sign(np.sin(step * 0.1)),
                    np.sign(np.cos(step * 0.13)),
                    np.sign(np.sin(step * 0.17)),
                    np.sign(np.cos(step * 0.19)),
                    np.sign(np.sin(step * 0.23)),
                    np.sign(np.cos(step * 0.29)),
                    np.sign(np.sin(step * 0.31)),
                ],
                dtype=np.float32,
            )
            

            obs, reward, done, info = env.step(action)

            frames.append(np.flipud(obs["galleryview_image"]))

            if step % 10 == 0:
                print(f"step={step}, reward={reward}")

            if done:
                print("Resetting the env")
                obs = env.reset()

        env.close()

    # Save video
    imageio.mimsave("LiberoRandom.mp4", frames, fps=50)
    print("LiberoRandom.mp4 saved")


if __name__ == "__main__":
    main()
