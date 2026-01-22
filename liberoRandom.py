import os
import numpy as np
import imageio
import time
import libero.libero.envs

os.environ['MUJOCO_GL'] = 'omesa'

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def main():
    task_suite = benchmark.get_benchmark("libero_10")()
    num_tasks = task_suite.get_num_tasks()
    task_names = task_suite.get_task_names()
    task = task_suite.get_task(0)

    # Construct BDDL file path manually  
    bddl_file_path = os.path.join(  
        get_libero_path("bddl_files"),   
        task.problem_folder,   
        task.bddl_file  
    )  
  
    # Create environment arguments dictionary  
    env_args = {  
        "bddl_file_name": bddl_file_path,  
        "camera_heights": 720,  # HD resolution
        "camera_widths": 1280,
        "camera_names": "galleryview"
    }  
  
    # Create environment  
    env = OffScreenRenderEnv(**env_args)
    action = np.zeros(7)
    amp = 0.9

    frames = []
    obs = env.reset()

    for step in range(num_tasks):
        time.sleep(0.05)
        print(f"Using task: {task_names[step]}")
       
        delta = np.array([ 0.8, -0.6, 0.7, -0.5, 0.6, -0.4, 0.0]) * np.sign(np.sin(step * 0.1))
        action = np.clip(delta, -1.0, 1)

        action = amp * np.array([
                 np.sign(np.sin(step * 0.1)),
                 np.sign(np.cos(step * 0.13)),
                 np.sign(np.sin(step * 0.17)),
                 np.sign(np.cos(step * 0.19)),
                 np.sign(np.sin(step * 0.23)),
                 np.sign(np.cos(step * 0.29)),
                 np.sign(np.sin(step * 0.31)),
                ], dtype=np.float32)

        obs, reward, done, info = env.step(action)

        frames.append(np.flipud(obs['galleryview_image']))

        if step % 10 == 0:
            print(f"step={step}, reward={reward}")

        if done:
            print("Resetting the env")
            obs = env.reset()

    env.close()

    # Save video
    imageio.mimsave('LiberoRandom.mp4', frames, fps=5)
    print("LiberoRandom.mp4 saved")


if __name__ == "__main__":
    main()

