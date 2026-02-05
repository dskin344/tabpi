from __future__ import annotations

import imageio
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv


def main():
    task_suite = benchmark.get_benchmark("libero_10")()
    num_tasks = task_suite.get_num_tasks()
    task_names = task_suite.get_task_names()
    frames = []
    max_steps = 300

    for t in range(num_tasks):
        task = task_suite.get_task(t)
        bddl_file_path = task_suite.get_task_bddl_file_path(t)
        print(f"Using task: {task_names[t]}")

        # Create environment arguments dictionary
        env_args = {
            "bddl_file_name": bddl_file_path,
            "camera_heights": 720,  # HD resolution
            "camera_widths": 1280,
            "camera_names": "galleryview",
        }

        # Create environment
        env = OffScreenRenderEnv(**env_args)

        done, step = False, 0
        while not done and step < max_steps:
            step += 1

            action = np.random.uniform(-1, 1, 7)
            obs, reward, done, _info = env.step(action)

            frames.append(obs["galleryview_image"][::-1])

            if step % 10 == 0:
                print(f"step={step}, reward={reward}")

            if done:
                print("Resetting the env")
                obs = env.reset()

    env.close()

    # Save video
    imageio.mimsave("LiberoRandom.mp4", frames, fps=5)
    print("LiberoRandom.mp4 saved")


if __name__ == "__main__":
    main()
