import gym
import numpy as np
import os
import yaml
import meshcat

from gym.envs.robot_locomotion_group.drake.shoe.open_loop import (
    get_instructions,
    instructions_to_x
)

shoe_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(shoe_dir, 'config.yaml')
config = yaml.safe_load(open(config_path, 'r'))
vis = meshcat.Visualizer()

env = gym.make("Shoe-v0", config=config)
env.reset()

open_loop = get_instructions()
i = 0
action = None
t = 0
env.systems["visualizer"].start_recording()
switchpoints = [0, 20, 40, 80, 120, 160, 200, 240, 290, 340, 390, 440, 490, 540, 590]
while True:
    print(f"Action {i} Time {t}")

    if int(10*t) in switchpoints:
        if i >= len(open_loop):
            break
        action = instructions_to_x([open_loop[i]])
        obs, _, success, _ = env.step(action)
        i += 1
    else:
        obs, _, success, _ = env.step(None)
    # If grippers not moving, do next stage
    # obs, _, success, _ = env.step(action)
    # action = None
    # if np.linalg.norm(obs) < 0.0001:
    #     if i >= len(open_loop):
    #         print("Done")
    #         break
    #     print(f"Executing move {i}")
    #     action = instructions_to_x([open_loop[i]])
    #     i += 1
    t += 0.1

    if not success:
        break
print(len(env.systems["visualizer"]._animation.clips))
print(env.systems["visualizer"]._recording_frame_num)
env.systems["visualizer"].stop_recording()
env.systems["visualizer"].publish_recording(repetitions=1)
f = open("saved.html", "w")
f.write(vis.static_html())
f.close()
