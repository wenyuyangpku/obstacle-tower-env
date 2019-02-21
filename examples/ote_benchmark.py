from obstacle_tower_env import ObstacleTowerEnv
import numpy as np
import time

env = ObstacleTowerEnv("ObstacleTower/obstacletower")
file = open("benchamrk.csv", "w")

num_episodes = 5
num_seeds = 5
steps_per_episode = 500

header = "Floor, Seed, Mean, Std\n"
print(header)
file.write(header)
for i in range(num_episodes):
	floor_num = i * 5
	for j in range(num_seeds):
		env.floor(floor_num)
		env.seed(j)
		env.reset()
		step_times = []
		for k in range(steps_per_episode):
			time_a = time.time()
			env.step(env.action_space.sample())
			time_b = time.time()
			step_times.append((time_b - time_a) * 1000)
		result = "{}, {}, {}, {}\n".format(floor_num, j, np.mean(step_times), np.std(step_times))
		print(result)
		file.write(result)
file.close()
