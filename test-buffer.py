from schedules import LinearSchedule
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

import gym
import numpy as np

prioritized_replay = True
buffer_size = 1000
demo_size = 100
max_timesteps = 1500
prioritized_replay_beta_iters = None
prioritized_replay_alpha=0.6
prioritized_replay_beta0=0.4
batch_size = 5
n_step = 10
max_train_steps = 1000
sample_full_trajectory = True

# Create the replay buffer
if prioritized_replay:
    replay_buffer = PrioritizedReplayBuffer(buffer_size, demo_size, alpha=prioritized_replay_alpha)
    if prioritized_replay_beta_iters is None:
        prioritized_replay_beta_iters = max_timesteps
    beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                   initial_p=prioritized_replay_beta0,
                                   final_p=1.0)
else:
    replay_buffer = ReplayBuffer(buffer_size, demo_size)
    beta_schedule = None


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()

# Add dummy trajectories to replay buffer, e.g., FrozenLake
env = gym.make("FrozenLake-v0")
agent = RandomAgent(env.action_space)

# Construct buffer for sanity check
# Transitions added to both replay and sanity buffer. We know how sanity buffer 
# works. Sanity buffer is a list of transitions, first demo_size demos and 
# thereafter a cyclic list of self generated transitions. If we know how replay 
# buffer works and are sampling from it correctly, the samples can be compared 
# with sanity buffer to catch misunderstandings.
sanity_buffer = []

# Add demo transitions
ob = env.reset()
for i in range(demo_size):
	action = agent.act()
	new_ob, reward, done, _ = env.step(action)
	replay_buffer.add(ob, action, reward, new_ob, float(done))
	sanity_buffer.append((ob, action, reward, new_ob, float(done)))
	if done:
		ob = env.reset()

# Add further "self generated" transitions 
selfgen_buffer_size = buffer_size - demo_size
ob = env.reset()
for i in range(max_timesteps):
	cyclic_id = (i % selfgen_buffer_size) + demo_size
	action = agent.act()
	new_ob, reward, done, _ = env.step(action)
	replay_buffer.add(ob, action, reward, new_ob, float(done))
	if cyclic_id < len(sanity_buffer):
		sanity_buffer[cyclic_id] = (ob, action, reward, new_ob, float(done))
	else:
		sanity_buffer.append((ob, action, reward, new_ob, float(done)))
	if done:
		ob = env.reset()
	else:
		ob = new_ob
	# Note there's no env.render() here. But the environment still can open window and
	# render if asked by env.monitor: it calls env.render('rgb_array') to record video.
	# Video is not recorded every episode, see capped_cubic_video_schedule for details.
# Close the env and write monitor result info to disk
env.close()

def verify_buffer_sampling_fulltraj(o_t, a, r, o_tp1, dones, batch_idxes):
	pass

def verify_buffer_sampling(o_t, a, r, o_tp1, dones, nstep_rewards, o_tpn, n_tpn, nstep_dones, batch_idxes):
	# To hold values from replay buffer
	recover_replay = []
	recover_replay_nstep_rewards = []
	recover_replay_o_tpn = []
	recover_replay_nstep_dones = []

	# To hold values from sanity buffer
	recover_sanity = []
	recover_sanity_nstep_rewards = []
	recover_sanity_o_tpn = []
	recover_sanity_nstep_dones = []

	# Args to lists
	o_t_list = o_t.tolist()
	a_lsit = a.tolist()
	r_list = r.tolist()
	o_tp1_list = o_tp1.tolist()
	dones_list = dones.tolist()
	nstep_rewards_list = nstep_rewards.tolist()
	o_tpn_list = o_tpn.tolist()
	n_tpn_list = n_tpn.tolist()
	nstep_dones_list = nstep_dones.tolist()

	# [1] NB. "nstep_rewards" contains tp1 to tpn rewards for each start index, 
	# reward at t corresponding to start index transition is "r"

	# Construct equatable lists from replay and sanity buffers
	for i, n in enumerate(n_tpn_list):
		n_as_id = n - 1 # [1] see comment above
		recover_replay.append((o_t_list[i], a_lsit[i], r_list[i], o_tp1_list[i], dones_list[i]))
		recover_replay_nstep_rewards.append(nstep_rewards_list[i][:n_as_id])
		recover_replay_o_tpn.append(o_tpn_list[i])
		recover_replay_nstep_dones.append(nstep_dones_list[i])

	for i, idx in enumerate(batch_idxes):
		recover_sanity.append(sanity_buffer[idx])
		# In case idx + n_tpn goes out of range of sanity_buffer
		traj_end_idx = idx + n_tpn_list[i]
		if traj_end_idx >= len(sanity_buffer):
			traj_idx = sanity_buffer[idx:]
			overshoot = traj_end_idx - len(sanity_buffer)
			traj_idx.extend(sanity_buffer[demo_size: demo_size + overshoot])
		else:
			traj_idx = sanity_buffer[idx: traj_end_idx]
		recover_sanity_nstep_rewards.append([elem[2] for elem in traj_idx[1:]]) # [1] see comment above
		recover_sanity_o_tpn.append(traj_idx[-1][3])
		recover_sanity_nstep_dones.append(traj_idx[-1][4])

	assert recover_replay == recover_sanity, "Error recovering (s, a, r, s')"
	assert recover_replay_nstep_rewards == recover_sanity_nstep_rewards, "Error recovering n-step rewards"
	assert recover_replay_o_tpn == recover_sanity_o_tpn, "Error recovering n-step obs"
	assert recover_replay_nstep_dones == recover_sanity_nstep_dones, "Error recovering n-step dones"

	# print(recover_replay, recover_sanity)
	# print(recover_replay_nstep_rewards, recover_sanity_nstep_rewards)
	# print(recover_replay_o_tpn, recover_sanity_o_tpn)
	# print(recover_replay_nstep_dones, recover_sanity_nstep_dones)

# Sample trajectories, verify correctness
num_iters = 0
if sample_full_trajectory:
	for num_iters in range(max_train_steps): 
		if prioritized_replay:
		    experience = replay_buffer.sample_nstep(batch_size, beta=beta_schedule.value(num_iters), n_step=n_step)
		    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes, demo_selfgens) = experience
		else:
		    experience = replay_buffer.sample_nstep(batch_size, n_step=n_step)
		    obses_t, actions, rewards, obses_tp1, dones, batch_idxes, demo_selfgens = experience
		    weights = np.ones_like(rewards)
	
		# Match tuples based on ids in sanity buffer and those returned by replay buffer
		print(obses_t, "\n", actions, "\n", rewards, "\n", obses_tp1, "\n", dones)
		break
		verify_buffer_sampling_fulltraj(obses_t, actions, rewards, obses_tp1, dones, batch_idxes)
else:
	for num_iters in range(max_train_steps): 
		if prioritized_replay:
		    experience = replay_buffer.sample_nstep(batch_size, beta=beta_schedule.value(num_iters), n_step=n_step)
		    (obses_t, actions, rewards, obses_tp1, dones, nstep_rewards, obses_tpn, n_tpn, nstep_dones, weights, batch_idxes, demo_selfgens) = experience
		else:
		    experience = replay_buffer.sample_nstep(batch_size, n_step=n_step)
		    obses_t, actions, rewards, obses_tp1, dones, nstep_rewards, obses_tpn, n_tpn, nstep_dones, batch_idxes, demo_selfgens = experience
		    weights = np.ones_like(rewards)
	
		# Match tuples based on ids in sanity buffer and those returned by replay buffer
		verify_buffer_sampling(obses_t, actions, rewards, obses_tp1, dones, nstep_rewards, obses_tpn, n_tpn, nstep_dones, batch_idxes)

print("Verified")

