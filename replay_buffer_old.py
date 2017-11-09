import numpy as np
import random
import logging

from segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, size, dsize):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped, except when the memories
            are from demonstrations.
        dsize: int
            Max number of demonstration transitions. These are retained in the 
            buffer permanently.
            https://arxiv.org/abs/1704.03732
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self._demosize = dsize
        self._traj_id = 0


    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done, self._traj_id)

        if done:
            # To prevent the trajectory id to grow to infititum, we reset it from time to time.
            self._traj_id = (self._traj_id +1) % self._maxsize

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        if self._next_idx >= self._demosize:
            self._next_idx = self._demosize + (self._next_idx - self._demosize + 1) % (self._maxsize - self._demosize)
        else:
            self._next_idx += 1

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, traj_id = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def _encode_full_trajectory(self, idxes, n_step):
        n_step_obses_t, n_step_actions, n_step_rewards, n_step_obses_tp1, n_step_dones = [], [], [], [], []
        
        for i in idxes:
            try:
                obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
                if i >= self._demosize:
                    j_last = (i + n_step - 1) % len(self._storage) + self._demosize
                else:
                    j_last = (i + n_step - 1)
                
                step = 1
                for j in range(i, i+n_step):
                    # cyclic storage
                    if j >= len(self._storage):
                        _j = j % len(self._storage) + self._demosize
                        _k = (j + 1) % len(self._storage) + self._demosize
                    else:
                        _j = j
                        _k = j + 1
                   
                    data = self._storage[_j]
                    obs_t, action, reward, obs_tp1, done, traj_id = data
                    obses_t.append(np.array(obs_t, copy=False))
                    actions.append(np.array(action, copy=False))
                    rewards.append(reward)
                    obses_tp1.append(np.array(obs_tp1, copy=False))
                    dones.append(done)
                    
                    # NB. Below is a workaround to fit with OpenAI baselines.
                    # It should not be reached under the parallel training (PAAC) 
                    # regimen because the replay buffer only contains demo data, and 
                    # storage is therefore not cyclic ~ not overwritten = no abrupt ends.   
                    # abrupt end (demo overshoot, s_this' != s_next)
                    if len(self._storage) > self._demosize:
                        data_tp1 = self._storage[_k]
                        if ((data[5] != data_tp1[5] and _j != j_last) or 
                            (i < self._demosize and _k >= self._demosize)):
                            # fill with zeros and ones accordingly
                            fill_span = n_step - step # j_last - _j
                            obses_t.extend([np.zeros_like(obs_t)] * fill_span)
                            actions.extend([np.zeros_like(action)] * fill_span)
                            rewards.extend([0.] * fill_span)
                            obses_tp1.extend([np.zeros_like(obs_tp1)] * fill_span)
                            dones.extend([1] * fill_span)              
                            break

                    step += 1
                n_step_obses_t.append(np.array(obses_t, copy=False))
                n_step_actions.append(np.array(actions, copy=False))
                n_step_rewards.append(np.array(rewards, copy=False))
                n_step_obses_tp1.append(np.array(obses_tp1, copy=False))
                n_step_dones.append(np.array(dones, copy=False))
            except IndexError:
                print("Something funky happened accessing replay storage.")
        print("Inside buffer: ", np.array(n_step_obses_t).shape)
        return np.array(n_step_obses_t), np.array(n_step_actions), np.array(n_step_rewards), np.array(n_step_obses_tp1), np.array(n_step_dones)

    def _encode_trajectory(self, idxes, n_step):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        n_step_rewards, obses_tpn, n_is, traj_dones = [], [], [], []
        #n_step = n_step - 1 # nth step is bootstrapped
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, traj_id = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            # Get rewards, end states and dones along trajectory starting at i
            try:
                _n_step = 1
                n_step_rewards_i, obses_tpn_i, n_is_i, traj_done = [], None, None, None
                if done == 1:
                    # Trajectory done in one step
                    n_step_rewards_i.extend([0.0]*(n_step - _n_step))
                    obses_tpn_i = obs_tp1
                    n_is_i = _n_step
                    traj_done = done
                else:
                    for j in range(i+1, i+n_step): 
                        # Trajectory overshoots demo |dataset|
                        if i < self._demosize and j >= self._demosize:
                            n_step_rewards_i.extend([0.0]*(n_step - _n_step))
                            obses_tpn_i = data[3]
                            n_is_i = _n_step
                            traj_done = data[4]
                            break

                        # Handling cyclic storage if not demo transition
                        if j >= len(self._storage):
                            _j = j % len(self._storage) + self._demosize
                        else:
                            _j = j
                        
                        data_tp1 = self._storage[_j]
                        # Unfinished trajectory -- exit loop
                        if (data[5] != data_tp1[5]):
                            n_step_rewards_i.extend([0.0]*(n_step - _n_step))
                            obses_tpn_i = data[3]
                            n_is_i = _n_step
                            traj_done = 0
                            break
                        
                        _n_step += 1
                        
                        if data_tp1[4] == 1:
                            # Trajectory complete -- exit loop
                            n_step_rewards_i.append(data_tp1[2])
                            n_step_rewards_i.extend([0.0]*(n_step - _n_step))
                            obses_tpn_i = data_tp1[3]
                            n_is_i = _n_step
                            traj_done = data_tp1[4]
                            break
                        else:
                            # Move along trajectory
                            n_step_rewards_i.append(data_tp1[2])
                            obses_tpn_i = data_tp1[3]
                            n_is_i = _n_step
                            traj_done = data_tp1[4]
                            data = data_tp1

                n_step_rewards.append(np.array(n_step_rewards_i, copy=False))
                obses_tpn.append(np.array(obses_tpn_i, copy=False))
                n_is.append(np.array(n_is_i, copy=False))
                traj_dones.append(np.array(traj_done, copy=False))
            except IndexError:
                print("Something funky happened accessing replay storage.")

        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), np.array(n_step_rewards), np.array(obses_tpn), np.array(n_is), np.array(traj_dones) 

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def sample_nstep(self, batch_size, n_step):
        """Sample a batch of experiences, n-step trajectory rewards, and tpn'th
        state, where n varies. Trajectory ends after n steps or until done/not 
        accessible (due to lack of data), whichever happens earlier.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        n_step: int
            How many steps to look into the future

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        n_step_rewards_batch: np.array
            n-step rewards vector batch
        tpn_obs_batch: np.array
            tpn set of observations
        n_tpn_step_batch: np.array
            n in n-step indicator to indicate if trajectory sampled 
            is unfinished or done -- trajectory is unfinished if 
            there are no more transitions to cover all n steps
        n_step_done_mask: np.array
            n_step_done_mask[i] = 1 if trajectory sampled reaches 
            the end of an episode, and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - n_step - 1) for _ in range(batch_size)]
        encoded_trajectory = self._encode_full_trajectory(idxes, n_step)
        demo_data = [1. if id < self._demosize else 0. for id in idxes]
        return tuple(list(encoded_trajectory) + [idxes, demo_data])

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, dsize, alpha):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        dsize: int
            Max number of demonstration transitions. These are retained in the 
            buffer permanently.
            https://arxiv.org/abs/1704.03732
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size, dsize)
        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size, n_step=0):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - n_step - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.


        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def sample_nstep(self, batch_size, beta, n_step):
        """Sample a batch of experiences, n-step trajectory rewards, and tpn'th
        state, where n varies. Trajectory ends after n steps or until done/not 
        accessible (due to lack of data), whichever happens earlier.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.


        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        n_step: int
            How many steps to look into the future

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        n_step_rewards_batch: np.array
            n-step rewards vector batch
        tpn_obs_batch: np.array
            tpn set of observations
        n_tpn_step_batch: np.array
            n in n-step indicator to indicate if trajectory sampled 
            is unfinished or done -- trajectory is unfinished if 
            there are no more transitions to cover all n steps
        n_step_done_mask: np.array
            n_step_done_mask[i] = 1 if trajectory sampled reaches 
            the end of an episode, and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size, n_step)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_trajectory = self._encode_full_trajectory(idxes, n_step)
        demo_selfgen = [1. if id < self._demosize else 0. for id in idxes]
        return tuple(list(encoded_trajectory) + [weights, idxes, demo_selfgen])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

