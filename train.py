from fog_env import Offload
from RL_brain import DeepQNetwork
import numpy as np
import random

np.set_printoptions(threshold=np.inf)


def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            return item


def reward_fun(delay, max_delay, unfinish_indi):
    penalty = - max_delay * 2
    if unfinish_indi:
        print(f"Task unfinished, returning penalty: {penalty}")
        return penalty
    else:
        print(f"Task finished, returning reward based on delay: {-delay}")
        return -delay


def train(iot_RL_list, env, NUM_EPISODE):
    RL_step = 0

       # Initialize tracking for performance and accuracy
    total_rewards = []
    total_dropped_tasks = []
    total_delays = []

    for episode in range(NUM_EPISODE):
        print(f'Episode {episode}, Epsilon: {iot_RL_list[0].epsilon}');


        total_reward = 0
        dropped_tasks = 0
        total_delay = 0
        
        
        # Generate random bit arrival tasks
        bitarrive = np.random.uniform(env.min_bit_arrive, env.max_bit_arrive, size=[env.n_time, env.n_iot])
        bitarrive *= np.random.uniform(0, 1, size=[env.n_time, env.n_iot]) < env.task_arrive_prob
        bitarrive[-env.max_delay:, :] = np.zeros([env.max_delay, env.n_iot])

        # Initialize observation matrix
        history = [
            [{'observation': np.zeros(env.n_features),
              'lstm': np.zeros(env.n_lstm_state),
              'action': np.nan,
              'observation_': np.zeros(env.n_features),
              'lstm_': np.zeros(env.n_lstm_state)} 
             for _ in range(env.n_iot)] 
            for _ in range(env.n_time)
        ]
        reward_indicator = np.zeros([env.n_time, env.n_iot])

        # Initial observation and LSTM states
        observation_all, lstm_state_all = env.reset(bitarrive)

        while True:
            action_all = np.zeros(env.n_iot)
            
            # Choose actions for each IoT device
            for iot_index in range(env.n_iot):
                observation = np.squeeze(observation_all[iot_index, :])

                if np.sum(observation) == 0:
                    action_all[iot_index] = 0
                else:
                    action_all[iot_index] = iot_RL_list[iot_index].choose_action(observation)
                
                if observation[0] != 0:
                    iot_RL_list[iot_index].do_store_action(episode, env.time_count, action_all[iot_index])

            # Observe next state and process delays
            observation_all_, lstm_state_all_, done = env.step(action_all)

            process_delay = env.process_delay
            unfinish_indi = env.process_delay_unfinish_ind

            # Store transition and reward
            for iot_index in range(env.n_iot):
                history[env.time_count - 1][iot_index].update({
                    'observation': observation_all[iot_index, :],
                    'lstm': np.squeeze(lstm_state_all[iot_index, :]),
                    'action': action_all[iot_index],
                    'observation_': observation_all_[iot_index],
                    'lstm_': np.squeeze(lstm_state_all_[iot_index, :])
                })

                update_index = np.where((1 - reward_indicator[:, iot_index]) * process_delay[:, iot_index] > 0)[0]

                for time_index in update_index:
                    reward = reward_fun(process_delay[time_index, iot_index], env.max_delay, unfinish_indi[time_index, iot_index])
                    print(f"Time Index {time_index}, IoT Device {iot_index}, Reward: {reward}")
                    iot_RL_list[iot_index].store_transition(
                        history[time_index][iot_index]['observation'],
                        history[time_index][iot_index]['lstm'],
                        history[time_index][iot_index]['action'],
                        reward,
                        history[time_index][iot_index]['observation_'],
                        history[time_index][iot_index]['lstm_']
                    )
                    reward_indicator[time_index, iot_index] = 1

            # Update state and LSTM states
            observation_all = observation_all_
            lstm_state_all = lstm_state_all_

            # Learning
            RL_step += 1
            if (RL_step > 200) and (RL_step % 10 == 0):
                for iot in range(env.n_iot):
                    iot_RL_list[iot].learn()

            # End the episode
            if done:
                break
 # Store metrics for the episode
        total_rewards.append(total_reward)
        total_dropped_tasks.append(dropped_tasks)
        total_delays.append(total_delay)

        # # Print performance and accuracy after each episode
        avg_reward = np.mean(total_rewards)
        avg_dropped_tasks = np.mean(total_dropped_tasks)
        avg_delay = np.mean(total_delays)
        total_reward += reward
        print(f"Accumulating reward for IoT {iot_index}, Total Reward: {total_reward}")

    print(f"Episode {episode}: Average Reward = {avg_reward:.2f}, Dropped Tasks = {avg_dropped_tasks}, Average Delay = {avg_delay:.2f}")

if __name__ == "__main__":
    NUM_IOT = 50
    NUM_FOG = 5
    NUM_EPISODE = 20
    NUM_TIME_BASE = 100
    MAX_DELAY = 10
    NUM_TIME = NUM_TIME_BASE + MAX_DELAY

    # Generate the environment
    env = Offload(NUM_IOT, NUM_FOG, NUM_TIME, MAX_DELAY)

    # Initialize RL agents for each IoT device
    iot_RL_list = [DeepQNetwork(env.n_actions, env.n_features, env.n_lstm_state, env.n_time,
                                learning_rate=0.01, reward_decay=0.9, e_greedy=0.99,
                                replace_target_iter=200, memory_size=500)
                   for _ in range(NUM_IOT)]

    # Train the system
    train(iot_RL_list, env, NUM_EPISODE)
    print('Training Finished')

