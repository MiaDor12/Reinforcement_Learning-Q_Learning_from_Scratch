import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import math


def create_q_table(low_value, high_value, size):
    q_table = np.random.uniform(low=low_value, high=high_value, size=size)
    return q_table


def get_discrete_state(env, state, discrete_os_win_size):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    # we use this tuple to look up the 3 Q values for the available actions in the q-table
    return tuple(discrete_state.astype(np.int)) 


def get_action(env, epsilon, discrete_state, q_table):
    
    if np.random.random() > epsilon:
        # Get action from Q table
        action = np.argmax(q_table[discrete_state])
    else:
        # Get random action
        action = np.random.randint(0, env.action_space.n)
        
    return action


def calculate_q_value(learning_rate, current_q, reward, max_future_q, DISCOUNT):
    new_q = ((1 - learning_rate) * current_q) + (learning_rate * (reward + DISCOUNT * max_future_q))
    return new_q


def update_q_table(q_table, discrete_state, new_discrete_state, action, reward, LEARNING_RATE, DISCOUNT):
    
    # Maximum possible Q value in next step (for new state)
    max_future_q = np.max(q_table[new_discrete_state])

    # Current Q value (for current state and performed action)
    current_q = q_table[discrete_state + (action,)]

    # And here's our equation for a new Q value for current state and action
    new_q = calculate_q_value(LEARNING_RATE, current_q, reward, max_future_q, DISCOUNT)

    # Update Q table with new Q value
    q_table[discrete_state + (action,)] = new_q
    
    return q_table


def update_epsilon(episode, epsilon, epsilon_decay_value, START_EPSILON_DECAYING, END_EPSILON_DECAYING):
    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    
    return epsilon

def update_learning_rate(episode, learning_rate, learning_rate_decay_value, START_LEARNING_RATE_DECAYING, END_LEARNING_RATE_DECAYING):
    # Decaying is being done every episode if episode number is within decaying range
    if END_LEARNING_RATE_DECAYING >= episode >= START_LEARNING_RATE_DECAYING:
        learning_rate -= learning_rate_decay_value
    
    return learning_rate


def render_env(env, episode, SHOW_EVERY, TO_RENDER):
    if TO_RENDER & (episode % SHOW_EVERY == 0):
        env.render()
        
        
def update_stats(q_table, episode, epsilon, ep_rewards, aggr_ep_rewards, 
                 learning_rate, STATS_EVERY, PRINT_EVERY, SAVE_EVERY,
                 TO_PRINT, folder, file_name):
    
    if (not episode % STATS_EVERY) & (episode > 0):
        average_reward = sum(ep_rewards[-STATS_EVERY:])/STATS_EVERY
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
        
        if TO_PRINT & (episode % PRINT_EVERY == 0):
            print(f'Episode: {episode:>6d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}, current learning_rate: {learning_rate:>1.2f}')

    if episode % SAVE_EVERY == 0:
        if not os.path.exists(folder):
            os.mkdir(folder)
        np.save(folder+file_name, q_table)
            
    return aggr_ep_rewards


def plot_learning_results(aggr_ep_rewards):
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()