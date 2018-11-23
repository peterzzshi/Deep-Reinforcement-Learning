import gym
import tensorflow as tf
import numpy as np
import random
import time
# General Parameters

ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy


GAMMA = 0.8 # discount factor
INITIAL_EPSILON = 0.6 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
EPSILON_DECAY_STEPS = 100 # decay period

HIDDEN_DIM = 50
STOP_NUM = 150
MEMORY_SIZE = 1000
MEMORY_BATCH_SIZE = 128

# Create environment
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])


# Define Network Graph
def dqn(state_in, STATE_DIM, HIDDEN_DIM, ACTION_DIM):
    W1 = tf.get_variable(name="W1", shape=[STATE_DIM, HIDDEN_DIM])
    b1 = tf.get_variable(name="b1", shape=[1, HIDDEN_DIM], initializer=tf.constant_initializer(0.1))

    W2 = tf.get_variable(name="W2", shape=[HIDDEN_DIM, ACTION_DIM])
    b2 = tf.get_variable(name="b2", shape=[1, ACTION_DIM], initializer=tf.constant_initializer(0.1))

    hidden_logits = tf.matmul(state_in, W1) + b1
    hidden_act = tf.tanh(hidden_logits)

    output_logits = tf.matmul(hidden_act, W2) + b2
    # output_act = tf.tanh(output_logits)

    return output_logits


# Network outputs
q_values = dqn(state_in, STATE_DIM, HIDDEN_DIM, ACTION_DIM)
q_action = tf.reduce_sum(tf.multiply(action_in, q_values), axis=1)

# Loss/Optimizer Definition
loss = tf.reduce_mean(tf.squared_difference(target_in,q_action))
optimizer = tf.train.AdamOptimizer(0.01, name="AdamOptimizer").minimize(loss=loss)

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


Brain = []
flag = True


def update(q_values, state_in, testbatch):
    target_q = []

    state_batch = [batch[0] for batch in testbatch]
    action_batch = [batch[1] for batch in testbatch]
    reward_batch = [batch[2] for batch in testbatch]
    next_state_batch = [batch[3] for batch in testbatch]

    Q_future = q_values.eval(feed_dict={
        state_in: next_state_batch
    })

    for i in range(0, len(testbatch)):
        if testbatch[i][4]:
            target_q.append(reward_batch[i])
        else:
            # set the target_val is used for  Q value update
            target_val = reward_batch[i] + GAMMA * np.max(Q_future[i])
            target_q.append(target_val)
    return np.array(target_q), np.array(action_batch), np.array(state_batch)



def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action


# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    epsilon -= epsilon / EPSILON_DECAY_STEPS

    # Move through env according to e-greedy policy
    for step in range(STEP):
        # env.render()
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))
        Brain.append((state, action, reward, next_state, done))
        if len(Brain) > MEMORY_SIZE:
            Brain.pop(0)

        if flag:
            if len(Brain) >= MEMORY_BATCH_SIZE:
                testbatch = random.sample(Brain, MEMORY_BATCH_SIZE)

            else:
                testbatch = Brain

            target, action, state = update(q_values, state_in, testbatch)
            session.run([optimizer], feed_dict={
                target_in: target,
                action_in: action,
                state_in: state
            })

        # Update
        state = next_state

        if done:
            if step > STOP_NUM:
                flag = False
            break
    # Test and view sample runs - can disable render to save time
    
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                # env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()