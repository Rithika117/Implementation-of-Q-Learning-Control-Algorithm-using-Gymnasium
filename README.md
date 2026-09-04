# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement
To implement and evaluate the Q-Learning control algorithm in the Gymnasium FrozenLake-v1 environment by modifying the learning rate, discount factor, and exploration decay parameters and comparing the resulting learning performance with the original configuration.

## Software Requirements
* Python 3.x
* Gymnasium
* NumPy
* Matplotlib
* Jupyter Notebook / Google Colab

## Environment Description
FrozenLake-v1 is a grid-world environment in which the agent starts from a starting state and must reach the goal while avoiding holes. The environment has 16 states and 4 possible actions: Left, Down, Right, and Up. The environment is slippery, so the agent's movement can be uncertain.

## Theory

Q-Learning estimates the optimal action-value function directly.

The action-value function $Q(s,a)$ represents the expected return obtained when the agent takes action $a$ in state $s$, and then follows the best possible policy afterward.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |
| $max_{a} Q(S_{t+1},a)$ | Maximum action value in the next state |

---

## Epsilon-Greedy Action Selection

During training, the agent uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_{a} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---

## Algorithm
1. Create the FrozenLake-v1 environment.

2. Initialize the Q-table with zeros.

3. Set the learning rate, discount factor, and exploration parameters.

4. Select actions using the epsilon-greedy strategy.

5. Execute the selected action and observe the reward and next state.

6. Update the Q-value using the Q-Learning update rule.

7. Repeat the process for the specified number of episodes.

8. Gradually reduce epsilon to shift from exploration to exploitation.

9. Calculate the state-value function using the maximum Q-value for each state.

10. Extract the learned policy using the action with the highest Q-value.
11. Compare the modified configuration with the original configuration using the learning curve and average reward.

## Python Program
```
# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------

env = gym.make("FrozenLake-v1", is_slippery=True)

n_states = env.observation_space.n
n_actions = env.action_space.n


# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

alpha = 0.5
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.99
epsilon_min = 0.01

episodes = 10000
max_steps = 100


# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

Q = np.zeros((n_states, n_actions))


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def choose_action(state):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])


# -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------

episode_rewards = []

for episode in range(episodes):

    state, info = env.reset()
    total_reward = 0

    for step in range(max_steps):

        # Choose action
        action = choose_action(state)

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)

        # Q-Learning update
        Q[state, action] = Q[state, action] + alpha * (
            reward
            + gamma * np.max(Q[next_state])
            - Q[state, action]
        )

        state = next_state
        total_reward += reward

        if terminated or truncated:
            break

    episode_rewards.append(total_reward)

    # Reduce exploration
    epsilon = max(epsilon_min, epsilon * epsilon_decay)


# -------------------------------------------------
# Calculate State Values and Policy
# -------------------------------------------------

state_values = np.max(Q, axis=1)

learned_policy = np.argmax(Q, axis=1)
```
## Output
<img width="838" height="515" alt="image" src="https://github.com/user-attachments/assets/9a759068-6054-4c57-95a1-eafe67a0a9be" />
<img width="681" height="602" alt="image" src="https://github.com/user-attachments/assets/2d3c16fb-d54f-4d67-be0c-4cbb7f5cfa72" />

After the gamma and alpha value change:

<img width="888" height="520" alt="image" src="https://github.com/user-attachments/assets/4f91f54f-db1a-42c3-8254-7ec0f730e5dc" />
<img width="613" height="588" alt="image" src="https://github.com/user-attachments/assets/7786a225-9b4c-404f-a015-1155ba00ed5b" />


## Result

The Q-Learning agent successfully learned a policy for the FrozenLake environment using the modified hyperparameters. Compared with the original configuration, the modified configuration produced a different learning curve, Q-table, and learned policy.

## Inference

The modified hyperparameters improved the learning behavior of the Q-Learning agent compared with the original configuration. Increasing the learning rate from 0.2 to 0.5 allowed the agent to learn more quickly from new experiences, while increasing the discount factor from 0.92 to 0.95 gave greater importance to future rewards. Reducing the epsilon decay from 0.995 to 0.99 caused exploration to decrease faster. Thus, changing the hyperparameters affected the learning curve, Q-values, and learned policy compared with the original setup.

