from gridworld import GridWorldMDP
from qlearn import QLearner

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    shape = (3, 4)
    goal = (0, -1)
    trap = (1, -1)
    obstacle = (1, 1)
    start = (2, 0)
    default_reward = -0.1
    goal_reward = 1
    trap_reward = -1

    reward_grid = np.zeros(shape) + default_reward
    reward_grid[goal] = goal_reward
    reward_grid[trap] = trap_reward
    reward_grid[obstacle] = 0

    terminal_mask = np.zeros_like(reward_grid, dtype=np.bool)
    terminal_mask[goal] = True
    terminal_mask[trap] = True

    obstacle_mask = np.zeros_like(reward_grid, dtype=np.bool)
    obstacle_mask[1, 1] = True

    gw = GridWorldMDP(reward_grid=reward_grid,
                      obstacle_mask=obstacle_mask,
                      terminal_mask=terminal_mask,
                      action_probabilities=[
                          (-1, 0.1),
                          (0, 0.8),
                          (1, 0.1),
                      ],
                      no_action_probability=0.0)

    mdp_solvers = {'Value Iteration': gw.run_value_iterations,
                   'Policy Iteration': gw.run_policy_iterations} 

    for solver_name, solver_fn in mdp_solvers.items():
        print('Final result of {}:'.format(solver_name))
        policy_grid, utility_grid = solver_fn(iterations=25, discount=0.5)
        print(policy_grid)
        print(utility_grid)
        plt.figure()
        gw.plot_policy(utility_grid)

    plt.show()

    ql = QLearner(num_states=(shape[0] * shape[1]),
                  num_actions=4,
                  learning_rate=0.8,
                  discount_rate=0.9,
                  random_action_prob=0.5,
                  random_action_decay_rate=0.99)

    start_state = gw.grid_coordinates_to_indices(start)

    ql.learn(start_state, gw.generate_experience, max_iterations=10000)
    print(ql._Q)
    ql_policy = np.argmax(ql._Q, axis=1).reshape(gw.shape)
    print(ql_policy)
