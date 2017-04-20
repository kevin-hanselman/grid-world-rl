import numpy as np


def mask_to_points(mask):
    return [tuple(point) for point in np.argwhere(mask)]


class GridWorldMDP:

    def __init__(self,
                 reward_grid,
                 terminal_mask,
                 obstacle_mask,
                 action_probabilities,
                 no_action_probability):

        self._reward_grid = reward_grid
        self._terminal_mask = terminal_mask
        self._obstacle_mask = obstacle_mask
        self._T = self._create_transition_matrix(
            action_probabilities,
            no_action_probability,
            obstacle_mask
        )

    def _create_transition_matrix(self,
                                  action_probabilities,
                                  no_action_probability,
                                  obstacle_mask):
        M, N = self._reward_grid.shape

        # up, right, down, left
        deltas = [
            (-1, 0),
            (0, 1),
            (1, 0),
            (0, -1),
        ]

        num_actions = len(deltas)

        T = np.zeros((M, N, num_actions, M, N))

        r0 = np.arange(M * N) // N
        c0 = np.arange(M * N) % N

        T[r0, c0, :, r0, c0] += no_action_probability

        for action in range(num_actions):
            for offset, P in action_probabilities:
                direction = (action + offset) % num_actions

                dr, dc = deltas[direction]
                r1 = np.clip(r0 + dr, 0, M - 1)
                c1 = np.clip(c0 + dc, 0, N - 1)

                temp_mask = obstacle_mask[r1, c1].flatten()
                r1[temp_mask] = r0[temp_mask]
                c1[temp_mask] = c0[temp_mask]

                T[r0, c0, action, r1, c1] += P

        return T

    def value_iteration(self, discount=1.0, utility_grid=None):
        if utility_grid is None:
            utility_grid = np.zeros_like(self._reward_grid)

        utility_grid = utility_grid.astype(np.float64)
        out = np.zeros_like(utility_grid)
        M, N = self._reward_grid.shape
        for i in range(M):
            for j in range(N):
                out[i, j] = self._calc_utility((i, j),
                                               discount,
                                               utility_grid)
        return out

    def _calc_utility(self, loc, discount, utility_grid):
        if self._terminal_mask[loc]:
            return self._reward_grid[loc]
        row, col = loc
        return np.max(
            discount * np.sum(
                np.sum(self._T[row, col, :, :, :] * utility_grid,
                       axis=-1),
                axis=-1)
        ) + self._reward_grid[loc]

    def _get_neighbors(self, loc):
        neighbors = [(loc[0] + n, loc[1]) for n in (-1, 1)] + \
                    [(loc[0], loc[1] + n) for n in (-1, 1)]

        return [(r, c) for (r, c) in neighbors
                if r >= 0 and r < self._reward_grid.shape[0]
                and c >= 0 and c < self._reward_grid.shape[1]]


if __name__ == '__main__':
    size = (3, 4)
    goal = (0, -1)
    trap = (1, -1)
    obstacle = (1, 1)
    default_reward = -3
    goal_reward = 100
    trap_reward = -100

    reward_grid = np.zeros(size) + default_reward
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

    utility_grid = np.zeros_like(reward_grid)
    for _ in range(10):
        utility_grid = gw.value_iteration(utility_grid=utility_grid)
        print(utility_grid)
