from typing import Tuple

from helpers.env import SlipperyGridWorld, ACTIONS, ACTION_TO_DELTA


class CustomSlipperyGridWorld(SlipperyGridWorld):
    """
    Extended version with:
    - Larger grid
    - Cliff (big penalty, reset to start)
    - Walls (impassable states)
    """
    
    def __init__(self, 
                 rows=7, 
                 cols=10, 
                 start=(0, 0), 
                 goal=(6, 9),
                 slip_prob=0.2,
                 step_reward=-1.0,
                 goal_reward=10.0,
                 cliff_reward=-50.0,      # ← new
                 max_steps=100,
                 seed=None):
        
        super().__init__(rows=rows, cols=cols, start=start, goal=goal,
                        slip_prob=slip_prob, step_reward=step_reward,
                        goal_reward=goal_reward, max_steps=max_steps, seed=seed)
        
        self.cliff_reward = cliff_reward
        
        # Define walls as set of (row, col) tuples
        self.walls = set()
        # Example walls in the center (you can modify this)
        for r in range(2, 4):
            for c in range(3, 7):
                self.walls.add((r, c))
        
        # Define cliff: entire bottom row except goal (you can customize)
        self.cliff_states = set()
        for c in range(self.cols):
            if (self.rows-1, c) != self.goal_row_column:
                self.cliff_states.add((self.rows-1, c))
        
        self.num_states = rows * cols   # still the same

    def is_wall(self, r: int, c: int) -> bool:
        return (r, c) in self.walls

    def is_cliff(self, r: int, c: int) -> bool:
        return (r, c) in self.cliff_states

    def _apply_action(self, r: int, c: int, a: int) -> Tuple[int, int]:
        """Override to respect walls"""
        dr, dc = ACTION_TO_DELTA[a]
        nr, nc = r + dr, c + dc
        
        # If next position is wall or out of bounds → stay in place
        if not self._in_bounds(nr, nc) or self.is_wall(nr, nc):
            return r, c
        return nr, nc

    def step(self, action: int):
        """Override step to handle cliff"""
        assert action in ACTIONS
        
        self._steps += 1
        intended = action
        executed = self._sample_action_with_slip(intended)

        r, c = self._agent_row_column
        nr, nc = self._apply_action(r, c, executed)
        
        # Move agent
        self._agent_row_column = (nr, nc)
        
        # Check if fell into cliff
        if self.is_cliff(nr, nc):
            reward = self.cliff_reward
            # Reset agent to start (classic cliff walking behavior)
            self._agent_row_column = self.start_row_column
            done = False                    # Important: do NOT terminate
        else:
            done = (self._agent_row_column == self.goal_row_column)
            reward = self.goal_reward if done else self.step_reward

        if self.max_steps is not None and self._steps >= self.max_steps:
            done = True

        info = {
            "intended_action": intended, 
            "executed_action": executed, 
            "steps": self._steps,
            "cliff": self.is_cliff(nr, nc)
        }
        
        return self.row_column_to_state(*self._agent_row_column), reward, done, info

    def reward(self, state: int, action: int, next_state: int) -> float:
        """Override reward to handle cliff"""
        r_next, c_next = self.state_to_row_column(next_state)
        
        if self.is_cliff(r_next, c_next):
            return self.cliff_reward
        if self.state_to_row_column(state) == self.goal_row_column:
            return 0.0
        if (r_next, c_next) == self.goal_row_column:
            return self.goal_reward
        return self.step_reward

    def is_terminal_state(self, state: int) -> bool:
        """Only goal is terminal (cliff is not)"""
        return self.state_to_row_column(state) == self.goal_row_column