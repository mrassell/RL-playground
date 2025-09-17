"""
Tic-Tac-Toe Environment for GRPO Training

Implements a 3x3 Tic-Tac-Toe game with configurable reward shaping,
opponent types, and action masking.
"""

import numpy as np
import random
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum


class OpponentType(Enum):
    RANDOM = "random"
    GREEDY = "greedy"
    POLICY = "policy"


class TicTacToeEnv:
    """
    Tic-Tac-Toe environment with configurable reward shaping.
    
    Board representation: 3x3 numpy array with values:
    - 0: empty
    - 1: agent (X)
    - -1: opponent (O)
    
    Action space: 9 positions (0-8) representing board positions:
    0 1 2
    3 4 5
    6 7 8
    """
    
    def __init__(
        self,
        win_reward: float = 10.0,
        draw_reward: float = 1.0,
        loss_penalty: float = -5.0,
        center_bonus: float = 0.5,
        fork_bonus: float = 1.0,
        illegal_move_penalty: float = -20.0,
        opponent_type: OpponentType = OpponentType.RANDOM,
        opponent_policy=None
    ):
        self.win_reward = win_reward
        self.draw_reward = draw_reward
        self.loss_penalty = loss_penalty
        self.center_bonus = center_bonus
        self.fork_bonus = fork_bonus
        self.illegal_move_penalty = illegal_move_penalty
        self.opponent_type = opponent_type
        self.opponent_policy = opponent_policy
        
        # Game state
        self.board = None
        self.current_player = 1  # Agent starts first
        self.game_over = False
        self.winner = None
        self.step_count = 0
        
        # Statistics
        self.total_games = 0
        self.agent_wins = 0
        self.opponent_wins = 0
        self.draws = 0
    
    def reset(self) -> np.ndarray:
        """Reset the environment and return initial state."""
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = 1  # Agent starts first
        self.game_over = False
        self.winner = None
        self.step_count = 0
        
        return self.board.copy()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (0-8)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.game_over:
            return self.board.copy(), 0.0, True, {"error": "Game already over"}
        
        # Check if action is valid
        if not self._is_valid_action(action):
            reward = self.illegal_move_penalty
            info = {"error": "Illegal move", "action": action}
            return self.board.copy(), reward, False, info
        
        # Agent's move
        self.board[action] = 1
        self.step_count += 1
        
        # Check for game end after agent's move
        if self._check_winner() == 1:
            self.game_over = True
            self.winner = 1
            reward = self.win_reward
            info = {"winner": 1, "game_over": True}
            return self.board.copy(), reward, True, info
        
        if self._is_board_full():
            self.game_over = True
            self.winner = 0  # Draw
            reward = self.draw_reward
            info = {"winner": 0, "game_over": True}
            return self.board.copy(), reward, True, info
        
        # Opponent's move
        opponent_action = self._get_opponent_action()
        if opponent_action is not None:
            self.board[opponent_action] = -1
            self.step_count += 1
            
            # Check for game end after opponent's move
            if self._check_winner() == -1:
                self.game_over = True
                self.winner = -1
                reward = self.loss_penalty
                info = {"winner": -1, "game_over": True}
                return self.board.copy(), reward, True, info
            
            if self._is_board_full():
                self.game_over = True
                self.winner = 0  # Draw
                reward = self.draw_reward
                info = {"winner": 0, "game_over": True}
                return self.board.copy(), reward, True, info
        
        # Calculate reward with shaping
        reward = self._calculate_shaping_reward(action)
        
        info = {
            "step_count": self.step_count,
            "agent_action": action,
            "opponent_action": opponent_action
        }
        
        return self.board.copy(), reward, False, info
    
    def _is_valid_action(self, action: int) -> bool:
        """Check if action is valid."""
        return 0 <= action < 9 and self.board[action] == 0
    
    def _is_board_full(self) -> bool:
        """Check if board is full."""
        return np.all(self.board != 0)
    
    def _check_winner(self) -> Optional[int]:
        """Check if there's a winner. Returns 1 (agent), -1 (opponent), or None."""
        # Check rows
        for i in range(0, 9, 3):
            if self.board[i] == self.board[i+1] == self.board[i+2] != 0:
                return self.board[i]
        
        # Check columns
        for i in range(3):
            if self.board[i] == self.board[i+3] == self.board[i+6] != 0:
                return self.board[i]
        
        # Check diagonals
        if self.board[0] == self.board[4] == self.board[8] != 0:
            return self.board[0]
        if self.board[2] == self.board[4] == self.board[6] != 0:
            return self.board[2]
        
        return None
    
    def _get_opponent_action(self) -> Optional[int]:
        """Get opponent's action based on opponent type."""
        valid_actions = [i for i in range(9) if self.board[i] == 0]
        
        if not valid_actions:
            return None
        
        if self.opponent_type == OpponentType.RANDOM:
            return random.choice(valid_actions)
        
        elif self.opponent_type == OpponentType.GREEDY:
            return self._get_greedy_action(valid_actions)
        
        elif self.opponent_type == OpponentType.POLICY and self.opponent_policy is not None:
            return self._get_policy_action(valid_actions)
        
        else:
            return random.choice(valid_actions)
    
    def _get_greedy_action(self, valid_actions: List[int]) -> int:
        """Get greedy opponent action (win if possible, block if necessary)."""
        # Try to win
        for action in valid_actions:
            temp_board = self.board.copy()
            temp_board[action] = -1
            if self._check_winner_on_board(temp_board) == -1:
                return action
        
        # Try to block agent from winning
        for action in valid_actions:
            temp_board = self.board.copy()
            temp_board[action] = 1
            if self._check_winner_on_board(temp_board) == 1:
                return action
        
        # Prefer center, then corners, then edges
        if 4 in valid_actions:  # Center
            return 4
        
        corners = [0, 2, 6, 8]
        for corner in corners:
            if corner in valid_actions:
                return corner
        
        # Random from remaining
        return random.choice(valid_actions)
    
    def _get_policy_action(self, valid_actions: List[int]) -> int:
        """Get action from opponent policy."""
        action_mask = np.zeros(9, dtype=bool)
        for action in valid_actions:
            action_mask[action] = True
        
        action, _ = self.opponent_policy.act(self.board, action_mask)
        return action
    
    def _check_winner_on_board(self, board: np.ndarray) -> Optional[int]:
        """Check winner on a given board state."""
        # Check rows
        for i in range(0, 9, 3):
            if board[i] == board[i+1] == board[i+2] != 0:
                return board[i]
        
        # Check columns
        for i in range(3):
            if board[i] == board[i+3] == board[i+6] != 0:
                return board[i]
        
        # Check diagonals
        if board[0] == board[4] == board[8] != 0:
            return board[0]
        if board[2] == board[4] == board[6] != 0:
            return board[2]
        
        return None
    
    def _calculate_shaping_reward(self, action: int) -> float:
        """Calculate reward shaping for the action."""
        reward = 0.0
        
        # Center bonus
        if action == 4:  # Center position
            reward += self.center_bonus
        
        # Fork bonus (creating two threats)
        if self._creates_fork(action):
            reward += self.fork_bonus
        
        return reward
    
    def _creates_fork(self, action: int) -> bool:
        """Check if action creates a fork (two winning threats)."""
        temp_board = self.board.copy()
        temp_board[action] = 1
        
        # Count how many winning lines this move completes
        winning_lines = 0
        
        # Check rows
        for i in range(0, 9, 3):
            if temp_board[i] == temp_board[i+1] == temp_board[i+2] == 1:
                winning_lines += 1
        
        # Check columns
        for i in range(3):
            if temp_board[i] == temp_board[i+3] == temp_board[i+6] == 1:
                winning_lines += 1
        
        # Check diagonals
        if temp_board[0] == temp_board[4] == temp_board[8] == 1:
            winning_lines += 1
        if temp_board[2] == temp_board[4] == temp_board[6] == 1:
            winning_lines += 1
        
        return winning_lines >= 2
    
    def get_action_mask(self) -> np.ndarray:
        """Get action mask for valid actions."""
        mask = np.zeros(9, dtype=bool)
        for i in range(9):
            if self.board[i] == 0:
                mask[i] = True
        return mask
    
    def render_state(self) -> str:
        """Render the current board state as a string."""
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        
        lines = []
        for i in range(0, 9, 3):
            line = ' | '.join([symbols[self.board[i+j]] for j in range(3)])
            lines.append(line)
        
        return '\n'.join(lines)
    
    def check_winner(self) -> Optional[int]:
        """Check current winner."""
        return self._check_winner()
    
    def is_game_over(self) -> bool:
        """Check if game is over."""
        return self.game_over
    
    def get_game_result(self) -> Dict[str, Any]:
        """Get game result information."""
        return {
            "winner": self.winner,
            "game_over": self.game_over,
            "step_count": self.step_count,
            "board": self.board.copy()
        }
    
    def evaluate_vs_opponent(
        self,
        policy,
        num_games: int = 100,
        opponent_type: Optional[OpponentType] = None
    ) -> Dict[str, float]:
        """
        Evaluate policy against opponent over multiple games.
        
        Args:
            policy: Policy to evaluate
            num_games: Number of games to play
            opponent_type: Opponent type (uses current if None)
            
        Returns:
            Dictionary with win rate and other metrics
        """
        if opponent_type is not None:
            original_opponent = self.opponent_type
            self.opponent_type = opponent_type
        
        wins = 0
        losses = 0
        draws = 0
        
        for _ in range(num_games):
            state = self.reset()
            done = False
            
            while not done:
                if self.current_player == 1:  # Agent's turn
                    action_mask = self.get_action_mask()
                    action, _ = policy.act(state, action_mask)
                else:  # Opponent's turn
                    action_mask = self.get_action_mask()
                    action = self._get_opponent_action()
                
                state, reward, done, info = self.step(action)
            
            # Count results
            if self.winner == 1:
                wins += 1
            elif self.winner == -1:
                losses += 1
            else:
                draws += 1
        
        if opponent_type is not None:
            self.opponent_type = original_opponent
        
        return {
            "win_rate": wins / num_games,
            "loss_rate": losses / num_games,
            "draw_rate": draws / num_games,
            "total_games": num_games
        }
    
    def get_evaluation_metrics(self, episodes: List) -> Dict[str, float]:
        """Get environment-specific evaluation metrics."""
        wins = 0
        losses = 0
        draws = 0
        
        for episode in episodes:
            if episode.metadata and "winner" in episode.metadata:
                winner = episode.metadata["winner"]
                if winner == 1:
                    wins += 1
                elif winner == -1:
                    losses += 1
                else:
                    draws += 1
        
        total = len(episodes)
        if total == 0:
            return {"win_rate": 0.0, "loss_rate": 0.0, "draw_rate": 0.0}
        
        return {
            "win_rate": wins / total,
            "loss_rate": losses / total,
            "draw_rate": draws / total
        }
