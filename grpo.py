import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class Episode:
    states: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    total_return: float
    episode_length: int
    metadata: Dict[str, Any] = None

class GRPOTrainer:
    def __init__(self, policy: nn.Module, lr: float = 2e-3, K: int = 64, top_m: int = 16):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.K = K
        self.top_m = top_m
        
    def rollout_episode(self, opponent: str = 'random') -> Dict[str, Any]:
        """Rollout a single episode and return episode data."""
        # Simplified Tic-Tac-Toe environment
        board = np.zeros(9, dtype=int)
        states = []
        actions = []
        rewards = []
        
        for step in range(9):
            if step % 2 == 0:  # Agent's turn
                state = board.copy()
                action_mask = (board == 0).astype(float)
                
                # Get action from policy
                with torch.no_grad():
                    logits = self.policy(torch.FloatTensor(state).unsqueeze(0))
                    masked_logits = logits - (1 - torch.FloatTensor(action_mask)) * 1e9
                    action = torch.multinomial(torch.softmax(masked_logits, dim=-1), 1).item()
                
                board[action] = 1
                states.append(state)
                actions.append(action)
                
                # Check for win
                if self._check_winner(board, 1):
                    rewards.append(10.0)
                    break
                elif np.all(board != 0):
                    rewards.append(1.0)  # Draw
                    break
                else:
                    rewards.append(0.0)
            else:  # Opponent's turn
                if opponent == 'random':
                    valid_moves = np.where(board == 0)[0]
                    if len(valid_moves) > 0:
                        action = np.random.choice(valid_moves)
                        board[action] = -1
                        
                        # Check for opponent win
                        if self._check_winner(board, -1):
                            rewards.append(-5.0)
                            break
                        elif np.all(board != 0):
                            rewards.append(1.0)  # Draw
                            break
                        else:
                            rewards.append(0.0)
        
        total_return = sum(rewards)
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'ret': total_return,
            'length': len(actions)
        }
    
    def _check_winner(self, board: np.ndarray, player: int) -> bool:
        """Check if player has won."""
        win_patterns = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]  # Diagonals
        ]
        
        for pattern in win_patterns:
            if all(board[i] == player for i in pattern):
                return True
        return False
    
    def update(self, episodes: List[Dict[str, Any]]) -> float:
        """Update policy using GRPO."""
        # Sort episodes by return
        sorted_episodes = sorted(episodes, key=lambda x: x['ret'], reverse=True)
        preferred_episodes = sorted_episodes[:self.top_m]
        
        # Compute advantages
        all_returns = [ep['ret'] for ep in episodes]
        baseline = np.mean(all_returns)
        advantages = [ep['ret'] - baseline for ep in episodes]
        
        # Collect all states and actions
        all_states = []
        all_actions = []
        all_advantages = []
        
        for episode, advantage in zip(episodes, advantages):
            all_states.extend(episode['states'])
            all_actions.extend(episode['actions'])
            all_advantages.extend([advantage] * len(episode['actions']))
        
        if len(all_states) == 0:
            return 0.0
        
        # Compute policy loss
        total_loss = 0.0
        for state, action, advantage in zip(all_states, all_actions, all_advantages):
            logits = self.policy(torch.FloatTensor(state).unsqueeze(0))
            action_mask = (torch.FloatTensor(state) == 0).float()
            masked_logits = logits - (1 - action_mask) * 1e9
            
            log_prob = torch.log_softmax(masked_logits, dim=-1)[0, action]
            total_loss += -advantage * log_prob
        
        # Update policy
        self.optimizer.zero_grad()
        loss = total_loss / len(all_states)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

class TTTPolicy(nn.Module):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(9, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 9)
        )
    
    def forward(self, x):
        return self.network(x)

class MinimaxOracle:
    """Perfect minimax player for Tic-Tac-Toe."""
    
    def __init__(self):
        self.cache = {}
    
    def get_best_move(self, board):
        """Get the best move for the current player (1 for agent, -1 for opponent)."""
        board_tuple = tuple(board)
        if board_tuple in self.cache:
            return self.cache[board_tuple]
        
        # Check for terminal states
        if self._check_winner(board, 1):
            return None, 1  # Agent wins
        if self._check_winner(board, -1):
            return None, -1  # Opponent wins
        if np.all(board != 0):
            return None, 0  # Draw
        
        # Find empty positions
        empty_positions = np.where(board == 0)[0]
        if len(empty_positions) == 0:
            return None, 0
        
        # Determine current player (count non-zero elements)
        # Agent (1) goes first, so even number of moves = agent's turn
        current_player = 1 if np.sum(board != 0) % 2 == 0 else -1
        
        best_move = None
        best_score = -float('inf') if current_player == 1 else float('inf')
        
        for pos in empty_positions:
            # Make move
            new_board = board.copy()
            new_board[pos] = current_player
            
            # Recursively evaluate
            _, score = self.get_best_move(new_board)
            
            if current_player == 1:  # Maximizing player (agent)
                if score > best_score:
                    best_score = score
                    best_move = pos
            else:  # Minimizing player (opponent)
                if score < best_score:
                    best_score = score
                    best_move = pos
        
        self.cache[board_tuple] = (best_move, best_score)
        return best_move, best_score
    
    def get_agent_move(self, board):
        """Get the best move for the agent (player 1) from any board state."""
        # Force agent's perspective
        best_move, _ = self.get_best_move(board)
        return best_move
    
    def _check_winner(self, board, player):
        """Check if player has won."""
        win_patterns = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]  # Diagonals
        ]
        
        for pattern in win_patterns:
            if all(board[i] == player for i in pattern):
                return True
        return False

class DistillationTrainer:
    """Train policy to mimic minimax oracle."""
    
    def __init__(self, policy, oracle, lr=1e-3):
        self.policy = policy
        self.oracle = oracle
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
    
    def generate_training_data(self, num_samples=5000):
        """Generate training data from minimax oracle."""
        states = []
        actions = []
        
        print(f"🔍 Generating {num_samples} training samples...")
        
        # Generate comprehensive training data covering all possible game states
        for sample_idx in range(num_samples):
            # Generate random board state
            board = np.zeros(9, dtype=int)
            num_moves = np.random.randint(0, 8)  # Leave at least one empty
            
            # Randomly place some moves
            positions = np.random.choice(9, num_moves, replace=False)
            for i, pos in enumerate(positions):
                board[pos] = 1 if i % 2 == 0 else -1
            
            # Skip if game is over
            if (self.oracle._check_winner(board, 1) or 
                self.oracle._check_winner(board, -1) or 
                np.all(board != 0)):
                continue
            
            # Get minimax move for agent's turn (assuming agent is player 1)
            if np.sum(board != 0) % 2 == 0:  # Agent's turn
                best_move, score = self.oracle.get_best_move(board)
                if best_move is not None:
                    states.append(board.copy())
                    actions.append(best_move)
                    
                    # Debug: Print some examples
                    if sample_idx < 5:
                        print(f"Sample {sample_idx}: Board={board.tolist()}, Move={best_move}, Score={score}")
        
        # Add critical opening moves
        critical_states = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Empty board - should go center
            [1, 0, 0, 0, 0, 0, 0, 0, 0],  # Agent center, opponent corner
            [0, 0, 0, 0, 1, 0, 0, 0, 0],  # Agent center, opponent edge
            [1, -1, 0, 0, 0, 0, 0, 0, 0], # Agent corner, opponent corner
            [0, 0, 0, 0, 1, 0, 0, 0, -1], # Agent center, opponent corner
        ]
        
        for board_state in critical_states:
            board = np.array(board_state)
            if np.sum(board != 0) % 2 == 0:  # Agent's turn
                best_move, score = self.oracle.get_best_move(board)
                if best_move is not None:
                    states.append(board.copy())
                    actions.append(best_move)
                    print(f"Critical state: Board={board.tolist()}, Move={best_move}, Score={score}")
        
        print(f"✅ Generated {len(states)} valid training samples")
        return states, actions
    
    def distill(self, num_epochs=10, num_samples=5000):
        """Distill minimax knowledge into policy."""
        states, actions = self.generate_training_data(num_samples)
        
        if len(states) == 0:
            print("❌ No training data generated!")
            return 0.0
        
        print(f"📚 Starting distillation with {len(states)} samples for {num_epochs} epochs...")
        
        # Convert to numpy arrays first to avoid the warning
        states_array = np.array(states)
        actions_array = np.array(actions)
        
        states = torch.FloatTensor(states_array)
        actions = torch.LongTensor(actions_array)
        
        total_loss = 0.0
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct_predictions = 0
            
            for i in range(len(states)):
                state = states[i].unsqueeze(0)
                action = actions[i]
                
                # Get action mask
                action_mask = (state == 0).float()
                logits = self.policy(state)
                masked_logits = logits - (1 - action_mask) * 1e9
                
                # Compute loss
                loss = self.criterion(masked_logits, action.unsqueeze(0))
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # Check if prediction is correct
                predicted_action = torch.argmax(masked_logits, dim=-1).item()
                if predicted_action == action.item():
                    correct_predictions += 1
            
            accuracy = correct_predictions / len(states)
            avg_loss = epoch_loss / len(states)
            total_loss += avg_loss
            
            print(f"Epoch {epoch + 1}/{num_epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2%}")
        
        final_loss = total_loss / num_epochs
        print(f"✅ Distillation complete! Final loss: {final_loss:.4f}")
        return final_loss
    
    def distill_with_chart(self, num_epochs=10, num_samples=5000, progress_placeholder=None, logs_container=None):
        """Distill minimax knowledge into policy with real-time accuracy chart."""
        states, actions = self.generate_training_data(num_samples)
        
        if len(states) == 0:
            print("❌ No training data generated!")
            return 0.0
        
        print(f"📚 Starting distillation with {len(states)} samples for {num_epochs} epochs...")
        
        # Convert to numpy arrays first to avoid the warning
        states_array = np.array(states)
        actions_array = np.array(actions)
        
        states = torch.FloatTensor(states_array)
        actions = torch.LongTensor(actions_array)
        
        total_loss = 0.0
        accuracy_data = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct_predictions = 0
            
            for i in range(len(states)):
                state = states[i].unsqueeze(0)
                action = actions[i]
                
                # Get action mask
                action_mask = (state == 0).float()
                logits = self.policy(state)
                masked_logits = logits - (1 - action_mask) * 1e9
                
                # Compute loss
                loss = self.criterion(masked_logits, action.unsqueeze(0))
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # Check if prediction is correct
                predicted_action = torch.argmax(masked_logits, dim=-1).item()
                if predicted_action == action.item():
                    correct_predictions += 1
            
            accuracy = correct_predictions / len(states)
            avg_loss = epoch_loss / len(states)
            total_loss += avg_loss
            accuracy_data.append(accuracy)
            
            print(f"Epoch {epoch + 1}/{num_epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2%}")
            
            # Update real-time chart
            if progress_placeholder is not None:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=accuracy_data, 
                    mode='lines+markers', 
                    name='Distillation Accuracy',
                    line=dict(color='#ff6b6b', width=3),
                    marker=dict(size=8, color='#ff6b6b')
                ))
                fig.update_layout(
                    title='Live Distillation Accuracy',
                    template='plotly_dark',
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis_title="Epoch",
                    yaxis_title="Accuracy",
                    font=dict(color='white')
                )
                fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
                fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
                
                progress_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Update logs
            if logs_container is not None:
                logs_container.write(f"**Epoch {epoch + 1}/{num_epochs}:** Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")
        
        final_loss = total_loss / num_epochs
        print(f"✅ Distillation complete! Final loss: {final_loss:.4f}")
        return final_loss
