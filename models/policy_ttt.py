"""
Tic-Tac-Toe Policy Network

Simple MLP policy for playing Tic-Tac-Toe with action masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Union


class TicTacToePolicy(nn.Module):
    """
    MLP policy for Tic-Tac-Toe.
    
    Input: 9-dimensional board state (flattened 3x3 board)
    Output: 9-dimensional action logits (one for each position)
    """
    
    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input layer
        self.input_layer = nn.Linear(9, hidden_size)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, 9)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, state: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: Board state (9-dimensional array)
            
        Returns:
            Action logits (9-dimensional)
        """
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32, device=next(self.parameters()).device)
        
        # Ensure state is 2D (batch_size, 9)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Forward pass
        x = F.relu(self.input_layer(state))
        x = self.dropout_layer(x)
        
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
            x = self.dropout_layer(x)
        
        logits = self.output_layer(x)
        
        return logits.squeeze(0) if state.dim() == 1 else logits
    
    def get_action_mask(self, state: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Get action mask for valid actions.
        
        Args:
            state: Board state
            
        Returns:
            Boolean mask for valid actions
        """
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        
        # Valid actions are empty positions (value == 0)
        mask = (state == 0).astype(bool)
        return mask
    
    def act(
        self,
        state: Union[np.ndarray, torch.Tensor],
        action_mask: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[int, float]:
        """
        Sample action from policy.
        
        Args:
            state: Board state
            action_mask: Boolean mask for valid actions
            
        Returns:
            Tuple of (action, log_prob)
        """
        # Get logits
        logits = self.forward(state)
        
        # Apply action mask
        if isinstance(action_mask, np.ndarray):
            action_mask = torch.tensor(action_mask, dtype=torch.bool, device=logits.device)
        
        # Set invalid actions to very negative values
        masked_logits = logits.clone()
        if masked_logits.dim() == 2:
            masked_logits = masked_logits.squeeze(0)
        masked_logits[~action_mask] = -1e9
        
        # Sample action
        action_probs = F.softmax(masked_logits, dim=-1)
        action = torch.multinomial(action_probs, 1).item()
        
        # Calculate log probability
        log_prob = F.log_softmax(masked_logits, dim=-1)[action].item()
        
        return action, log_prob
    
    def log_prob(
        self,
        state: Union[np.ndarray, torch.Tensor],
        action: int,
        action_mask: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Calculate log probability of action.
        
        Args:
            state: Board state
            action: Action taken
            action_mask: Boolean mask for valid actions
            
        Returns:
            Log probability of the action
        """
        # Get logits
        logits = self.forward(state)
        
        # Apply action mask
        if isinstance(action_mask, np.ndarray):
            action_mask = torch.tensor(action_mask, dtype=torch.bool, device=logits.device)
        
        # Set invalid actions to very negative values
        masked_logits = logits.clone()
        if masked_logits.dim() == 2:
            masked_logits = masked_logits.squeeze(0)
        masked_logits[~action_mask] = -1e9
        
        # Calculate log probability
        log_prob = F.log_softmax(masked_logits, dim=-1)[action].item()
        
        return log_prob
    
    def get_action_probs(
        self,
        state: Union[np.ndarray, torch.Tensor],
        action_mask: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Get action probabilities.
        
        Args:
            state: Board state
            action_mask: Boolean mask for valid actions
            
        Returns:
            Action probabilities
        """
        # Get logits
        logits = self.forward(state)
        
        # Apply action mask
        if isinstance(action_mask, np.ndarray):
            action_mask = torch.tensor(action_mask, dtype=torch.bool, device=logits.device)
        
        # Set invalid actions to very negative values
        masked_logits = logits.clone()
        if masked_logits.dim() == 2:
            masked_logits = masked_logits.squeeze(0)
        masked_logits[~action_mask] = -1e9
        
        # Get probabilities
        probs = F.softmax(masked_logits, dim=-1)
        
        return probs.cpu().numpy()
    
    def save(self, filepath: str):
        """Save policy to file."""
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath: str):
        """Load policy from file."""
        self.load_state_dict(torch.load(filepath, map_location=next(self.parameters()).device))


def create_ttt_policy(
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1
) -> TicTacToePolicy:
    """Factory function to create a Tic-Tac-Toe policy."""
    return TicTacToePolicy(hidden_size, num_layers, dropout)
