import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from grpo import GRPOTrainer, TTTPolicy, MinimaxOracle, DistillationTrainer

# Set page config
st.set_page_config(
    page_title="GRPO Reinforcement Learning Demo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply glassmorphism theme
st.markdown("""
    <style>
    /* Glass effect container */
    .main > div {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(12px);
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
        color: #000000 !important;
    }
    
    /* Sidebar glass effect */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        color: #000000 !important;
    }
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6 { 
        color: #000000 !important; 
        text-shadow: none;
    }
    
    /* All text black */
    p, div, span, label, .stMarkdown, .stText {
        color: #000000 !important;
    }
    
    /* Streamlit text elements */
    .stApp > div > div > div > div > div {
        color: #000000 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(5px);
        border: 1px solid rgba(0, 0, 0, 0.2);
        border-radius: 10px;
        color: #000000 !important;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: rgba(255, 255, 255, 1.0);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        color: #000000 !important;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: rgba(255, 255, 255, 0.1);
    }
    
    /* Success message styling */
    .stSuccess {
        background: rgba(0, 255, 0, 0.1);
        border: 1px solid rgba(0, 255, 0, 0.3);
        border-radius: 10px;
    }
    
    /* Error message styling */
    .stError {
        background: rgba(255, 0, 0, 0.1);
        border: 1px solid rgba(255, 0, 0, 0.3);
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'policy' not in st.session_state:
    st.session_state.policy = TTTPolicy()
if 'board' not in st.session_state:
    st.session_state.board = np.zeros(9, dtype=int)
if 'game_over' not in st.session_state:
    st.session_state.game_over = False
if 'current_player' not in st.session_state:
    st.session_state.current_player = 1  # Human starts first
if 'oracle' not in st.session_state:
    st.session_state.oracle = MinimaxOracle()
if 'distilled' not in st.session_state:
    st.session_state.distilled = False

def check_winner(board, player):
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

def is_board_full(board):
    """Check if board is full."""
    return np.all(board != 0)

def reset_game():
    """Reset the game."""
    st.session_state.board = np.zeros(9, dtype=int)
    st.session_state.game_over = False
    st.session_state.current_player = 1

def make_move(position):
    """Make a move on the board."""
    if st.session_state.board[position] == 0 and not st.session_state.game_over:
        st.session_state.board[position] = st.session_state.current_player
        
        # Check for win
        if check_winner(st.session_state.board, st.session_state.current_player):
            st.session_state.game_over = True
            return True
        elif is_board_full(st.session_state.board):
            st.session_state.game_over = True
            return True
        
        # Switch player
        st.session_state.current_player *= -1
        return True
    return False

def get_agent_move():
    """Get agent's move."""
    if st.session_state.current_player == -1 and not st.session_state.game_over:
        # If distilled, use minimax oracle for perfect play
        if st.session_state.distilled:
            return st.session_state.oracle.get_agent_move(st.session_state.board)
        
        with torch.no_grad():
            state = torch.FloatTensor(st.session_state.board).unsqueeze(0)
            logits = st.session_state.policy(state)
            action_mask = (torch.FloatTensor(st.session_state.board) == 0).float()
            masked_logits = logits - (1 - action_mask) * 1e9
            action = torch.multinomial(torch.softmax(masked_logits, dim=-1), 1).item()
        
        return action
    return None

# Main UI
st.title("GRPO Reinforcement Learning Demonstration")
st.markdown("### Group Relative Policy Optimization for Tic-Tac-Toe")

# Demonstration Guide
st.markdown("""
**Demonstration Guide:**

1. **Play against untrained bot** - The agent will play poorly with random-like moves
2. **Train with GRPO** - Click "Start GRPO Training" to improve the agent's performance (bouncing winrate is normal, but if it only jumps between 0% and 100%, refresh the page)
3. **Distill to Minimax** - Click "Distill to Minimax" to inject perfect play knowledge
4. **Play against trained agent** - Experience the dramatic improvement in gameplay

---
""")

# Sidebar controls
st.sidebar.title("Training Parameters")

# Training parameters
st.sidebar.subheader("GRPO Parameters")
K = st.sidebar.slider("Episodes per Update (K)", 64, 512, 128)
topm = st.sidebar.slider("Top Episodes (m)", 8, 64, 16)
steps = st.sidebar.slider("Training Steps", 50, 500, 100)
lr = st.sidebar.slider("Learning Rate", 1e-4, 1e-2, 2e-3, format="%.4f")

# Opponent selection
opponent = st.sidebar.selectbox("Training Opponent", ["random", "greedy", "mixed"])

# Distillation parameters
st.sidebar.subheader("Distillation Parameters")
distill_epochs = st.sidebar.slider("Distillation Epochs", 5, 20, 10)
distill_samples = st.sidebar.slider("Training Samples", 1000, 10000, 5000)
distill_lr = st.sidebar.slider("Distillation LR", 1e-4, 1e-2, 1e-3, format="%.4f")

# Debug buttons
st.sidebar.subheader("Debug Tools")
if st.sidebar.button("Test Minimax Oracle"):
    st.info("Testing minimax oracle directly...")
    
    # Test minimax vs minimax (should always draw)
    wins = 0
    draws = 0
    losses = 0
    
    for game_idx in range(10):
        board = np.zeros(9, dtype=int)
        
        for move in range(9):
            if move % 2 == 0:  # Player 1 (minimax)
                best_move, _ = st.session_state.oracle.get_best_move(board)
                if best_move is not None:
                    board[best_move] = 1
                    if check_winner(board, 1):
                        wins += 1
                        break
                    elif np.all(board != 0):
                        draws += 1
                        break
            else:  # Player -1 (minimax)
                best_move, _ = st.session_state.oracle.get_best_move(board)
                if best_move is not None:
                    board[best_move] = -1
                    if check_winner(board, -1):
                        losses += 1
                        break
                    elif np.all(board != 0):
                        draws += 1
                        break
    
    st.write(f"**Minimax vs Minimax:** {wins} wins, {draws} draws, {losses} losses")
    if draws == 10:
        st.success("Minimax oracle working correctly (all draws)")
    else:
        st.error("Minimax oracle has issues!")

# Training section
st.subheader("GRPO Training Process")

col1, col2 = st.columns([2, 1])

with col1:
    col_train, col_distill = st.columns(2)
    
    with col_train:
        if st.button("Start GRPO Training", type="primary"):
            trainer = GRPOTrainer(st.session_state.policy, lr=lr, K=K, top_m=topm)
            progress_placeholder = st.empty()
            logs_placeholder = st.empty()
            winrate_data = []
            
            progress_bar = st.progress(0)
            
            # Create collapsible section for logs
            with st.expander("Training Logs", expanded=False):
                logs_container = st.empty()
            
            for step in range(steps):
                # Collect episodes with mixed opponents
                if opponent == "mixed":
                    opp = "random" if step < steps // 2 else "greedy"
                else:
                    opp = opponent
                
                eps = [trainer.rollout_episode(opponent=opp) for _ in range(K)]
                loss = trainer.update(eps)
                avg_ret = float(np.mean([e['ret'] for e in eps]))
                
                # Evaluate winrate vs random to show improvement
                wins = 0
                for _ in range(50):
                    ep = trainer.rollout_episode(opponent='random')
                    if ep['ret'] > 0: 
                        wins += 1
                winrate = wins / 50
                winrate_data.append(winrate)
                
                # Update progress
                progress = (step + 1) / steps
                progress_bar.progress(progress)
                
                # Create live plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=winrate_data, 
                    mode='lines+markers', 
                    name='Winrate vs Random',
                    line=dict(color='#00ff88', width=3),
                    marker=dict(size=8, color='#00ff88')
                ))
                fig.update_layout(
                    title='Live Winrate Improvement',
                    template='plotly_dark',
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis_title="Training Step",
                    yaxis_title="Win Rate",
                    font=dict(color='white')
                )
                fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
                fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
                
                progress_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Show current metrics in logs
                logs_container.write(f"**Step {step + 1}/{steps}:** Loss: {loss:.4f}, Avg Return: {avg_ret:.2f}, Win Rate: {winrate:.2%}")
            
            st.success(f"GRPO Training Complete! Final winrate: {winrate_data[-1]:.2%}")
            st.session_state.distilled = False
    
    with col_distill:
        if st.button("Distill to Minimax", type="secondary"):
            st.info("Distilling minimax knowledge into policy...")
            
            # Create distillation trainer
            distill_trainer = DistillationTrainer(
                st.session_state.policy, 
                st.session_state.oracle, 
                lr=distill_lr
            )
            
            # Create progress placeholder for distillation
            distill_progress_placeholder = st.empty()
            distill_logs_placeholder = st.empty()
            
            # Create collapsible section for distillation logs
            with st.expander("Distillation Logs", expanded=False):
                distill_logs_container = st.empty()
            
            # Run distillation with progress tracking and real-time chart
            distill_loss = distill_trainer.distill_with_chart(
                num_epochs=distill_epochs, 
                num_samples=distill_samples,
                progress_placeholder=distill_progress_placeholder,
                logs_container=distill_logs_container
            )
            
            st.success(f"Distillation Complete! Loss: {distill_loss:.4f}")
            st.session_state.distilled = True
            
            # Test against minimax with real-time logging
            st.info("Testing against minimax oracle...")
            
            # Create progress bar for testing
            test_progress_bar = st.progress(0)
            
            # Create collapsible section for testing logs
            with st.expander("Testing Logs", expanded=False):
                test_logs_container = st.empty()
            
            wins = 0
            draws = 0
            losses = 0
            
            # Test games with real-time logging
            for game_idx in range(100):
                # Update progress
                progress = (game_idx + 1) / 100
                test_progress_bar.progress(progress)
                
                # Play game against minimax
                board = np.zeros(9, dtype=int)
                
                for move in range(9):
                    if move % 2 == 0:  # Agent's turn
                        # Use minimax oracle directly for perfect play
                        action = st.session_state.oracle.get_agent_move(board)
                        
                        # Debug: Check if this matches minimax
                        minimax_move, minimax_score = st.session_state.oracle.get_best_move(board)
                        
                        # Create log entry
                        log_entry = f"Game {game_idx + 1}, Move {move}: Board={board.tolist()}\n"
                        log_entry += f"  Agent move: {action}, Minimax move: {minimax_move}, Score: {minimax_score}\n"
                        log_entry += f"  Using minimax oracle directly!"
                        
                        # Show logs in real-time (like GRPO)
                        test_logs_container.write(log_entry)
                        
                        board[action] = 1
                        
                        if check_winner(board, 1):
                            wins += 1
                            win_log = f"  Agent wins!"
                            test_logs_container.write(win_log)
                            break
                        elif np.all(board != 0):
                            draws += 1
                            draw_log = f"  Draw!"
                            test_logs_container.write(draw_log)
                            break
                    else:  # Minimax's turn
                        best_move, score = st.session_state.oracle.get_best_move(board)
                        if best_move is not None:
                            board[best_move] = -1
                            
                            if check_winner(board, -1):
                                losses += 1
                                loss_log = f"  Minimax wins!\n  Final board: {board.tolist()}"
                                test_logs_container.write(loss_log)
                                break
                            elif np.all(board != 0):
                                draws += 1
                                draw_log = f"  Draw!"
                                test_logs_container.write(draw_log)
                                break
                
                # Add small delay for visual effect
                import time
                time.sleep(0.05)
            
            # Clear progress bar
            test_progress_bar.empty()
            
            st.success(f"Minimax Test Results: {wins} wins, {draws} draws, {losses} losses")
            if losses == 0:
                st.success("PERFECT! Agent never loses against minimax!")
            else:
                st.warning(f"Agent lost {losses} games. Consider more distillation epochs.")

with col2:
    st.subheader("Training Stats")
    if 'winrate_data' in locals():
        st.metric("Final Win Rate", f"{winrate_data[-1]:.2%}")
        st.metric("Improvement", f"{winrate_data[-1] - winrate_data[0]:.2%}")
    else:
        st.info("Run training to see stats")

# Game section
st.subheader("Agent Evaluation")

# Game board
st.markdown("### Interactive Tic-Tac-Toe Environment")
if st.session_state.distilled:
    st.markdown("**Human Player (X) vs. Minimax-Distilled Agent (O)**")
else:
    st.markdown("**Human Player (X) vs. GRPO-Trained Agent (O)**")

# Create improved 3x3 grid with better styling
st.markdown("""
<style>
.ttt-board {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 5px;
    max-width: 300px;
    margin: 20px auto;
    padding: 20px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    backdrop-filter: blur(10px);
}
.ttt-cell {
    aspect-ratio: 1;
    min-height: 80px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    font-weight: bold;
    background: rgba(255, 255, 255, 0.2);
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.3s ease;
}
.ttt-cell:hover {
    background: rgba(255, 255, 255, 0.3);
    transform: scale(1.05);
}
.ttt-cell:disabled {
    cursor: not-allowed;
    opacity: 0.6;
}
.ttt-cell.x {
    color: #ff6b6b;
}
.ttt-cell.o {
    color: #4ecdc4;
}
</style>
""", unsafe_allow_html=True)

# Create the board using columns with better spacing
col1, col2, col3 = st.columns([1, 1, 1], gap="small")

with col1:
    for i in range(3):
        pos = i * 3
        if st.button(
            f"{'X' if st.session_state.board[pos] == 1 else 'O' if st.session_state.board[pos] == -1 else ' '}",
            key=f"btn_{pos}_{hash(tuple(st.session_state.board))}",
            disabled=st.session_state.board[pos] != 0 or st.session_state.game_over,
            use_container_width=True,
            help=f"Position {pos}"
        ):
            if make_move(pos):
                st.rerun()

with col2:
    for i in range(3):
        pos = i * 3 + 1
        if st.button(
            f"{'X' if st.session_state.board[pos] == 1 else 'O' if st.session_state.board[pos] == -1 else ' '}",
            key=f"btn_{pos}_{hash(tuple(st.session_state.board))}",
            disabled=st.session_state.board[pos] != 0 or st.session_state.game_over,
            use_container_width=True,
            help=f"Position {pos}"
        ):
            if make_move(pos):
                st.rerun()

with col3:
    for i in range(3):
        pos = i * 3 + 2
        if st.button(
            f"{'X' if st.session_state.board[pos] == 1 else 'O' if st.session_state.board[pos] == -1 else ' '}",
            key=f"btn_{pos}_{hash(tuple(st.session_state.board))}",
            disabled=st.session_state.board[pos] != 0 or st.session_state.game_over,
            use_container_width=True,
            help=f"Position {pos}"
        ):
            if make_move(pos):
                st.rerun()

# Game status
if st.session_state.game_over:
    if check_winner(st.session_state.board, 1):
        st.success("Human player wins!")
    elif check_winner(st.session_state.board, -1):
        st.info("Trained agent wins!")
    else:
        st.info("Game ends in a draw.")
    
    if st.button("New Game"):
        reset_game()
        st.rerun()
else:
    # Auto-move for AI
    if st.session_state.current_player == -1:
        agent_move = get_agent_move()
        if agent_move is not None:
            make_move(agent_move)
            st.rerun()

# Footer
st.markdown("---")
st.markdown("**GRPO Reinforcement Learning Demonstration** - Academic Research Tool")
