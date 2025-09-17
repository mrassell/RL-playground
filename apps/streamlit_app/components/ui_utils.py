"""
UI utility components for Streamlit app.
"""

import streamlit as st
import time
from typing import Any, Optional, Callable
import pandas as pd


def show_spinner(message: str, func: Callable, *args, **kwargs) -> Any:
    """
    Show spinner while executing function.
    
    Args:
        message: Spinner message
        func: Function to execute
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
    """
    with st.spinner(message):
        return func(*args, **kwargs)


def show_progress_bar(total: int, message: str = "Progress"):
    """
    Show progress bar.
    
    Args:
        total: Total number of steps
        message: Progress message
        
    Returns:
        Progress bar object
    """
    return st.progress(0, text=message)


def update_progress_bar(progress_bar, current: int, total: int, message: str = "Progress"):
    """
    Update progress bar.
    
    Args:
        progress_bar: Progress bar object
        current: Current step
        total: Total steps
        message: Progress message
    """
    progress = current / total
    progress_bar.progress(progress, text=f"{message}: {current}/{total}")


def show_status_message(message: str, message_type: str = "info"):
    """
    Show status message.
    
    Args:
        message: Message text
        message_type: Type of message (info, success, warning, error)
    """
    if message_type == "success":
        st.success(message)
    elif message_type == "warning":
        st.warning(message)
    elif message_type == "error":
        st.error(message)
    else:
        st.info(message)


def create_sidebar_config(
    title: str = "Configuration",
    seed: int = 1337,
    num_episodes: int = 64,
    top_m: int = 16,
    learning_rate: float = 1e-3,
    num_updates: int = 10
) -> dict:
    """
    Create sidebar configuration.
    
    Args:
        title: Sidebar title
        seed: Random seed
        num_episodes: Number of episodes per update
        top_m: Number of top episodes for GRPO
        learning_rate: Learning rate
        num_updates: Number of training updates
        
    Returns:
        Configuration dictionary
    """
    st.sidebar.header(title)
    
    config = {}
    
    config['seed'] = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=9999,
        value=seed,
        help="Random seed for reproducibility"
    )
    
    config['num_episodes'] = st.sidebar.number_input(
        "Episodes per Update (K)",
        min_value=1,
        max_value=200,
        value=num_episodes,
        help="Number of episodes to collect per training update"
    )
    
    config['top_m'] = st.sidebar.number_input(
        "Top Episodes (m)",
        min_value=1,
        max_value=50,
        value=top_m,
        help="Number of top episodes to use for GRPO update"
    )
    
    config['learning_rate'] = st.sidebar.number_input(
        "Learning Rate",
        min_value=1e-5,
        max_value=1e-1,
        value=learning_rate,
        format="%.2e",
        help="Learning rate for policy optimization"
    )
    
    config['num_updates'] = st.sidebar.number_input(
        "Training Updates",
        min_value=1,
        max_value=100,
        value=num_updates,
        help="Number of training updates to perform"
    )
    
    return config


def create_ttt_reward_config() -> dict:
    """
    Create Tic-Tac-Toe reward configuration.
    
    Returns:
        Reward configuration dictionary
    """
    st.subheader("Reward Shaping")
    
    config = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        config['win_reward'] = st.number_input(
            "Win Reward",
            min_value=0.0,
            max_value=50.0,
            value=10.0,
            step=1.0,
            help="Reward for winning"
        )
        
        config['draw_reward'] = st.number_input(
            "Draw Reward",
            min_value=-5.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Reward for draw"
        )
        
        config['loss_penalty'] = st.number_input(
            "Loss Penalty",
            min_value=-20.0,
            max_value=0.0,
            value=-5.0,
            step=0.5,
            help="Penalty for losing"
        )
    
    with col2:
        config['center_bonus'] = st.number_input(
            "Center Bonus",
            min_value=0.0,
            max_value=2.0,
            value=0.5,
            step=0.1,
            help="Bonus for taking center position"
        )
        
        config['fork_bonus'] = st.number_input(
            "Fork Bonus",
            min_value=0.0,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="Bonus for creating fork (two threats)"
        )
        
        config['illegal_move_penalty'] = st.number_input(
            "Illegal Move Penalty",
            min_value=-50.0,
            max_value=0.0,
            value=-20.0,
            step=1.0,
            help="Penalty for illegal moves"
        )
    
    return config


def create_ttt_board_ui(board: list, on_click: Callable = None) -> list:
    """
    Create interactive Tic-Tac-Toe board UI.
    
    Args:
        board: 3x3 board state
        on_click: Click handler function
        
    Returns:
        List of clicked positions
    """
    if board is None:
        return []
    
    clicked_positions = []
    
    # Create a visually appealing board with CSS
    st.markdown("""
    <style>
    .ttt-container {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }
    .ttt-board {
        display: grid;
        grid-template-columns: repeat(3, 80px);
        grid-template-rows: repeat(3, 80px);
        gap: 3px;
        background-color: #333;
        padding: 3px;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .ttt-cell {
        background-color: #fff;
        border: none;
        border-radius: 4px;
        font-size: 2rem;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .ttt-cell:hover {
        background-color: #f0f0f0;
        transform: scale(1.05);
    }
    .ttt-cell:disabled {
        cursor: not-allowed;
        transform: none;
    }
    .ttt-cell.x {
        color: #e74c3c;
    }
    .ttt-cell.o {
        color: #3498db;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create 3x3 grid with individual buttons
    for i in range(3):
        cols = st.columns(3)
        for j in range(3):
            pos = i * 3 + j
            
            if pos >= len(board):
                continue
                
            symbol = "X" if board[pos] == 1 else "O" if board[pos] == -1 else ""
            
            # Use different button types for better visual feedback
            if board[pos] == 1:
                button_type = "primary"  # X is red/primary
            elif board[pos] == -1:
                button_type = "secondary"  # O is blue/secondary
            else:
                button_type = "secondary"  # Empty cells
            
            # Create button with unique key that includes board state
            board_state_str = "_".join(map(str, board))
            button_key = f"ttt_pos_{pos}_{board_state_str}"
            
            if cols[j].button(
                symbol, 
                key=button_key, 
                disabled=(board[pos] != 0),
                help=f"Click to place X" if board[pos] == 0 else "Occupied",
                use_container_width=True,
                type=button_type
            ):
                clicked_positions.append(pos)
                if on_click:
                    on_click(pos)
    
    return clicked_positions


def create_edit_preview(data: pd.DataFrame, num_rows: int = 5) -> None:
    """
    Create edit data preview.
    
    Args:
        data: Edit data DataFrame
        num_rows: Number of rows to show
    """
    st.subheader("Edit Data Preview")
    
    if len(data) > 0:
        preview_data = data.head(num_rows)
        
        for idx, row in preview_data.iterrows():
            with st.expander(f"Task {row['id']}: {row['instruction'][:50]}..."):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Before:**")
                    st.code(row['before_text'], language='text')
                
                with col2:
                    st.write("**After:**")
                    st.code(row['after_text'], language='text')
                
                st.write("**Instruction:**")
                st.write(row['instruction'])
    else:
        st.warning("No edit data available")


def create_diff_viewer(before_text: str, after_text: str, title: str = "Diff View") -> None:
    """
    Create diff viewer for text changes.
    
    Args:
        before_text: Original text
        after_text: Modified text
        title: Viewer title
    """
    st.subheader(title)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Before:**")
        st.code(before_text, language='text')
    
    with col2:
        st.write("**After:**")
        st.code(after_text, language='text')
    
    # Show unified diff
    if before_text != after_text:
        st.write("**Unified Diff:**")
        import difflib
        diff = difflib.unified_diff(
            before_text.splitlines(keepends=True),
            after_text.splitlines(keepends=True),
            fromfile='Before',
            tofile='After',
            lineterm=''
        )
        diff_text = ''.join(diff)
        st.code(diff_text, language='diff')


def create_metrics_display(metrics: dict, title: str = "Metrics") -> None:
    """
    Create metrics display.
    
    Args:
        metrics: Dictionary of metrics
        title: Display title
    """
    st.subheader(title)
    
    if metrics:
        # Create columns for metrics
        cols = st.columns(len(metrics))
        
        for i, (name, value) in enumerate(metrics.items()):
            with cols[i]:
                st.metric(
                    label=name.replace('_', ' ').title(),
                    value=f"{value:.4f}" if isinstance(value, float) else str(value)
                )
    else:
        st.info("No metrics available")


def create_file_uploader(
    label: str = "Upload File",
    accept_types: list = None,
    help_text: str = None
) -> Optional[bytes]:
    """
    Create file uploader.
    
    Args:
        label: Uploader label
        accept_types: Accepted file types
        help_text: Help text
        
    Returns:
        Uploaded file bytes or None
    """
    if accept_types is None:
        accept_types = ['txt', 'csv', 'json']
    
    return st.file_uploader(
        label,
        type=accept_types,
        help=help_text
    )


def create_text_input(
    label: str,
    default: str = "",
    height: int = 100,
    help_text: str = None
) -> str:
    """
    Create text input area.
    
    Args:
        label: Input label
        default: Default value
        height: Input height
        help_text: Help text
        
    Returns:
        Input text
    """
    return st.text_area(
        label,
        value=default,
        height=height,
        help=help_text
    )


def create_action_buttons() -> dict:
    """
    Create action buttons.
    
    Returns:
        Dictionary of button states
    """
    col1, col2, col3 = st.columns(3)
    
    buttons = {}
    
    with col1:
        buttons['train'] = st.button("Train", type="primary")
    
    with col2:
        buttons['evaluate'] = st.button("Evaluate")
    
    with col3:
        buttons['reset'] = st.button("Reset")
    
    return buttons


def show_training_progress(update: int, total: int, metrics: dict = None) -> None:
    """
    Show training progress.
    
    Args:
        update: Current update
        total: Total updates
        metrics: Current metrics
    """
    progress = update / total
    st.progress(progress, text=f"Training Update: {update}/{total}")
    
    if metrics:
        st.write("**Current Metrics:**")
        for name, value in metrics.items():
            st.write(f"- {name}: {value:.4f}")


def create_log_display(logs: list, title: str = "Training Logs") -> None:
    """
    Create log display.
    
    Args:
        logs: List of log messages
        title: Display title
    """
    st.subheader(title)
    
    if logs:
        log_text = "\n".join(logs[-10:])  # Show last 10 logs
        st.text_area("", value=log_text, height=200, disabled=True)
    else:
        st.info("No logs available")
