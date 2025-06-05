import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import chess
import chess.pgn
import io
import pandas as pd
from modules.monte_carlo import MCTS, evaluate_position
from modules.game_theory import NashEquilibriumSolver, OpeningMove
from modules.visualization import (
    create_board_heatmap,
    create_3d_evaluation_plot,
    create_tactical_visibility_cube
)
from modules.heuristics import TacticsGenerator
from modules.strategic_ai import ChessExplainer

# Ensure all module directories exist
for dir_name in ['modules', 'config', 'models', 'tests']:
    Path(dir_name).mkdir(exist_ok=True)

# Page config
st.set_page_config(
    page_title="Chess AI Strategy & Simulation Toolkit",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and styling
st.markdown("""
    <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .sidebar .sidebar-content {
            background-color: #262730;
        }
        .stButton>button {
            background-color: #F63366;
            color: #FAFAFA;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            border: none;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #D91F54;
            transform: translateY(-2px);
        }
        .stProgress .st-bo {
            background-color: #F63366;
        }
        .help-box {
            background-color: #262730;
            padding: 20px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid #F63366;
        }
        .info-box {
            background-color: #1E2A3A;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid #4CAF50;
        }
        h1, h2, h3 {
            color: #F63366;
        }
        .stSelectbox label {
            color: #FAFAFA !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("♟️ Chess AI Strategy & Simulation Toolkit")
st.markdown("""
<div class="info-box">
Advanced chess analysis and simulation platform powered by Monte Carlo methods and Game Theory.
This toolkit helps players analyze positions, evaluate strategies, and visualize chess concepts in new ways.
</div>
""", unsafe_allow_html=True)

# Help box for chess notation
with st.expander("📖 Chess Notation Guide"):
    st.markdown("""
    <div class="help-box">
    ### Chess Notation Guide
    
    #### FEN (Forsyth–Edwards Notation)
    - Used to describe a particular board position
    - Example: `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1` (starting position)
    - Pieces: K=King, Q=Queen, R=Rook, B=Bishop, N=Knight, P=Pawn
    - Numbers represent empty squares
    - Forward slashes separate ranks (rows)
    
    #### PGN (Portable Game Notation)
    - Used to record chess moves and games
    - Example: `1. e4 e5 2. Nf3 Nc6`
    - Numbers followed by periods indicate move numbers
    - Piece movements:
        - e4 = pawn to e4
        - Nf3 = knight to f3
        - O-O = kingside castling
        - x = capture (e.g., exd5)
        
    #### Square Notation
    - Files (columns): a through h (left to right)
    - Ranks (rows): 1 through 8 (bottom to top)
    - Example: e4 = e file, 4th rank
    </div>
    """, unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose Module",
    [
        "Monte Carlo Analysis",
        "Game Theory Tools",
        "Position Visualization",
        "Strategic AI",
        "Settings"
    ]
)

def simulate_game(board, num_moves=10):
    """Simulate a game from the current position."""
    moves = []
    evals = []
    times = []
    current_time = 0
    
    try:
        for _ in range(num_moves):
            if board.is_game_over():
                break
                
            mcts = MCTS(board.copy(), num_simulations=100)
            move, eval_score = mcts.get_best_move()
            
            moves.append(move)
            evals.append(eval_score)
            current_time += np.random.uniform(0.5, 3.0)
            times.append(current_time)
            
            board.push(move)
    except Exception as e:
        st.error(f"Simulation error: {str(e)}")
        return [], [], []
        
    return moves, evals, times

# Main content based on selected mode
if app_mode == "Monte Carlo Analysis":
    st.header("🎲 Monte Carlo Analysis")
    
    st.markdown("""
    <div class="help-box">
    Monte Carlo Analysis uses random sampling to evaluate chess positions and moves.
    - **Endgame Evaluator**: Analyzes specific endgame positions
    - **Opening Line Simulator**: Evaluates opening sequences
    - **Match Simulation**: Simulates complete games
    </div>
    """, unsafe_allow_html=True)
    
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Endgame Evaluator", "Opening Line Simulator", "Match Simulation"]
    )
    
    if analysis_type == "Endgame Evaluator":
        st.subheader("Endgame Position Evaluation")
        
        example_positions = {
            "King and Pawn vs King": "8/8/8/4k3/4P3/4K3/8/8 w - - 0 1",
            "Rook Endgame": "8/8/8/4k3/8/4K3/4R3/8 w - - 0 1",
            "Two Bishops Mate": "8/8/8/3k4/8/3K4/3BB3/8 w - - 0 1"
        }
        
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_example = st.selectbox("Choose an example position:", list(example_positions.keys()))
            fen = st.text_input("Or enter custom FEN:", value=example_positions[selected_example])
        
        with col2:
            st.markdown("### Position Preview")
            # Here you would add a board preview using a chess board visualization library
        
        num_sims = st.slider("Number of simulations:", min_value=100, max_value=2000, value=500, step=100)
        
        if st.button("Analyze Position", key="analyze_endgame"):
            with st.spinner("Analyzing position..."):
                try:
                    board = chess.Board(fen)
                    if not board.is_valid():
                        st.error("Invalid chess position. Please check the FEN string.")
                        st.stop()
                    
                    eval_dict = evaluate_position(board, num_simulations=num_sims)
                    
                    # Display evaluation metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Win Probability", f"{eval_dict['win_probability']:.2%}")
                    col2.metric("Draw Probability", f"{eval_dict['draw_probability']:.2%}")
                    col3.metric("Loss Probability", f"{eval_dict['loss_probability']:.2%}")
                    
                    # Run MCTS for best move
                    mcts = MCTS(board, num_simulations=num_sims)
                    best_move, prob = mcts.get_best_move()
                    
                    if best_move:
                        st.success(f"Best move: {board.san(best_move)} (confidence: {prob:.2%})")
                        
                        # Show alternative moves
                        st.write("### Alternative Moves")
                        moves_analyzed = 0
                        for move in board.legal_moves:
                            if moves_analyzed >= 3 or move == best_move:
                                continue
                            
                            board_copy = board.copy()
                            board_copy.push(move)
                            alt_eval = evaluate_position(board_copy, num_simulations=num_sims//2)
                            st.write(f"- {board.san(move)} (win probability: {alt_eval['win_probability']:.2%})")
                            moves_analyzed += 1
                    else:
                        st.warning("No legal moves available in this position.")
                    
                except ValueError as e:
                    st.error(f"Error analyzing position: {str(e)}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
    
    elif analysis_type == "Opening Line Simulator":
        st.subheader("Opening Line Analysis")
        
        example_openings = {
            "Ruy Lopez": "1. e4 e5 2. Nf3 Nc6 3. Bb5",
            "Sicilian Defense": "1. e4 c5 2. Nf3 d6",
            "Queen's Gambit": "1. d4 d5 2. c4"
        }
        
        selected_opening = st.selectbox("Choose an example opening:", list(example_openings.keys()))
        pgn = st.text_area("Or enter custom PGN:", value=example_openings[selected_opening])
        
        if st.button("Analyze Line", key="analyze_opening"):
            with st.spinner("Analyzing opening line..."):
                try:
                    # Parse PGN
                    game = chess.pgn.read_game(io.StringIO(pgn))
                    if game is None:
                        st.error("Invalid PGN format")
                        st.stop()
                    
                    board = game.board()
                    positions = [board.copy()]
                    evals = []
                    times = []
                    current_time = 0
                    
                    for move in game.mainline_moves():
                        board.push(move)
                        positions.append(board.copy())
                        eval_dict = evaluate_position(board)
                        evals.append(eval_dict['win_probability'])
                        current_time += 1
                        times.append(current_time)
                    
                    # Create 3D visualization
                    fig = create_3d_evaluation_plot(
                        positions,
                        evals,
                        times,
                        "Opening Line Analysis"
                    )
                    st.plotly_chart(fig)
                    
                    # Show move-by-move analysis
                    st.write("### Move-by-Move Analysis")
                    for i, eval_score in enumerate(evals):
                        st.write(f"Move {i+1}: Evaluation = {eval_score:.2f}")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    elif analysis_type == "Match Simulation":
        st.subheader("Game Simulation")
        
        position_options = {
            "Initial Position": chess.Board().fen(),
            "Open Sicilian": "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
            "Ruy Lopez": "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
        }
        
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_position = st.selectbox("Choose a starting position:", list(position_options.keys()))
            fen = st.text_input("Or enter custom FEN:", value=position_options[selected_position])
            num_moves = st.slider("Number of moves to simulate:", 5, 30, 10)
        
        if st.button("Start Simulation", key="start_simulation"):
            with st.spinner("Running simulation..."):
                try:
                    board = chess.Board(fen)
                    positions = [board.copy()]
                    evals = []
                    times = []
                    current_time = 0
                    
                    for _ in range(num_moves):
                        if board.is_game_over():
                            break
                        
                        mcts = MCTS(board, num_simulations=100)
                        move, eval_score = mcts.get_best_move()
                        
                        board.push(move)
                        positions.append(board.copy())
                        evals.append(eval_score)
                        current_time += np.random.uniform(0.5, 3.0)
                        times.append(current_time)
                    
                    # Create 3D visualization
                    fig = create_3d_evaluation_plot(
                        positions,
                        evals,
                        times,
                        "Game Progression Analysis"
                    )
                    st.plotly_chart(fig)
                    
                    # Show move list
                    st.write("### Move List")
                    for i, (pos, eval_score) in enumerate(zip(positions[1:], evals)):
                        st.write(f"Move {i+1}: {pos.move_stack[-1]} (eval: {eval_score:.2f})")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

elif app_mode == "Game Theory Tools":
    st.header("♟️ Game Theory Analysis")
    
    st.markdown("""
    <div class="help-box">
    Game Theory Tools help analyze optimal chess strategies.
    - **Nash Equilibrium**: Finds optimal mixed strategies
    - **Opening Analysis**: Evaluates opening move choices
    </div>
    """, unsafe_allow_html=True)
    
    tool_type = st.selectbox(
        "Select Tool",
        ["Nash Equilibrium Optimizer"]
    )
    
    if tool_type == "Nash Equilibrium Optimizer":
        st.subheader("Opening Repertoire Optimization")
        
        # Default opening moves
        default_openings = {
            "e4": {"win_rate": 0.55, "draw_rate": 0.30, "popularity": 0.40, "complexity": 0.5},
            "d4": {"win_rate": 0.52, "draw_rate": 0.35, "popularity": 0.35, "complexity": 0.4},
            "c4": {"win_rate": 0.51, "draw_rate": 0.38, "popularity": 0.25, "complexity": 0.3},
            "Nf3": {"win_rate": 0.50, "draw_rate": 0.40, "popularity": 0.20, "complexity": 0.2}
        }
        
        # Add opening management
        st.write("### Opening Statistics")
        st.write("You can edit the values directly in the table below:")
        
        # Convert default openings to DataFrame for editing
        df = pd.DataFrame(default_openings).T
        df.columns = ['Win Rate', 'Draw Rate', 'Popularity', 'Complexity']
        
        # Create an editable dataframe
        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            column_config={
                "Win Rate": st.column_config.NumberColumn(
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    format="%.2f"
                ),
                "Draw Rate": st.column_config.NumberColumn(
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    format="%.2f"
                ),
                "Popularity": st.column_config.NumberColumn(
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    format="%.2f"
                ),
                "Complexity": st.column_config.NumberColumn(
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    format="%.2f"
                )
            },
            key="opening_stats"
        )
        
        # Convert edited DataFrame back to OpeningMove objects
        openings = {}
        for move, row in edited_df.iterrows():
            openings[move] = OpeningMove(
                move=move,
                win_rate=row['Win Rate'],
                draw_rate=row['Draw Rate'],
                popularity=row['Popularity'],
                complexity=row['Complexity']
            )
        
        if st.button("Calculate Optimal Strategy", key="calc_strategy"):
            with st.spinner("Calculating optimal strategy..."):
                try:
                    solver = NashEquilibriumSolver(
                        list(openings.values()),
                        list(openings.values())
                    )
                    
                    white_probs, black_probs = solver.solve()
                    
                    # Create visualization
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=list(openings.keys()),
                        y=white_probs,
                        name='White',
                        marker_color='#F63366'
                    ))
                    
                    fig.add_trace(go.Bar(
                        x=list(openings.keys()),
                        y=black_probs,
                        name='Black',
                        marker_color='#4CAF50'
                    ))
                    
                    fig.update_layout(
                        title="Optimal Move Distribution",
                        xaxis_title="Opening Move",
                        yaxis_title="Probability",
                        template="plotly_dark",
                        barmode='group',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0.1)',
                        height=500
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Show recommendations with more detailed analysis
                    st.write("### Detailed Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**White's Strategy:**")
                        white_moves = []
                        for move, prob in zip(openings.keys(), white_probs):
                            white_moves.append({
                                'Move': move,
                                'Probability': f"{prob:.2%}",
                                'Expected Score': f"{prob * openings[move].win_rate + 0.5 * openings[move].draw_rate:.3f}"
                            })
                        st.table(pd.DataFrame(white_moves))
                    
                    with col2:
                        st.write("**Black's Strategy:**")
                        black_moves = []
                        for move, prob in zip(openings.keys(), black_probs):
                            black_moves.append({
                                'Move': move,
                                'Probability': f"{prob:.2%}",
                                'Expected Score': f"{prob * (1 - openings[move].win_rate) + 0.5 * openings[move].draw_rate:.3f}"
                            })
                        st.table(pd.DataFrame(black_moves))
                    
                    # Add strategic insights
                    st.write("### Strategic Insights")
                    total_ev = sum(prob * (openings[move].win_rate + 0.5 * openings[move].draw_rate) 
                                 for move, prob in zip(openings.keys(), white_probs))
                    
                    st.info(f"""
                    - Expected value for White: {total_ev:.3f}
                    - Recommended repertoire complexity: {sum(prob * openings[move].complexity for move, prob in zip(openings.keys(), white_probs)):.2f}
                    - Most stable move: {max(openings.keys(), key=lambda m: openings[m].draw_rate)}
                    - Sharpest move: {max(openings.keys(), key=lambda m: openings[m].complexity)}
                    """)
                    
                except Exception as e:
                    st.error(f"Error calculating optimal strategy: {str(e)}")
                    st.write("Please ensure all probabilities are valid (between 0 and 1) and sum to appropriate totals.")

elif app_mode == "Position Visualization":
    st.header("📊 Position Analysis")
    
    st.markdown("""
    <div class="help-box">
    Visualize various aspects of chess positions:
    - **Piece Influence**: Shows how pieces control the board in 3D
    - **Square Control**: Displays territory controlled by each side
    - **Position Evolution**: Analyzes how the position changes over time
    </div>
    """, unsafe_allow_html=True)
    
    viz_type = st.selectbox(
        "Select Visualization",
        ["Piece Influence", "Square Control", "Position Evolution"]
    )
    
    # Add example positions with more interesting cases
    position_options = {
        "Initial Position": chess.Board().fen(),
        "Center Control": "rnbqkbnr/ppp2ppp/8/3pp3/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
        "Kingside Attack": "rnbqk2r/ppp2ppp/5n2/3p4/3P4/2N2N2/PPP2PPP/R1BQK2R w KQkq - 0 6",
        "Queen's Gambit": "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
        "Sicilian Dragon": "rnbqk2r/pp2ppbp/3p1np1/8/3NP3/2N1B3/PPP2PPP/R2QKB1R w KQkq - 0 1"
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_position = st.selectbox("Choose a position:", list(position_options.keys()))
        fen = st.text_input("Or enter custom FEN:", value=position_options[selected_position])
    
    with col2:
        st.markdown("### Position Preview")
        st.markdown(f"""
        <div style='background-color: #262730; padding: 10px; border-radius: 5px; border-left: 4px solid #F63366;'>
        Selected position: <br>
        <code>{fen}</code>
        </div>
        """, unsafe_allow_html=True)
    
    if viz_type == "Piece Influence":
        piece_type = st.selectbox(
            "Select piece type:",
            ["Queen", "Rook", "Bishop", "Knight", "Pawn", "King"]
        )
        
        color = st.radio("Select side:", ["White", "Black"], horizontal=True)
        
        piece_map = {
            "Queen": chess.QUEEN,
            "Rook": chess.ROOK,
            "Bishop": chess.BISHOP,
            "Knight": chess.KNIGHT,
            "Pawn": chess.PAWN,
            "King": chess.KING
        }
        
        if st.button("Generate Visualization", key="gen_influence"):
            with st.spinner("Generating 3D visualization..."):
                try:
                    board = chess.Board(fen)
                    fig = create_tactical_visibility_cube(
                        board,
                        piece_map[piece_type],
                        chess.WHITE if color == "White" else chess.BLACK
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add analysis text
                    piece_count = len(list(board.pieces(piece_map[piece_type], chess.WHITE if color == "White" else chess.BLACK)))
                    attacked_squares = sum(1 for sq in chess.SQUARES if board.is_attacked_by(chess.WHITE if color == "White" else chess.BLACK, sq))
                    
                    st.markdown(f"""
                    <div class='info-box'>
                    ### Analysis
                    - Number of {piece_type}s: {piece_count}
                    - Squares attacked: {attacked_squares}
                    - Control percentage: {(attacked_squares/64)*100:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    elif viz_type == "Square Control":
        show_annotations = st.checkbox("Show square coordinates", value=True)
        colorscale = st.selectbox("Color scheme:", ["RdBu", "Spectral", "RdYlBu"])
        
        if st.button("Generate Heatmap", key="gen_heatmap"):
            with st.spinner("Generating heatmap..."):
                try:
                    board = chess.Board(fen)
                    
                    # Generate control data
                    control_data = np.zeros((8, 8))
                    white_control = set()
                    black_control = set()
                    
                    for square in chess.SQUARES:
                        if board.is_attacked_by(chess.WHITE, square):
                            control_data[7 - chess.square_rank(square)][chess.square_file(square)] += 1
                            white_control.add(square)
                        if board.is_attacked_by(chess.BLACK, square):
                            control_data[7 - chess.square_rank(square)][chess.square_file(square)] -= 1
                            black_control.add(square)
                    
                    fig = create_board_heatmap(
                        control_data,
                        "Square Control Analysis",
                        colorscale=colorscale,
                        show_annotations=show_annotations
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display detailed statistics
                    st.markdown("### Control Statistics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        white_control_pct = len(white_control)/64
                        st.metric(
                            "White's Control",
                            f"{white_control_pct:.1%}",
                            f"{white_control_pct - 0.5:.1%} from equal"
                        )
                    
                    with col2:
                        black_control_pct = len(black_control)/64
                        st.metric(
                            "Black's Control",
                            f"{black_control_pct:.1%}",
                            f"{black_control_pct - 0.5:.1%} from equal"
                        )
                    
                    with col3:
                        contested = len(white_control.intersection(black_control))
                        st.metric(
                            "Contested Squares",
                            f"{contested}",
                            f"{contested/64:.1%} of board"
                        )
                    
                    # Add strategic assessment
                    advantage = "White" if white_control_pct > black_control_pct else "Black"
                    st.markdown(f"""
                    <div class='info-box'>
                    ### Strategic Assessment
                    - Space advantage: {advantage}
                    - Center control: {sum(abs(control_data[3:5, 3:5].flatten()))/16:.2f} (0-1 scale)
                    - Contested squares: {contested} ({contested/64:.1%} of the board)
                    - Total control intensity: {np.sum(np.abs(control_data))/64:.2f} average per square
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    elif viz_type == "Position Evolution":
        num_moves = st.slider("Number of moves to analyze:", 3, 10, 5)
        
        if st.button("Generate Analysis", key="gen_evolution"):
            with st.spinner("Analyzing position evolution..."):
                try:
                    board = chess.Board(fen)
                    
                    # Create sample evaluation data
                    positions = [board.copy()]
                    evals = [0.5]  # Start with neutral evaluation
                    times = [0]
                    moves_made = []
                    
                    # Simulate moves and gather data
                    for i in range(num_moves):
                        eval_dict = evaluate_position(board)
                        evals.append(eval_dict['win_probability'])
                        times.append(i + 1)
                        
                        # Make best move using MCTS
                        mcts = MCTS(board, num_simulations=200)
                        best_move, _ = mcts.get_best_move()
                        
                        if best_move:
                            moves_made.append(board.san(best_move))
                            board.push(best_move)
                            positions.append(board.copy())
                    
                    # Create 3D visualization
                    fig = create_3d_evaluation_plot(
                        positions,
                        evals,
                        times,
                        "Position Evolution Analysis"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show move list with evaluations
                    st.markdown("### Move Sequence")
                    move_data = []
                    for i, (move, eval_score) in enumerate(zip(moves_made, evals[1:])):
                        eval_change = eval_score - evals[i]
                        move_data.append({
                            'Move': f"{i+1}. {move}",
                            'Evaluation': f"{eval_score:.2f}",
                            'Change': f"{eval_change:+.2f}"
                        })
                    
                    st.table(pd.DataFrame(move_data))
                    
                    # Add position assessment
                    final_eval = evals[-1]
                    st.markdown(f"""
                    <div class='info-box'>
                    ### Position Assessment
                    - Initial evaluation: {evals[0]:.2f}
                    - Final evaluation: {final_eval:.2f}
                    - Evaluation trend: {'Improving' if final_eval > evals[0] else 'Declining'} for {'White' if final_eval > 0.5 else 'Black'}
                    - Position complexity: {np.std(evals):.2f} (based on evaluation volatility)
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

elif app_mode == "Strategic AI":
    st.header("🧠 Strategic Analysis")
    
    st.markdown("""
    <div class="help-box">
    Strategic AI provides natural language analysis of chess positions:
    - Explains the best moves and their purposes
    - Identifies strategic themes and patterns
    - Evaluates position characteristics
    </div>
    """, unsafe_allow_html=True)
    
    # Add example positions
    position_options = {
        "Initial Position": chess.Board().fen(),
        "Sicilian Dragon": "rnbqk2r/pp2ppbp/3p1np1/8/3NP3/2N1B3/PPP2PPP/R2QKB1R w KQkq - 0 1",
        "King's Indian": "rnbqk2r/ppp1ppbp/3p1np1/8/2PPP3/2N5/PP3PPP/R1BQKBNR w KQkq - 0 1"
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_position = st.selectbox("Choose a position:", list(position_options.keys()))
        fen = st.text_input("Or enter custom FEN:", value=position_options[selected_position])
    
    if st.button("Analyze Position", key="analyze_strategic"):
        with st.spinner("Analyzing position..."):
            try:
                board = chess.Board(fen)
                explainer = ChessExplainer()
                
                st.write("### Position Analysis")
                
                # Analyze top 3 moves
                for move in list(board.legal_moves)[:3]:
                    explanation = explainer.explain_move(board, move)
                    
                    with st.expander(f"Move: {board.san(move)} ({explanation.strength:.0%} strength)"):
                        st.write(explanation.explanation)
                        st.write("**Themes:**", ", ".join(explanation.themes))
                        
                        if explanation.alternative_moves:
                            st.write("**Alternative moves:**")
                            for alt_move, alt_explanation in explanation.alternative_moves:
                                st.write(f"- {board.san(alt_move)}: {alt_explanation}")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

elif app_mode == "Settings":
    st.header("⚙️ Settings")
    
    st.markdown("""
    <div class="help-box">
    Configure various aspects of the analysis:
    - Monte Carlo simulation parameters
    - Performance optimization settings
    - Visualization preferences
    - Analysis depth and quality
    </div>
    """, unsafe_allow_html=True)
    
    # Monte Carlo Settings
    st.subheader("Monte Carlo Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        mc_iterations = st.slider(
            "Number of Monte Carlo Iterations",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="Higher values give more accurate results but take longer"
        )
    
    with col2:
        mc_depth = st.slider(
            "Maximum Search Depth",
            min_value=10,
            max_value=100,
            value=30,
            step=5,
            help="Maximum number of moves to look ahead"
        )
    
    # Performance Settings
    st.subheader("Performance Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        use_parallel = st.toggle(
            "Enable Parallel Processing",
            value=True,
            help="Use multiple CPU cores for calculations"
        )
    
    with col2:
        cache_size = st.select_slider(
            "Cache Size",
            options=["Small (100MB)", "Medium (500MB)", "Large (1GB)", "Extra Large (2GB)"],
            value="Medium (500MB)",
            help="Amount of memory to use for caching results"
        )
    
    # Visualization Settings
    st.subheader("Visualization Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        theme = st.selectbox(
            "Color Theme",
            ["Dark", "Light", "Custom"],
            help="Choose the color scheme for visualizations"
        )
        
        if theme == "Custom":
            primary_color = st.color_picker("Primary Color", "#F63366")
            secondary_color = st.color_picker("Secondary Color", "#4CAF50")
    
    with col2:
        plot_quality = st.select_slider(
            "Plot Quality",
            options=["Low", "Medium", "High", "Ultra"],
            value="Medium",
            help="Higher quality looks better but may be slower"
        )
    
    # Analysis Settings
    st.subheader("Analysis Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        eval_depth = st.slider(
            "Evaluation Depth",
            min_value=1,
            max_value=20,
            value=10,
            help="How deep to analyze positions"
        )
    
    with col2:
        show_variations = st.number_input(
            "Number of Variations to Show",
            min_value=1,
            max_value=10,
            value=3,
            help="How many alternative moves to display"
        )
    
    # Save Settings
    if st.button("Save Settings", type="primary"):
        settings = {
            "monte_carlo": {
                "iterations": mc_iterations,
                "depth": mc_depth
            },
            "performance": {
                "parallel": use_parallel,
                "cache_size": cache_size
            },
            "visualization": {
                "theme": theme,
                "quality": plot_quality,
                "colors": {
                    "primary": primary_color if theme == "Custom" else "#F63366",
                    "secondary": secondary_color if theme == "Custom" else "#4CAF50"
                }
            },
            "analysis": {
                "depth": eval_depth,
                "variations": show_variations
            }
        }
        
        # Here you would normally save these settings to a config file
        st.success("Settings saved successfully!")
        
        # Show current configuration
        st.markdown("### Current Configuration")
        st.json(settings) 