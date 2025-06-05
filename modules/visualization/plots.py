import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, List, Tuple, Optional
import chess
import pandas as pd

def create_board_heatmap(
    data: np.ndarray,
    title: str,
    colorscale: str = "RdBu",
    show_annotations: bool = True
) -> go.Figure:
    """Create a chess board heatmap."""
    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    ranks = ['1', '2', '3', '4', '5', '6', '7', '8']
    
    # Create checkerboard pattern for the background
    pattern = np.zeros((8, 8))
    pattern[::2, ::2] = 1
    pattern[1::2, 1::2] = 1
    
    # Create figure with two traces
    fig = go.Figure()
    
    # Add checkerboard pattern
    fig.add_trace(go.Heatmap(
        z=pattern,
        colorscale=[[0, '#1a1a1a'], [1, '#404040']],
        showscale=False,
        hoverongaps=False
    ))
    
    # Add data heatmap with transparency
    fig.add_trace(go.Heatmap(
        z=data,
        colorscale=colorscale,
        opacity=0.8,
        showscale=True,
        hoverongaps=False,
        colorbar=dict(
            title="Strength",
            titleside="right",
            thickness=15,
            len=0.7
        )
    ))
    
    if show_annotations:
        annotations = []
        for i in range(8):
            for j in range(8):
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=f"{files[j]}{ranks[i]}",
                        showarrow=False,
                        font=dict(
                            color="white",
                            size=10
                        )
                    )
                )
        fig.update_layout(annotations=annotations)
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=20)
        ),
        xaxis=dict(
            ticktext=files,
            tickvals=list(range(8)),
            scaleanchor="y",
            scaleratio=1,
            constrain="domain",
            showgrid=False
        ),
        yaxis=dict(
            ticktext=ranks,
            tickvals=list(range(8)),
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
            showgrid=False
        ),
        width=600,
        height=600,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

def create_3d_evaluation_plot(
    positions: List[chess.Board],
    evaluations: List[float],
    move_times: List[float],
    title: str = "3D Position Evaluation"
) -> go.Figure:
    """Create a 3D plot of position evaluations over time."""
    # Calculate complexity for each position
    complexities = []
    for pos in positions:
        # Enhanced complexity metric
        num_legal_moves = len(list(pos.legal_moves))
        num_pieces = len(pos.piece_map())
        attacked_squares = sum(1 for sq in chess.SQUARES if pos.is_attacked_by(chess.WHITE, sq))
        complexity = (num_legal_moves * num_pieces + attacked_squares) / 1500  # Normalize
        complexities.append(complexity)
    
    # Create the 3D scatter plot
    fig = go.Figure()
    
    # Add main trace
    fig.add_trace(go.Scatter3d(
        x=move_times,
        y=evaluations,
        z=complexities,
        mode='lines+markers',
        marker=dict(
            size=6,
            color=evaluations,
            colorscale='RdBu',
            showscale=True,
            colorbar=dict(
                title="Evaluation",
                titleside="right",
                thickness=15,
                len=0.7
            ),
            opacity=0.8
        ),
        line=dict(
            color='rgba(100,100,100,0.8)',
            width=2
        ),
        hovertemplate=
        '<b>Time</b>: %{x:.1f}s<br>' +
        '<b>Evaluation</b>: %{y:.2f}<br>' +
        '<b>Complexity</b>: %{z:.2f}<br>'
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title=dict(text='Time (seconds)', font=dict(size=12)),
            yaxis_title=dict(text='Evaluation', font=dict(size=12)),
            zaxis_title=dict(text='Position Complexity', font=dict(size=12)),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='cube'
        ),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=40, b=0),
        width=800,
        height=800,
        showlegend=False
    )
    
    return fig

def create_tactical_visibility_cube(
    board: chess.Board,
    piece_type: chess.PieceType,
    color: chess.Color
) -> go.Figure:
    """Create a 3D visualization of piece influence."""
    # Create 8x8x3 grid for piece influence
    influence = np.zeros((8, 8, 3))
    
    # Calculate influence for the specified piece type
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type == piece_type and piece.color == color:
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            
            # Add influence based on legal moves and attacks
            for target_square in chess.SQUARES:
                if board.is_attacked_by(color, target_square):
                    to_file = chess.square_file(target_square)
                    to_rank = chess.square_rank(target_square)
                    
                    # Calculate influence strength based on distance and piece value
                    distance = max(abs(to_file - file), abs(to_rank - rank))
                    strength = 1.0 / (distance + 1)
                    
                    # Add to influence grid with height variation
                    influence[to_rank, to_file, 0] += strength  # Base influence
                    influence[to_rank, to_file, 1] += strength * 0.8  # Mid-layer
                    influence[to_rank, to_file, 2] += strength * 0.6  # Top layer
    
    # Create 3D surface plot
    x, y = np.mgrid[0:8, 0:8]
    
    fig = go.Figure()
    
    # Add three layers with different heights and transparencies
    for i, (z_data, opacity) in enumerate(zip(
        influence.transpose(2, 0, 1),
        [0.9, 0.6, 0.3]
    )):
        fig.add_trace(go.Surface(
            x=x,
            y=y,
            z=z_data * (i + 1),  # Scale height by layer
            colorscale='Viridis',
            opacity=opacity,
            showscale=(i == 0),  # Only show colorbar for first layer
            colorbar=dict(
                title="Influence",
                titleside="right",
                thickness=15,
                len=0.7
            ),
            hovertemplate=
            '<b>File</b>: %{x}<br>' +
            '<b>Rank</b>: %{y}<br>' +
            '<b>Influence</b>: %{z:.2f}<br>'
        ))
    
    piece_names = {
        chess.PAWN: "Pawn",
        chess.KNIGHT: "Knight",
        chess.BISHOP: "Bishop",
        chess.ROOK: "Rook",
        chess.QUEEN: "Queen",
        chess.KING: "King"
    }
    
    fig.update_layout(
        title=dict(
            text=f"{piece_names[piece_type]} Influence Map",
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=20)
        ),
        scene=dict(
            xaxis=dict(
                title='File',
                ticktext=list('abcdefgh'),
                tickvals=list(range(8)),
                gridcolor='rgba(128,128,128,0.2)',
                showbackground=False
            ),
            yaxis=dict(
                title='Rank',
                ticktext=list('12345678'),
                tickvals=list(range(8)),
                gridcolor='rgba(128,128,128,0.2)',
                showbackground=False
            ),
            zaxis=dict(
                title='Influence',
                gridcolor='rgba(128,128,128,0.2)',
                showbackground=False
            ),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.75, y=1.75, z=1.75)
            ),
            aspectmode='cube'
        ),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=40, b=0),
        width=800,
        height=800,
        showlegend=False
    )
    
    return fig

def create_phase_space_radar(
    position_features: Dict[str, float],
    title: str = "Position Phase Space"
) -> go.Figure:
    """Create a radar chart of position characteristics.
    
    Args:
        position_features: Dictionary of position features and their values
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    categories = list(position_features.keys())
    values = list(position_features.values())
    
    fig = go.Figure(data=[go.Scatterpolar(
        r=values + [values[0]],  # Close the polygon
        theta=categories + [categories[0]],
        fill='toself'
    )])
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title=title,
        template="plotly_dark",
        showlegend=False,
        width=600,
        height=600
    )
    
    return fig

def create_blunder_rate_heatmap(
    moves: List[str],
    evaluations: List[float],
    threshold: float = 1.0
) -> go.Figure:
    """Create a heatmap showing blunder likelihood.
    
    Args:
        moves: List of moves in the game
        evaluations: List of position evaluations
        threshold: Evaluation difference threshold for blunders
        
    Returns:
        Plotly figure object
    """
    # Calculate evaluation differences
    eval_diffs = np.abs(np.diff(evaluations))
    blunder_matrix = np.zeros((8, 8))
    
    # Map moves to board squares and accumulate blunder rates
    for i, move in enumerate(moves[:-1]):
        try:
            # Parse move to get target square
            move = chess.Move.from_uci(move)
            file = chess.square_file(move.to_square)
            rank = chess.square_rank(move.to_square)
            
            # Add blunder probability
            if eval_diffs[i] > threshold:
                blunder_matrix[rank, file] += 1
        except:
            continue
    
    # Normalize blunder rates
    total_moves = len(moves)
    blunder_matrix = blunder_matrix / total_moves
    
    return create_board_heatmap(
        blunder_matrix,
        "Blunder Rate by Square",
        colorscale="Reds",
        show_annotations=True
    ) 