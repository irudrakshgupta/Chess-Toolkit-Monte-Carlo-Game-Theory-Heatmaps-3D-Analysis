import chess
import numpy as np
from typing import Dict, List, Optional, Tuple
import math
from dataclasses import dataclass
import random

@dataclass
class MCTSNode:
    """Node in the Monte Carlo Tree Search."""
    state: chess.Board
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = None
    wins: int = 0
    visits: int = 0
    untried_moves: List[chess.Move] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.untried_moves is None:
            self.untried_moves = list(self.state.legal_moves)
    
    def ucb1(self, exploration: float = math.sqrt(2)) -> float:
        """Calculate the UCB1 value for this node."""
        if self.visits == 0:
            return float('inf')
        if self.parent is None:  # Root node
            return float('inf')
        return (self.wins / self.visits) + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
    
    def add_child(self, move: chess.Move, state: chess.Board) -> 'MCTSNode':
        """Add a child node with the given move and state."""
        child = MCTSNode(state=state, parent=self)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child
    
    def update(self, result: float):
        """Update node statistics."""
        self.visits += 1
        self.wins += result

class MCTS:
    """Monte Carlo Tree Search implementation for chess."""
    
    def __init__(self, board: chess.Board, num_simulations: int = 1000):
        self.board = board.copy()
        self.num_simulations = num_simulations
        self.root = MCTSNode(state=self.board)
    
    def get_best_move(self) -> Tuple[chess.Move, float]:
        """Run MCTS and return the best move with its evaluation."""
        # If only one legal move, return it immediately
        legal_moves = list(self.board.legal_moves)
        if len(legal_moves) == 0:
            raise ValueError("No legal moves available")
        if len(legal_moves) == 1:
            return legal_moves[0], 1.0
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = self._select(self.root)
            node = self._expand(node)
            if node:  # Only simulate if node was expanded
                result = self._simulate(node.state)
                self._backpropagate(node, result)
        
        # If root has no children (rare case), make a random move
        if not self.root.children:
            move = random.choice(legal_moves)
            return move, 0.5
        
        # Select best child based on visit count
        best_child = max(self.root.children, key=lambda n: n.visits)
        best_move = None
        for move in legal_moves:
            board_copy = self.board.copy()
            board_copy.push(move)
            if board_copy.fen() == best_child.state.fen():
                best_move = move
                break
        
        if best_move is None:  # Fallback if no matching move found
            best_move = random.choice(legal_moves)
        
        return best_move, best_child.wins / max(best_child.visits, 1)
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a leaf node using UCB1."""
        while not node.state.is_game_over():
            if node.untried_moves:  # If there are untried moves, select this node
                return node
            if not node.children:  # If no children, select this node
                return node
            # Select child with highest UCB1 value
            node = max(node.children, key=lambda n: n.ucb1())
        return node
    
    def _expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Expand the node by adding a child."""
        if node.untried_moves and not node.state.is_game_over():
            move = random.choice(node.untried_moves)
            new_state = node.state.copy()
            new_state.push(move)
            return node.add_child(move, new_state)
        return None
    
    def _simulate(self, state: chess.Board) -> float:
        """Run a random simulation from the given state."""
        state = state.copy()
        max_moves = 100  # Prevent infinite games
        moves_played = 0
        
        while not state.is_game_over() and moves_played < max_moves:
            moves = list(state.legal_moves)
            if not moves:
                break
            move = random.choice(moves)
            state.push(move)
            moves_played += 1
        
        if moves_played >= max_moves:
            return 0.5  # Draw if max moves reached
        
        result = state.result()
        if result == "1-0":
            return 1.0 if state.turn == chess.BLACK else 0.0
        elif result == "0-1":
            return 1.0 if state.turn == chess.WHITE else 0.0
        else:  # Draw
            return 0.5
    
    def _backpropagate(self, node: MCTSNode, result: float):
        """Backpropagate the simulation result."""
        while node:
            node.update(result)
            node = node.parent
            if node:  # Flip result for parent's perspective
                result = 1 - result

def evaluate_position(board: chess.Board, num_simulations: int = 100) -> Dict[str, float]:
    """Evaluate a chess position using Monte Carlo simulation."""
    if board.is_game_over():
        result = board.result()
        if result == "1-0":
            return {'win_probability': 1.0, 'draw_probability': 0.0, 'loss_probability': 0.0}
        elif result == "0-1":
            return {'win_probability': 0.0, 'draw_probability': 0.0, 'loss_probability': 1.0}
        else:  # Draw
            return {'win_probability': 0.0, 'draw_probability': 1.0, 'loss_probability': 0.0}
    
    try:
        mcts = MCTS(board, num_simulations=num_simulations)
        _, win_prob = mcts.get_best_move()
        
        # Calculate probabilities
        draw_prob = 0.5 - abs(win_prob - 0.5)  # Higher near 0.5, lower at extremes
        loss_prob = 1.0 - win_prob - draw_prob
        
        return {
            'win_probability': win_prob,
            'draw_probability': draw_prob,
            'loss_probability': max(0.0, loss_prob)  # Ensure non-negative
        }
    except ValueError:
        # Fallback to simple evaluation if MCTS fails
        simple_eval = evaluate_simple_position(board)
        return {
            'win_probability': simple_eval,
            'draw_probability': 0.2,
            'loss_probability': 0.8 - simple_eval
        }

def evaluate_simple_position(board: chess.Board) -> float:
    """Simple position evaluation based on material and piece positions."""
    if board.is_checkmate():
        return 1.0 if board.turn else 0.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.5
    
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    # Central squares for piece position evaluation
    central_squares = {
        chess.E4, chess.E5,
        chess.D4, chess.D5
    }
    
    score = 0.0
    
    # Material evaluation
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            if piece.color == chess.WHITE:
                score += value
                if square in central_squares:
                    score += 0.1
            else:
                score -= value
                if square in central_squares:
                    score -= 0.1
    
    # Normalize to [0, 1]
    return 1 / (1 + math.exp(-score/3))

def analyze_opening_line(pgn: str, num_variations: int = 3) -> List[Dict[str, float]]:
    """Analyze an opening line and suggest variations."""
    game = chess.pgn.read_game(chess.pgn.StringIO(pgn))
    if not game:
        raise ValueError("Invalid PGN")
    
    board = game.board()
    analysis = []
    
    for move in game.mainline_moves():
        board.push(move)
        eval_dict = evaluate_position(board)
        
        # Find alternative moves
        alternatives = []
        for alt_move in board.legal_moves:
            if len(alternatives) >= num_variations:
                break
            board_copy = board.copy()
            board_copy.push(alt_move)
            alt_eval = evaluate_position(board_copy)
            alternatives.append({
                'move': board.san(alt_move),
                'evaluation': alt_eval['win_probability']
            })
        
        analysis.append({
            'position': board.fen(),
            'evaluation': eval_dict,
            'alternatives': alternatives
        })
    
    return analysis 