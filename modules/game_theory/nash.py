import numpy as np
from typing import Dict, List, Tuple
import chess
import chess.pgn
from dataclasses import dataclass

@dataclass
class OpeningMove:
    """Represents a chess opening move with its properties."""
    move: str
    win_rate: float
    draw_rate: float
    popularity: float
    complexity: float

def create_payoff_matrix(moves: List[OpeningMove]) -> np.ndarray:
    """Create a payoff matrix for the opening moves."""
    n = len(moves)
    matrix = np.zeros((n, n))
    
    for i, white_move in enumerate(moves):
        for j, black_move in enumerate(moves):
            # Calculate payoff based on win rates and draw rates
            white_score = white_move.win_rate + 0.5 * white_move.draw_rate
            black_score = black_move.win_rate + 0.5 * black_move.draw_rate
            
            # Adjust for popularity and complexity
            popularity_factor = (white_move.popularity + black_move.popularity) / 2
            complexity_factor = (white_move.complexity + black_move.complexity) / 2
            
            # Final payoff calculation
            matrix[i][j] = white_score * (1 + 0.1 * popularity_factor) * (1 + 0.05 * complexity_factor)
    
    return matrix

def solve_zero_sum_game(payoff_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve a zero-sum game using a simplified method."""
    n = len(payoff_matrix)
    
    # Initialize mixed strategy with uniform distribution
    strategy = np.ones(n) / n
    
    # Simple iterative method
    for _ in range(100):
        # Best response for opponent
        opponent_best_response = np.zeros(n)
        opponent_best_response[np.argmin(payoff_matrix.T @ strategy)] = 1
        
        # Update strategy
        new_strategy = np.zeros(n)
        new_strategy[np.argmax(payoff_matrix @ opponent_best_response)] = 1
        
        # Mix with previous strategy
        strategy = 0.9 * strategy + 0.1 * new_strategy
        
        # Normalize
        strategy = strategy / strategy.sum()
    
    return strategy, 1 - strategy

class NashEquilibriumSolver:
    """Solver for finding Nash Equilibrium in chess openings."""
    
    def __init__(self, white_moves: List[OpeningMove], black_moves: List[OpeningMove]):
        self.white_moves = white_moves
        self.black_moves = black_moves
        self.payoff_matrix = create_payoff_matrix(white_moves)
    
    def solve(self) -> Tuple[List[float], List[float]]:
        """Find Nash Equilibrium strategies for both players."""
        white_strategy, black_strategy = solve_zero_sum_game(self.payoff_matrix)
        return white_strategy.tolist(), black_strategy.tolist()

def calculate_exploitability(strategy: List[float], payoff_matrix: np.ndarray) -> float:
    """Calculate the exploitability of a strategy."""
    strategy = np.array(strategy)
    best_response_value = max(payoff_matrix.T @ strategy)
    strategy_value = strategy @ payoff_matrix @ strategy
    return best_response_value - strategy_value

class OpeningRepertoireOptimizer:
    """Optimize chess opening repertoire using game theory principles."""
    
    def __init__(self, database: List[Dict]):
        """Initialize with a database of chess games/openings."""
        self.database = database
        self.white_moves: Dict[str, OpeningMove] = {}
        self.black_moves: Dict[str, OpeningMove] = {}
        self._process_database()
    
    def _process_database(self):
        """Process the game database to extract opening statistics."""
        for game in self.database:
            if 'moves' not in game or 'result' not in game:
                continue
            
            first_move = game['moves'].split()[0]
            result = game['result']
            
            # Update white moves
            if first_move not in self.white_moves:
                self.white_moves[first_move] = OpeningMove(
                    move=first_move,
                    win_rate=0.0,
                    draw_rate=0.0,
                    popularity=0.0,
                    complexity=self._calculate_complexity(game['moves'])
                )
            
            # Update statistics
            if result == "1-0":
                self.white_moves[first_move].win_rate += 1
            elif result == "1/2-1/2":
                self.white_moves[first_move].draw_rate += 1
            
            self.white_moves[first_move].popularity += 1
    
    def _calculate_complexity(self, moves: str) -> float:
        """Calculate the complexity of a move sequence."""
        # Simple complexity metric based on number of candidate moves
        board = chess.Board()
        complexity = 0.0
        
        for move in moves.split()[:5]:  # Look at first 5 moves
            try:
                board.push_san(move)
                complexity += len(list(board.legal_moves)) * 0.1
            except:
                break
        
        return complexity
    
    def optimize_repertoire(self, num_variations: int = 3) -> Dict[str, List[Tuple[str, float]]]:
        """Optimize the opening repertoire for both colors.
        
        Args:
            num_variations: Number of variations to include in repertoire
            
        Returns:
            Dictionary with recommended moves and their probabilities
        """
        # Normalize statistics
        total_games = sum(move.popularity for move in self.white_moves.values())
        
        for move in self.white_moves.values():
            move.win_rate /= move.popularity
            move.draw_rate /= move.popularity
            move.popularity /= total_games
        
        # Create solver
        solver = NashEquilibriumSolver(
            list(self.white_moves.values()),
            list(self.white_moves.values())  # Use same moves for black
        )
        
        white_probs, black_probs = solver.solve()
        
        # Get top variations
        white_repertoire = [
            (move.move, prob)
            for move, prob in zip(self.white_moves.values(), white_probs)
        ]
        
        black_repertoire = [
            (move.move, prob)
            for move, prob in zip(self.white_moves.values(), black_probs)
        ]
        
        # Sort by probability and take top variations
        white_repertoire.sort(key=lambda x: x[1], reverse=True)
        black_repertoire.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'white': white_repertoire[:num_variations],
            'black': black_repertoire[:num_variations]
        } 