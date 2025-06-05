import chess
import chess.engine
import numpy as np
from typing import List, Tuple, Optional
import random
from dataclasses import dataclass
import math

@dataclass
class TacticalPosition:
    """Represents a chess position with tactical themes."""
    board: chess.Board
    score: float = 0.0
    themes: List[str] = None
    difficulty: float = 0.0
    
    def __post_init__(self):
        if self.themes is None:
            self.themes = []

class TacticsGenerator:
    """Generate chess tactics using simulated annealing."""
    
    def __init__(
        self,
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.95,
        min_temperature: float = 0.01,
        iterations_per_temp: int = 100
    ):
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.iterations_per_temp = iterations_per_temp
        
        # Tactical themes and their weights
        self.themes = {
            'fork': 1.0,
            'pin': 1.0,
            'skewer': 1.0,
            'discovered_attack': 1.2,
            'double_attack': 1.1,
            'overloaded_piece': 1.0,
            'trapped_piece': 0.9
        }
    
    def _calculate_energy(self, position: TacticalPosition) -> float:
        """Calculate the energy (inverse of quality) of a tactical position."""
        energy = 0.0
        
        # Penalize positions with too few pieces
        piece_count = len(position.board.piece_map())
        if piece_count < 6:
            energy += 10.0
        
        # Reward positions with tactical themes
        theme_score = sum(self.themes.get(theme, 0.0) for theme in position.themes)
        energy -= theme_score
        
        # Penalize very unbalanced material
        white_material = self._calculate_material(position.board, chess.WHITE)
        black_material = self._calculate_material(position.board, chess.BLACK)
        material_diff = abs(white_material - black_material)
        if material_diff > 5:  # More than a rook difference
            energy += material_diff
        
        # Penalize positions that are too forcing
        if len(list(position.board.legal_moves)) < 3:
            energy += 5.0
        
        return energy
    
    def _calculate_material(self, board: chess.Board, color: chess.Color) -> float:
        """Calculate material value for a side."""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }
        
        return sum(
            piece_values[piece.piece_type]
            for piece in board.piece_map().values()
            if piece.color == color
        )
    
    def _detect_themes(self, board: chess.Board) -> List[str]:
        """Detect tactical themes in a position."""
        themes = []
        
        # Detect forks
        for move in board.legal_moves:
            board.push(move)
            attacked_pieces = 0
            valuable_pieces = 0
            for square in chess.SQUARES:
                if board.is_attacked_by(not board.turn, square):
                    piece = board.piece_at(square)
                    if piece and piece.color == board.turn:
                        attacked_pieces += 1
                        if piece.piece_type in [chess.ROOK, chess.QUEEN, chess.KING]:
                            valuable_pieces += 1
            board.pop()
            
            if attacked_pieces >= 2 and valuable_pieces >= 1:
                themes.append('fork')
                break
        
        # Detect pins
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                if self._is_pinned(board, square):
                    themes.append('pin')
                    break
        
        # Detect discovered attacks
        for move in board.legal_moves:
            if self._is_discovered_attack(board, move):
                themes.append('discovered_attack')
                break
        
        return themes
    
    def _is_pinned(self, board: chess.Board, square: int) -> bool:
        """Check if a piece is pinned."""
        if not board.piece_at(square):
            return False
            
        color = board.piece_at(square).color
        return board.is_pinned(color, square)
    
    def _is_discovered_attack(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if a move is a discovered attack."""
        piece = board.piece_at(move.from_square)
        if not piece:
            return False
            
        # Make the move and check for new attacks
        board.push(move)
        new_attacks = False
        
        for square in chess.SQUARES:
            target = board.piece_at(square)
            if target and target.color != piece.color:
                if board.is_attacked_by(piece.color, square):
                    new_attacks = True
                    break
        
        board.pop()
        return new_attacks
    
    def _neighbor(self, position: TacticalPosition) -> TacticalPosition:
        """Generate a neighboring position by making a random legal move."""
        new_board = position.board.copy()
        
        if len(list(new_board.legal_moves)) == 0:
            return position
        
        # Make a random move
        move = random.choice(list(new_board.legal_moves))
        new_board.push(move)
        
        # Detect themes in new position
        new_themes = self._detect_themes(new_board)
        
        return TacticalPosition(
            board=new_board,
            themes=new_themes
        )
    
    def _acceptance_probability(
        self,
        old_energy: float,
        new_energy: float,
        temperature: float
    ) -> float:
        """Calculate probability of accepting a new state."""
        if new_energy < old_energy:
            return 1.0
        return math.exp((old_energy - new_energy) / temperature)
    
    def generate_tactical_position(
        self,
        initial_position: Optional[chess.Board] = None
    ) -> TacticalPosition:
        """Generate a tactical position using simulated annealing.
        
        Args:
            initial_position: Starting position (default: standard chess position)
            
        Returns:
            TacticalPosition with tactical themes
        """
        if initial_position is None:
            initial_position = chess.Board()
        
        current = TacticalPosition(
            board=initial_position.copy(),
            themes=self._detect_themes(initial_position)
        )
        best = current
        best_energy = self._calculate_energy(best)
        
        temperature = self.initial_temperature
        
        while temperature > self.min_temperature:
            for _ in range(self.iterations_per_temp):
                neighbor = self._neighbor(current)
                current_energy = self._calculate_energy(current)
                neighbor_energy = self._calculate_energy(neighbor)
                
                if self._acceptance_probability(
                    current_energy,
                    neighbor_energy,
                    temperature
                ) > random.random():
                    current = neighbor
                    if neighbor_energy < best_energy:
                        best = neighbor
                        best_energy = neighbor_energy
            
            temperature *= self.cooling_rate
        
        # Calculate difficulty based on themes and material
        best.difficulty = len(best.themes) * 0.3 + self._calculate_material(
            best.board,
            chess.WHITE
        ) * 0.1
        
        return best

def generate_puzzle_sequence(
    generator: TacticsGenerator,
    num_puzzles: int = 10,
    min_difficulty: float = 0.5,
    max_difficulty: float = 3.0
) -> List[TacticalPosition]:
    """Generate a sequence of tactical puzzles with increasing difficulty.
    
    Args:
        generator: TacticsGenerator instance
        num_puzzles: Number of puzzles to generate
        min_difficulty: Minimum puzzle difficulty
        max_difficulty: Maximum puzzle difficulty
        
    Returns:
        List of tactical positions sorted by difficulty
    """
    puzzles = []
    target_difficulties = np.linspace(min_difficulty, max_difficulty, num_puzzles)
    
    for target_diff in target_difficulties:
        # Generate puzzles until we find one with appropriate difficulty
        while True:
            puzzle = generator.generate_tactical_position()
            if abs(puzzle.difficulty - target_diff) < 0.3:
                puzzles.append(puzzle)
                break
    
    return puzzles 