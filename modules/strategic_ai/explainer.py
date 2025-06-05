import chess
import chess.pgn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class PositionFeatures:
    """Features extracted from a chess position."""
    material_balance: float
    piece_activity: float
    king_safety: float
    pawn_structure: float
    center_control: float
    space_advantage: float
    development: float
    initiative: float

@dataclass
class MoveExplanation:
    """Explanation of a chess move."""
    move: chess.Move
    explanation: str
    strength: float
    themes: List[str]
    alternative_moves: List[Tuple[chess.Move, str]]

def evaluate_piece_mobility(board: chess.Board) -> Dict[chess.PieceType, float]:
    """Calculate mobility scores for each piece type."""
    mobility = {
        chess.PAWN: 0.0,
        chess.KNIGHT: 0.0,
        chess.BISHOP: 0.0,
        chess.ROOK: 0.0,
        chess.QUEEN: 0.0,
        chess.KING: 0.0
    }
    
    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)
        if piece:
            mobility[piece.piece_type] += 1.0
    
    # Normalize scores
    max_score = max(mobility.values()) if mobility.values() else 1.0
    return {k: v/max_score if max_score > 0 else 0 for k, v in mobility.items()}

def identify_themes(board: chess.Board, move: chess.Move) -> List[str]:
    """Identify tactical and strategic themes in a move."""
    themes = []
    
    # Make the move on a copy of the board
    test_board = board.copy()
    test_board.push(move)
    
    # Check basic tactical themes
    if test_board.is_check():
        themes.append("Check")
    if board.is_capture(move):
        themes.append("Capture")
    
    # Check piece-specific themes
    piece = board.piece_at(move.from_square)
    if piece:
        # Pawn themes
        if piece.piece_type == chess.PAWN:
            if chess.square_rank(move.to_square) in [0, 7]:
                themes.append("Promotion")
            if abs(chess.square_file(move.from_square) - chess.square_file(move.to_square)) == 1:
                themes.append("Pawn Structure")
        
        # Knight themes
        if piece.piece_type == chess.KNIGHT:
            if len(list(test_board.attackers(not board.turn, move.to_square))) == 0:
                themes.append("Outpost")
        
        # Bishop/Rook/Queen themes
        if piece.piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
            attackers = board.attackers(board.turn, move.to_square)
            if len(list(attackers)) > 1:
                themes.append("Battery")
        
        # King themes
        if piece.piece_type == chess.KING:
            if abs(chess.square_file(move.from_square) - chess.square_file(move.to_square)) == 2:
                themes.append("Castling")
    
    return themes

def evaluate_move_strength(board: chess.Board, move: chess.Move) -> float:
    """Evaluate the relative strength of a move."""
    # Basic material counting
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    # Make move on copy of board
    test_board = board.copy()
    test_board.push(move)
    
    # Calculate material difference
    material_diff = sum(
        len(test_board.pieces(piece_type, test_board.turn)) * value
        for piece_type, value in piece_values.items()
    ) - sum(
        len(test_board.pieces(piece_type, not test_board.turn)) * value
        for piece_type, value in piece_values.items()
    )
    
    # Calculate positional factors
    center_control = len(list(test_board.attackers(test_board.turn, chess.E4)))
    center_control += len(list(test_board.attackers(test_board.turn, chess.E5)))
    center_control += len(list(test_board.attackers(test_board.turn, chess.D4)))
    center_control += len(list(test_board.attackers(test_board.turn, chess.D5)))
    
    # Normalize and combine factors
    strength = (material_diff + center_control/4) / 10  # Scale to roughly 0-1
    return max(0.1, min(1.0, strength + 0.5))  # Ensure between 0.1 and 1.0

class ChessExplainer:
    """Explains chess positions and moves."""
    
    def explain_move(self, board: chess.Board, move: chess.Move) -> MoveExplanation:
        """Generate an explanation for a chess move."""
        themes = identify_themes(board, move)
        strength = evaluate_move_strength(board, move)
        
        # Generate explanation text
        piece = board.piece_at(move.from_square)
        piece_name = piece.symbol().upper() if piece else 'Piece'
        
        explanation_parts = []
        
        # Basic move description
        explanation_parts.append(f"{piece_name} moves to {chess.square_name(move.to_square)}")
        
        # Add theme-based explanations
        if "Check" in themes:
            explanation_parts.append("giving check")
        if "Capture" in themes:
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                explanation_parts.append(f"capturing the {captured_piece.symbol().upper()}")
        if "Promotion" in themes:
            explanation_parts.append("with pawn promotion")
        if "Battery" in themes:
            explanation_parts.append("creating a battery")
        if "Outpost" in themes:
            explanation_parts.append("establishing an outpost")
        if "Castling" in themes:
            explanation_parts.append("castling to safety")
        
        # Find alternative moves
        alternatives = []
        for alt_move in board.legal_moves:
            if alt_move != move:
                alt_themes = identify_themes(board, alt_move)
                if len(alt_themes) > 0:
                    alt_str = f"Alternative: {board.san(alt_move)} ({', '.join(alt_themes)})"
                    alternatives.append((alt_move, alt_str))
                if len(alternatives) >= 2:  # Limit to 2 alternatives
                    break
        
        return MoveExplanation(
            move=move,
            explanation=" ".join(explanation_parts),
            strength=strength,
            themes=themes,
            alternative_moves=alternatives
        )

    def _extract_features(self, board: chess.Board) -> PositionFeatures:
        """Extract strategic features from a position."""
        return PositionFeatures(
            material_balance=self._evaluate_material(board),
            piece_activity=self._evaluate_piece_activity(board),
            king_safety=self._evaluate_king_safety(board),
            pawn_structure=self._evaluate_pawn_structure(board),
            center_control=self._evaluate_center_control(board),
            space_advantage=self._evaluate_space(board),
            development=self._evaluate_development(board),
            initiative=self._evaluate_initiative(board)
        )
    
    def _evaluate_material(self, board: chess.Board) -> float:
        """Evaluate material balance."""
        white_material = sum(
            len(board.pieces(piece_type, chess.WHITE)) * value
            for piece_type, value in self.piece_values.items()
        )
        black_material = sum(
            len(board.pieces(piece_type, chess.BLACK)) * value
            for piece_type, value in self.piece_values.items()
        )
        return white_material - black_material
    
    def _evaluate_piece_activity(self, board: chess.Board) -> float:
        """Evaluate piece activity and mobility."""
        activity = 0.0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if not piece:
                continue
                
            # Count attacked squares
            attacks = len([
                target for target in chess.SQUARES
                if board.is_attacked_by(piece.color, target)
            ])
            
            # Weight by piece value
            activity += (
                attacks * self.piece_values.get(piece.piece_type, 0) *
                (1 if piece.color == chess.WHITE else -1)
            )
        
        return activity / 100.0  # Normalize
    
    def _evaluate_king_safety(self, board: chess.Board) -> float:
        """Evaluate king safety."""
        safety = 0.0
        
        for color in [chess.WHITE, chess.BLACK]:
            king_square = board.king(color)
            if not king_square:
                continue
            
            # Count defender density around king
            defenders = 0
            for square in board.attacks(king_square):
                if board.is_attacked_by(color, square):
                    defenders += 1
            
            # Count attacker density
            attackers = 0
            for square in board.attacks(king_square):
                if board.is_attacked_by(not color, square):
                    attackers += 1
            
            safety += (defenders - attackers) * (1 if color == chess.WHITE else -1)
        
        return safety / 8.0  # Normalize
    
    def _evaluate_pawn_structure(self, board: chess.Board) -> float:
        """Evaluate pawn structure."""
        structure = 0.0
        
        for color in [chess.WHITE, chess.BLACK]:
            pawns = board.pieces(chess.PAWN, color)
            
            # Count doubled pawns
            files = [chess.square_file(square) for square in pawns]
            doubled = len(files) - len(set(files))
            
            # Count isolated pawns
            isolated = 0
            for pawn in pawns:
                file = chess.square_file(pawn)
                has_neighbors = False
                for neighbor_file in [file - 1, file + 1]:
                    if 0 <= neighbor_file < 8:
                        if any(chess.square_file(p) == neighbor_file for p in pawns):
                            has_neighbors = True
                            break
                if not has_neighbors:
                    isolated += 1
            
            structure += (
                -doubled - isolated * 0.5
            ) * (1 if color == chess.WHITE else -1)
        
        return structure / 8.0  # Normalize
    
    def _evaluate_center_control(self, board: chess.Board) -> float:
        """Evaluate center control."""
        center_squares = [
            chess.E4, chess.E5,
            chess.D4, chess.D5
        ]
        
        control = 0.0
        for square in center_squares:
            if board.is_attacked_by(chess.WHITE, square):
                control += 1
            if board.is_attacked_by(chess.BLACK, square):
                control -= 1
        
        return control / 8.0  # Normalize
    
    def _evaluate_space(self, board: chess.Board) -> float:
        """Evaluate space advantage."""
        space = 0.0
        
        for color in [chess.WHITE, chess.BLACK]:
            # Count squares controlled beyond 4th/5th rank
            rank_limit = 3 if color == chess.WHITE else 4
            controlled = 0
            
            for rank in range(8):
                if (color == chess.WHITE and rank > rank_limit) or \
                   (color == chess.BLACK and rank < rank_limit):
                    for file in range(8):
                        square = chess.square(file, rank)
                        if board.is_attacked_by(color, square):
                            controlled += 1
            
            space += controlled * (1 if color == chess.WHITE else -1)
        
        return space / 32.0  # Normalize
    
    def _evaluate_development(self, board: chess.Board) -> float:
        """Evaluate piece development."""
        development = 0.0
        
        for color in [chess.WHITE, chess.BLACK]:
            # Count developed minor pieces
            home_rank = 0 if color == chess.WHITE else 7
            developed = 0
            
            for piece_type in [chess.KNIGHT, chess.BISHOP]:
                for piece in board.pieces(piece_type, color):
                    if chess.square_rank(piece) != home_rank:
                        developed += 1
            
            development += developed * (1 if color == chess.WHITE else -1)
        
        return development / 4.0  # Normalize
    
    def _evaluate_initiative(self, board: chess.Board) -> float:
        """Evaluate initiative and tempo."""
        # Simple approximation based on number of legal moves
        board.push(chess.Move.null())
        opponent_moves = len(list(board.legal_moves))
        board.pop()
        
        own_moves = len(list(board.legal_moves))
        
        return (own_moves - opponent_moves) / 30.0  # Normalize
    
    def _identify_themes(
        self,
        current: PositionFeatures,
        new: PositionFeatures,
        previous: Optional[PositionFeatures]
    ) -> List[str]:
        """Identify strategic themes in a move."""
        themes = []
        
        # Material changes
        if abs(new.material_balance - current.material_balance) > 0.5:
            themes.append('material')
        
        # Development improvement
        if new.development > current.development + 0.2:
            themes.append('development')
        
        # Center control
        if new.center_control > current.center_control + 0.2:
            themes.append('center_control')
        
        # King safety
        if new.king_safety > current.king_safety + 0.2:
            themes.append('king_safety')
        
        # Pawn structure
        if new.pawn_structure > current.pawn_structure + 0.2:
            themes.append('pawn_structure')
        
        # Piece activity
        if new.piece_activity > current.piece_activity + 0.2:
            themes.append('piece_activity')
        
        # Initiative
        if new.initiative > current.initiative + 0.2:
            themes.append('initiative')
        
        # Prophylaxis (if we have previous position)
        if previous and \
           abs(new.piece_activity - current.piece_activity) < 0.1 and \
           abs(new.material_balance - current.material_balance) < 0.1:
            themes.append('prophylaxis')
        
        return themes
    
    def _generate_explanation(
        self,
        board: chess.Board,
        move: chess.Move,
        themes: List[str],
        current: PositionFeatures,
        new: PositionFeatures
    ) -> str:
        """Generate natural language explanation for a move."""
        piece = board.piece_at(move.from_square)
        if not piece:
            return "Invalid move"
            
        piece_name = chess.piece_name(piece.piece_type).capitalize()
        from_square = chess.square_name(move.from_square)
        to_square = chess.square_name(move.to_square)
        
        # Basic move description
        explanation = f"{piece_name} moves from {from_square} to {to_square}"
        
        # Add tactical elements
        captures = board.is_capture(move)
        gives_check = board.gives_check(move)
        if captures:
            explanation += ", capturing a piece"
        if gives_check:
            explanation += " and giving check"
        
        # Add strategic themes
        if themes:
            explanation += ". This move "
            theme_explanations = [
                self.themes[theme] for theme in themes
                if theme in self.themes
            ]
            explanation += " and ".join(theme_explanations)
        
        return explanation
    
    def _find_alternatives(
        self,
        board: chess.Board,
        main_move: chess.Move,
        num_alternatives: int = 2
    ) -> List[Tuple[chess.Move, str]]:
        """Find and explain alternative moves."""
        alternatives = []
        
        for move in board.legal_moves:
            if move == main_move:
                continue
                
            # Get position features before and after move
            current_features = self._extract_features(board)
            board.push(move)
            new_features = self._extract_features(board)
            board.pop()
            
            # Identify themes
            themes = self._identify_themes(
                current_features,
                new_features,
                None
            )
            
            # Generate brief explanation
            piece = board.piece_at(move.from_square)
            if not piece:
                continue
                
            explanation = f"Alternative: {chess.piece_name(piece.piece_type).capitalize()} to " + \
                         f"{chess.square_name(move.to_square)}"
            
            if themes:
                explanation += f" ({', '.join(themes)})"
            
            alternatives.append((move, explanation))
            
            if len(alternatives) >= num_alternatives:
                break
        
        return alternatives
    
    def _calculate_move_strength(
        self,
        current: PositionFeatures,
        new: PositionFeatures,
        themes: List[str]
    ) -> float:
        """Calculate relative strength of a move."""
        strength = 0.5  # Base strength
        
        # Adjust for material changes
        material_change = new.material_balance - current.material_balance
        strength += np.clip(material_change / 3.0, -0.3, 0.3)
        
        # Adjust for positional improvements
        positional_change = (
            (new.piece_activity - current.piece_activity) +
            (new.king_safety - current.king_safety) +
            (new.pawn_structure - current.pawn_structure) +
            (new.center_control - current.center_control)
        ) / 4.0
        
        strength += np.clip(positional_change, -0.2, 0.2)
        
        # Bonus for multiple themes
        strength += len(themes) * 0.05
        
        return np.clip(strength, 0.0, 1.0) 