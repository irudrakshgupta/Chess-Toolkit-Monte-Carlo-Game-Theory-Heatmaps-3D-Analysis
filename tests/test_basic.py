import unittest
import chess
import numpy as np
from modules.monte_carlo import MCTS, evaluate_position
from modules.game_theory import NashEquilibriumSolver, OpeningMove
from modules.visualization import create_board_heatmap
from modules.heuristics import TacticsGenerator
from modules.strategic_ai import ChessExplainer

class TestChessAI(unittest.TestCase):
    def setUp(self):
        self.board = chess.Board()
        
    def test_mcts(self):
        """Test Monte Carlo Tree Search."""
        mcts = MCTS(self.board)
        move, prob = mcts.get_best_move()
        
        self.assertIsInstance(move, chess.Move)
        self.assertIsInstance(prob, float)
        self.assertTrue(0 <= prob <= 1)
        
    def test_position_evaluation(self):
        """Test position evaluation."""
        eval_dict = evaluate_position(self.board)
        
        self.assertIn('win_probability', eval_dict)
        self.assertIn('draw_probability', eval_dict)
        self.assertIn('loss_probability', eval_dict)
        
        total_prob = sum(eval_dict.values())
        self.assertAlmostEqual(total_prob, 1.0, places=5)
        
    def test_nash_equilibrium(self):
        """Test Nash Equilibrium solver."""
        moves = [
            OpeningMove("e4", 0.55, 0.3, 0.4, 1.0),
            OpeningMove("d4", 0.52, 0.35, 0.3, 0.9)
        ]
        
        solver = NashEquilibriumSolver(moves, moves)
        white_probs, black_probs = solver.solve()
        
        self.assertEqual(len(white_probs), len(moves))
        self.assertEqual(len(black_probs), len(moves))
        
        self.assertAlmostEqual(sum(white_probs), 1.0, places=5)
        self.assertAlmostEqual(sum(black_probs), 1.0, places=5)
        
    def test_heatmap(self):
        """Test heatmap creation."""
        data = np.random.rand(8, 8)
        fig = create_board_heatmap(data, "Test Heatmap")
        
        self.assertIsNotNone(fig)
        
    def test_tactics_generator(self):
        """Test tactics generation."""
        generator = TacticsGenerator()
        position = generator.generate_tactical_position()
        
        self.assertIsNotNone(position.board)
        self.assertIsInstance(position.themes, list)
        
    def test_move_explanation(self):
        """Test move explanation."""
        explainer = ChessExplainer()
        
        # Make a move
        move = chess.Move.from_uci("e2e4")
        explanation = explainer.explain_move(self.board, move)
        
        self.assertIsNotNone(explanation.explanation)
        self.assertIsInstance(explanation.themes, list)
        self.assertTrue(0 <= explanation.strength <= 1)

if __name__ == '__main__':
    unittest.main()