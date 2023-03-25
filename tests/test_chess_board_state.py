"""Unittest for Chess State."""
import unittest
import chess
from titan.mcts.chess_state import Chess


class TestChess(unittest.TestCase):

    def test_encode_epd(self) -> None:
        """."""
        # Starting position of the game.
        state = Chess(chess.Board())

        result = game.encode_epd()
        expected = [
            6, 7, 8, 9, 10, 8, 7, 6,
            5, 5, 5, 5, 5, 5, 5, 5,
            None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None,
            11, 11, 11, 11, 11, 11, 11, 11,
            0, 1, 2, 3, 4, 2, 1, 0
        ]
        # Tests the encoding of the starting position.
        self.assertEqual(result, expected)
