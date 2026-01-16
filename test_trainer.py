"""
test_trainer.py

Developer test harness for GoMemoryTrainer.
Uses pytest for test discovery and reporting.
Visual tests can be skipped with: pytest -m "not visual"
"""

import pytest
from trainer import (
    GoMemoryTrainer,
    Stone,
    Phase,
    score_position,
)


# ---------- Fixtures ----------

@pytest.fixture
def empty_board_factory():
    """Factory fixture for creating empty boards."""
    def _empty_board(size):
        return [[Stone.EMPTY for _ in range(size)] for _ in range(size)]
    return _empty_board


@pytest.fixture
def place_stones_helper():
    """Helper fixture for placing stones on a board."""
    def _place_stones(board, stones):
        for r, c, s in stones:
            board[r][c] = s
        return board
    return _place_stones


@pytest.fixture
def trainer():
    """Fixture providing a fresh GoMemoryTrainer instance."""
    return GoMemoryTrainer()


# ---------- Scoring Tests (Unit Tests) ----------

class TestScoring:
    """All scoring tests - no UI required."""

    @pytest.mark.parametrize("original,user,expected", [
        (
            [(0, 0, Stone.BLACK), (1, 1, Stone.WHITE)],
            [(0, 0, Stone.BLACK), (1, 1, Stone.WHITE)],
            0,
        ),
        ([(0, 0, Stone.BLACK)], [], -2),
        ([], [(0, 0, Stone.BLACK)], -2),
        ([(0, 0, Stone.BLACK)], [(0, 0, Stone.WHITE)], -1),
        ([(3, 3, Stone.BLACK)], [(3, 4, Stone.BLACK)], -1),
        ([(3, 3, Stone.WHITE)], [(4, 3, Stone.WHITE)], -1),
        (
            [(0, 0, Stone.BLACK), (1, 1, Stone.WHITE), (3, 3, Stone.BLACK)],
            [(0, 0, Stone.BLACK), (2, 2, Stone.WHITE), (3, 4, Stone.BLACK)],
            -5,
        ),
        (
            [(0, 0, Stone.BLACK), (1, 1, Stone.BLACK), (2, 2, Stone.BLACK)],
            [],
            -6,
        ),
        (
            [],
            [(0, 0, Stone.BLACK), (1, 1, Stone.BLACK), (2, 2, Stone.BLACK)],
            -6,
        ),
    ], ids=[
        "all_correct",
        "one_missing",
        "one_hallucinated",
        "wrong_color",
        "offset_horizontal",
        "offset_vertical",
        "mixed",
        "multiple_missing",
        "multiple_hallucinated",
    ])
    def test_scoring(self, empty_board_factory, place_stones_helper, original, user, expected):
        """Parametrized scoring tests."""
        original_board = empty_board_factory(9)
        user_board = empty_board_factory(9)
        
        place_stones_helper(original_board, original)
        place_stones_helper(user_board, user)
        
        actual = score_position(original_board, user_board)
        assert actual == expected, f"Score mismatch: expected {expected}, got {actual}"


# ---------- Visual Tests (Integration Tests) ----------

@pytest.mark.visual
class TestVisual:
    """Visual tests that render the UI. Skip with: pytest -m 'not visual'"""

    def _setup_trainer(self, trainer, board_size, original_stones, user_stones):
        """Helper to configure trainer for visual testing."""
        if getattr(trainer, "hide_timer_id", None):
            trainer.root.after_cancel(trainer.hide_timer_id)

        trainer.board_size = board_size
        trainer.original_state = [[Stone.EMPTY for _ in range(board_size)] for _ in range(board_size)]
        trainer.user_state = [[Stone.EMPTY for _ in range(board_size)] for _ in range(board_size)]

        for r, c, s in original_stones:
            trainer.original_state[r][c] = s
        for r, c, s in user_stones:
            trainer.user_state[r][c] = s

        trainer.canvas_size = (
            2 * trainer.margin + (board_size - 1) * trainer.cell_size
        )
        trainer.canvas.config(width=trainer.canvas_size, height=trainer.canvas_size)

        trainer.phase = Phase.DONE
        trainer.score = score_position(trainer.original_state, trainer.user_state)
        trainer.score_label.config(text=f"Score: {trainer.score}")
        trainer._update_button_states()
        trainer._draw()

    def test_correct_position(self, trainer):
        """Render: two correct stones with green checkmarks."""
        self._setup_trainer(
            trainer,
            board_size=9,
            original_stones=[(3, 3, Stone.BLACK), (5, 5, Stone.WHITE)],
            user_stones=[(3, 3, Stone.BLACK), (5, 5, Stone.WHITE)],
        )
        trainer.run()

    def test_almost_correct_offset(self, trainer):
        """Render: stone offset by one with yellow marker."""
        self._setup_trainer(
            trainer,
            board_size=9,
            original_stones=[(3, 3, Stone.BLACK)],
            user_stones=[(3, 4, Stone.BLACK)],
        )
        trainer.run()

    def test_wrong_color_same_place(self, trainer):
        """Render: wrong color stone with yellow marker."""
        self._setup_trainer(
            trainer,
            board_size=9,
            original_stones=[(4, 4, Stone.BLACK)],
            user_stones=[(4, 4, Stone.WHITE)],
        )
        trainer.run()

    def test_hallucinated_and_missing(self, trainer):
        """Render: hallucinated stone (red X) and missing ghost stone."""
        self._setup_trainer(
            trainer,
            board_size=9,
            original_stones=[(5, 5, Stone.WHITE)],
            user_stones=[(2, 2, Stone.BLACK)],
        )
        trainer.run()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
