"""
test_trainer.py

Developer test harness for GoMemoryTrainer.
Allows explicit specification of original and user board positions
and renders the result directly in DONE mode.
"""

from trainer import (
    GoMemoryTrainer,
    Stone,
    Phase,
    score_position,
)


# ---------- Helpers ----------

def empty_board(size):
    """Create an empty Go board."""
    return [[Stone.EMPTY for _ in range(size)] for _ in range(size)]


def place_stones(board, stones):
    """
    Place stones on a board.

    stones: list of (row, col, Stone)
    """
    for r, c, s in stones:
        board[r][c] = s


# ---------- Core test runner ----------

def run_test_case(
    trainer,
    board_size,
    original_stones,
    user_stones,
):
    """
    Force the trainer into DONE mode with predefined positions.

    trainer         : GoMemoryTrainer instance
    board_size      : int
    original_stones : [(row, col, Stone), ...]
    user_stones     : [(row, col, Stone), ...]
    """

    # --- Cancel timers if any ---
    if getattr(trainer, "hide_timer_id", None):
        trainer.root.after_cancel(trainer.hide_timer_id)

    # --- Configure board ---
    trainer.board_size = board_size
    trainer.original_state = empty_board(board_size)
    trainer.user_state = empty_board(board_size)

    place_stones(trainer.original_state, original_stones)
    place_stones(trainer.user_state, user_stones)

    # --- Resize canvas ---
    trainer.canvas_size = (
        2 * trainer.margin + (board_size - 1) * trainer.cell_size
    )
    trainer.canvas.config(
        width=trainer.canvas_size,
        height=trainer.canvas_size,
    )

    # --- Force DONE phase ---
    trainer.phase = Phase.DONE
    trainer.score = score_position(
        trainer.original_state,
        trainer.user_state,
    )

    trainer.score_label.config(text=f"Score: {trainer.score}")
    trainer._update_button_states()
    trainer._draw()


# ---------- Example test cases ----------

def test_almost_correct_offset():
    """
    Generated: black at (3,3)
    User:      black at (3,4)
    Expect:    yellow plus + missing black ghost stone
    """
    trainer = GoMemoryTrainer()

    original = [
        (3, 3, Stone.BLACK),
    ]

    user = [
        (3, 4, Stone.BLACK),
    ]

    run_test_case(
        trainer=trainer,
        board_size=9,
        original_stones=original,
        user_stones=user,
    )

    trainer.run()


def test_wrong_color_same_place():
    """
    Generated: black at (4,4)
    User:      white at (4,4)
    Expect:    yellow plus (wrong color)
    """
    trainer = GoMemoryTrainer()

    original = [
        (4, 4, Stone.BLACK),
    ]

    user = [
        (4, 4, Stone.WHITE),
    ]

    run_test_case(
        trainer=trainer,
        board_size=9,
        original_stones=original,
        user_stones=user,
    )

    trainer.run()


def test_hallucinated_and_missing():
    """
    Generated: white at (5,5)
    User:      black at (2,2)
    Expect:    red X at (2,2), off-white ghost at (5,5)
    """
    trainer = GoMemoryTrainer()

    original = [
        (5, 5, Stone.WHITE),
    ]

    user = [
        (2, 2, Stone.BLACK),
    ]

    run_test_case(
        trainer=trainer,
        board_size=9,
        original_stones=original,
        user_stones=user,
    )

    trainer.run()


# ---------- Entry point ----------

if __name__ == "__main__":
    # Choose which test to run:
    test_almost_correct_offset()
    test_wrong_color_same_place()
    test_hallucinated_and_missing()
