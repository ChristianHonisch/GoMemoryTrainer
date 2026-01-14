import tkinter as tk
import random
from enum import Enum
from functools import partial


class Stone(Enum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2


class Phase(Enum):
    SHOWING = 1
    INPUT = 2
    DONE = 3


RANK_PRESETS = {
    "30k": {"board": 9,  "stones": 6,  "time": 10},
    "20k": {"board": 9,  "stones": 10, "time": 8},
    "10k": {"board": 13, "stones": 20, "time": 7},
    "5k":  {"board": 13, "stones": 30, "time": 6},
    "1k":  {"board": 19, "stones": 45, "time": 5},
    "1d":  {"board": 19, "stones": 60, "time": 4},
}

MISSING_COLORS = {
    Stone.BLACK: "#555555",   # dark gray
    Stone.WHITE: "#EAEAE0",   # off-white
}

class GoMemoryTrainer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Go Memory Trainer")

        self.board_size = 9
        self.num_random_stones = 12
        self.show_time_sec = 5

        self.cell_size = 50
        self.margin = 40

        self.phase = Phase.SHOWING
        self.placement_mode = "alternating"
        self.current_player = Stone.BLACK

        self.hide_timer_id = None
        self.remaining_time = 0
        self.score = 0

        self.original_state = []
        self.user_state = []

        self._build_controls()
        self._build_canvas()
        self._reset_game()

    # ---------- UI ----------

    def _build_controls(self):
        top = tk.Frame(self.root)
        top.pack(pady=4)

        def labeled_entry(label, default):
            frame = tk.Frame(top)
            frame.pack(side=tk.LEFT, padx=4)
            tk.Label(frame, text=label).pack()
            e = tk.Entry(frame, width=5)
            e.insert(0, str(default))
            e.pack()
            return e

        self.board_size_entry = labeled_entry("Board", self.board_size)
        self.stones_entry = labeled_entry("Stones", self.num_random_stones)
        self.timer_entry = labeled_entry("Time (s)", self.show_time_sec)

        self.time_label = tk.Label(self.root, text="Time: –")
        self.time_label.pack()

        self.score_label = tk.Label(self.root, text="Score: –")
        self.score_label.pack()

        modes = tk.Frame(self.root)
        modes.pack(pady=4)

        self.mode_buttons = {}

        for mode in ("alternating", "black", "white", "delete"):
            btn = tk.Button(
                modes,
                text=mode,
                width=10,
                command=partial(self._set_mode, mode),
                relief=tk.RAISED,
            )
            btn.pack(side=tk.LEFT, padx=2)
            self.mode_buttons[mode] = btn

        buttons = tk.Frame(self.root)
        buttons.pack(pady=4)

        self.hide_button = tk.Button(
            buttons, text="Hide now", command=self._hide_board
        )
        self.hide_button.pack(side=tk.LEFT)

        self.done_button = tk.Button(
            buttons, text="Done", command=self._finish
        )
        self.done_button.pack(side=tk.LEFT)

        self.reset_button = tk.Button(
            buttons, text="Reset / New", command=self._reset_game
        )
        self.reset_button.pack(side=tk.LEFT)

        self._set_mode("alternating") ## not sure if theis is the right location

    def _build_canvas(self):
        self.canvas = tk.Canvas(self.root, bg="#DEB887")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Button-3>", self._on_right_click)

    def _set_mode(self, mode):
        self.placement_mode = mode

        for m, btn in self.mode_buttons.items():
            btn.config(relief=tk.SUNKEN if m == mode else tk.RAISED)


    def _update_button_states(self):
        if self.phase == Phase.SHOWING:
            self.done_button.config(state=tk.DISABLED)
            self.hide_button.config(state=tk.NORMAL)
        elif self.phase == Phase.INPUT:
            self.done_button.config(state=tk.NORMAL)
            self.hide_button.config(state=tk.DISABLED)
        else:  # DONE
            self.done_button.config(state=tk.DISABLED)
            self.hide_button.config(state=tk.DISABLED)
            
    # ---------- Game setup ----------

    def _read_parameters(self):
        self.board_size = max(5, int(self.board_size_entry.get()))
        self.num_random_stones = max(0, int(self.stones_entry.get()))
        self.show_time_sec = max(0, int(self.timer_entry.get()))

    def _reset_game(self):
        if self.hide_timer_id:
            self.root.after_cancel(self.hide_timer_id)

        self._read_parameters()

        self.canvas_size = (
            2 * self.margin + (self.board_size - 1) * self.cell_size
        )
        self.canvas.config(width=self.canvas_size, height=self.canvas_size)

        self.phase = Phase.SHOWING
        self._update_button_states()
        self.current_player = Stone.BLACK
        self.remaining_time = self.show_time_sec
        self.score = 0

        self.original_state = self._empty_state()
        self.user_state = self._empty_state()

        self._generate_random_position()
        self._draw()
        self._update_timer()

    def _empty_state(self):
        return [
            [Stone.EMPTY for _ in range(self.board_size)]
            for _ in range(self.board_size)
        ]

    def _generate_random_position(self):
        points = [(r, c) for r in range(self.board_size) for c in range(self.board_size)]
        random.shuffle(points)

        for i, (r, c) in enumerate(points[: self.num_random_stones]):
            self.original_state[r][c] = Stone.BLACK if i % 2 == 0 else Stone.WHITE

    # ---------- Timer ----------

    def _update_timer(self):
        if self.phase != Phase.SHOWING:
            return

        self.time_label.config(text=f"Time: {self.remaining_time}s")

        if self.remaining_time <= 0:
            self._hide_board()
            return

        self.remaining_time -= 1
        self.hide_timer_id = self.root.after(1000, self._update_timer)

    # ---------- Drawing ----------

    def _draw(self):
        self.canvas.delete("all")
        self._draw_grid()
        self._draw_hoshi()

        if self.phase == Phase.SHOWING:
            self._draw_stones(self.original_state)
        elif self.phase == Phase.INPUT:
            self._draw_stones(self.user_state)
        elif self.phase == Phase.DONE:
            self._draw_result_view()

    def _draw_result_view(self):
        """
        Draws:
        - User stones
        - Missing stones (ghosted)
        - Evaluation symbols
        """
        r = self.cell_size * 0.45

        # --- Draw stones (user + missing originals) ---
        for row in range(self.board_size):
            for col in range(self.board_size):
                u = self.user_state[row][col]
                o = self.original_state[row][col]

                cx = self.margin + col * self.cell_size
                cy = self.margin + row * self.cell_size

                if u != Stone.EMPTY:
                    color = "black" if u == Stone.BLACK else "white"
                    self.canvas.create_oval(
                        cx - r, cy - r,
                        cx + r, cy + r,
                        fill=color,
                        outline="black",
                        width=2,
                    )

                elif o != Stone.EMPTY:
                    # Missing stone
                    self.canvas.create_oval(
                        cx - r, cy - r,
                        cx + r, cy + r,
                        fill=MISSING_COLORS[o],
                        outline="black",
                        width=1,
                    )

        # --- Draw evaluation symbols for user stones only ---
        self._draw_evaluation()


    def _draw_grid(self):
        for i in range(self.board_size):
            x = self.margin + i * self.cell_size
            self.canvas.create_line(x, self.margin, x, self.canvas_size - self.margin, width=2)
            self.canvas.create_line(self.margin, x, self.canvas_size - self.margin, x, width=2)

    def _draw_hoshi(self):
        hoshi_map = {
            9:  [2, 4, 6],
            13: [3, 6, 9],
            19: [3, 9, 15],
        }
        if self.board_size not in hoshi_map:
            return

        r = 4
        for i in hoshi_map[self.board_size]:
            for j in hoshi_map[self.board_size]:
                x = self.margin + j * self.cell_size
                y = self.margin + i * self.cell_size
                self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")

    def _draw_stones(self, state):
        r = self.cell_size * 0.45
        for i in range(self.board_size):
            for j in range(self.board_size):
                s = state[i][j]
                if s == Stone.EMPTY:
                    continue
                cx = self.margin + j * self.cell_size
                cy = self.margin + i * self.cell_size
                color = "black" if s == Stone.BLACK else "white"
                self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                        fill=color, outline="black", width=2)

    def _draw_evaluation(self):
        for r in range(self.board_size):
            for c in range(self.board_size):
                u = self.user_state[r][c]
                if u == Stone.EMPTY:
                    continue

                cx = self.margin + c * self.cell_size
                cy = self.margin + r * self.cell_size

                status = classify_stone(self.original_state, self.user_state, r, c)

                if status == "correct":
                    text, color = "✓", "green"
                elif status == "almost":
                    text, color = "✚", "#B8860B"
                else:
                    text, color = "✗", "red"

                self.canvas.create_text(cx, cy, text=text, fill=color,
                                        font=("Arial", 24, "bold"))

    # ---------- Interaction ----------

    def _on_click(self, event):
        if self.phase != Phase.INPUT:
            return

        col = round((event.x - self.margin) / self.cell_size)
        row = round((event.y - self.margin) / self.cell_size)

        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return

        if self.placement_mode == "delete":
            self.user_state[row][col] = Stone.EMPTY
            self._draw()
            return

        if self.placement_mode == "alternating":
            stone = self.current_player
            self.current_player = (
                Stone.WHITE if self.current_player == Stone.BLACK else Stone.BLACK
            )
        elif self.placement_mode == "black":
            stone = Stone.BLACK
        else:  # white
            stone = Stone.WHITE

        self.user_state[row][col] = stone
        self._draw()

    def _on_right_click(self, event):
        if self.phase != Phase.INPUT:
            return

        col = round((event.x - self.margin) / self.cell_size)
        row = round((event.y - self.margin) / self.cell_size)

        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return

        self.user_state[row][col] = Stone.EMPTY
        self._draw()
    # ---------- Phase control ----------

    def _hide_board(self):
        self.phase = Phase.INPUT
        self._update_button_states()
        self._draw()

    def _finish(self):
        self.phase = Phase.DONE
        self._update_button_states()
        self.score = score_position(self.original_state, self.user_state)
        self.score_label.config(text=f"Score: {self.score}")
        self._draw()

    def run(self):
        self.root.mainloop()


# ---------- Evaluation helpers ----------

def classify_stone(original, user, r, c):
    u = user[r][c]
    o = original[r][c]

    if u == o:
        return "correct"

    if o != Stone.EMPTY and u != Stone.EMPTY:
        # wrong color, right place
        return "almost"

    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < len(original) and 0 <= cc < len(original):
            if original[rr][cc] == u:
                return "almost"

    return "wrong"


def score_position(original, user) -> int:
    score = 0
    size = len(original)

    used_original = [[False]*size for _ in range(size)]

    for r in range(size):
        for c in range(size):
            u = user[r][c]
            o = original[r][c]

            if u == Stone.EMPTY and o != Stone.EMPTY:
                score -= 2
            elif u != Stone.EMPTY:
                if u == o:
                    score += 3
                    used_original[r][c] = True
                elif o != Stone.EMPTY:
                    score += 1
                else:
                    score -= 2

    return score


if __name__ == "__main__":
    GoMemoryTrainer().run()
