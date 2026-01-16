import tkinter as tk
import random
from enum import Enum
from functools import partial
from collections import deque

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
DIRECTIONS_DIAGONAL = DIRECTIONS + [(-1, -1), (-1, 1), (1, -1), (1, 1)]

COLORS = {
    "black": "black",
    "white": "white",
    "black_missing": "#555555",
    "white_missing": "#EAEAE0",
    "board": "#DEB887",
    "grid": "black",
}

STONE_RADIUS_FACTOR = 0.45
HOSHI_RADIUS = 5
EVALUATION_FONT = ("Arial", 24, "bold")


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
        self.num_random_stones = 6
        self.show_time_sec = 0

        self.cell_size = 50
        self.margin = 40
        self.stone_radius = self.cell_size * STONE_RADIUS_FACTOR

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
        self.difficulty_label = tk.Label(self.root, text="Difficulty: –")
        self.difficulty_label.pack()

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

        self._set_mode("alternating")

    def _build_canvas(self):
        self.canvas = tk.Canvas(self.root, bg=COLORS["board"])
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
        self.remaining_time = self.show_time_sec

        self._update_button_states()
        self.current_player = Stone.BLACK
        self.score = 0

        self.original_state = self._empty_state()
        self.user_state = self._empty_state()

        self._generate_random_position()
        self.difficulty = estimate_difficulty(self.original_state)
        self.difficulty_label.config(text=f"Difficulty: {self.difficulty:.2f}")

        self._draw()

        if self.show_time_sec != 0:
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
        """Draw user stones, missing stones (ghosted), and evaluation symbols."""
        for row in range(self.board_size):
            for col in range(self.board_size):
                u = self.user_state[row][col]
                o = self.original_state[row][col]
                cx, cy = self._get_canvas_coords(row, col)

                if u != Stone.EMPTY:
                    color = COLORS["black"] if u == Stone.BLACK else COLORS["white"]
                    self._draw_single_stone(cx, cy, color)
                elif o != Stone.EMPTY:
                    color = COLORS["black_missing"] if o == Stone.BLACK else COLORS["white_missing"]
                    self._draw_single_stone(cx, cy, color)

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

        for i in hoshi_map[self.board_size]:
            for j in hoshi_map[self.board_size]:
                cx, cy = self._get_canvas_coords(i, j)
                self.canvas.create_oval(
                    cx - HOSHI_RADIUS, cy - HOSHI_RADIUS,
                    cx + HOSHI_RADIUS, cy + HOSHI_RADIUS,
                    fill="black"
                )

    def _draw_stones(self, state):
        for i in range(self.board_size):
            for j in range(self.board_size):
                s = state[i][j]
                if s == Stone.EMPTY:
                    continue
                cx, cy = self._get_canvas_coords(i, j)
                color = COLORS["black"] if s == Stone.BLACK else COLORS["white"]
                self._draw_single_stone(cx, cy, color)

    def _draw_single_stone(self, cx, cy, color):
        """Draw a single stone at canvas coordinates."""
        self.canvas.create_oval(
            cx - self.stone_radius, cy - self.stone_radius,
            cx + self.stone_radius, cy + self.stone_radius,
            fill=color, outline="black", width=2,
        )

    def _draw_evaluation(self):
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.user_state[r][c] == Stone.EMPTY:
                    continue

                cx, cy = self._get_canvas_coords(r, c)
                status = classify_stone(self.original_state, self.user_state, r, c)

                if status == "correct":
                    text, color = "✓", "green"
                elif status == "almost":
                    text, color = "o", "#B8860B"
                else:
                    text, color = "✗", "red"

                self.canvas.create_text(cx, cy, text=text, fill=color, font=EVALUATION_FONT)

    def _get_canvas_coords(self, row, col):
        """Convert board coordinates to canvas coordinates."""
        cx = self.margin + col * self.cell_size
        cy = self.margin + row * self.cell_size
        return cx, cy

    def _get_board_coords(self, event_x, event_y):
        """Convert canvas coordinates to board coordinates."""
        col = round((event_x - self.margin) / self.cell_size)
        row = round((event_y - self.margin) / self.cell_size)
        return row, col

    def _is_valid_position(self, row, col):
        """Check if position is within board bounds."""
        return 0 <= row < self.board_size and 0 <= col < self.board_size

    # ---------- Interaction ----------

    def _on_click(self, event):
        if self.phase != Phase.INPUT:
            return

        row, col = self._get_board_coords(event.x, event.y)

        if not self._is_valid_position(row, col):
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

        row, col = self._get_board_coords(event.x, event.y)

        if not self._is_valid_position(row, col):
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
        return "almost"

    for dr, dc in DIRECTIONS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < len(original) and 0 <= cc < len(original):
            if original[rr][cc] == u:
                return "almost"

    return "wrong"


def score_position(original, user) -> int:
    size = len(original)
    score = 0

    used_original = [[False]*size for _ in range(size)]
    used_user = [[False]*size for _ in range(size)]

    # 1. Exact matches (correct position + color)
    for r in range(size):
        for c in range(size):
            if user[r][c] != Stone.EMPTY and user[r][c] == original[r][c]:
                used_original[r][c] = True
                used_user[r][c] = True
                # 0 points for correct stone

    # 2. Almost-correct matches
    for r in range(size):
        for c in range(size):
            if used_user[r][c]:
                continue
            u = user[r][c]
            if u == Stone.EMPTY:
                continue

            # Check same position, wrong color
            if original[r][c] != Stone.EMPTY and not used_original[r][c]:
                score -= 1
                used_original[r][c] = True
                used_user[r][c] = True
                continue

            # Check offset by one (Manhattan distance = 1)
            for dr, dc in ((1,0), (-1,0), (0,1), (0,-1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < size and 0 <= cc < size:
                    if (original[rr][cc] != Stone.EMPTY
                        and not used_original[rr][cc]):
                        score -= 1
                        used_original[rr][cc] = True
                        used_user[r][c] = True
                        break

    # 3. Missing stones
    for r in range(size):
        for c in range(size):
            if original[r][c] != Stone.EMPTY and not used_original[r][c]:
                score -= 2

    # 4. Hallucinated stones
    for r in range(size):
        for c in range(size):
            if user[r][c] != Stone.EMPTY and not used_user[r][c]:
                score -= 2

    return score

def estimate_difficulty(board) -> float:
    size = len(board)

    stones = [(r, c) for r in range(size) for c in range(size)
              if board[r][c] != Stone.EMPTY]

    N = len(stones)
    if N == 0:
        return 0.0

    # ---- connected components ----
    visited = [[False]*size for _ in range(size)]
    clusters = []

    for r, c in stones:
        if visited[r][c]:
            continue

        queue = deque([(r, c)])
        visited[r][c] = True
        cluster = [(r, c)]

        while queue:
            rr, cc = queue.popleft()
            for dr, dc in DIRECTIONS:
                nr, nc = rr + dr, cc + dc
                if (0 <= nr < size and 0 <= nc < size
                    and not visited[nr][nc]
                    and board[nr][nc] != Stone.EMPTY):
                    visited[nr][nc] = True
                    queue.append((nr, nc))
                    cluster.append((nr, nc))

        clusters.append(cluster)

    num_clusters = len(clusters)
    isolated_stones = sum(1 for c in clusters if len(c) == 1)
    mean_cluster_size = N / num_clusters

    # ---- bounding box ----
    rows = [r for r, _ in stones]
    cols = [c for _, c in stones]
    bounding_box_area = (
        (max(rows) - min(rows) + 1) *
        (max(cols) - min(cols) + 1)
    )

    # ---- nearest neighbor distance ----
    def manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    nn_distances = []
    for i, p in enumerate(stones):
        d = min(manhattan(p, q)
                for j, q in enumerate(stones) if i != j)
        nn_distances.append(d)

    avg_nn_distance = sum(nn_distances) / len(nn_distances)

    # ---- symmetry ----
    sym = 0.0
    if all(board[r][c] == board[size-1-r][c]
           for r in range(size) for c in range(size)):
        sym += 1
    if all(board[r][c] == board[r][size-1-c]
           for r in range(size) for c in range(size)):
        sym += 1
    if all(board[r][c] == board[size-1-r][size-1-c]
           for r in range(size) for c in range(size)):
        sym += 1
    sym /= 3.0

    difficulty = (
        1.0 * N
        + 2.0 * isolated_stones
        + 1.5 * num_clusters
        + 0.05 * bounding_box_area
        + 1.0 * avg_nn_distance
        - 3.0 * sym
        - 2.0 * mean_cluster_size
    )

    return max(0.0, difficulty)


if __name__ == "__main__":
    GoMemoryTrainer().run()
