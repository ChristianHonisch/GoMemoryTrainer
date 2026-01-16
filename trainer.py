import tkinter as tk
import random
from enum import Enum
from functools import partial
from collections import deque
import base64
import csv
import os
import time
from pathlib import Path
from tkinter import simpledialog

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
        self.delay_before_input_sec = 0
        self.username = "default"

        self.cell_size = 50
        self.margin = 40
        self.stone_radius = self.cell_size * STONE_RADIUS_FACTOR

        self.phase = Phase.SHOWING
        self.placement_mode = "alternating"
        self.current_player = Stone.BLACK

        self.hide_timer_id = None
        self.delay_timer_id = None
        self.remaining_time = 0
        self.remaining_delay = 0
        self.score = 0

        self.original_state = []
        self.user_state = []
        
        self.trial_epoch = None
        self.time_used = 0.0

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

        # User selection dropdown
        user_frame = tk.Frame(top)
        user_frame.pack(side=tk.LEFT, padx=4)
        tk.Label(user_frame, text="User").pack()
        
        self.existing_users = load_existing_users()
        self.user_var = tk.StringVar(value=self.existing_users[0])
        self.user_dropdown = tk.OptionMenu(user_frame, self.user_var, *self.existing_users, command=self._on_user_changed)
        self.user_dropdown.pack(side=tk.LEFT)
        
        # Add new user button
        add_user_btn = tk.Button(user_frame, text="+ Add", width=5, command=self._add_new_user)
        add_user_btn.pack(side=tk.LEFT, padx=2)

        self.board_size_entry = labeled_entry("Board", self.board_size)
        self.stones_entry = labeled_entry("Stones", self.num_random_stones)
        self.timer_entry = labeled_entry("Time (s)", self.show_time_sec)
        self.delay_entry = labeled_entry("Delay (s)", self.delay_before_input_sec)

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

    def _on_user_changed(self, value):
        """Handle user selection change."""
        self.username = value

    def _add_new_user(self):
        """Open dialog to add a new user."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add New User")
        dialog.geometry("300x100")
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="Username:").pack(pady=5)
        entry = tk.Entry(dialog, width=30)
        entry.pack(pady=5)
        entry.focus()
        
        def save_user():
            new_username = entry.get().strip()
            if new_username and new_username not in self.existing_users:
                self.existing_users.append(new_username)
                self.existing_users.sort()
                
                # Recreate dropdown with new user
                self.user_dropdown['menu'].delete(0, 'end')
                for user in self.existing_users:
                    self.user_dropdown['menu'].add_command(
                        label=user,
                        command=partial(self._on_user_changed, user)
                    )
                    self.user_var.set(user)
                
                # Ensure CSV file exists for new user
                ensure_csv_headers(new_username)
                dialog.destroy()
        
        tk.Button(dialog, text="OK", command=save_user).pack(pady=5)

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
            # Disable input buttons if delay is active
            if self.remaining_delay > 0:
                self.done_button.config(state=tk.DISABLED)
            else:
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
        self.delay_before_input_sec = max(0, int(self.delay_entry.get()))

    def _reset_game(self):
        if self.hide_timer_id:
            self.root.after_cancel(self.hide_timer_id)
        if self.delay_timer_id:
            self.root.after_cancel(self.delay_timer_id)

        self._read_parameters()

        self.canvas_size = (
            2 * self.margin + (self.board_size - 1) * self.cell_size
        )
        self.canvas.config(width=self.canvas_size, height=self.canvas_size)

        self.phase = Phase.SHOWING
        self.remaining_time = self.show_time_sec
        self.remaining_delay = 0

        self._update_button_states()
        self.current_player = Stone.BLACK
        self.score = 0

        self.original_state = self._empty_state()
        self.user_state = self._empty_state()

        self._generate_random_position()
        self.difficulty = estimate_difficulty(self.original_state)
        self.difficulty_label.config(text=f"Difficulty: {self.difficulty:.2f}")

        self.trial_epoch = time.time()
        self.time_used = 0.0

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
        if self.phase != Phase.INPUT or not self._is_input_allowed():
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
        if self.phase != Phase.INPUT or not self._is_input_allowed():
            return

        row, col = self._get_board_coords(event.x, event.y)

        if not self._is_valid_position(row, col):
            return

        self.user_state[row][col] = Stone.EMPTY
        self._draw()
    # ---------- Phase control ----------

    def _hide_board(self):
        self.phase = Phase.INPUT
        self.time_used = time.time() - self.trial_epoch
        self._update_button_states()
        
        # Start delay timer if configured
        if self.delay_before_input_sec > 0:
            self.remaining_delay = self.delay_before_input_sec
            self._update_delay_timer()
        
        self._draw()

    def _update_delay_timer(self):
        """Update the countdown timer before user can input."""
        if self.phase != Phase.INPUT:
            return

        self.time_label.config(text=f"Wait: {self.remaining_delay}s")

        if self.remaining_delay <= 0:
            self.time_label.config(text="Time: –")
            self._update_button_states()
            return

        self.remaining_delay -= 1
        self.delay_timer_id = self.root.after(1000, self._update_delay_timer)

    def _is_input_allowed(self) -> bool:
        """Check if user input is currently allowed."""
        if self.phase != Phase.INPUT:
            return False
        return self.remaining_delay <= 0

    def _finish(self):
        self.phase = Phase.DONE
        self._update_button_states()
        self.score = score_position(self.original_state, self.user_state)
        self.score_label.config(text=f"Score: {self.score}")
        
        # Save trial data to CSV
        original_b64 = board_to_base64(self.original_state)
        user_b64 = board_to_base64(self.user_state)
        
        save_trial(
            username=self.username,
            epoch=self.trial_epoch,
            board_size=self.board_size,
            stones=self.num_random_stones,
            original_b64=original_b64,
            user_b64=user_b64,
            show_time=self.show_time_sec,
            time_used=self.time_used,
            score=self.score,
            difficulty=self.difficulty
        )
        
        self._draw()

    def run(self):
        self.root.mainloop()


# ---------- Board encoding/decoding ----------

def board_to_base64(board) -> str:
    """Encode board state to base64 string using 2 bits per cell.
    
    Each cell needs 2 bits:
    - 00 = EMPTY (0)
    - 01 = BLACK (1)
    - 10 = WHITE (2)
    Packs 4 cells per byte.
    """
    size = len(board)
    data = bytearray([size])  # first byte is board size
    
    # Pack cells into bytes (4 cells per byte)
    cells_flat = []
    for row in board:
        for cell in row:
            cells_flat.append(cell.value)
    
    for i in range(0, len(cells_flat), 4):
        byte_val = 0
        for j in range(4):
            if i + j < len(cells_flat):
                byte_val |= (cells_flat[i + j] & 0x3) << (j * 2)
        data.append(byte_val)
    
    return base64.b64encode(data).decode('ascii')


def base64_to_board(encoded: str):
    """Decode base64 string back to board state (2 bits per cell)."""
    data = base64.b64decode(encoded.encode('ascii'))
    size = data[0]
    board = []
    
    # Unpack bytes into cells (4 cells per byte)
    cells_flat = []
    for i in range(1, len(data)):
        byte_val = data[i]
        for j in range(4):
            cell_val = (byte_val >> (j * 2)) & 0x3
            cells_flat.append(Stone(cell_val))
    
    # Reconstruct board
    idx = 0
    for _ in range(size):
        row = []
        for _ in range(size):
            if idx < len(cells_flat):
                row.append(cells_flat[idx])
                idx += 1
        board.append(row)
    
    return board


# ---------- CSV file management ----------

def get_trials_dir():
    """Get or create the trials directory."""
    trials_dir = Path("trials")
    trials_dir.mkdir(exist_ok=True)
    return trials_dir


def get_user_csv_path(username: str) -> Path:
    """Get the CSV file path for a specific user."""
    return get_trials_dir() / f"{username}.csv"


def load_existing_users() -> list:
    """Scan trials directory and return list of existing users."""
    trials_dir = get_trials_dir()
    users = []
    
    if trials_dir.exists():
        for csv_file in trials_dir.glob("*.csv"):
            username = csv_file.stem
            users.append(username)
    
    # Always include default
    if "default" not in users:
        users.insert(0, "default")
    else:
        users.remove("default")
        users.insert(0, "default")
    
    return sorted(users)


def ensure_csv_headers(username: str):
    """Create CSV file with headers if it doesn't exist."""
    csv_path = get_user_csv_path(username)
    
    if not csv_path.exists():
        headers = [
            "epoch", "board_size", "stones", "original_b64", "user_b64",
            "show_time", "time_used", "score", "difficulty"
        ]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def save_trial(username: str, epoch: float, board_size: int, stones: int,
               original_b64: str, user_b64: str, show_time: int, time_used: float,
               score: int, difficulty: float):
    """Save a single trial to the user's CSV file."""
    ensure_csv_headers(username)
    csv_path = get_user_csv_path(username)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, board_size, stones, original_b64, user_b64,
            show_time, time_used, score, difficulty
        ])


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
