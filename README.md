# Go Memory Trainer

A tkinter-based application for training Go pattern recognition and memorization skills. Track your progress across multiple users with detailed trial data.

## About

This trainer is based on the instructional lecture by **Jonas Welticke (6d)**, available on the [GoMagic YouTube channel](https://www.youtube.com/@GoMagic). The lecture includes a PDF with example board positions for practice. This application provides a simple way to work through memory exercises and track your progress over time.

## Features

- **Customizable board sizes**: 5×5 up to 19×19 (or larger)
- **Adjustable difficulty**: Control number of stones and time limit
- **Input delay**: Optional delay before memorized position input (useful for practice variations)
- **Multiple users**: Support for multiple learners with individual progress tracking
- **Trial tracking**: CSV-based storage with board positions encoded in compact format
- **Difficulty estimation**: Automatic calculation based on stone distribution and patterns
- **Real-time feedback**: Visual evaluation (✓ correct, ○ almost, ✗ wrong)

## Installation

### Prerequisites

- Python 3.8 or later

### Development Setup

```bash
# Clone or navigate to the repository
cd GoMemoryTrainer

# Install dependencies
pip install -r requirements.txt
```

### Run from Source

```bash
python trainer.py
```

## Building an Executable

### Option 1: Python (Cross-platform)

```bash
python build.py
```

The executable will be created in `dist/GoMemoryTrainer.exe`.

### Option 2: PowerShell (Windows)

```powershell
.\build.ps1
```

## Usage

1. **Select a user** from the dropdown (or click "+ Add" to create a new one)
2. **Configure settings**:
   - Board size (5-19)
   - Number of stones
   - Memorization time (seconds)
   - Delay before input (seconds)
3. **Click "Reset / New"** to generate a random position
4. **Memorize** the board during the showing phase
5. **Click "Hide now"** (or wait for auto-hide if timer is set)
6. **Wait** if a delay is configured, then recreate the position
7. **Click "Done"** to submit and see evaluation

## Data Storage

All trial data is stored in `trials/` folder:
- Each user has a `username.csv` file
- Columns: epoch, board_size, stones, original_b64, user_b64, show_time, time_used, score, difficulty
- Board positions are encoded in compact base64 format (2 bits per cell)
- The `trials/` folder is not tracked by git

## Controls

| Button | Action |
|--------|--------|
| **alternating** | Alternate black/white stones |
| **black** | Place only black stones |
| **white** | Place only white stones |
| **delete** | Erase stones (right-click also erases) |
| **Hide now** | Hide board and start input phase |
| **Done** | Submit answer and see evaluation |
| **Reset / New** | Generate new position and start over |

## Scoring

- **Correct position**: 0 points
- **Wrong color at correct position**: -1 point
- **Off by 1 (Manhattan distance)**: -1 point
- **Missing stone**: -2 points
- **Hallucinated stone**: -2 points

## Development

### Project Structure

```
GoMemoryTrainer/
├── trainer.py              # Main application
├── build.py               # Python build script
├── build.ps1              # PowerShell build script
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project metadata
├── pytest.ini             # Pytest configuration
├── test_trainer.py        # Unit tests
├── trials/                # User data (git-ignored)
└── dist/                  # Executables (git-ignored)
```

### Building for Release

1. Test thoroughly: `python -m pytest`
2. Build executable: `python build.py`
3. Test the executable
4. Distribute `dist/GoMemoryTrainer.exe`

## License

See LICENSE file
