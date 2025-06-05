# Chess AI Strategy & Simulation Toolkit

A comprehensive chess analysis platform powered by Monte Carlo methods and Game Theory.

## Features

- **Monte Carlo Analysis**
  - Endgame Position Evaluation
  - Opening Line Simulation
  - Match Simulation with 3D Visualization

- **Game Theory Tools**
  - Nash Equilibrium Solver for Opening Repertoire
  - Strategy Optimization
  - Exploitability Analysis

- **Position Analysis**
  - Interactive Heatmaps
  - 3D Piece Influence Visualization
  - Strategic Position Assessment

- **AI Explanations**
  - Natural Language Move Analysis
  - Tactical Theme Detection
  - Alternative Move Suggestions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/chess-ai-toolkit.git
cd chess-ai-toolkit
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
python -m pip install -r requirements.txt
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. Select a module from the sidebar:
   - Monte Carlo Analysis
   - Game Theory Tools
   - Simulated Annealing
   - Heatmap Analysis
   - 3D Visualization
   - Strategic AI

## Module Descriptions

### Monte Carlo Analysis
- Analyze endgame positions using Monte Carlo Tree Search
- Simulate opening lines and evaluate their strength
- Run full game simulations with position evaluation

### Game Theory Tools
- Find optimal opening repertoires using Nash Equilibrium
- Analyze move choices in critical positions
- Calculate strategy exploitability

### Position Analysis
- Generate heatmaps showing piece influence
- Visualize tactical patterns in 3D
- Analyze pawn structures and piece placement

### Strategic AI
- Get natural language explanations of moves
- Identify tactical and strategic themes
- Explore alternative moves and variations

## Input Formats

### FEN (Forsyth–Edwards Notation)
Used to input specific positions. Example:
```
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
```

### PGN (Portable Game Notation)
Used to input game moves. Example:
```
1. e4 e5 2. Nf3 Nc6 3. Bb5
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 