# Snake AI — Q-Learning

Snake game trained with a deep Q-learning neural network built from scratch using PyTorch.

## Architecture
- Input: 11 neurons (food direction ×4, danger ×3, heading ×4)
- Hidden: 16 neurons (ReLU)
- Output: 4 neurons (up/right/down/left)

## How to run
```bash
pip install torch
python snake.py
```
