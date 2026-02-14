# Grid World!

We all remember *Wumpus World* from the first AI book we read!

Here is GridWorld.

gridworld.py — DQN navigation in a simple 10×10 gridworld.

This script trains a Deep Q-Network (DQN) agent to navigate from a start square ("S")
to a goal ("G") while avoiding walls ("#"), traps ("T"), and a static adversary ("A").

## Environment

### Grid encoding (characters in `world`):
    - " " : empty (step cost -1)
    - "G" : goal (terminal, +100)
    - "T" : trap (non-terminal penalty -25)
    - "A" : adversary (terminal, -100)
    - "#" : wall (impassable; actions that would enter are illegal)

### Episode termination:
    - Reaching the goal.
    - Colliding with the adversary.
    - Exceeding `max_steps` (default: width*height); returns reward 0 and ends.

## State Representation (current code)

The agent encodes state as a 43-dimensional float vector:

1) Local 3×3 observation patch around the agent (9 cells):
   Each cell becomes 4 binary bits (A, T, G, #) => 9 * 4 = 36 dims

2) Relative position to the start (row_delta_norm, col_delta_norm) => 2 dims
   Normalised by grid height/width so values are typically in [-1, 1] for small maps.

3) One-hot previous action over the current Action enum (N/E/S/W/WAIT) => 5 dims

Total: 36 + 2 + 5 = 43.

## Action Masking
Illegal moves (into walls) are handled in two places:
    1) During action selection: only Q-values for currently available actions are considered.
    2) During bootstrap target computation: next-state Q-values are masked before max().

## Training Setup
- Online network: MLP(43 → 64 → 64 → |Action|)
- Target network: periodically synced with online network (every 100 environment steps)
- Replay buffer: deque of transitions, uniform random mini-batch sampling
- Loss: mean-squared TD error (DQN / Bellman regression)

## Notes / Known Simplifications
- The target update schedule is step-based rather than episode-based.
- Traps are non-terminal (-25) in this implementation

## Run
Executing python.py this file will train for 5000 episodes and save:
    - rewards.png
    - loss.png

## To do
- Make the adversary non-static, possibly with its own DQN policy net
- Test on generalisation (new maps)
