# 2048_Agent

## Running Docker
Run once to grant permissions
```bash
> chmod +x buildrun.sh
> chmod +x clear_docker.sh
```

Run to build and run container
```bash
> sudo ./buildrun.sh
```

Run to delete image and container
```bash
> sudo ./clear_docker.sh
```

## World
The game is defined in the world.py file. The Board class contains the game state and logic. You can play the game in the terminal using
```bash
> python3 world.py
``` 
The game uses WASD inputs and the enter key must be pressed to lock in a move.

The most important attributes of the Board class for the robot are:
- board: the current board state
- score: the current score
- legal(): returns a list of legal moves in the current board state
- move_left()/move_right()/move_up()/move_down(): slide the tiles in the given direction

Additional attributes that can be used to create rollouts or simulate future states
- get_empty(): returns a list of empty spaces
- spawn(): creates either a 2 or a 4 in a random empty space
- copy(): creates a copy of the Board
