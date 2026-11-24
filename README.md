# 2048_Agent

## Running Docker
Run once to grant permissions
```bash
> chmod +x buildrun.bash
> chmod +x clear_docker.bash
> chmod +x clear_container_files.bash
```

Run to build and run container
```bash
> sudo ./buildrun.bash
```

Run to delete image and container
```bash
> sudo ./clear_docker.bash
```

Run to delete files created by the container
```bash
> sudo ./clear_container_files.bash
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
- rotate(): rotates the board 90 degrees counterclockwise

Additional attributes that can be used to create rollouts or simulate future states
- get_empty(): returns a list of empty spaces
- spawn(): creates either a 2 or a 4 in a random empty space
- copy(): creates a copy of the Board

## Policy
The Policy object is defined in the policy.py file. It holds:
- table: dictionary of states and expected values
- cnn: CNN representation of a policy

These functions can be used to load or save a policy:
- save_policy(filename): saves dictionary to a pkl file
- load_policy(filename): loads dictionary from a pkl file
- save_cnn(filename): saves CNN
- load_cnn(filename): loads CNN

The functions used for training a dictionary are:
- rotate_state(state): takes in a string representation of a board and returns the string representations of every possible board state that is identical to the input except a rotation has been applied. Also returns the mapping of directions between the input state and the rotated states. This function is used to make sure effectively identical states are not considered different.
- find(rotations): Takes in the states and direction mappings exactly as outputted by rotate_state. Checks the policy's table for the states in rotations. If it finds one, it returns the matched state, the expected values, and the direction mapping. Otherwise, it returns None. 

The function used for generating rollouts and testing is:
- expected_values(state): takes in a string representation of a board and returns the expected values of each move, sorted in descending order with ties being broken randomly. If the policy has no information about a move, the expected value is 0.

## Generating Data
The generating code is in the generate_runs.py file. 
- generate(n,policy,exploration_rate): generates n rollouts using policy and exploration_rate. Each rollout is saved as a line in rollouts.txt. Each line starts with the number of turns in the rollout, then the final score, then the first state, then the first move selected, then the first reward, then the second state, etc. Everything is seperated by commas. The function returns the number of times exploitation was chosen, the number of times exploration was chosen, and the number of times each direction was chosen.
- play(index): displays the record in rollouts.txt that corresponds with index. The record is replayed turn-by-turn in the terminal.

Running generate_runs.py directly creates 100 rollouts with a blank policy and replays the first one.

## Training the Policy
The code for training the dictionary is in the train_policy.py file.
- train_mc(weight_iterations,n_per_iteration,exploration_rate,a,lam): weight iterations is the number of times new rollouts are generated. n_per_iteration is the number of rollouts in each batch. exploration_rate is the exploration rate used for generating rollouts. a is the alpha in the weight update formula and lam is the discount factor. Logs the average length, average score, percent of encountered states that were previously unseen, and length of the dictionary for every iteration. Saves the final dictionary to container_files/mc_policy.pkl and returns the log.

Running train_policy.py directly trains a policy from scratch with 5000 iterations, 100 rollouts per iteration, an exploration rate of .4, an alpha of .3, and a discount factor of .9. Graphs average score and percent of states that are unseen in container_files/plot.png.

## Problem with Monte Carlo
As the figure in saved/plot.png shows, over 5000 iterations the average score does not increases. In addition, the percentage of unseen states is about 70% at the end. The percentage seems to be decaying exponentially, so an extremely high number of iterations would be necessary for most encountered states to have been seen before. This is likely why the average score is not increasing, as when most states are unseen, most decisions are random. In addition, if a state does not show up many times, the model does not accurately learn the expected values associated with that state. The number of iterations needed to solve this seems unfeasibly high.