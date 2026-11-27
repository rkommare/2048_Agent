# 2048_Agent

## Running Docker
Run once to grant permissions
```bash
> chmod +x build_run.bash
> chmod +x build_dev.bash
> chmod +x clear_docker.bash
> chmod +x clear_container_files.bash
```

Run to build and run container
```bash
> sudo ./build_run.bash
```

Run to build and run container in dev mode
```bash
> sudo ./build_dev.bash
```

Run to delete image and container
```bash
> sudo ./clear_docker.bash
```

Run to delete files created by the container
```bash
> sudo ./clear_container_files.bash
```

Dev mode mounts the entire directory, allowing changes to the source code to be made without rebuilding or restarting the container. The default mode only mounts the `/container_files` directory, where the container saves graphs and weights so they can be accessed outside the container. `clear_container_files.bash` only clears the content of the `/container_files` directory, and will not clear files created outside the directory in dev mode. Default mode only copies necessary files into the container, allowing for a quicker build and cleaner environment. Files that can end up outside `/container_files` during dev mode are ignored by default mode, so it is not necessary to delete them. Those files, as well as all graphs and weights, are ignored by git. In order to push graphs or weights, put them in the `/saved` directory.

## World
The game is defined in the `world.py` file. The `Board` class contains the game state and logic. You can play the game in the terminal using
```bash
> python3 world.py
``` 
The game uses WASD inputs and the enter key must be pressed to lock in a move.

The `Board` class is initialized with:
- `__init__(self,state:str|list[list]=None,score=0)`: creates a `Board` using a string  or 4 x 4 list representation of a board state, or creates an empty `Board` if state is not given or `None`

The most important attributes of the `Board` class for the robot are:
- `board`: the current board state
- `score`: the current score
- `legal():` returns a list of legal moves in the current board state
- `move_left()`/`move_right()`/`move_up()`/`move_down()`: slide the tiles in the given direction
- `rotate()`: rotates the board 90 degrees counterclockwise
- `flip_horizontal()`: flips the board across the horizontal axis

Additional attributes that can be used to create rollouts or simulate future states
- `get_empty()`: returns a list of empty spaces
- `spawn()`: creates either a 2 or a 4 in a random empty space
- `copy()`: creates a copy of the `Board`
- `__str__()`: returns a string representation of the current board state. The representation is flattened to a single line, with 0 translated to -1 and every other number translated with a base 2 logarithm to save memory. The tiles are separated with |.
- `display_board()`: prints the current board state to the terminal

## Policy
The `Policy` object is defined in the `policy.py` file. It holds:
- `table`: dictionary of states and expected values
- `cnn`: Pytorch CNN representation of a policy
- `device`: device for CNN. Defaults to CPU
- `index_to_action`: dictionary that maps indices ranging from 0 to 3 to directions on the board
- `action_to_index`: dictionary that maps directions on the board to list indices.

These functions can be used to load or save a policy:
- `save_policy(filename)`: saves dictionary to file
- `load_policy(filename)`: loads dictionary from a file
- `save_cnn(filename)`: saves CNN to file
- `load_cnn(filename)`: loads CNN from file and puts it on GPU if available

The functions used for training a dictionary are:
- `rotate_state(state)`: takes in a string or list representation of a board and returns the string representations of every possible board state that is identical to the input except a rotation has been applied. Also returns the mapping of directions between the input state and the rotated states. This function is used to make sure effectively identical states are not considered different.
- `find(rotations)`: Takes in the states and direction mappings exactly as outputted by `rotate_state`. Checks the policy's table for the states in rotations. If it finds one, it returns the matched state, the expected values, and the direction mapping. Otherwise, it returns `None`. 

The function used for generating rollouts and testing is:
- `expected_values(state)`: takes in a string representation of a board and returns the expected values of each move, sorted in descending order with ties being broken randomly. If the policy has no information about a move, the expected value is 0. If the policy is a CNN, state can also be an iterable of board states. The iterable must contain actual board states, not string representations. Inputting an iterable is much faster than calling expected_values multiple time when running the policy on GPU.

## Generating Data
The generating code is in the `generate_runs.py` file. 
- `generate(n,policy,exploration_rate)`: generates `n` rollouts using `policy` and `exploration_rate`. Each rollout is saved as a line in `rollouts.txt`. Each line starts with the number of turns in the rollout, then the final score, then the first state, then the first move selected, then the first reward, then the second state, etc. Everything is seperated by commas. The function returns the number of times exploitation was chosen, the number of times exploration was chosen, and the number of times each direction was chosen.
- `generate_batches(batch_size,batches,policy,exploration_rate):` same as `generate`, but generates `batch_size` rollouts simultaneously. Generates `batch_size` * `batches` rollouts in total. Only works with a CNN policy, and is much faster than `generate` when running the policy on GPU.
- `play(index)`: displays the record in `rollouts.txt` that corresponds with `index`. The record is replayed turn-by-turn in the terminal.

Running `generate_runs.py` directly creates 100 rollouts with a blank policy and replays the first one.

## Training the Policy
The code for training the dictionary is in the `train_policy.py` file.
- `train_mc(weight_iterations,n_per_iteration,exploration_rate,a,lam)`: `weight iterations` is the number of times new rollouts are generated. `n_per_iteration` is the number of rollouts in each batch. `exploration_rate` is the exploration rate used for generating rollouts. `a` is the alpha in the weight update formula and lam is the discount factor. Logs the average length, average score, percent of encountered states that were previously unseen, and length of the dictionary for every iteration. Saves the final dictionary to `container_files/mc_policy.pkl` and returns the log.

Running `train_policy.py` directly trains a policy from scratch with 5000 iterations, 100 rollouts per iteration, an exploration rate of .4, an alpha of .3, and a discount factor of .9. Graphs average score and percent of states that are unseen in `container_files/plot.png`.

## Testing Parameters
The `test_parameters.py` file runs 1000 iterations of 100 rollouts with a discount factor of .9 for each combination of exploration rate and alpha ranging from 0 to .9 in intervals of .1. The graphs are saved to `container_files/test_plots/exploration rate=\*,a=\*.png`

## CNN
The `cnn.py` file defines the `CNN` class. The `CNN` class is a PyTorch `Module` with a variable number of convolution layers, followed by three linear layers of decreasing size. The convolution layers do not change the size of the input as the board is already 4 x 4, which is very small. The final layer outputs 4 values, which are the expected value of moving in the four directions.
- `__init__(channels,conv_layers=3)`: creates the network with `channels` internal channels for the convolution layers. The model will have `max(1,conv_layers)` convolution layers.

## Training the CNN
The code for training the CNN are in the `train_cnn.py` file. In order to preproccess data, use:
- `dataset(lam,device='cpu')`: converts the data in the `rollouts.txt` file into a PyTorch `tensor` of states that can be input directly into the CNN and a targets `tensor` that holds the future reward associated with each state and action. `policy.action_to_index` is used to map a reward and action to a single vector. The other directions are `nan`. `lam` is the discount factor used to calculate future rewards. `dataset` also returns the total length and score of the rollouts.
- `rotated_dataset(lam,device='cpu')`: the same as `dataset`, except the data is augmented with random rotation.
- `loss_fn(output,target)`: a custom loss function that calculates mean square error while ignoring `nan`, effectively only calculating gradient for the action that was chosen.
- `train_cnn(epochs,subepoch,batch_size,num_batches,exploration_rate,lr,lam,channels)`: Trains the CNN for `epochs` epochs, generating new rollouts every `subepoch` epochs. `batch_size`, `num_batches`, and `exploration_rate` are used to generate rollouts. `channels` is a hyperparameter for the CNN, `lr` is the learning rate, and `lam` is the discount factor. Saves the weights to `./container_files/cnn_weights.pth` and returns the average length, average score, and average loss of every epoch.

Running `train_cnn.py` runs `train_cnn(100,10,500,1,.2,.05,.9,4)` and plots the average loss, score, and length. The graph is saved to `./container_files/cnn_plot.png`.

## Problem with Monte Carlo
As the figure in `saved/plot.png` shows, over 5000 iterations the average score does not increases. In addition, the percentage of unseen states is about 70% at the end. The percentage seems to be decaying exponentially, so an extremely high number of iterations would be necessary for most encountered states to have been seen before. This is likely why the average score is not increasing, as when most states are unseen, most decisions are random. In addition, if a state does not show up many times, the model does not accurately learn the expected values associated with that state. The number of iterations needed to solve this seems unfeasibly high.
