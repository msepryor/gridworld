import numpy as np
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
from typing import cast
import matplotlib.pyplot as plt


# 0 = empty
# 1 = goal - 100 points
# 2 = trap - game over, -100 points
# 4 = adversary
# 8 = wall

empty =     " "
goal =      "G"
trap =      "T"
adversary = "A"
wall =      "#"

start_world = np.array(
    [
        ['#', 'G','#','#','#','#','#','#','#', '#'],
        ['#', ' ',' ',' ','#',' ','#','T',' ', '#'],
        ['#', ' ',' ',' ',' ',' ',' ',' ',' ', '#'],
        ['#', '#','#','#','#',' ','#','#','#', '#'],
        ['#', 'A',' ',' ',' ',' ',' ',' ',' ', '#'],
        ['#', ' ',' ',' ',' ','#','#',' ','#', '#'],
        ['#', '#','#','#',' ','#',' ',' ',' ', '#'],
        ['#', ' ',' ',' ',' ','#',' ','T',' ', '#'],
        ['#', 'S',' ','#',' ','#',' ',' ',' ', '#'],
        ['#', '#','#','#','#','#','#','#','#', '#']
    ]
                
)



class Action (Enum):
    NORTH =  (0, (-1, 0))
    EAST  =  (1, (0,  1))
    SOUTH =  (2, (1,  0))
    WEST  =  (3, (0, -1))
    WAIT  =  (4, (0,  0))

    def __init__ (self, index:int, delta:tuple[int,int]):
      self.index = index;
      self.delta = delta;


    def encode(self):
      return np.array (dtype=np.float32, object =
        [
            1 if self == Action.NORTH else 0,
            1 if self == Action.EAST  else 0,
            1 if self == Action.SOUTH else 0,
            1 if self == Action.WEST  else 0,
            1 if self == Action.WAIT  else 0
        ]
      )

    def as_int (self) -> int:
      return list(Action).index(self)


class GameState (Enum):
    RUNNING = "*"
    COMPLETE = "/"

class Environment:

  game_state:GameState

  def __init__ (self):

    self.bit_dict:dict[str,int] = {}
    self.bit_dict[empty]     = 0b0000
    self.bit_dict[goal]      = 0b0001
    self.bit_dict[trap]      = 0b0010
    self.bit_dict[adversary] = 0b0100
    self.bit_dict[wall]      = 0b1000
    self.game_state = GameState.RUNNING
    self.steps = 0
    self.world = np.copy(start_world);
    #print (self.world)
    self.max_steps = self.world.shape[0] * self.world.shape[1]
    self.agent_start = self.find_start_position()
    self.agent_position = np.copy(self.agent_start)
    self.local_observation:np.ndarray = self.create_local_observation()
    self.last_reward = 0
    #print ("Start location = ", self.agent_position)

  def get_DQN_friendly_local_observation (self) -> np.ndarray:
    local_grid = self.local_observation
    arr = np.zeros (4*9, dtype=np.float32)
    for pos, char_cell in enumerate (local_grid.flatten()):
      cell = self.bit_dict[char_cell];
      p = 4 * pos
      arr[p]   = 1 if cell & self.bit_dict[adversary] else 0
      arr[p+1] = 1 if cell & self.bit_dict[trap] else 0
      arr[p+2] = 1 if cell & self.bit_dict[goal] else 0
      arr[p+3] = 1 if cell & self.bit_dict[wall] else 0
    return arr;
  
  def create_local_observation (self) -> np.ndarray:
    # just a demonstration of how much easier it is to do things if you know what you'r doing.
    # I had:
    # agent_row = self.agent_position[0];
    # agent_col = self.agent_position[1];

    # nw_sq = np.array ([agent_row-1, agent_col -1])
    # n_sq  = np.array ([agent_row-1, agent_col   ])
    # ne_sq = np.array ([agent_row-1, agent_col+1 ])
    # w_sq  = np.array ([agent_row,   agent_col-1 ])
    # e_sq  = np.array ([agent_row,   agent_col+1 ])
    # sw_sq = np.array ([agent_row+1, agent_col-1 ])
    # s_sq  = np.array ([agent_row+1, agent_col   ])
    # se_sq = np.array ([agent_row+1, agent_col+1 ])

    # local_grid = np.empty(shape=(3,3),dtype=str);

    # local_grid [0,0] = self.world[nw_sq [0],nw_sq[1]];
    # local_grid [0,1] = self.world[n_sq  [0], n_sq[1]];
    # local_grid [0,2] = self.world[ne_sq [0],ne_sq[1]];

    # local_grid [1,0] = self.world[w_sq  [0], w_sq[1]];
    # local_grid [1,1] = self.world[agent_row, agent_col];
    # local_grid [1,2] = self.world[e_sq  [0], e_sq[1]];

    # local_grid [2,0] = self.world[sw_sq[0], sw_sq[1]];
    # local_grid [2,1] = self.world[s_sq [0],  s_sq[1]];
    # local_grid [2,2] = self.world[se_sq[0], se_sq[1]];

    # return local_grid;

    # ChatGPT suggested replacing it with this - TWO LINES!!:

    r, c = self.agent_position
    return self.world[r-1:r+2, c-1:c+2].copy() # we don't want to accidentally mess with the world array, so we copy it.



  def act (self, action:Action):
    self.agent_position [0] += action.delta[0];
    self.agent_position [1] += action.delta[1];
    self.last_reward = self.calc_reward()
    self.local_observation = self.create_local_observation()
    self.steps += 1

  def calc_reward (self) -> int:
    if self.steps >= self.max_steps:
      self.game_state = GameState.COMPLETE
      return 0
    
    on_square = self.world [self.agent_position[0], self.agent_position[1]];
    reward = 0
    if on_square == adversary:
      reward = -100
      self.game_state = GameState.COMPLETE
    elif on_square == goal:
      reward = 100
      self.game_state = GameState.COMPLETE
    elif on_square == trap:
      reward = -25
    elif on_square == empty:
      reward = -1
    return reward

  def get_available_actions (self) -> list [Action]:
    grid = self.local_observation;
    actions:list[Action] = [];
    if grid[0,1] != wall:
      actions.append(Action.NORTH)
    if grid[1,2] != wall:
      actions.append(Action.EAST)
    if grid[2,1] != wall:
      actions.append(Action.SOUTH)
    if grid[1,0] != wall:
      actions.append(Action.WEST)
    actions.append(Action.WAIT)
    return actions

  def find_start_position (self):
    array = np.argwhere(self.world == "S")
    if array.size == 0:
      raise Exception("No start position defined - put an S in there!")
    pos = array[0]
    self.world[pos[0],pos[1]] = " "
    return pos


class DQN(nn.Module):
    def __init__(self, input_dim:int=43, num_actions:int=len(Action)):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, num_actions)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim) or (input_dim,)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)  # (batch, num_actions)


class MemoryItem():
  def __init__ (self, local_observation:np.ndarray, pre_state:np.ndarray, action:Action, reward:int, post_state:np.ndarray, done:bool, post_valid_actions_mask:np.ndarray):
    self.observation:np.ndarray = local_observation
    self.pre_state:np.ndarray = pre_state
    self.action:Action = action
    self.reward:int = reward
    self.post_state:np.ndarray = post_state
    self.done:bool = done
    self.post_valid_actions_mask:np.ndarray = post_valid_actions_mask

MIN_BUFFER = 1000
EPSILON = 0.2
GAMMA = 0.2

class Agent:

  def __init__ (self, min_buffer:int = 1000, epsilon_start:float = 0.1, gamma:float = 0.99, learning_rate:float = 0.001):
    self.online_network = DQN()
    self.target_network = DQN()
    self.MIN_BUFFER = min_buffer
    self.epsilon_min = 0.05
    self.epsilon = epsilon_start
    self.GAMMA = gamma
    self.learning_rate = learning_rate
    self.loss_history:list[float] = []
    self.total_steps = 0
    self.rewards = 0
    self.environment:Environment|None = None
    self.location_in_world:np.ndarray|None = None
    self.start_position:np.ndarray|None = None
    self.last_action:Action|None = None
    self.memory:deque[MemoryItem] = deque(maxlen=50_000)
    self.optimizer:torch.optim.Optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.learning_rate)

  def explore (self, episodes:int=100):
    self.rewards = 0
    self.scores:dict[int,int] = {}
    for episode in range (episodes):
      self.environment = Environment()
      self.start_position = self.environment.agent_start
      self.location_in_world = self.start_position
      while self.environment.game_state == GameState.RUNNING:
        state_before:np.ndarray = self.encode_state()
        act = self.choose_action (state_before)
        self.environment.act(act)
        self.location_in_world = self.environment.agent_position
        #print (act," -> ", self.environment.agent_position, self.environment.last_reward)
        reward = self.environment.last_reward
        state_after = self.encode_state()
        self.last_action = act
        state:GameState = cast(GameState, self.environment.game_state)
        done:bool = state is GameState.COMPLETE
        next_actions:list[Action] = [] if done else self.environment.get_available_actions()
        self.remember (self.environment.local_observation, state_before, act, reward, state_after, done, next_actions);
        self.rewards += reward
        self.total_steps += 1

      if episode % 100 == 0:
          print (f"Episode {episode} complete. Total reward: {self.rewards}")

      self.scores[episode] = self.rewards
      self.rewards = 0

  def remember (self, observation:np.ndarray, pre_state:np.ndarray, action:Action, reward:int, post_state:np.ndarray, done:bool, valid_actions:list[Action]):
    post_valid_actions_mask = np.zeros (len(Action), dtype=int)
    for a in valid_actions:
      post_valid_actions_mask[a.index] = 1

    i = MemoryItem (observation, pre_state, action, reward, post_state, done, post_valid_actions_mask=post_valid_actions_mask)
    self.memory.append(i)
    self.training_loop()

  def choose_action (self, state_before:np.ndarray) -> Action:
      assert self.environment is not None
      
      actions:list[Action] = self.environment.get_available_actions()
      action:Action = random.choice(actions)
      self.epsilon = max (self.epsilon_min, self.epsilon * 0.9999) # decay epsilon
      if random.random() > self.epsilon:
        #todo: How often should I decrease epsilon? And by how much? I'll ask ChatGPT.
        with torch.no_grad():
          state_tensor = torch.tensor(state_before.copy(), dtype=torch.float32)
          q_values = self.online_network(state_tensor)
          if q_values.dim() == 1:
            q_values = q_values.unsqueeze(0)
          
          masked_q_values = torch.full_like(q_values, float('-inf'))
          for a in actions:
            masked_q_values[0, a.index] = q_values[0, a.index]

          best_action_index = int (masked_q_values.argmax().item())
          action = list(Action)[best_action_index]
      
      return action

  def encode_state (self) -> np.ndarray:
    assert self.environment is not None
    assert self.location_in_world is not None
    assert self.start_position is not None

    view:np.ndarray = self.environment.get_DQN_friendly_local_observation()
    agent_pos_relative = np.array (dtype=np.float32,
        object=[(self.location_in_world[0]-self.start_position[0]) / self.environment.world.shape[0],
         (self.location_in_world[1]-self.start_position[1]) / self.environment.world.shape[1]])
    if self.last_action is None:
      enc_last_action = np.zeros (len(Action), dtype=np.float32)
    else:
      enc_last_action = self.last_action.encode()

    full_state = np.concatenate([view, agent_pos_relative, enc_last_action])
    return full_state

  def training_loop (self):
    assert self.environment is not None

    if len(self.memory) < MIN_BUFFER:
      return
    
    # pick random sample of say 40
    training_sample = random.sample(self.memory, k=40)
    # turn these into tensors!
    S_arr:list[np.ndarray] =  []
    A_arr:list[int] =  []
    R_arr:list[float] =  []
    S2_arr:list[np.ndarray] = []
    D_arr:list[bool]  = []
    V_arr: list[np.ndarray] = [] # valid next actions

    for t in training_sample:
      S_arr.append (t.pre_state.copy());
      A_arr.append (t.action.index)
      R_arr.append (t.reward)
      S2_arr.append (t.post_state.copy())
      D_arr.append (t.done)
      V_arr.append (t.post_valid_actions_mask.copy())
    
    S: torch.Tensor = torch.tensor(np.array(S_arr), dtype=torch.float32)
    S2: torch.Tensor = torch.tensor(np.array(S2_arr), dtype=torch.float32)
    A = torch.tensor(np.array(A_arr), dtype=torch.int64)
    R = torch.tensor(np.array(R_arr), dtype=torch.float32)
    D = torch.tensor(np.array(D_arr), dtype=torch.float32)
    V = torch.tensor(np.array(V_arr), dtype=torch.bool)

    q_values = self.online_network(S)
    q_sa = q_values.gather (1, A.unsqueeze (1)).squeeze(1)

    with torch.no_grad():
      q_next = self.target_network(S2)
      q_next_masked:torch.Tensor = q_next.masked_fill(~V, -1e9)
      max_q_next = q_next_masked.max(dim=1).values
      target = R + self.GAMMA * (1-D) * max_q_next

    loss:torch.Tensor = torch.nn.functional.mse_loss(q_sa, target)
    self.loss_history.append(loss.item())
    self.optimizer.zero_grad()
    loss.backward() # type: ignore
    self.optimizer.step()

    if self.total_steps % 100 == 0:
      self.target_network.load_state_dict(self.online_network.state_dict())

def show_training_stats(agent:Agent):

  plt.figure(figsize=(10,5))
  plt.title("Rewards per episode")
  plt.plot(list(agent.scores.keys()), list(agent.scores.values()))
  plt.xlabel("Episode")
  plt.ylabel("Reward")
  plt.tight_layout()
  plt.savefig("rewards.png", dpi=150)
  plt.close()

  plt.figure(figsize=(10,5))
  plt.title("Loss history")
  plt.plot(agent.loss_history)
  plt.xlabel("Training step")
  plt.ylabel("Loss")
  plt.tight_layout()
  plt.savefig("loss.png", dpi=150)
  plt.close()


agent:Agent = Agent()
agent.explore(episodes=5000)
show_training_stats(agent)

