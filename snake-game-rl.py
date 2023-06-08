import itertools
import time
import tkinter
import random
import numpy as np
from agents import *
import argparse
from gym import spaces, Wrapper
from typing import Sequence

from utils import compare_results
from utils import plot_deaths
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace
from ma_gym.envs.utils.action_space import MultiAgentActionSpace


CANVAS_WIDTH = 50  # Width of drawing canvas in pixels
CANVAS_HEIGHT = 50  # Height of drawing canvas in pixels
SPEED = 5  # Greater value here increases the speed of motion of the snakes
UNIT_SIZE = 10  # Decides how thick the snake is
INITIAL_SNAKE_SIZE = 1


ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
}


def train_eval_loop_single(train_environment, eval_environment, agent, n_evaluations, n_training_episodes, n_eval_episodes,canvas,canvas2):

    print(f"Train-Eval Loop for {agent.name}\n")

    results = np.zeros((n_evaluations, n_eval_episodes))

    for evaluation in range(n_evaluations):

        print(f"\tIteration {evaluation+1}/{n_evaluations}")

        # Train
        print(f"\t\tTraining {agent.name} for {n_training_episodes} episodes.")
        agent.train()   # Enables training mode

        run_single(train_environment, agent, n_training_episodes)

        # Eval
        print(f"\t\tEvaluating {agent.name} for {n_eval_episodes} episodes.")
        agent.eval()    # Disables training mode

        results[evaluation] = run_single(eval_environment,agent,n_eval_episodes)

        print(f"\t\tAverage Steps To Capture: {round(results[evaluation].mean(), 2)}")
        print()

    return results


def run_single(environment, agent, n_episodes):

    results = np.zeros(n_episodes)

    for episode in range(n_episodes):

        steps = 0
        terminal = False
        observation = environment.reset()
        while not terminal:
            steps += 1
            agent.see(observation)
            action = agent.action()
            next_observation, reward, terminal, info = environment.step(action)
            if agent.training:
                agent.next(observation, action, next_observation, reward, terminal, info)
            observation = next_observation

        environment.close()
        results[episode] = steps

    return results



def results_by_type(results):
    step_results = []
    score_results = []
    efficiency_results = []
    death_results = []
    
    for team in results:
        team_step = []
        team_score = []
        team_efficiency = []
        team_death = []
        
        for result in team:
            team_step += [result[0]]
            team_score += [result[1]]
            team_efficiency += [result[1]/result[0]]
            team_death += [result[2]]
        
        step_results += [team_step]
        score_results += [team_score]
        efficiency_results += [team_efficiency]
        death_results += [team_death]
    
    return [step_results, score_results, efficiency_results, death_results]


def create_team(agent_type, canvas, debug):

    if agent_type in ["random", "fully_greedy", "part_greedy", "social_convention", "intention_comm", "rl"]:
        return [Snake(1, 'brown', canvas, agent_type, debug), Snake(2, 'green', canvas, agent_type, debug)]

    else:
        print("Invalid agent type provided. Please refer to the README.md for further instructions")
        exit()

def make_canvas(width, height, title, root):
        """
        Method to create a canvas that acts as a base for all the objects in the game
        """
        root.minsize(width=width, height=height)
        root.title(title)

        canvas = tkinter.Canvas(root, width=width + 1, height=height + 1, bg='black')
        canvas.pack(padx=10, pady=10)
        return canvas

class Snake:
    """
    This class defines the properties of a snake object for the game and contains methods for creating the snake,
    dynamically increasing its size and its movements
    """
    def __init__(self, id, color, canvas, agent_type, debug):

        self.canvas = canvas
        self.id = id
        self.color = color
        self.direction_x = 1
        self.direction_y = 0
        self.body = []
        self.death = None
        self.initialize_snake()
        self.food = 0
        self.communicates = False

        if (agent_type == "random"):
            self.agent = RandomAgent()
        elif (agent_type == "fully_greedy"):
            self.agent = FullyGreedyAgent(id, debug)
        elif (agent_type == "part_greedy"):
            self.agent = PartiallyGreedyAgent(id, debug)
        elif (agent_type == "social_convention"):
            self.agent = SocialConventionAgent(id, debug)
        elif (agent_type == "intention_comm"):
            self.agent = IntentionCommunicationAgent(id, debug)
            self.communicates = True
        elif (agent_type == "rl"):
            self.agent = QLearning(id)

    def new_food(self, food_id):
        self.food = food_id

    def new_canvas(self, canvas):
        self.canvas = canvas

    def initialize_snake(self):
        """
         Method to instantiate the initial snake :
         Each Snake is instantiated as a chain of squares appearing as a single long creature.

         This method creates a circular head(tagged as 'snake_<num>' and 'head' for future reference)
         and n no.of blocks based on start_snake_size.

         Each snake block is stored as an object in the list body[]
        """
        initial_x = (INITIAL_SNAKE_SIZE - 1)*UNIT_SIZE
        initial_y = self.id*CANVAS_HEIGHT / 3 - UNIT_SIZE
        
        # create head
        self.body.append(self.canvas.create_oval(
            initial_x, 
            initial_y,
            initial_x + UNIT_SIZE, 
            initial_y + UNIT_SIZE,
            fill='orange', outline='brown',
            tags=('snake_' + str(self.id), 'head')
        ))

        # complete body
        for block_index in range(1,INITIAL_SNAKE_SIZE):
            x0 = initial_x - block_index * UNIT_SIZE
            snake_block = self.create_block(
                x0, 
                initial_y, 
                x0+UNIT_SIZE, 
                initial_y + UNIT_SIZE,
            )
            self.body.append(snake_block)

    def create_block(self, x0, y0, x1, y1):
        """
         Method to create a single block of each snake based on the coordinates passed to it.
         Each block is tagged as 'snake' to be accessed in future.
        """
        return self.canvas.create_rectangle(x0, y0, x1, y1, fill=self.color, tags='snake')
    
    def body_position(self):
        position = []
        for block in self.body:
            pos = self.canvas.coords(block)[:2]
            position.append([int(x) for x in pos])
        return position

    '''
     move_* methods below control the snake's navigation. These functions are invoked based on user's key presses.
     Special checks are done in each of them to ensure invalid turns are blocked 
     (Ex: Block right turn if the snake is currently going to the left, and so on)
    '''
    def move(self, direction):
        """
        In each frame, the snake's position is grabbed in a dictionary chain_position{}.
        'Key:Value' pairs here are each of the 'Snake_block(Object ID):Its coordinates'.

        Algorithm to move snake:
        1) The ‘snake head’ is repositioned based on the player controls.
        2) The block following the snake head is programmed to take
        snake head’s previous position in the subsequent frame.
        Similarly, the 3rd block takes the 2nd block position and so on.
        """
        move_x, move_y = direction
        
        self.direction_x=move_x
        self.direction_y=move_y

        # move body block to the position of block in front
        blocks = self.body[:]
        blocks.reverse()
        for i, block in enumerate(blocks[:len(blocks)-1]):
            self.canvas.moveto(
                block, 
                self.canvas.coords(blocks[i+1])[0] -1,
                self.canvas.coords(blocks[i+1])[1] -1
            )

        # move head
        snake_head_tag = self.get_head_tag()
        self.canvas.move(snake_head_tag, self.direction_x * UNIT_SIZE, self.direction_y * UNIT_SIZE)

    def get_head_tag(self):
        return 'snake_' + str(self.id) + '&&head'

class Game:
    """
    Creates a canvas and contains attributes for all the objects on the Canvas(food, score_board, etc).
    The methods in it handle everything for the game right from instantiating the snakes, score_board to
    handling player controls, placing food, processing events happening during the game
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, root, snakes, canvas):
        self.root = root
        self.canvas = canvas
        self.snake1 = snakes[0]
        self.snake2 = snakes[1]
        self.food1 = 0
        self.food2 = 0
        self.steps = 0
        self.score = 0
        self.reward1 = 0
        self.reward2 = 0
        self.game_over = False

        self._grid_shape = (10,10)
        self.n_agents = 2
        self.n_food = 2
        self._max_steps = 1000
        self._step_count = None
        self._penalty = -0.5
        self._step_cost = -0.01
        self._prey_capture_reward = 5
        self._agent_view_mask = (4, 4)
        self.reward_range = [-100,100]

        self.action_space = MultiAgentActionSpace([spaces.Discrete(4) for _ in range(self.n_agents)])
        self.agent_pos = {_: None for _ in range(self.n_agents)}
        self.prey_pos = {_: None for _ in range(self.n_food)}
        self._prey_alive = None

        self._agent_dones = [False for _ in range(self.n_agents)]

        self.viewer = None


        # agent pos (2), prey (25), step (1)
        mask_size = np.prod(self._agent_view_mask)
        self._obs_high = np.array([1., 1.] + [1.] * mask_size + [1.0], dtype=np.float32)
        self._obs_low = np.array([0., 0.] + [0.] * mask_size + [0.0], dtype=np.float32)
        self._obs_high = np.tile(self._obs_high, self.n_agents)
        self._obs_low = np.tile(self._obs_low, self.n_agents)
        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

        self._total_episode_reward = [0,0]



        self.create_boards()
        self.play_game()
        


    def simplified_features(self):

        current_grid = np.array(self._full_obs)

        agent_pos = []
        for agent_id in range(self.n_agents):
            tag = f"A{agent_id + 1}"
            row, col = np.where(current_grid == tag)
            row = row[0]
            col = col[0]
            agent_pos.append((col, row))

        prey_pos = []
        for prey_id in range(self.n_food):
            tag = f"P{prey_id + 1}"
            row, col = np.where(current_grid == tag)
            row = row[0]
            col = col[0]
            prey_pos.append((col, row))

        features = np.array(agent_pos + prey_pos).reshape(-1)

        return features
    
    def get_action_meanings(self, agent_i=None):
        if agent_i is not None:
            assert agent_i <= self.n_agents
            return [ACTION_MEANING[i] for i in range(self.action_space[agent_i].n)]
        else:
            return [[ACTION_MEANING[i] for i in range(ac.n)] for ac in self.action_space]

    def get_results(self):
        death = None
        if (self.snake1.death != None):
            death = self.snake1.death
        else:
            death = self.snake2.death

        return [self.steps, self.score, death]

    def create_boards(self):
        """
        Method to position score_board text on the canvas
        """
        y_offset = 0.02
        self.canvas.create_text(
            0.15 * CANVAS_WIDTH, 
            y_offset * CANVAS_HEIGHT,
            text=('Steps : ' + str(self.steps)), 
            font=("Times", 12, 'bold'), 
            fill='white',
            tags='steps_board'
        )
        self.canvas.create_text(
            0.85 * CANVAS_WIDTH, 
            y_offset * CANVAS_HEIGHT,
            text=('Score : ' + str(self.score)), 
            font=("Times", 12, 'bold'), 
            fill='white',
            tags='score_board'
        )

    def place_food(self, color):
        """
        Method to randomly place a circular 'food' object anywhere on Canvas.
        The tag on it is used for making decisions in the program
        """
        x1 = random.randrange(2*UNIT_SIZE, CANVAS_WIDTH - UNIT_SIZE, step=UNIT_SIZE)
        y1 = random.randrange(2*UNIT_SIZE, CANVAS_HEIGHT - UNIT_SIZE, step=UNIT_SIZE)
        id = self.canvas.create_oval(x1, y1, x1 + UNIT_SIZE, y1 + UNIT_SIZE, fill= color, tags='food')
        return id, [x1, y1]
    
    def move_snake(self, snake):
        direction = snake.agent.move_direction()
        snake_moved = snake.move(direction)

    def update_scores(self):
        self.canvas.itemconfig("score_board", text='Score : ' + str(self.score))
        self.canvas.itemconfig("steps_board", text='Steps : ' + str(self.steps))

    def snake_check(self, snake):
        """
        Method to handle events during the Snake's motion.
        Makes use of 'tags' given to each object to filter out what's overlapping.

        1. Hit food --> Check if the hit object is food: If yes, eat it, increment snake size and delete food object
        2. Hit wall --> Check if Snake head went past the wall coordinates: If yes, kill snake
        3. Hit snake --> Check if Snake hit itself or other snake: If yes, kill this snake
        """
        flag = True
        snake_head_tag = snake.get_head_tag()
        x0, y0, x1, y1 = self.canvas.coords(snake_head_tag)

        if (x0 <= 0) or (y0 <= 0) or (x1 >= CANVAS_WIDTH) or (y1 >= CANVAS_HEIGHT):
            self.handle_hit_wall(snake)
            if (snake.id == 1):
                self.reward1 = -50
            else:
                self.reward2 = -50
            

        overlapping_objects = self.canvas.find_overlapping(x0+1, y0+1, x1-1, y1-1)
        for obj in overlapping_objects:
            if 'food' in self.canvas.gettags(obj):
                self.handle_hit_food(obj, snake)
                flag = False
                break
            elif 'snake' in self.canvas.gettags(obj):
                self.handle_hit_snake(snake)
                if (snake.id == 1):
                    self.reward1 = -50
                else:
                    self.reward2 = -50
        if(flag):
            if (snake.id == 1):
                self.reward1 = -0.5
            else:
                self.reward2 = -0.5

    def handle_hit_food(self, food_id, snake):
        if (snake.food == food_id):
            self.canvas.delete(food_id)
            self.score += 1
            
            new_id, position = self.place_food(snake.color)
            snake.new_food(new_id)
            if (snake.id == 1):
                self.food1 = position
                self.reward1= 10
            else:
                self.food2 = position
                self.reward2 = 10

    def handle_hit_snake(self, snake):
        snake.death = "SNAKE"

    def handle_hit_wall(self, snake):
        snake.death = "WALL"

    def update_game(self):
        self.snake_check(self.snake1)
        self.snake_check(self.snake2)
        self.update_scores()
        if self.snake1.death or self.snake2.death:
            self.game_over=True

    def handle_episode_over(self):
        """
        Method to print out the final message and declare the winner based on player scores
        """
        print("Episode Over!")
        print(f"\nSteps: {self.steps} \nScore: {self.score} \nCase of death snake 1: {self.snake1.death} \nCase of death snake 2: {self.snake2.death} "
        )
        widget = tkinter.Label(
            self.canvas, 
            text='Episode Over!',
            fg='white', bg='black', 
            font=("Times", 20, 'bold'
        ))
        widget.pack()
        widget.place(relx=0.5, rely=0.5, anchor='center')       

    def display_label(self, message, display_time):
        """
        Method to display introductory messages on screen before the start of the game
        """
        widget = tkinter.Label(
            self.canvas, 
            text=message, 
            fg='white', 
            bg='black',
            font=("Times", 20, 'bold')
        )
        widget.place(relx=0.5, rely=0.5, anchor='center')
        self.canvas.update()
        time.sleep(display_time)
        widget.place_forget()
        self.canvas.update()  

    def get_snake_positions(self):
        position1 = self.snake1.body_position()
        position2 = self.snake2.body_position()
        return [position1, position2]

    def get_food_positions(self):
        return [self.food1, self.food2]

    def step(self, action):
       
        self.move_snake(self.snake1)
        self.move_snake(self.snake2)
        self.steps+=1
        self.canvas.update()
        self.update_game()
        self._total_episode_reward[0] += self.reward1
        self._total_episode_reward[1] += self.reward2

        snakes_pos = self.get_snake_positions()
        food_pos = self.get_food_positions()
        positions = [snakes_pos, food_pos]
        rewards = [self.reward1, self.reward2]

        done = self.game_over
        return positions, rewards, [done,done], True


    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def reset(self):
        id1, food1 = self.place_food('brown')
        id2, food2 = self.place_food('green')

        self.food1 = food1
        self.food2 = food2
        self.snake1.new_food(id1)
        self.snake2.new_food(id2)

        snakes_pos = self.get_snake_positions()
        food_pos = self.get_food_positions()
        positions = ([snakes_pos, food_pos])
        rewards = [0, 0]

        done = self.game_over
        #acho q so da as positions
        return positions, rewards

    def play_game(self):
        self.display_label('Welcome to the Snake World!', 0.5)
        
        observation = self.reset()
        while not self.game_over:
        # Update World
            #print("Observation: ", observation)
            self.snake1.agent.see(observation)
            self.snake2.agent.see(observation)
            observation = self.step(0)
            time.sleep(1/SPEED)
        self.handle_episode_over()



class JointActionWrapper(Wrapper):

    """ A Wrapper for centralized multi-agent environments.

    * Allows a single agent to control all agents via a global joint-action.
    * Reduces the N action spaces (where N is the number of agents) to a single joint-action space.

    Example
    -------
    >> N = 2
    >> Actions Agent 0 = Actions Agent 1 = [ Move (0), Stay (1) ]

    | Action 1 | Action 2 | Team Action |
    |----------|----------|--------------|
    | 0        | 0        | 0            |
    | 0        | 1        | 1            |
    | 1        | 0        | 2            |
    | 1        | 1        | 3            |
    """

    def __init__(self, env):

        super(JointActionWrapper, self).__init__(env)

        self.n_agents = env.n_agents

        self.snake1 = env.snake1
        self.snake2 = env.snake2

        self.action_spaces = [list(range(env.action_space[a].n)) for a in range(self.n_agents)]
        self.joint_action_space = list(itertools.product(*self.action_spaces))

        self.action_meanings = [env.get_action_meanings(a) for a in range(self.n_agents)]
        self.joint_action_meanings = list(itertools.product(*self.action_meanings))

        self.n_joint_actions = len(self.joint_action_meanings)

    def reset(self):
        observations = super(JointActionWrapper, self).reset()
        observation = observations[0]   # For the predator-prey domain, the observations are shared.
        return observation

    def step(self, joint_action: int):

        individual_actions: Sequence[int] = self.joint_action_space[joint_action]
        next_observations, rewards, terminals, info = super(JointActionWrapper, self).step(individual_actions)

        next_observation = next_observations[0]    # For the predator-prey domain, the observations are shared.
        equal_rewards = all(rewards[0] == reward for reward in rewards)
        assert equal_rewards, "Multi-Agent RL requires same reward signal for all agents"
        reward = rewards[0]
        terminal = all(terminals)

        return next_observation, reward, terminal, info

    def get_action_meanings(self):
        return self.team_action_meanings

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-training", type=int, default=100)
    parser.add_argument("--episodes-per-evaluation", type=int, default=64)
    parser.add_argument("--evaluations", type=int, default=10)
    opt = parser.parse_args()

    root = tkinter.Tk()
    root2 = tkinter.Tk()

    canvas = make_canvas(CANVAS_WIDTH, CANVAS_HEIGHT, 'Snake Game', root)
    canvas2 = make_canvas(CANVAS_WIDTH, CANVAS_HEIGHT, 'Snake Game', root)

    team = create_team("rl", canvas, False)
    team2 = create_team("rl", canvas2, False)

    train_run = Game(root, team, canvas)
    eval_run = Game(root2, team, canvas2)
        

    joint_train_environment = JointActionWrapper(train_run)
    joint_eval_environment = JointActionWrapper(eval_run)
    centralized_multi_agent_learner = QLearning(joint_train_environment.n_joint_actions)
    
        #results = run.get_results()
        #root.destroy() # uncomment for automatic closure of the window after the game 
    train_eval_loop_single(
        joint_train_environment, joint_eval_environment, centralized_multi_agent_learner,
        opt.evaluations, opt.episodes_per_training, opt.episodes_per_evaluation,canvas,canvas2)
        

if __name__ == '__main__':
    main()