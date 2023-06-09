import time
import tkinter
import random
import numpy as np
from agents import *
import argparse
from tqdm import tqdm

from utils import compare_results
from utils import plot_deaths

CANVAS_WIDTH = 600  # Width of drawing canvas in pixels
CANVAS_HEIGHT = 600  # Height of drawing canvas in pixels
SPEED = 15  # Greater value here increases the speed of motion of the snakes
UNIT_SIZE = 20  # Decides how thick the snake is
MAX_STEPS = 500 # Maximum steps in an episode
INITIAL_SNAKE_SIZE = 7

def results_by_type(results):
    """
    Organizes the given results into separate lists based on their types.

    Args:
        results (list): List of team results, where each team result is a list of individual episode results.

    Returns:
        list: A list containing separate lists for each metric (steps, scores, efficiency, deaths).
    """
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
    """
    Creates a team of two snakes based on the specified agent type.
    """
    if agent_type in ["random", "fully_greedy", "part_greedy", "social_convention", "intention_comm"]:
        return [Snake(1, 'brown', canvas, agent_type, debug), Snake(2, 'green', canvas, agent_type, debug)]

    else:
        print("Invalid agent type provided. Please refer to the README.md for further instructions")
        exit()

def make_canvas(width, height, title, root):
    """
    Creates a canvas that serves as the base for all the objects in the game.
    """
    root.minsize(width=width, height=height)
    root.title(title)

    canvas = tkinter.Canvas(root, width=width + 1, height=height + 1, bg='black')
    canvas.pack(padx=10, pady=10)
    return canvas

class Snake:
    """
    Represents a snake object in the game.

    Attributes:
        id (int): The identifier for the snake.
        color (str): The color of the snake.
        canvas (tkinter.Canvas): The canvas on which the snake is displayed.
        agent_type (str): The type of agent controlling the snake.
        debug (bool): A flag indicating whether debug mode is enabled or not.
        direction_x (int): The horizontal direction of the snake's movement (-1 for left, 1 for right).
        direction_y (int): The vertical direction of the snake's movement (-1 for up, 1 for down).
        body (list): A list of object IDs representing the snake's body blocks.
        death: Placeholder for the snake's death status.
        food (int): The ID of the food the snake is currently targeting.
        communicates (bool): Indicates whether the snake can communicate with other snakes.

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

    def new_food(self, food_id):
        """
        Sets the ID of the food the snake is targeting.
        """
        self.food = food_id

    def new_canvas(self, canvas):
        """
        Updates the canvas on which the snake is displayed.
        """
        self.canvas = canvas

    def initialize_snake(self):
        """
        Initializes the snake's body by creating the head and initial blocks.
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
        Creates a single block for the snake based on the given coordinates.
        """
        return self.canvas.create_rectangle(x0, y0, x1, y1, fill=self.color, tags='snake_' + str(self.id))
    
    def body_position(self):
        """
        Retrieves the current positions of the snake's body blocks.

        Returns:
            list: A list of lists containing the x and y coordinates of each body block.
        """
        position = []
        for block in self.body:
            pos = self.canvas.coords(block)[:2]
            position.append([int(x) for x in pos])
        return position

    def move(self, direction):
        """
        Moves the snake in the specified direction.

        Args:
            direction (tuple): A tuple containing the horizontal and vertical movement values (move_x, move_y).
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
        """
        Returns the tag of the snake's head.
        """
        return 'snake_' + str(self.id) + '&&head'

class Game:
    """
    Represents the game environment.

    Attributes:
        root (tkinter.Tk): The root window of the game.
        canvas (tkinter.Canvas): The canvas on which the game objects are displayed.
        snake1 (Snake): The first snake in the game.
        snake2 (Snake): The second snake in the game.
        food1 (int): The ID of the first food object (targeted by snake 1).
        food2 (int): The ID of the second food object (targeted by snake 2).
        steps (int): The number of steps taken in the game.
        score (int): The score of the game.
        game_over (bool): Indicates whether the game is over or not.
    """
    def __init__(self, root, snakes, canvas):
        self.root = root
        self.canvas = canvas
        self.snake1 = snakes[0]
        self.snake2 = snakes[1]
        
        self.food1 = 0
        self.food2 = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.create_boards()
        self.play_game()
        
    def get_results(self):
        """
        Returns the results of the game.

        Returns:
            list: A list containing the number of steps, the score, and the cause of death of the snake.
        """
        death = None
        if (self.snake1.death != None):
            death = self.snake1.death
        else:
            death = self.snake2.death

        return [self.steps, self.score, death]

    def create_boards(self):
        """
        Positions score and steps boards on the canvas
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
        Randomly places a circular 'food' object anywhere on the canvas.
        
        Args:
            color (str): The color of the food object.
        
        Returns:
            tuple: A tuple containing the ID of the created food object and its position as [x, y].
        """
        x1 = random.randrange(2*UNIT_SIZE, CANVAS_WIDTH - UNIT_SIZE, step=UNIT_SIZE)
        y1 = random.randrange(2*UNIT_SIZE, CANVAS_HEIGHT - UNIT_SIZE, step=UNIT_SIZE)
        id = self.canvas.create_oval(x1, y1, x1 + UNIT_SIZE, y1 + UNIT_SIZE, fill= color, tags='food')
        return id, [x1, y1]
    
    def move_snake(self, snake):
        """
        Moves the specified snake based on its agent's move direction.
        """
        direction = snake.agent.move_direction()
        snake.move(direction)

    def update_score_board(self):
        """
        Updates score and steps boards in the canvas
        """
        self.canvas.itemconfig("score_board", text='Score : ' + str(self.score))
        self.canvas.itemconfig("steps_board", text='Steps : ' + str(self.steps))

    def snake_check(self, snake):
        """
        Handles events during the snake's motion.
        Checks for collisions with food, wall, self or other snakes.
        """
        snake_head_tag = snake.get_head_tag()
        x0, y0, x1, y1 = self.canvas.coords(snake_head_tag)

        if (x0 <= 0) or (y0 <= 0) or (x1 >= CANVAS_WIDTH) or (y1 >= CANVAS_HEIGHT):
            snake.death = "WALL"

        overlapping_objects = self.canvas.find_overlapping(x0+1, y0+1, x1-1, y1-1)
        for obj in overlapping_objects:
            overlapping_object_tags = self.canvas.gettags(obj)
            if 'food' in overlapping_object_tags:
                self.handle_hit_food(obj, snake)
                break
            elif 'snake_'+str(snake.id) in overlapping_object_tags:
                if "head" not in overlapping_object_tags:
                    snake.death = "SELF"
            elif 'snake_' in str(overlapping_object_tags):
                snake.death = "SNAKE"

    def handle_hit_food(self, food_id, snake):
        """
        Handles the event when a snake hits a food object. 
        If it is the snakes targeted food, deletes current food and creates a new one.
        """
        if (snake.food == food_id):
            self.canvas.delete(food_id)
            self.score += 1
            new_id, position = self.place_food(snake.color)
            snake.new_food(new_id)
            if (snake.id == 1):
                self.food1 = position
            else:
                self.food2 = position

    def update_game(self):
        """
        Updates the game state by checking for collisions, updating the score board,
        and determining if the game is over.
        """
        self.snake_check(self.snake1)
        self.snake_check(self.snake2)
        self.update_score_board()
        if self.snake1.death or self.snake2.death:
            self.game_over=True
        elif self.steps == MAX_STEPS:
            self.snake1.death = self.snake2.death = "MAX_STEPS"
            self.game_over=True

    def handle_episode_over(self):
        """
        Prints out the final results.
        """
        print("\n\nEpisode Over!")
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
        Displays messages on the canvas.
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
        """
        Retrieves the positions of both snakes.
        """
        position1 = self.snake1.body_position()
        position2 = self.snake2.body_position()
        return [position1, position2]

    def get_food_positions(self):
        """
        Retrieves the positions of the food objects.
        """
        return [self.food1, self.food2]

    def step(self):
        """
        Performs a single step in the game.
        Moves both snakes, updates the steps counter, and updates the game state.

        Returns:
            tuple: A tuple containing the positions of the snakes and food objects, rewards for each snake,
                and a boolean indicating if the game is over.
        """
        if (self.snake1.communicates and len(self.snake1.agent.intention) == 0):
            intention = self.snake1.agent.make_new_intention()
            self.snake2.agent.receive_intention(intention)
            _ = self.snake2.agent.make_new_intention()
        
        if (self.snake2.communicates and len(self.snake2.agent.intention) == 0):
            _ = self.snake2.agent.make_new_intention()

        self.move_snake(self.snake1)
        self.move_snake(self.snake2)
        self.steps+=1
        self.canvas.update()
        self.update_game()

        snakes_pos = self.get_snake_positions()
        food_pos = self.get_food_positions()
        positions = [snakes_pos, food_pos]
        rewards = [0, 0]

        done = self.game_over
        return positions, rewards, done

    def reset(self):
        """
        Resets the game to its initial state.

        Creates new food objects, updates the snake's food IDs and positions,
        and retrieves the positions of the snakes and food objects.

        Returns:
            tuple: A tuple containing the positions of the snakes and food objects, rewards for each snake,
                and a boolean indicating if the game is over.
        """
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
        return positions, rewards, done

    def play_game(self):
        """
        Plays the game until one of the snakes dies or the maximum number of steps is reached.
        """
        self.display_label('Welcome to the Snake World!', 0.5)
        
        observation = self.reset()
        while not self.game_over:
            # move snakes and update game
            self.snake1.agent.see(observation)
            self.snake2.agent.see(observation)
            observation = self.step()
            time.sleep(1/SPEED)
        self.handle_episode_over()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--agents", default="")
    parser.add_argument("--debug", default="")
    parser.add_argument("--ghost", default="")
    opt = parser.parse_args()

    debug = False
    if opt.debug == "true":
        debug = True
    
    if opt.agents == "all":
        print("Compare results for different teams")

        teams = { "Random team": "random", "Fully Greedy team": "fully_greedy", "Partially Greedy team": "part_greedy", "Social Convention Team" : "social_convention", "Intention Communication Team" : "intention_comm"}
    
        results = []
        for team, agents in tqdm(teams.items(), desc="Agent", leave=True):
            team_results = []
            for episode in tqdm(range(opt.episodes), desc="Episode", position=0):
                # Create a new root and canvas for each episode
                new_root = tkinter.Tk()
                if opt.ghost: 
                    new_root.withdraw()
                new_canvas = make_canvas(CANVAS_WIDTH, CANVAS_HEIGHT, 'Snake Game', new_root)

                team = create_team(agents, new_canvas, debug)

                run = Game(new_root, team, new_canvas)
                result = run.get_results()
                new_root.destroy()
                if debug:
                    print(result)
                team_results += [result]
            
            results += [team_results]
        if debug:
            print("Results: ", results)
        
        # Analyze and compare the results
        results = results_by_type(results)
        colors=["orange", "green", "blue", "red", "black"]

        compare_results(
            results[0],
            title="Average Steps Comparison",
            colors=colors,
            metric="Steps per Episode"
        ) 

        compare_results(
            results[1],
            title="Average Score Comparison",
            colors=colors,
            metric="Score per Episode"
        )

        compare_results(
            results[2],
            title="Score Efficiency Comparison",
            colors=colors,
            metric="Score/Steps per Episode"
        )


        plot_deaths(
            results[3],
            colors=colors,
        )

    else:
        # Create a root and canvas for a single-team game
        root = tkinter.Tk()
        canvas = make_canvas(CANVAS_WIDTH, CANVAS_HEIGHT, 'Snake Game', root)
        if opt.ghost:
            root.withdraw()
        team = create_team(opt.agents, canvas, debug)
        run = Game(root, team, canvas)
        if opt.ghost:
            root.destroy()
        root.mainloop()
        

if __name__ == '__main__':
    main()