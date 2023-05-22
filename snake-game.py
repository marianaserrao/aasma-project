import time
import tkinter
import random
import numpy as np
from agents import *

CANVAS_WIDTH = 300  # Width of drawing canvas in pixels
CANVAS_HEIGHT = 300  # Height of drawing canvas in pixels
SPEED = 20  # Greater value here increases the speed of motion of the snakes
UNIT_SIZE = 10  # Decides how thick the snake is
INITIAL_SNAKE_SIZE = 7


class Snake:
    """
    This class defines the properties of a snake object for the game and contains methods for creating the snake,
    dynamically increasing its size and its movements
    """
    def __init__(self, id, color, canvas, agent_type):
        if (agent_type == "random"):
                self.agent = RandomAgent()
        elif (agent_type == "greedy"):
                self.agent = GreedyAgent(id)
             
        self.canvas = canvas
        self.id = id
        self.color = color
        self.direction_x = 1
        self.direction_y = 0
        self.body = []
        self.death = None
        self.initialize_snake()
        self.food = 0

    def new_food(self, food_id):
        self.food = food_id

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

        if (move_x != 0 and move_x == -self.direction_x) or (move_y != 0 and move_y == -self.direction_y):
            return False
        
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
    def __init__(self, root):
        self.root = root
        self.canvas = self.make_canvas(CANVAS_WIDTH, CANVAS_HEIGHT, 'Snake Game')
        #self.snake1 = Snake(1, 'brown', self.canvas, "random")
        #self.snake2 = Snake(2, 'green', self.canvas, "random")
        self.snake1 = Snake(1, 'brown', self.canvas, "greedy")
        self.snake2 = Snake(2, 'green', self.canvas, "greedy")
        self.food1 = 0
        self.food2 = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.create_boards()
        self.play_game()

    def make_canvas(self, width, height, title):
        """
        Method to create a canvas that acts as a base for all the objects in the game
        """
        self.root.minsize(width=width, height=height)
        self.root.title(title)

        canvas = tkinter.Canvas(self.root, width=width + 1, height=height + 1, bg='black')
        canvas.pack(padx=10, pady=10)
        return canvas

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
        snake_moved = False
        while snake_moved == False:
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
        snake_head_tag = snake.get_head_tag()
        x0, y0, x1, y1 = self.canvas.coords(snake_head_tag)

        if (x0 <= 0) or (y0 <= 0) or (x1 >= CANVAS_WIDTH) or (y1 >= CANVAS_HEIGHT):
            self.handle_hit_wall(snake)

        overlapping_objects = self.canvas.find_overlapping(x0+1, y0+1, x1-1, y1-1)
        for obj in overlapping_objects:
            if 'food' in self.canvas.gettags(obj):
                self.handle_hit_food(obj, snake)
                break
            elif 'snake' in self.canvas.gettags(obj):
                self.handle_hit_snake(snake)

    def handle_hit_food(self, food_id, snake):
        if (snake.food == food_id):
            self.canvas.delete(food_id)
            self.score += 1
            new_id, position = self.place_food(snake.color)
            snake.new_food(new_id)
            if (snake.id == 1):
                self.food1 = position
            else:
                self.food2 = position

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

    def handle_game_over(self):
        """
        Method to print out the final message and declare the winner based on player scores
        """
        print("Game Over!")
        print(f"\nSteps: {self.steps} \nScore: {self.score} \nCase of death snake 1: {self.snake1.death} \nCase of death snake 2: {self.snake2.death} "
        )
        widget = tkinter.Label(
            self.canvas, 
            text='Game Over!',
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

    def step(self):
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
        self.display_label('Welcome to the Snake World!', 0.5)
        
        observation = self.reset()
        while not self.game_over:
        # Update World
            #print("Observation: ", observation)
            self.snake1.agent.see(observation)
            self.snake2.agent.see(observation)
            observation = self.step()
            time.sleep(1/SPEED)
        self.handle_game_over()


def main():
    root = tkinter.Tk()
    Game(root)
    #root.destroy() # uncomment for automatic closure of the window after the game 
    root.mainloop()

if __name__ == '__main__':
    main()