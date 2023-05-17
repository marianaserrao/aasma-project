import random

def random_movement():
    directions = [[0,1], [0,-1], [1,0], [-1,0]]
    choice = random.choice(directions)
    return choice

def move_agent(snake):
    return random_movement()