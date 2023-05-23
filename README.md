# 2-Agent Snake Game

## Description

## Dependencies and Module installation

## How to Run

The project can be run with the command **python *snake-game.py* --agents arg1 --episodes arg2**.
Keep in mind that python3 can replace python depending on the python version you have installed.

After the agent run is over you need to close the visual game window to properly end the process.

#### Arguments

**arg1** corresponds to the **type of agent** that will be created to explore the environment.
The possible values this argument can take are:

- <span style="color:blue">random</span>: Run **1 episode** of the environment with **2 random agents**
- <span style="color:blue">greedy</span>: Run **1 episode** of the environment with **2 greedy agents**
- <span style="color:blue">all</span>: Run **_arg2_ episodes** of the environment with **different teams of agents** and **compare the results** obtained in terms of **score**, **number of steps** and **cause of death/loss**

**arg2** corresponds to the **number of episodes** of the environment that will be considered to draw results from.
This argument should only be provided if you wish to run the environment with **all** teams of agents.
Even then, this is an **optional argument** that can take **any positive integer**, with the **default being 30**.
