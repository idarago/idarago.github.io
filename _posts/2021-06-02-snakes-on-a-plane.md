---
title: "Snakes on a plane"
mathjax: true
layout: post
excerpt_separator: <!--more-->
---

In this post we will explain how to implement a simple Snake game on Python using Pygame, and propose a very straightforward strategy for the computer to play the game on its own. In a later post, we will use <i>Reinforcement Learning</i> techniques!

<center>
<img src="/assets/images/bfsgameplay.gif" alt="Snake using BFS" width="400"/>
</center>

<center>
<i>Gameplay using our simple BFS algorithm.</i>
</center>
<!--more-->
<h3>Building the game logic</h3>

First we have to create the basic logic making the game work. Our main class will be the ```Head``` of the snake. We can set up some constants first: our gameboard will consist of $$m\times n$$ squares of a given size. We use ```m=40```, ```n=30``` and ```SIZE=20``` to get a screen of size $$800\times 600$$.

{% highlight python %}
class Head:
    """Head of the snake"""
    def __init__(self):
        self._direction = [0, 0]
        self._head_position = [200, 200]
{% endhighlight %}

It will have an ```update``` method, that moves the head of the snake according to some input (either user input, or what our algorithm generates in case it's playing on its own). To handle inputs given in some way that's not the usual keypress event, we will need to have our personalized PyGame events.

{% highlight python %}
MOVE_UP = pygame.USEREVENT + 1    # Personalized PyGame events. We will use them to feed
MOVE_DOWN = pygame.USEREVENT + 2  # information using algorithms/reinforcement learning
MOVE_LEFT = pygame.USEREVENT + 3  # instead of usual keypress events
MOVE_RIGHT = pygame.USEREVENT + 4 #
{% endhighlight %}

Using these personalized events, we can simply update the direction accordingly.

{% highlight python %}
def update(self, pygame_event):
        """Moves the head according to the event"""
        if pygame_event.type in [pygame.KEYDOWN, MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT]:
            # Checking self._direction makes sure the snake doesn't move back into itself
            if pygame_event.type == MOVE_UP and self._direction != [0,SIZE]:
                self._direction = [0,-SIZE]

            if pygame_event.type == MOVE_DOWN and self._direction != [0,-SIZE]:
                self._direction = [0,SIZE]

            if pygame_event.type == MOVE_LEFT and self._direction != [SIZE,0]:
                self._direction = [-SIZE,0]

            if pygame_event.type == MOVE_RIGHT and self._direction != [-SIZE,0]:
                self._direction = [SIZE,0]

        self._head_position[0] += self._direction[0] # Updates the position of the head
        self._head_position[1] += self._direction[1] #
{% endhighlight %}

We can similarly handle user input using the ```KEYDOWN``` event in PyGame.

{% highlight python %}
if pygame_event.type == pygame.KEYDOWN: # Handles user input (usual gameplay)
    key = pygame_event.dict["key"]
    if (key == KEYUP or key == ord('w')) and self._direction != [0,SIZE]:
        self._direction = [0,-SIZE]
    
    if (key == KEYDOWN or key == ord('s')) and self._direction != [0,-SIZE]:
        self._direction = [0,SIZE]

    if (key == KEYLEFT or key == ord('a')) and self._direction != [SIZE,0]:
        self._direction = [-SIZE,0]
    
    if (key == KEYRIGHT or key == ord('d')) and self._direction != [-SIZE,0]:
        self._direction = [SIZE,0]
{% endhighlight %}


Now we can start the main game loop. We need to initialize the screen for the game and set the initial position of the snake and the apple. We keep track of the position of the head of the snake with ```snaketip``` and of the whole body with ```snake```.

{% highlight python %}
pygame.init()        
my_clock = pygame.time.Clock()

# Create the snake
head = Head()
head._direction = [0,0]
snake = 5*[list(head._head_position)] # This starts the body of the snake with length 5
snaketip = list(head._head_position)

# Create the apple
apple = [0,0]
apple[0] = SIZE*np.random.randint(0,m)        
apple[1] = SIZE*np.random.randint(0,n)
{% endhighlight %}

We also need to initialize the screen and direction of movement of the snake.

{% highlight python %}
# Keeps track of the score
score = 0
# Initialize the screen and movement direction
main_surface = pygame.display.set_mode((WIDTH,HEIGHT))
my_event = pygame.event.Event(pygame.NOEVENT)
direction = 0
{% endhighlight %}

The advantage of keeping track of the direction as a number is that our events are consecutive and we can very concisely trigger our events as follows.

{% highlight python %}
my_event = pygame.event.Event(MOVE_UP + direction)
{% endhighlight %}

We only need to worry about the main loop of the game. We update the position of the snake by forgetting the first square and adding the ```snaketip``` at the end (that is, we update the squares consisting of the body of the snake in a <i>first in, first out</i> way).

{% highlight python %}
head.update(my_event) # We update the head of the snake according to the event just triggered
snaketip=list(head._head_position) # This makes a copy of head._head_position
snake = snake[1:]
snake.append(snaketip)
{% endhighlight %}

Finally, we need to make a new apple appear once the snake eats the current apple on the screen. This is simply handled by the following

{% highlight python %}
if snaketip == apple:
    score += 1
    while snake.count(apple)>0:
        apple[0] = SIZE*(random.uniform(0,m)//1)
        apple[1] = SIZE*(random.uniform(0,n)//1)
    snake = snake + 2*[snaketip]
{% endhighlight %}

This is the basic logic behind the game. The whole code can be found on my Github. We can now try to figure out how to make the computer play on its own!

<h3>The algorithm</h3>

The idea is simple: just take the shortest path from the head of the snake towards the apple. Of course, we don't want the path to go through squares which are occupied by the body of the snake. Such a naive algorithm works decently, despite the obvious drawbacks it has: sometimes the shortest path will trap the snake and the only possible outcome will be biting its own tail.

There's also a much simpler thing we can do. We could transverse the whole screen everytime, without even bothering about the position of the apple. This would be extremely slow. The ideal solution to this problem would involve some sort of trade-off between time taken to get the apple and unraveling the body so that the snake doesn't trap itself. We will discuss this in terms of <i>rewards</i> when we approach this problem using reinforcement learning.

The idea for finding the shortest path is to model our problem in terms of graphs and use <b>breadth-first search</b>. Each square in the screen will consist of a node, and two nodes will be joined by an edge if the corresponding squares are adjacent and neither of them is occupied by some part of the body of the snake.

To do this, we will keep track of the occupied squares in a ```grid``` of size $$40\times 30$$ (because of the dimensions of our gamescreen) where the $$(i,j)$$-th entry is $$1$$ if it's occupied and $$0$$ otherwise.

{% highlight python %}
ds = [[0,-1],[0,1],[1,0],[-1,0]] # All the possible directions, 
                                 # the order matters for the function we use
                                 # to format the path for input
def neighbors(grid, pos):
    i,j = pos
    neighbors = []
    for [dx,dy] in ds:
        if (0 <=(i+dx) and i+dx < len(grid)) and (0<= j+dy and j+dy < len(grid[0])):
            if grid[i+dx][j+dy] == 0:
                neighbors.append([i+dx, j+dy])
    return neighbors
{% endhighlight %}

We also need a way of translating from the positions we used before (that are the ones used for rendering nicely in PyGame) and the corresponding position in the grid. This is because coordinates in Pygame set the $$(0,0)$$ in the upper-left corner, the $$x$$-axis moves to the right in the positive direction (as usual), whereas the $$y$$-axis moves down in the positive direction. The following simple function deals with this issue.

{% highlight python %}
def coords_to_grid(pos):
    return [int(pos[0]/SIZE), n-1-int(pos[1]/SIZE)]
{% endhighlight %}

Having all the necessary ingredients, the algorithm we use for finding the shortest path is a simple breadth-first search.

{% highlight python %}
def grid_bfs(grid, start, end):
    visited = [[False for _ in range(len(grid[0]))] for _ in range(len(grid))]
    queue = []
    queue.append(start)
    prev = [[None for _ in range(len(grid[0]))] for _ in range(len(grid))] # Stores the previous nodes in the path
    path = []
    while (len(queue) != 0):
        current_pos = queue[0]
        queue.pop(0)
        for neighbor in neighbors(grid,current_pos):
            neighbor_x,neighbor_y = neighbor
            if visited[neighbor_x][neighbor_y] == False:
                visited[neighbor_x][neighbor_y] = True
                prev[neighbor_x][neighbor_y] = current_pos
                queue.append([neighbor_x,neighbor_y])
                if [neighbor_x,neighbor_y] == end:
                    break
    pos = end
    path.append(end)
    while pos != start and pos != None:
        pos_x, pos_y = pos
        path.append(prev[pos_x][pos_y])
        pos = prev[pos_x][pos_y]
    return path
{% endhighlight %}

And this is it! We simply need to format the path we obtain to be able to use it in the game loop, but if you're interested you can check those details on the Github repository.