---
title: "Knight moves"
mathjax: true
layout: post
excerpt_separator: <!--more-->
---

In chess, the <i>knight</i> is a piece that moves in an $$L$$-shaped fashion: that is, it may move two squares vertically and one square horizontally, or two squares horizontally and one square vertically.

The following snippet lets you <b>click any two tiles</b> on a chessboard and shows the shortest path a knight can take to go from one to the other.


<center>
<iframe id="chess" width="340" height="340" frameBorder="0"
src="https://math.uchicago.edu/~idarago/dijkstra/index.html">
</iframe>
</center>

<!--more-->

<h3>How do we get the shortest path?</h3>

We model the problem in terms of <i>graphs</i>: each square in the chessboard will be a node, and we connect two nodes with an edge if a knight can move between the corresponding squares.

{% highlight python %}
def neighbors(n,posx,posy):
    directions = [[1,2], [1,-2], [-1,2], [-1,-2], [2,1], [2,-1], [-2,1], [-2,-1]]
    visitable = []
    for [v1, v2] in directions:
        if posx + v1 < n and 0 <= posx + v1 and posy + v2 < n and 0 <= posy + v2:
            visitable.append([posx+v1, posy+v2])
    return visitable
{% endhighlight %}

Now, we can use <a href="https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm">Dijkstra's algorithm</a> to find the shortest path in the graph.
That means that we are going to transverse the graph in a <i>breath-first</i> way, while keeping track of the visited nodes and the previous node in the shortest path.

{% highlight python %}
from heapq import heappush, heappop # Priority queue
def Dijkstra(n, source, target):
    dist = [[None for _ in range(0,n)] for _ in range(0,n)] # Unknown distance from the source vertex to the vertex (i,j)
    prev = [[None for _ in range(0,n)] for _ in range(0,n)] # Predecessor of vertex in optimal path

    # Create vertex priority queue
    candidates = [(0, source)]

    while len(candidates) > 0:
        path_len, [v1,v2] = heappop(candidates)
        if [v1,v2] == target:
            return path_len, prev
        if dist[v1][v2] is None: # This means that v is unvisited
            dist[v1][v2] = path_len
            for [w1,w2] in neighbors(n,v1,v2):
                if dist[w1][w2] is None:
                    prev[w1][w2] = [v1,v2]
                    heappush(candidates, (path_len + 1, [w1,w2]))
    return -1  # If it completed the While loop and didn't return anything then it couldn't have reached the target.
{% endhighlight %}

Running this code will keep track of the previous node to each node along the shortest path, so we need to implement the following method to read the path.

{% highlight python %}
def readPath(prev, source, target):
    [c1,c2] = target
    path = [target]
    while [c1,c2] != source:
        [c1,c2] = prev[c1][c2]
        path.append([c1,c2])
    return path[::-1]
{% endhighlight %}

And that's it! A nice thing to notice is that no matter what two squares we choose, we can always move a knight from one to the other.

If you liked the snippet, you can check out the code on my Github.