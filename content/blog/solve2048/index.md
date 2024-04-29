---
title: "Solving 2048"
date: 2024-04-25T01:00:35-07:00
draft: true
toc: true
comments: true
katex: true
summary: |
    If you've been on the internet during the 2010's, you're probably familiar
    with the addictive game, **2048**. Today, we're going see how high we can
    score with the power of machine learning.
---

{{< game_2048 >}}

If you've been on the internet during the 2010's, you're probably familiar
with the addictive game, **2048**. Let's see how high we can
score with the power of machine learning.

## Game mechanics

The game works as follows. The board begins with 2 random tiles.
Each random tile has a 90% chance of being `2` and a 10% chance
of being `4`. 

There are 4 possible moves: Up, Down, Left, and Right. Each move corresponds
to a "sliding" motion on the board, where all tiles are shifted in that direction
until it collides with another tile. If two tiles have the same value $N$, the tiles
merge into a higher tile with value $2N$. Every time we merge, our score increases
by $2N$.

## Is it even that hard?

*Yes, it is*. At least to get an arbitrarily high tile. To see why, let's say you have
achieved 1024. Then, you have used 16 squares as a "scratchpad" to produce that score.
To get 2048, however, you have to produce *another* 1024 with $16-1=15$ squares as
the scratchpad. To get 4096, you need to do it with 14 tiles, and so forth. So there is
in fact a 
[theoretical maximum score of 3,932,156](https://www.reddit.com/r/2048/comments/214njx/highest_possible_score_for_2048_warning_math/),
that can be calculated solely from space constraints. We won't be getting there today, but
we will try to beat the best humans.

## Methods

There are two classes of methods that we will be using: search based and
learning based. Search-based methods try to explore a multitude of possible
game states and make a guess based on which move *most likely* will result
a desired outcome. Learning-based methods define a player model and simulate a
large number of games to learn its parameters.

### Random Player

This is our baseline. It is quite bad.

### Monte Carlo

This method works as follows:

- We have a **main** board, where we only play optimal moves
- For each possible move $m$, make a copy of the board, and on that make the move $m$
- Then, let a random player make moves on that board
until it loses. Then, we save the sum of the tiles of the final board. This will be
our **score**.
- Repeat that $N$ times, where a higher $N$ would mean more games were explored.
- Choose the move with the highest average score as the optimal move.

Here is the pseudocode:

```python
def choose_best_move(board):
    scores = [] # List of average sum of tiles
    moves = [UP, DOWN, LEFT, RIGHT]
    for move in moves:
        # Make a copy of the board
        board_copy = copy(board)
        # Make move (includes tile spawn)
        board_copy.make_move(move)
        tile_sums = []
        # Given this move, we let a random player
        # play until the game is over. Then, we record
        # the average sum of the final tiles on the board.
        for n in range(num_games):
            board_copy2 = copy(board_copy)
            # Have a random agent play until it loses
            board_copy2.random_game()
            sums.append(board_copy2.tile_sum())
        # Save the average sum of tiles in the losing configuration
        scores.append(sum(tile_sums) / len(tile_sums))
    return moves[argmax(scores)]
```


