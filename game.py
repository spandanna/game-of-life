from random import randint
import sys
from time import sleep

import numpy as np


def neighbour_coords() -> list[tuple]:
    """
    Returns coords of neighbouring cells.
    """
    return [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def count_living_neighbours(arr: np.array, x: int, y: int) -> int:
    """
    Counts the number of living neighbours around a single cell.
    """
    coords = neighbour_coords()
    neighbours = [arr[x + coord[0], y + coord[1]] for coord in coords]
    n_neighbours = sum(neighbours)
    return n_neighbours


def add_border(arr: np.array) -> np.array:
    """
    Adds a border of 0s around the inputted array.
    This allows us to easily iterate over all target cells with a 3x3 matrix.
    """
    x = arr.shape[0]
    y = arr.shape[1]

    new_arr = np.repeat([0], (x + 2) * (y + 2)).reshape(x + 2, y + 2)
    new_arr[1 : x + 1, 1 : y + 1] = arr.copy()
    return new_arr


def count_all_living_neighbours(arr: np.array) -> np.array:
    """
    Replaces the value of all cells with the count of their living neighbours.
    """
    output = arr.copy()
    border_arr = add_border(arr)
    # the location of the first cell whose neighbours we want to count is (1,1)
    first_x = 1
    first_y = 1
    # stopping point is at the last point of the original array (border array coords - 1)
    last_x = border_arr.shape[0] - 1
    last_y = border_arr.shape[1] - 1
    # loop through all the x and ys of target cells and replace the value in the original array
    for x in range(first_x, last_x):
        for y in range(first_y, last_y):
            s = count_living_neighbours(border_arr, x, y)
            output[x - 1, y - 1] = s
    return output


def apply_rules(arr: np.array, summed: np.array) -> np.array:
    """
    Applies the rules for Conway's game of life.
    """
    lonely = ((summed == 0) | (summed == 1)) & (arr == 1)
    overcrowded = (summed > 3) & (arr == 1)
    kill = np.where(lonely | overcrowded)
    dead_coords = list(zip(kill[0], kill[1]))

    revive = (summed == 3) & (arr == 0)
    sustain = ((summed == 2) | (summed == 3)) & (arr == 1)
    alive = np.where(revive | sustain)
    living_coords = list(zip(alive[0], alive[1]))
    for c in dead_coords:
        arr[c] = 0
    for c in living_coords:
        arr[c] = 1
    return arr


def render(grid, stay=0.1):
    """
    Write out the given grid to the terminal. Stay for `stay` in seconds.
    """
    formatting = {"1": "x", "0": " ", "[": " ", "]": " "}
    rendered = grid.copy()
    for old, new in formatting.items():
        rendered = str(rendered).replace(old, new)

    sys.stdout.write(rendered + "\n")
    sleep(stay)
    for _ in range(grid.shape[1]):
        sys.stdout.write("\x1b[2K")
        sys.stdout.write("\033[F")
        sys.stdout.flush()


def generate(dims: tuple, mode: str = "glider"):
    modes = ["glider", "random", "sparse"]

    if not mode:
        choose_random_mode = randint(0, len(modes) - 1)
        mode = modes[choose_random_mode]

    if mode == "glider":
        out = np.repeat([0], dims[0] * dims[1]).reshape(dims)
        glider = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        out[0:5, 0:5] = glider

    elif mode == "random":
        out = np.random.randint(2, size=dims)

    elif mode == "sparse":
        total = dims[0] * dims[1]
        zeroes = int(total * 0.9)
        out = np.append(np.repeat([0], zeroes), np.repeat([1], total - zeroes))
        np.random.shuffle(out)
        out = out.reshape(dims)

    return out


def run(gens: int = 500, dims: tuple = (30, 30), mode: str = None):
    """
    Runs the game of life for `gen` number of generations.
    """
    start = generate(dims=dims, mode=mode)
    render(start)
    for g in range(gens):
        summed = count_all_living_neighbours(start)
        out = apply_rules(start, summed)
        render(out)
        start = out
