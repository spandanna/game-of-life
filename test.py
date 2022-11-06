import numpy as np
import pytest

from game import (
    add_border,
    apply_rules,
    count_all_living_neighbours,
    count_living_neighbours,
)


@pytest.mark.parametrize(
    "input_arr, expected, target_x, target_y",
    [
        # sums to 0
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), 0, 1, 1),
        # sum to 3
        (np.array([[1, 1, 0], [0, 0, 0], [0, 0, 1]]), 3, 1, 1),
        # larger matrix with different target coords, sums to 2
        (np.array([[1, 1, 1, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0]]), 2, 2, 2),
    ],
)
def test_count_living_neighbours(input_arr, expected, target_x, target_y):
    actual = count_living_neighbours(input_arr, target_x, target_y)
    assert expected == actual


def test_add_border():
    input_arr = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
    expected = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    actual = add_border(input_arr)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "input_arr, expected",
    [
        # 1 living cell in the middle
        (
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
        ),
        # multiple living cells
        (
            np.array([[1, 1, 0], [1, 1, 0], [1, 0, 0]]),
            np.array([[3, 3, 2], [4, 4, 2], [2, 3, 1]]),
        ),
    ],
)
def test_count_all_living_neighbours(input_arr, expected):
    actual = count_all_living_neighbours(input_arr)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "prev_gen, summed_arr, expected",
    [
        # still life block - doesn't change
        (
            np.array(
                [
                    [1, 1, 0],
                    [1, 1, 0],
                    [0, 0, 0],
                ]
            ),
            np.array(
                [
                    [3, 3, 2],
                    [3, 3, 2],
                    [2, 2, 1],
                ]
            ),
            np.array(
                [
                    [1, 1, 0],
                    [1, 1, 0],
                    [0, 0, 0],
                ]
            ),
        ),
        # kill lonely cell
        (
            np.array(
                [
                    [1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ]
            ),
            np.array(
                [
                    [0, 1, 0],
                    [1, 1, 0],
                    [0, 0, 0],
                ]
            ),
            np.array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ]
            ),
        ),
        # kill overcrowded cell
        (
            np.array(
                [
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                ]
            ),
            np.array(
                [
                    [3, 5, 3],
                    [5, 8, 5],
                    [3, 5, 3],
                ]
            ),
            np.array(
                [
                    [1, 0, 1],
                    [0, 0, 0],
                    [1, 0, 1],
                ]
            ),
        ),
        # revive a cell
        (
            np.array(
                [
                    [1, 1, 0],
                    [1, 0, 0],
                    [0, 0, 0],
                ]
            ),
            np.array(
                [
                    [2, 2, 1],
                    [2, 3, 1],
                    [1, 1, 0],
                ]
            ),
            np.array(
                [
                    [1, 1, 0],
                    [1, 1, 0],
                    [0, 0, 0],
                ]
            ),
        ),
    ],
)
def test_apply_rules(prev_gen, summed_arr, expected):
    actual = apply_rules(prev_gen, summed_arr)
    np.testing.assert_array_equal(actual, expected)