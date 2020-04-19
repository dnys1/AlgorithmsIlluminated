from random import randint


comparisons = 0


def swap(x, a, b):
    temp = x[a]
    x[a] = x[b]
    x[b] = temp


def quick_sort(x, l, r):
    """
    Quick Sort implementation
    :param x: Array of integers to sort.
    :return: Sorted array of integers.
    """
    global comparisons
    # Base case
    if l >= r:
        return

    i = choose_pivot(x, l, r)
    swap(x, l, i)
    j = partition(x, l, r)
    quick_sort(x, l, j - 1)
    quick_sort(x, j + 1, r)


def choose_pivot(x, l, r):
    """
    Chooses a pivot for quick sort.
    :param x: The array of integers.
    :return: The pivot index.
    """
    # return randint(l, r)
    return l


def partition(x, l, r):
    """
    Partition the array x around pivot p.
    :param x: The array of integers to be partitioned.
    :param p: The index of the pivot.
    :return: A partitioned array x around pivot p.
    """
    global comparisons
    comparisons += (r - l)
    p = x[l]
    i = l + 1
    for j in range(l + 1, r + 1):
        if x[j] < p:
            swap(x, i, j)
            i += 1

    swap(x, l, i - 1)
    return i - 1


if __name__ == '__main__':
    comparisons = 0
    arr = []
    with open('problem5.6test1.txt', 'r') as file:
        for line in file.readlines():
            arr.append(int(line))

    quick_sort(arr, 0, len(arr) - 1)
    print((comparisons, arr))

    comparisons = 0
    arr = []
    with open('problem5.6test2.txt', 'r') as file:
        for line in file.readlines():
            arr.append(int(line))

    quick_sort(arr, 0, len(arr) - 1)
    print((comparisons, arr))

    # comparisons = 0
    # arr = []
    # with open('problem5.6.txt', 'r') as file:
    #     for line in file.readlines():
    #         arr.append(int(line))
    #
    # quick_sort(arr, 0, len(arr) - 1)
    # print(comparisons)
