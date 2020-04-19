def largest_two(arr):
    if len(arr) == 2:
        if arr[0] > arr[1]:
            return arr[0], arr[1]
        else:
            return arr[1], arr[0]

    split = int(len(arr) / 2)
    left = largest_two(arr[:split])
    right = largest_two(arr[split:])

    largest = []
    if left[0] > right[0]:
        largest.append(left.pop(0))
    else:
        largest.append(right.pop(0))

    if left[0] > largest[0]:
        largest.append(left[0])
    else:
        largest.append(right[0])

    return largest


if __name__ == '__main__':
    a = [0, 8, 9, 20, 13, 1, 40, 21]
    print(largest_two(a))