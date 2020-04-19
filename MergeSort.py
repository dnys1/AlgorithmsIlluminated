def merge_sort(A):
    """
    Merge Sort implementation
    :param A: Array of unsorted integers
    :return: Array of sorted integers
    """
    # Base case
    if len(A) <= 1:
        return A

    split = round(len(A)/2)
    C = merge_sort(A[:split])
    D = merge_sort(A[split:])
    return merge(C, D)


def merge(C, D):
    i = 0
    j = 0
    X = []

    for k in range(len(C) + len(D)):
        if C[i] < D[j]:
            X.append(C[i])
            if i + 1 == len(C):
                for _ in range(len(D[j:])):
                    X.append(D[j])
                    j += 1
                break
            else:
                i += 1
        else:
            X.append(D[j])
            if j + 1 == len(D):
                for _ in range(len(C[i:])):
                    X.append(C[i])
                    i += 1
                break
            else:
                j += 1

    return X


if __name__ == "__main__":
    A = [1, 4, 5, 6, 2, 8, 10, 21, 4, 5, 2, 11]
    print(merge_sort(A))
