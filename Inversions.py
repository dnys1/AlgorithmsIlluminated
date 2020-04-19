def count_inversions(A):
    """
    Merge Sort implementation
    :param A: Array of unsorted integers
    :return: Array of sorted integers
    """
    # Base case
    if len(A) <= 1:
        return A, 0

    half = round(len(A) / 2)
    C, left = count_inversions(A[:half])
    D, right = count_inversions(A[half:])
    B, split = count_split_inversions(C, D)
    return B, left + right + split


def count_split_inversions(C, D):
    i = 0
    j = 0
    X = []
    split_inv = 0

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
                    split_inv += 1
                    i += 1
                break
            else:
                split_inv += len(C[i:])
                j += 1

    return X, split_inv


if __name__ == "__main__":
    A = []
    answer = 2504602956
    with open('problem3.5.txt', 'r') as nums:
        for num in nums.readlines():
            A.append(int(num))
    print(count_inversions(A)[1])
    print(answer)
