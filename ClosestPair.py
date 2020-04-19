def merge(C, D, by_x):
    """
    Merge the sorted lists into one.
    :param C: A list of sorted points.
    :param D: A list of sorted points.
    :return: The merged list of sorted points.
    """
    i = 0
    j = 0
    X = []
    idx = 0 if by_x else 1

    for k in range(len(C) + len(D)):
        if C[i][idx] < D[j][idx]:
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


def sort_by(p, by_x):
    # Base case
    if len(p) <= 1:
        return p

    split = int(len(p) / 2)
    left = sort_by(p[:split], by_x)
    right = sort_by(p[split:], by_x)
    return merge(left, right, by_x)


def closest_split_pair(p_x, p_y, delta):
    split = int(len(p_x) / 2) - 1
    x_ = p_x[split][0]
    s_y = [pt for pt in p_y if x_ + delta >= pt[0] > x_ - delta]
    best = delta
    best_pair = (None, None)
    for i in range(len(s_y)):
        for j in range(i + 1, min(6, len(s_y))):
            d = dist(s_y[i], s_y[j])
            if d < best:
                best = d
                best_pair = (s_y[i], s_y[j])
    return best_pair


def closest_pair(p_x, p_y):
    # Base case
    if len(p_x) <= 3:
        min_dist = None
        min_pair = None
        p = p_x + p_y
        for i, p1 in enumerate(p):
            for p2 in p[i + 1:]:
                d = dist(p1, p2)
                if min_dist is None or d < min_dist:
                    min_pair = (p1, p2)
        return min_pair

    split = int(len(p_x) / 2)
    l_x = p_x[:split]
    r_x = p_x[split:]

    l_y, r_y = [], []
    for i in range(len(p_y)):
        if p_y[i] in l_x:
            l_y.append(p_y[i])
        else:
            r_y.append(p_y[i])

    l1, l2 = closest_pair(l_x, l_y)
    r1, r2 = closest_pair(r_x, r_y)

    pts = [(l1, l2), (r1, r2)]
    m = [dist(p1, p2) for p1, p2 in pts]
    min_idx, delta = min(enumerate(m), key=lambda pt: pt[1])

    s1, s2 = closest_split_pair(p_x, p_y, delta)

    if s1 is not None and s2 is not None:
        return s1, s2

    return pts[min_idx]


def dist(p1, p2):
    return (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2


if __name__ == '__main__':
    p = [(1.0, 2.0), (1.1, 3.6), (2.2, 4.9), (0.1, 0.7), (2.0, 12.4), (0.5, 0.6), (2.2, 4.8)]
    p_x = sort_by(p, by_x=True)
    p_y = sort_by(p, by_x=False)
    print(p_x)
    print(p_y)
    print(closest_pair(p_x, p_y))
