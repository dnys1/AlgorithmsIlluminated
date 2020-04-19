import math


def pad(x, n):
    x_str = str(x)
    if len(x_str) >= n:
        return x_str
    return '0' * (n - len(x_str)) + x_str


def karatsuba(x, y, n):
    # Base case
    if n == 1:
        return int(x) * int(y)

    if x == 0 or y == 0:
        return 0

    print(f'x: {x}\t\ty: {y}\t\tn: {n}')

    n = 2 ** math.ceil(math.log2(n))

    split = int(n/2)
    a = pad(x, n)[:split]
    b = pad(x, n)[split:]
    c = pad(y, n)[:split]
    d = pad(y, n)[split:]

    ac = karatsuba(a, c, split)
    bd = karatsuba(b, d, split)

    a_plus_b = int(a) + int(b)
    c_plus_d = int(c) + int(d)
    length = max(len(str(a_plus_b)), len(str(c_plus_d)))
    ad_bc = karatsuba(a_plus_b, c_plus_d, length) - ac - bd

    return int((10 ** n) * ac + (10 ** split) * ad_bc + bd)


if __name__ == '__main__':
    x = 3141592653589793238462643383279502884197169399375105820974944592
    y = 2718281828459045235360287471352662497757247093699959574966967627

    print(karatsuba(x, y, len(str(x))))
    print(x * y)
