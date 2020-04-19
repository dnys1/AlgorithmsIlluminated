from typing import List, Tuple
from datetime import time
import math


def parse_time(time_str: str) -> time:
    """
    Sample input: "18:00" --> time(6 PM)
    :param time_str:
    :return:
    """
    return time(hour=int(time_str.split(':')[0]), minute=int(time_str.split(':')[1]))


def find_available_blocks(blockA: List[Tuple[str, str]], blockB: List[Tuple[str, str]], length: int) -> List[Tuple[time, time]]:
    blockA = [[parse_time(t) for t in block] for block in blockA]
    blockB = [[parse_time(t) for t in block] for block in blockB]

    free_blockA = []
    free_blockB = []

    for i, t_pair in enumerate(blockA):
        t_next = blockA[(i + 1) % len(blockA)]
        free_start = t_pair[1]
        free_end = t_next[0]

        if free_end > free_start:
            free_blockA.append((free_start, free_end))

    for i, t_pair in enumerate(blockB):
        t_next = blockB[(i + 1) % len(blockB)]
        free_start = t_pair[1]
        free_end = t_next[0]

        if free_end > free_start:
            free_blockB.append((free_start, free_end))

    free_blocks = []

    for t_a in free_blockA:
        for t_b in free_blockB:
            if t_b[0] <= t_a[0] <= t_b[1] or t_a[0] <= t_b[0] <= t_a[1]:
                overlap = (max(t_a[0], t_b[0]), min(t_a[1], t_b[1]))
                if (overlap[1] - overlap[0]) > length:
                    free_blocks.append(overlap)

    return free_blocks


def caesarCipherEncryptor(string, key):
    new_string = ''
    for char in string:
        new_ord = ord(char) + key
        if new_ord > ord('z'):
            new_ord = new_ord % ord('z') % 26
            new_string += chr(ord('a') + new_ord - 1)
        else:
            new_string += chr(new_ord)
    return new_string


def threeNumberSum(array, targetSum):
    triplets = []
    for i in range(len(array) - 2):
        for j in range(i + 1, len(array) - 1):
            for k in range(j + 1, len(array)):
                if array[i] + array[j] + array[k] == targetSum:
                    triplet = [array[i], array[j], array[k]]
                    triplet.sort()
                    triplets.append(triplet)
    return triplets


def twoSum(nums: List[int], target: int) -> List[int]:
    copy = nums[:]
    copy.sort()
    length = len(nums)
    solution = None
    for idx, num in enumerate(copy):
        subtarget = target - num
        for subnum in copy[min(idx + 1, length - 1):]:
            if subnum > subtarget:
                break
            if subnum == subtarget:
                solution = [num, subnum]
                break
        if solution is not None:
            break

    sol_idx = [None, None]
    for idx, num in enumerate(nums):
        if num == solution[0]:
            sol_idx[0] = idx
        elif num == solution[1]:
            sol_idx[1] = idx

    return sol_idx


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


def add_two_linked_lists(l1: ListNode, l2: ListNode) -> ListNode:
    num1 = 0
    i = 0
    while l1.next is not None:
        num1 += l1.val * 10 ** i
        i += 1
    num1 += l1.val * 10 ** i

    num2 = 0
    j = 0
    while l2.next is not None:
        num2 += l2.val * 10 ** j
        j += 1
    num2 += l2.val * 10 ** j

    ans = num1 + num2
    str_ans = str(ans)
    l3 = None
    for node in str_ans[:len(str_ans) - 1:-1]:
        new_node = ListNode(node)
        new_node.next = l3
        l3 = new_node

    return l3

def lengthOfLongestSubstring(s: str) -> int:
    n = len(s)
    if n == 0:
        return ''
    longest = ''
    for idx, char in enumerate(s):
        i, j = idx, idx
        length = 0
        while j < n and s[j] == s[i]:
            j += 1
            length = j - i
            if length > len(longest):
                longest = char * (j - i)
        while j < n:
            if s[i] == s[j]:
                length += 2
                if length > len(longest):
                    longest = s[i:j]
            else:
                break
            i -= 1
            j += 1
            if i < 0:
                break
    return longest


def riverSizes(matrix):
    # Write your code here.
    rows = len(matrix)
    cols = len(matrix[0])
    checked = [[False] * cols] * rows
    rivers = []

    def river_length(row, col):
        if not checked[row][col]:
            checked[row][col] = True
        else:
            return 0
        if not matrix[row][col]:
            return 0
        length = 1
        if row - 1 > 0:
            length += river_length(row - 1, col)
        if row + 1 < rows:
            length += river_length(row + 1, col)
        if col - 1 > 0:
            length += river_length(row, col - 1)
        if col + 1 < cols:
            length += river_length(row, col + 1)
        return length

    for row in range(rows):
        for col in range(cols):
            el = matrix[row][col]
            if el and not checked[row][col]:
                rivers.append(river_length(row, col))

    return rivers


class Graph:
    def __init__(self, jobs, deps):
        self.order = {}
        self.nodes = {}
        self.visited = {}
        self.visiting = {}
        self.curr_order = len(jobs)
        for job in jobs:
            self.nodes[job] = []
            self.visited[job] = False
            self.visiting[job] = False
            self.order[job] = -1
        for dep in deps:
            self.nodes[dep[0]].append(dep[1])

    def topological_order(self):
        for node in self.nodes:
            if not self.visited[node]:
                is_cyclic = self.search(node)
                if is_cyclic:
                    return True
        return False

    def search(self, start):
        if self.visiting[start]:
            return True
        self.visiting[start] = True
        for node in self.nodes[start]:
            if not self.visited[node]:
                is_cyclic = self.search(node)
                if is_cyclic:
                    return True
        self.order[start] = self.curr_order
        self.curr_order -= 1
        self.visited[start] = True
        self.visiting[start] = False
        return False


def topologicalSort(jobs, deps):
    graph = Graph(jobs, deps)
    is_cyclic = graph.topological_order()
    if -1 in graph.order.values() or is_cyclic:
        return []
    answer = [-1 for _ in range(len(graph.nodes))]
    for node in graph.order:
        answer[graph.order[node] - 1] = node
    return answer


def find_pos(arr, x, i=0):
    for idx in range(i, len(arr)):
        if arr[idx] == x:
            return idx

# Complete the minimumBribes function below.
def minimumBribes(q):
    seen = set()
    swaps = 0
    for i in range(len(q)):
        if q[i] > i + 1 and abs((i + 1)  - q[i]) > 2:
            print("Too chaotic")
            return
        seen.add(q[i])
        for j in range(q[i] - 1, 0, -1):
            if j not in seen:
                swaps += 1
    print(swaps)


def sherlockAndAnagrams(s):
    import math
    anagrams = 0
    for i, char in enumerate(s):
        if char in s[i + 1:]:
            end = math.ceil((len(s) + i) / 2) + 1
            for j in range(i + 1, end + 1):
                if j > len(s):
                    break
                sub = s[i:j]
                rev = sub[::-1]
                if sub not in s[i + 1:] and rev not in s[i + 1:]:
                    break
                anagrams += (sub in s[i + 1:])
                anagrams += len(sub) > 1 and (rev in s)
    return anagrams

def isMatch(s: str, p: str) -> bool:
    if '.' not in p and '*' not in p:
        return s == p
    ptr = 0
    match = True
    i = 0
    while i < len(p):
        if ptr >= len(s):
            return False
        curr = p[i]
        modifier = None if i + 1 >= len(p) else p[i+1]
        if modifier == '*':
            while ptr < len(s) and (s[ptr] == curr or curr == '.'):
                ptr += 1
            i += 1
        else:
            if curr != s[ptr] and curr != '.':
                return False
            ptr += 1
        i += 1
    return match and ptr == len(s)

def getHint(secret: str, guess: str) -> str:
    bulls, cows = 0, 0
    matched = [False for _ in range(len(secret))]
    for i in range(len(secret)):
        this_secret = secret[i]
        this_guess = guess[i]
        if this_secret == this_guess:
            bulls += 1
            matched[i] = True
    for i in range(len(secret)):
        this_secret = secret[i]
        this_guess = guess[i]
        if this_secret != this_guess:
            for j in range(len(secret)):
                if not matched[j] and secret[j] == this_guess:
                    matched[j] = True
                    cows += 1
                    break

    return f'{bulls}A{cows}B'

def groupThePeople(groupSizes: List[int]) -> List[List[int]]:
    answer = []
    for i in range(len(groupSizes)):
        size = groupSizes[i]
        for group in answer:
            if len(group) == size and None in group:
                for j in range(size):
                    if group[j] is None:
                        group[j] = i
                        break
                break
        else:
            answer.append([i] + [None for _ in range(size - 1)])

    return answer

def partitionLabels(S: str) -> List[int]:
    def last_index(char):
        for i, s in enumerate(S[::-1]):
            if s == char:
                return len(S) - i - 1

    end = -1
    ans = []
    while end < len(S):
        start = end + 1
        if start >= len(S):
            break
        curr = S[start]
        end = last_index(curr)

        idx = start + 1

        while idx < end:
            if last_index(S[idx]) > end:
                end = last_index(S[idx])
            idx += 1

        if len(ans):
            ans.append(end - sum(ans) + 1)
        else:
            ans.append(end + 1)

    return ans


def deckRevealedIncreasing(deck: List[int]) -> List[int]:
    import collections
    d = collections.deque()
    for x in sorted(deck)[::-1]:
        d.rotate()
        d.appendleft(x)
    return list(d)

def allPathsSourceTarget(graph):
    N = len(graph)

    def solve(node):
        if node == N-1: return [[N-1]]
        ans = []
        for nei in graph[node]:
            for path in solve(nei):
                ans.append([node] + path)
        return ans

    return solve(0)

def balancedStringSplit(s: str) -> int:
    splits = 0
    i = 0
    track = {'L': 0, 'R': 0}
    while i < len(s):
        curr = s[i]
        j = i + 1
        track[curr] = 1
        while j < len(s):
            if s[j] == curr:
                j += 1
                track[curr] += 1
            else:
                break
        while j < len(s) and track['L' if curr == 'R' else 'R'] != track[curr]:
            track[s[j]] += 1
            j += 1

        splits += 1
        i = j
        track['L'] = 0
        track['R'] = 0

    return splits

def rotate(matrix: List[List[int]]) -> None:
    """
    Do not return anything, modify matrix in-place instead.
    """

    def shift(level):
        i, j = level, level
        temp0 = matrix[i][j]
        first_cycle = True
        while first_cycle or (i, j) != (level, level):
            first_cycle = False
            if i == level:
                j += 1
                if j == n - level:
                    j -= 1
                    i += 1
            elif i == n - level - 1:
                j -= 1
                if j == level - 1:
                    j += 1
                    i -= 1
            else:
                if j == n - level - 1:
                    i += 1
                else:  # j == level
                    i -= 1
            temp1 = matrix[i][j]
            matrix[i][j] = temp0
            temp0 = temp1

    n = len(matrix)
    i = 0
    levels = int(n / 2) - 1
    while i <= levels:
        num_shifts = n - 2 * i - 1
        if num_shifts == 0:
            break

        for x in range(num_shifts):
            shift(i)
        i += 1

def spiralMatrixIII(R: int, C: int, r0: int, c0: int) -> List[List[int]]:
    visited = [[r0, c0]]
    length = 1
    ticker = 0
    steps = 0
    direction = 'E'

    def rotate(direction):
        if direction == 'N': return 'E'
        if direction == 'E': return 'S'
        if direction == 'S': return 'W'
        if direction == 'W': return 'N'

    def in_board(r0, c0):
        return r0 >= 0 and r0 < R and c0 >= 0 and c0 < C

    while len(visited) < R * C:
        if steps == length:
            ticker += 1
            steps = 0
            direction = rotate(direction)

            if ticker == 2:
                ticker = 0
                length += 1

        if direction == 'E':
            c0 += 1
        elif direction == 'S':
            r0 += 1
        elif direction == 'W':
            c0 -= 1
        elif direction == 'N':
            r0 -= 1

        if in_board(r0, c0):
            visited.append([r0, c0])

        steps += 1

    return visited

def selfDividingNumbers(left: int, right: int) -> List[int]:
    ans = []
    for num in range(left, right + 1):
        this_ans = 0
        for digit in [int(char) for char in str(num)]:
            if digit == 0: break
            this_ans |= (num % digit == 0)
        else:
            if this_ans == 0:
                ans.append(num)

    return ans


def maxProfit(prices: List[int]) -> int:
    profit = 0
    buy_day = 0
    while buy_day < len(prices):
        buy_price = prices[buy_day]
        for sell_day, sell_price in enumerate(prices[buy_day + 1:]):
            if sell_price <= buy_price:
                continue
            profit += sell_price - buy_price
            buy_day = buy_day + 1 + sell_day + 1
            break
        else:
            buy_day += 1

    return profit

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

def getIntersectionNode(headA: ListNode, headB: ListNode) -> ListNode:
    listA = []
    listB = []

    while headA is not None:
        listA.append(headA)
        headA = headA.next

    while headB is not None:
        listB.append(headB)
        headB = headB.next

    if not len(listA) or not len(listB):
        return None

    i = len(listA) - 1
    j = len(listB) - 1
    while i >= 0 and j >= 0:
        if listA[i] != listB[j]:
            i += 1
            break
        i -= 1
        j -= 1

    return listA[i] if 0 <= i < len(listA) else None

def powerfulIntegers(x: int, y: int, bound: int) -> List[int]:
    ans = []
    for num in range(1, bound + 1):
        max_pow = int(math.log(num) / math.log(max(x, y)))
        for i in range(max_pow + 1):
            for j in range(max_pow + 1):
                if num == x ** i + y ** j:
                    ans.append(num)
                    break
            else:
                continue
            break

    return ans

def checkPossibility(nums: List[int]) -> bool:
    non_decreasing = 0
    i, n = 0, len(nums)
    while i < n - 1:
        if i == 0:
            last = -float('inf')
        else:
            last = nums[i - 1]
        num = nums[i]
        next = nums[i + 1]
        if next < num:
            if num == last or next >= last:
                non_decreasing += 1
            else:
                return False
        i += 1

    return non_decreasing <= 1

def isPalindrome(n: str):
    return n == n[::-1]

def nearestPalindromic(n: str) -> str:
    test = int(n)
    if (math.log10(test) % 1 == 0):
        return str(test - 1)

    length = len(n)
    if n == '11': return '9'
    if length > 1 and n == '9' * length: return str(10 ** length + 1)

    if not isPalindrome(n):
        for idx in range(length):
            r_idx = length - 1 - idx
            if n[idx] != n[r_idx]:
                l = list(n)
                l[r_idx] = n[idx]
                test = ''.join(l)
                if isPalindrome(test):
                    return test
                else:
                    n = test
    else:
        mid = int(length / 2)
        if length % 2:
            l = list(n)
            if l[mid] != '0':
                l[mid] = str(int(l[mid]) - 1)
            else:
                l[mid] = '1'
            return ''.join(l)
        else:
            mid = int(length / 2)
            l, r = mid - 1, mid
            test = list(n)
            if n[l] == '0':
                test[l] = test[r] = '1'
            else:
                test[l] = test[r] = str(int(n[l]) - 1)
            test = ''.join(test)
            return test

def shortestSubarray(A: List[int], K: int) -> int:
    i, j = -1, -1
    min_count = float('inf')
    while min_count == float('inf') and i < len(A):
        i += 1
        j = i
        sum = 0
        while j < len(A):
            sum += A[j]
            if sum >= K:
                if i == j:
                    return 1
                min_count = min(j - i + 1, min_count)
                while i < j and sum - A[i] >= K:
                    sum -= A[i]
                    i += 1
                    min_count = min(min_count, j - i + 1)
            j += 1

    j -= 1
    while i < j:
        if sum >= K:
            if i == j:
                return 1
            min_count = min(j - i + 1, min_count)
        sum -= A[i]
        i += 1

    return min_count if min_count != float('inf') else -1

def backspaceCompare(S: str, T: str) -> bool:
    a = len(S) - 1
    b = len(T) - 1
    a_back = 0
    b_back = 0
    while a >= 0 and b >= 0:
        if S[a] == '#' and T[b] == '#':
            a_back += 1
            b_back += 1
        elif S[a] == '#':
            a_back += 1
            a -= 1
            if b_back > 0:
                b -= b_back
                b_back = 0
            continue
        elif T[b] == '#':
            b_back += 1
            b -= 1
            if a_back > 0:
                a -= a_back
                a_back = 0
            continue
        else:
            if a_back > 0:
                a -= 1
                a_back -= 1
                continue
            if b_back > 0:
                b -= 1
                b_back -= 1
                continue
            if a < 0 and b < 0:
                return True
            if a < 0 or b < 0:
                return False
            if S[a] != T[b]:
                return False

        a -= 1
        b -= 1

    return True

def evalRPN(tokens: List[str]) -> int:
    ans = 0
    ops = []
    for token in tokens:
        if token == '+' or token == '-' or token == '*' or token == '/':
            y, x = ops.pop(), ops.pop()
            if token == '+':
                ans = x + y
            elif token == '-':
                ans = x - y
            elif token == '*':
                ans = x * y
            elif token == '/':
                ans = int(x / y)
            ops.append(ans)
        else:
            ops.append(int(token))

    return ans

def coinChange(coins: List[int], amount: int) -> int:
    if amount <= 0:
        return -1
    ans = -1
    for coin in coins[::-1]:
        this = coinChange(coins, amount - coin)
        if this != -1:
            if ans != -1:
                ans = min(ans, this)
            else:
                ans = this

    return ans

def rotate(nums: List[int], k: int) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    k = k % len(nums)
    if k == 0:
        return
    prev = len(nums) - k
    curr = nums[prev]
    while True:
        next = (prev + k) % len(nums)
        temp = nums[next]
        nums[next] = curr
        prev = next
        curr = temp
        if prev == len(nums) - k:
            if k % 2:
                break
            else:
                prev = (prev + 1) % len(nums)
                next = (next + 1) % len(nums)
                curr = nums[prev]
                temp = nums[next]
                nums[next] = curr
                prev = next
                curr = temp
        if prev == len(nums) - k + 1 and k % 2 == 0:
            nums[next] = curr
            break

if __name__ == '__main__':
    # block1 = [('10:00', '12:00'), ('13:00', '14:00')]
    # block2 = [('9:00', '11:00'), ('13:00', '13:30')]
    #
    # schedule1 = ('6:00', '18:00')
    # schedule2 = ('8:00', '19:00')
    #
    # block1.append((schedule1[1], schedule1[0]))
    # block2.append((schedule2[1], schedule2[0]))
    #
    # print(find_available_blocks(block1, block2, 30))

    # print(threeNumberSum([12, 3, 1, 2, -6, 5, -8, 6], 0))
    # l1 = ListNode(2)
    # l1.next = ListNode(4)
    # l1.next.next = ListNode(3)
    #
    # l2 = ListNode(5)
    # l2.next = ListNode(6)
    # l2.next.next = ListNode(4)

    # print(add_two_linked_lists(l1, l2).val)
    # print(lengthOfLongestSubstring("babad"))

    # matrix = [[1, 0, 0, 1, 0], [1, 0, 1, 0, 0], [0, 0, 1, 0, 1], [1, 0, 1, 0, 1]]
    # print(riverSizes(matrix))
    # {}.val

    # jobs= [1, 2, 3, 4, 5, 6, 7, 8]
    # deps= [[3,1], [8,1], [8,7],[5,7],[5,2],[1,4],[1,6],[1,2],[7,6]]
    # print(topologicalSort(jobs, deps))

    # q = [1, 2, 5, 3, 7, 8, 6, 4]
    # print(minimumBribes(q))
    # print(sherlockAndAnagrams('ifailuhkqq'))
    # print(sherlockAndAnagrams('kkkk'))

    # print(isMatch("ab", ".*c"))
    #
    # print(getHint('1122', '0001'))
    # print(groupThePeople([3,3,3,3,3,1,3]))

    # print(partitionLabels("qiejxqfnqceocmy"))
    # print(deckRevealedIncreasing([17,13,11,2,3,5,7]))
    # print(allPathsSourceTarget([[1,2], [3], [3], []]))
    # print(balancedStringSplit("LLLLRRRR"))
    # mat = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
    # rotate(mat)
    # print(mat)

    # print(spiralMatrixIII(5, 6, 1, 4)
    # print(selfDividingNumbers(1, 22))
    # print(maxProfit([7,1,5,3,6,4]))

    # tail = ListNode(8)
    # tail.next = ListNode(4)
    # tail.next.next = ListNode(5)
    # headA = ListNode(4)
    # headA.next = ListNode(1)
    # headA.next.next = tail
    # headB = ListNode(5)
    # headB.next = ListNode(0)
    # headB.next.next = ListNode(1)
    # headB.next.next.next = tail
    # headA = ListNode(1)
    # headB = headA
    #
    # getIntersectionNode(headA, headB)
    # print(powerfulIntegers(2, 3, 10))
    # print(nearestPalindromic("11011"))

    # print(shortestSubarray([1, 2], 4))
    # backspaceCompare("bxj##tw","bxj###tw")
    # evalRPN(["10","6","9","3","+","-11","*","/","*","17","+","5","+"])
    rotate([1,2,3], 2)