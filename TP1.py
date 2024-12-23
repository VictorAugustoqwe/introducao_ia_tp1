
import copy
from collections import deque
import heapq
import time
import sys

solution = [[1,2,3],[4,5,6],[7,8,0]]

def findEmpty(matrix):
  for line in range(3):
    for column in range(3):
      if (matrix[line][column] == 0):
        return (line,column)

  raise Exception('sem espaço vazio')


def expand(node):
  matrix = node.matrix
  (line,column) = findEmpty(matrix)
  validresults = []
  #Up
  if(line != 0):
    matrix_copy = copy.deepcopy(matrix)
    matrix_copy[line][column] = matrix_copy[line-1][column]
    matrix_copy[line-1][column] = 0

    new = Node(matrix_copy)
    new.parent = node
    validresults.append(new)

  #Right
  if(column != 2):
    matrix_copy = copy.deepcopy(matrix)
    matrix_copy[line][column] = matrix_copy[line][column+1]
    matrix_copy[line][column+1] = 0

    new = Node(matrix_copy)
    new.parent = node
    validresults.append(new)

  #Down
  if(line != 2):
    matrix_copy = copy.deepcopy(matrix)
    matrix_copy[line][column] = matrix_copy[line+1][column]
    matrix_copy[line+1][column] = 0

    new = Node(matrix_copy)
    new.parent = node
    validresults.append(new)

  #Left
  if(column != 0):
    matrix_copy = copy.deepcopy(matrix)
    matrix_copy[line][column] = matrix_copy[line][column-1]
    matrix_copy[line][column-1] = 0

    new = Node(matrix_copy)
    new.parent = node
    validresults.append(new)

  return validresults

def print_matrix(matrix):
  for line in matrix:
    for column in line:
      print(column, end = ' ')
    print()

class Node:
  def __init__(self, matrix):
      self.matrix = matrix
      self.parent = None
      self.path = None
      self.cost = -1

  def __str__(self):
    string = ""
    for line in self.matrix:
      for column in line:
        if column != 0:
          string += str(column) + " "
        else:
          string += "  "
      string += "\n"
    return string

  def __eq__(self, other):
      if(isinstance(other, Node)):
        return isinstance(other, Node) and self.matrix == other.matrix
      else:
        return self.matrix == other

  def __lt__(self, other):
        return self.cost < other.cost

class Stack:
  def __init__(self):
    self.data = []
  def push(self, item):
    self.data.append(item)
  def pushMany(self, itens):
    self.data.extend(itens)
  def pop(self):
    return self.data.pop()
  def isEmpty(self):
    return len(self.data) == 0
  def isInStack(self, item):
    return item in self.data

class Queue:
  def __init__(self):
    self.data = deque()
  def push(self, item):
    self.data.append(item)
  def pushMany(self, itens):
    self.data.extend(itens)
  def pop(self):
    return self.data.popleft()
  def isEmpty(self):
    return len(self.data) == 0
  def isInQueue(self, item):
    return item in self.data

def breadthFirst(matrix):
  frontier = Queue()
  expanded = set()

  frontier.push(Node(matrix))
  count = 0
  while not(frontier.isEmpty()):
    cur_node = frontier.pop()
    if str(cur_node.matrix) in expanded:
      continue
    count = count + 1
    expanded.add(str(cur_node.matrix))

    if cur_node == solution:
      return(cur_node, count)

    cur_expand = expand(cur_node)
    valid_expansions = []
    for exp in cur_expand:
      if not(str(exp.matrix) in expanded):
        valid_expansions.append(exp)


    frontier.pushMany(valid_expansions)

  print("Not found")

def getDepth(node):
  cur = node
  depth = 0
  while cur != None:
    cur = cur.parent
    depth = depth + 1
  return depth

def iterativeDeepening(matrix):
  limit = 0
  count = 0
  while True:
    limit = limit + 1
    frontier = Stack()

    firstNode = Node(matrix)
    firstNode.path = set()
    firstNode.path.add(str(matrix))
    frontier.push(firstNode)

    while not(frontier.isEmpty()):
      cur_node = frontier.pop()
      count = count + 1

      if cur_node == solution:
        return(cur_node, count)

      cur_path = cur_node.path
      if(len(cur_path) >= limit):
        continue

      cur_expand = expand(cur_node)
      valid_expansions = []
      for exp in cur_expand:
        if not(str(exp.matrix) in cur_path):
          valid_expansions.append(exp)

      for valid in valid_expansions:
        this_path = cur_path.copy()
        this_path.add(str(valid.matrix))
        valid.path = this_path

      #Evita uso de memória desnecessário
      cur_node.path = None

      frontier.pushMany(valid_expansions)

def uniformCost(matrix):
  frontier = []
  expanded = set()
  costDict = {}

  newNode = Node(matrix)
  newNode.cost = 0
  heapq.heappush(frontier, newNode)

  count = 0
  while len(frontier) > 0:
    cur_node = heapq.heappop(frontier)
    if str(cur_node.matrix) in expanded:
      continue
    count = count + 1
    if cur_node == solution:
      return(cur_node, count)

    expanded.add(str(cur_node.matrix))

    cur_expand = expand(cur_node)
    valid_expansions = []
    for exp in cur_expand:
      if not(str(exp.matrix) in expanded):
        if not(str(exp.matrix) in costDict):
          exp.cost = cur_node.cost + 1
          costDict[str(exp.matrix)] = exp.cost
          heapq.heappush(frontier, exp)
        else:
          exp.cost = cur_node.cost + 1
          if costDict[str(exp.matrix)] > exp.cost:
            costDict[str(exp.matrix)] = exp.cost
            heapq.heappush(frontier, exp)

def outOfPlace(matrix):
  solution = [[1,2,3],[4,5,6],[7,8,0]]
  count = 0
  for i in range(3):
    for j in range(3):
      if(matrix[i][j] != solution[i][j] and solution[i][j] != 0):
        count = count + 1
  return count

def manhattanDistance(matrix):
  cost = 0
  for i in range(3):
    for j in range(3):
      if matrix[i][j] != 0:
        correctI = (matrix[i][j]-1) // 3
        correctJ = (matrix[i][j]-1) % 3
        cost = cost + abs(i - correctI) + abs(j - correctJ)
  return cost

def greedySearch(matrix, heuristic):
  frontier = []
  expanded = set()
  newNode = Node(matrix)
  newNode.cost = heuristic(newNode.matrix)
  heapq.heappush(frontier, newNode)

  count = 0
  while len(frontier) > 0:
    cur_node = heapq.heappop(frontier)
    if str(cur_node.matrix) in expanded:
      continue
    count = count + 1
    if cur_node == solution:
      return(cur_node, count)

    expanded.add(str(cur_node.matrix))
    cur_expand = expand(cur_node)
    valid_expansions = []
    for exp in cur_expand:
      if not(str(exp.matrix) in expanded):
        exp.cost = heuristic(exp.matrix)
        heapq.heappush(frontier, exp)

def aStar(matrix, heuristic):
  frontier = []
  expanded = set()
  costDict = {}

  newNode = Node(matrix)
  newNode.cost = heuristic(newNode.matrix) + (getDepth(newNode) - 1)
  heapq.heappush(frontier, newNode)

  count = 0
  while len(frontier) > 0:
    cur_node = heapq.heappop(frontier)
    if str(cur_node.matrix) in expanded:
      continue

    count = count + 1

    if cur_node == solution:
      return(cur_node, count)

    expanded.add(str(cur_node.matrix))

    cur_expand = expand(cur_node)
    valid_expansions = []
    for exp in cur_expand:
      if not(str(exp.matrix) in expanded):
        if not(str(exp.matrix) in costDict):
          exp.cost = heuristic(exp.matrix) + (getDepth(exp) - 1)
          costDict[str(exp.matrix)] = exp.cost
          heapq.heappush(frontier, exp)
        else:
          exp.cost = heuristic(exp.matrix) + (getDepth(exp) - 1)
          if costDict[str(exp.matrix)] > exp.cost:
            costDict[str(exp.matrix)] = exp.cost
            heapq.heappush(frontier, exp)

def hillClimbing(matrix, heuristic, k):
  current = Node(matrix)
  current.cost = heuristic(current.matrix)

  count = 0
  notImproving = 0
  while notImproving < k:
    count = count + 1
    if current == solution:
      return(current, count)

    neighbours = expand(current)
    best = neighbours[0]
    for neighbor in neighbours:
      neighbor.cost = heuristic(neighbor.matrix)
      if neighbor.cost < best.cost:
        best = neighbor

    if best.cost < current.cost:
      current = best
      notImproving = 0
    elif best.cost == current.cost:
      current = best
      notImproving = notImproving+1
    else:
      return(current, count)
  return(current, count)

def treatInput(input):
  display = False
  inputs = input.split(' ')
  algorithm = inputs[0]

  puzzle = inputs[1:]
  if inputs[-1] == 'PRINT':
    puzzle = inputs[1:-1]
    display = True

  puzzle = [int(x) for x in puzzle]

  matrix = list()
  matrix.append(puzzle[:3])
  puzzle = puzzle[3:]

  matrix.append(puzzle[:3])
  puzzle = puzzle[3:]

  matrix.append(puzzle[:3])
  return (algorithm, matrix, display)

def solve(matrix, algorithm, display, heuristic = manhattanDistance, k = 5):
  ans_found = 0
  count = 0

  start_time = time.time()

  if algorithm == 'B':
    (ans_found, count) = breadthFirst(matrix)
  if algorithm == 'I':
    (ans_found, count) = iterativeDeepening(matrix)
  if algorithm == 'U':
    (ans_found, count) = uniformCost(matrix)
  if algorithm == 'A':
    (ans_found, count) = aStar(matrix, heuristic)
  if algorithm == 'G':
    (ans_found, count) = greedySearch(matrix, heuristic)
  if algorithm == 'H':
    (ans_found, count) = hillClimbing(matrix, manhattanDistance, 5)

  end_time = time.time()
  execution_time = end_time - start_time

  path = []
  cur = ans_found
  while cur != None:
    path.append(cur)
    cur = cur.parent
  path.reverse()
  quality = len(path)-1
  print(quality,'\n')
  if display:
    if display == True:
      for state in path:
        print(state)

  return (quality, count, execution_time)

query = " ".join(sys.argv[1:])
(algorithm, matrix, display) = treatInput(query)
(quality, count, execution_time) = solve(matrix, algorithm, display)
