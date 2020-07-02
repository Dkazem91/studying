a = [2,4,5,7,1,2,3,6]
from random import randint, sample
# MERGE SORT

def insertionSort(array):
    """
    worst case O(n^2)
    shifts everything up or down
    """
    for x in range(1, len(array)):
        key = array[x]
        i = x - 1
        while i >= 0 and array[i] > array[x]:
            array[i] = array[x]
            i -= 1 
        array[i + 1] = key
    return array

def selectionSort(array):
    """
    swap high low values as you progress through the list
    O(n^2)
    """
    for i in range(len(array)):
        smallest = []

        for j in range(i, len(array)):
            print(j, "J")
            if not smallest:
                smallest = [j, array[j]]
            elif array[j] < smallest[1]:
                smallest = [j, array[j]]

        array[smallest[0]] = array[i]
        array[i] = smallest[1]

    return array

def merge(array, start, mid, end):
    left = array[start:mid]
    right = array[mid:end]

    left_cursor, right_cursor = 0, 0

    for cursor in range(start, end):
        if left_cursor < len(left):
            left_item = left[left_cursor]
        else:
            left_item = None

        if right_cursor < len(right):
            right_item = right[right_cursor]
        else:
            right_item = None
        
        if not left_item:
            array[cursor] = right[right_cursor]
            right_cursor += 1
        elif not right_item:
            array[cursor] = left[left_cursor]
            left_cursor += 1
        elif left_item < right_item:
            array[cursor] = left[left_cursor]
            left_cursor += 1
        else:
            array[cursor] = right[right_cursor]
            right_cursor += 1
        


def mergeSort(array, start, end):
    """
    O(nlogn)
    """
    if start < end:
        mid = (start + end) // 2

        mergeSort(array, start, mid)
        mergeSort(array, mid + 1, end)
        merge(array, start, mid, end)


a = [5,1,4,6,9,4,3,2,8,1]

def binarySearch(a,start,end, num):
    """
    O(logn)
    """
    if start < end:
        mid = (start + end) // 2
        if a[mid] == num:
            return mid
        if a[mid] > num:
            return binarySearch(a, start, mid, num)
        else:
            return binarySearch(a, mid, end, num)
    else:
        return - 1

def bubbleSort(a):
    """
    O(n^2)
    """
    for i in range(len(a)):
        for j in range(len(a) - 1, i, -1):
            if a[j] < a[j-1]:
                temp = a[j]
                a[j] = a[j-1]
                a[j-1] = temp
            
a = [-13,-3,-25,-20,-3,-16,-23,-18,-20,-7,-12,0,-22,-15,-4,-7]

def maxCrossingSubArray(a, low, mid, high):
    max_left_index, max_right_index = 0, 0
    left_sum, right_sum = 0, 0

    sum = 0

    for i in range(mid,low-1, -1):
        sum += a[i]

        if not left_sum or sum > left_sum:
            left_sum = sum
            max_left_index = i
    
    sum = 0

    for j in range(mid + 1, high + 1):
        sum += a[j]

        if not right_sum or sum > right_sum:
            right_sum = sum
            max_right_index = j

    return (max_left_index, max_right_index, left_sum + right_sum)

def maxSubArray(a, low, high):
    if high == low:
        return(low, high, a[low])
    else:
        mid = (low + high) // 2

        left_low, left_high, left_sum = maxSubArray(a, low, mid)
        right_low, right_high, right_sum = maxSubArray(a, mid + 1, high)

        cross_low, cross_high, cross_sum = maxCrossingSubArray(a, low, mid, high)

        if left_sum >= right_sum and left_sum >= cross_sum:
            return (left_low, left_high, left_sum)
        elif right_sum >= left_sum and right_sum >= cross_sum:
            return (right_low, right_high, right_sum)
        else:
            return (cross_low, cross_high, cross_sum)

a = [13]

def bruteForceMaxSubArray(a):
    low, high = 0, 0
    max = a[0]

    for i in range(len(a)):
        if a[i] > max:
            max = a[i]
            low, high = i, i
        sum = a[i]

        for j in range(i + 1, len(a)):
            sum += a[j]
            if sum > max:
                low, high = i, j
                max = sum

    return low, high, max
            


heap = [16,4,10,14,7,9,3,2,8,1]

def left(i):
    return (i * 2) + 1
def right(i):
    return (i * 2) + 2

def parent(i):
    return i // 2

# MIN AND MAX HEAPIFY JUST CHECK IF HEAP[LEFT] OR HEAP[RIGHT] IS SMALLER THAN HEAP[i]
# flip the operaters for min vs max
def maxHeapify(heap, i):
    i_left = left(i)
    i_right = right(i)
    if i_left < len(heap) and heap[i_left] < heap[i]:
        largest = i_left
    else:
        largest = i 

    if i_right < len(heap) and heap[i_right] < heap[largest]:
        largest = i_right
    
    if largest != i:
        temp = heap[i]
        heap[i] = heap[largest]
        heap[largest] = temp
        maxHeapify(heap, largest)

def iterativeMaxHeapify(heap, i, heap_size=0):
    # needs work
    while i != -1:
        i_left = left(i)
        i_right = right(i)

        if i_left < (len(heap) - heap_size) and heap[i_left] > heap[i]:
            largest = i_left
        else:
            largest = i 

        if i_right < (len(heap) - heap_size) and heap[i_right] > heap[largest]:
            largest = i_right

        if largest != i:
            temp = heap[i]
            heap[i] = heap[largest]
            heap[largest] = temp

            i = largest
        else:
            i = -1


array = [5,13,2,25,7,17,20,8,4]

def buildMaxHeapify(array):
    for i in range((len(array)-1)//2,-1,-1):
        maxHeapify(array, i)


def heapSort(array):
    """
    O(nlogn) time 
    """
    buildMaxHeapify(array)
    heap_constraint = 0 
    for i in range(len(array) - 1, 0, -1):
        temp = array[0]
        array[0] = array[i]
        array[i] = temp
        heap_constraint += 1
        iterativeMaxHeapify(array, 0, heap_size=heap_constraint)


# how to use heapsort with a priority que??
class PriorityQueueMax():
    def __init__(self):
        self.array = [25, 13, 20, 8, 7, 17, 2, 5, 4]
    
    def max(self):
        print(self.array, "MAX: CURRENT HEAP")

        return self.array[0]
    
    def extractMax(self):
        max = self.array[0]

        self.array[0] = self.array[len(self.array) - 1]
        del self.array[-1]
        self.iterativeMaxHeapify(self.array, 0)

        print(self.array, "EXTRACT MAX: CURRENT HEAP")
        print(max)
        return max


    def iterativeMaxHeapify(self, heap, i):
        while i != -1:
            i_left = left(i)
            i_right = right(i)

            if i_left < len(self.array) and heap[i_left] > heap[i]:
                largest = i_left
            else:
                largest = i 

            if i_right < len(self.array) and heap[i_right] > heap[largest]:
                largest = i_right

            if largest != i:
                temp = heap[i]
                heap[i] = heap[largest]
                heap[largest] = temp

                i = largest
            else:
                i = -1
    
    def increaseKey(self, i, k):
        if self.array[i] and self.array[i] > k:
            print("has to be at least bigger than current key {}".format(self.array[i]))
            return

        self.array[i] = k

        while self.array[i] > self.array[parent(i)]:
            temp = self.array[i]

            self.array[i] = self.array[parent(i)]
            self.array[parent(i)] = temp

            i = parent(i)

        print(self.array, "INCREASE KEY CURRENT HEAP")

    def heapInsert(self, k):
        self.array.append(False)

        self.increaseKey(len(self.array) - 1, k)


class PriorityQueueMin():
    def __init__(self):
        self.array = [10, 30, 20, 400]
    
    def min(self):
        print(self.array, "MIN: CURRENT HEAP")

        return self.array[0]
    
    def extractMin(self):
        max = self.array[0]

        self.array[0] = self.array[len(self.array) - 1]
        del self.array[-1]
        self.iterativeMinHeapify(self.array, 0)

        print(self.array, "EXTRACT MIN: CURRENT HEAP")
        print(max)
        return max

    def iterativeMinHeapify(self, heap, i):
        while i != -1:
            i_left = left(i)
            i_right = right(i)

            if i_left < len(self.array) and heap[i_left] < heap[i]:
                largest = i_left
            else:
                largest = i 

            if i_right < len(self.array) and heap[i_right] < heap[largest]:
                largest = i_right

            if largest != i:
                temp = heap[i]
                heap[i] = heap[largest]
                heap[largest] = temp

                i = largest
            else:
                i = -1
    
    def decreaseKey(self, i, k):
        if self.array[i] and self.array[i] < k:
            print("has to be at least smaller than current key {}".format(self.array[i]))
            return

        self.array[i] = k

        while self.array[i] < self.array[parent(i)]:
            temp = self.array[i]

            self.array[i] = self.array[parent(i)]
            self.array[parent(i)] = temp

            i = parent(i)

        print(self.array, "INCREASE KEY CURRENT HEAP")

    def heapInsert(self, k):
        self.array.append(False)

        self.decreaseKey(len(self.array) - 1, k)
array = [0] * 4

def partition(a, left, right):
    pivot = a[right]
    i = left - 1
    for j in range(left, right):
        if a[j] <= pivot:
            i += 1
            a[i], a[j] = a[j], a[i]
    a[i + 1], a[right] = a[right], a[i + 1]
    return i + 1

def quickSort(a, left, right):
    if left < right:
        middle = partition(a, left, right)
        quickSort(a, left, middle - 1)
        quickSort(a,middle + 1, right)
        


array = [6,0,2,0,1,3,4,6,1,3,2]
holder_array = [0] * len(array)

def countingSort(array, max_digit):
    holder = [None] * (len(array))
    digit_counter = [0] * (max_digit + 1)
    
    for i in range(len(array)):
        digit_counter[array[i]] += 1

    for i in range(1, max_digit + 1):
        digit_counter[i] += digit_counter[i -1]

    for i in range(len(array) - 1, -1, -1):

        holder[digit_counter[array[i]] - 1] = array[i]
        digit_counter[array[i]] -= 1

    print(holder)

array =  [170, 45, 75, 90, 802, 24, 2, 66, 1000]


print(array, "BEFORE")
countingSort(array, max(array))

def radixCountingSort(array, digit):
    """
    counting sort modified to fit radix sort
    """
    digit_counter = [0] * 10
    holder = [0] * len(array)

    for i in range(len(array)):
        counter = (array[i] // digit) % 10
        digit_counter[counter] += 1


    for i in range(1, 10):
        digit_counter[i] += digit_counter[i -1]

    for i in range(len(array) - 1, -1, -1):
        counter = (array[i] // digit) % 10
        holder[ digit_counter[counter] - 1] = array[i]
        digit_counter[counter] -= 1

    for i in range(len(array)):
        array[i] = holder[i]
    print(array, "ARRAY AFTER {} RUN OF COUNTING SORT".format(digit))

def radixSort(array):
    max_digit = max(array)
    digit = 1

    while max_digit // digit > 0:
        radixCountingSort(array, digit)
        digit *= 10

def randomizedPartition(array, left, right):
    i = randint(left, right)
    array[i], array[right] = array[right], array[i]

    return partition(array, left, right)

def minimum(array):
    min = array[0]
    for i in range(1, len(array)):
        if array[i] < min:
            min = array[i]
    return min

def randomizedSelect(array, left, right, i):
    if left == right:
        return array,[left]

    pivot = randomizedPartition(array, left, right)

    num_elements = pivot - left + 1

    if i == num_elements:
        return array[pivot]
    elif i < num_elements:
        return randomizedSelect(array, left, pivot - 1, i)
    else:
        return randomizedSelect(array,pivot + 1, right, i - num_elements)

# array = [3,2,9,0,7,5,4,8,6,1]

# print(randomizedSelect(array, 0, len(array)-1, 5))

class Stack():
    def __init__(self, n):
        self.array = [None] * n
        self.top = -1
        
    def stackEmpty(self):
        if self.top == -1:
            return True
        return False
    
    def push(self, value):
        self.top += 1
        self.array[self.top] = value
        # print("pushed: ", value)
        # print("top right now: ", self.top)
        # print(self.array[:self.top +1])
        # print("******************************")

    def pop(self):
        if self.stackEmpty():
            return "underflow, nothing to pop bitch"

        popped_value = self.array[self.top]
        self.array[self.top] = None
        self.top -= 1

        # print("popped: ", popped_value)
        # print("top right now: ", self.top)
        # print(self.array[:self.top + 1])
        # print("******************************")
        return popped_value

class Queue():
    def __init__(self, n):
        self.array = [None] * n
        self.tail = 0
        self.head = 0
    
    def enqueue(self, value):
        self.array[self.tail] = value
        if self.tail == len(self.array) - 1:
            self.tail = 0
        else:
            self.tail += 1
        print('added: ', value)
        print("array right now: ", self.array)
        print("tail right now: ", self.tail)
        print("******************************")

    def dequeue(self):
        decked_value = self.array[self.head]

        if self.head == len(self.array) - 1:
            self.head = 0
        else:
            self.head += 1
        print('returning decked value: ', decked_value)
        print("array right now: ", self.array)
        print("head right now: ", self.head)
        print("******************************")

        return decked_value

# stack = Stack(6)
# stack.push(4)
# stack.push(1)
# stack.push(3)
# stack.pop()
# stack.push(8)
# stack.pop()

# queue = Queue(6)
# queue.enqueue(4)
# queue.enqueue(1)
# queue.enqueue(3)
# queue.dequeue()
# queue.enqueue(8)
# queue.dequeue()
# queue.enqueue(10)
# queue.enqueue(11)
# queue.enqueue(12)
class Element():
    def __init__(self, value):
        self.value = value
        self.next = None
        self.prev = None

    def __str__(self):
        return "element [{}]: next: [{}], prev: [{}]".format(self.value, self.next, self.prev)

class LinkedList():
    def __init__(self):
        self.head = None
    
    def insert(self, element):
        if self.head:
            element.next = self.head
            self.head.prev = element

        self.head = element
        self.head.prev = None
    
    def delete(self, element):
        if element.prev != None:
            element.prev.next = element.next
        if element.next != None:
            element.next.prev = element.prev

    def search(self, search_value):
        element = self.head
        while element.next != None and element.value != search_value:
            element = element.next
        
        return element

    def __str__(self):
        string_repr= ""
        element = self.head
        while element:
            string_repr += "[ element {} ]".format(element.value)
            if not element.next:
                break
            element = element.next

        return string_repr

# test_list = LinkedList()
# for x in range(5):
#     new_element = Element(x)
#     test_list.insert(new_element)
#     print(test_list)

# delete_element = test_list.search(2)
# test_list.delete(delete_element)
# delete_element = test_list.search(2)
# print(test_list)
# test_list.delete(delete_element)
# print(test_list)

class Node():
    def __init__(self, value):
        self.value = value
        self.parent = None
        self.left = None
        self.right = None

    def __str__(self):
        node_printed = "[NODE: {}]".format(self.value)
        return node_printed

    def __repr__(self):
        return "(Node[{}])".format(self.value)

class Tree():
    def __init__(self):
        self.count = 10
        self.root = None

    def createBinarySearchTree(self):
        self.node_values = sample(range(1, 100), 10)
        self.root = None
        for x in self.node_values:
            new_node = Node(x)
            if not self.root:
                self.root = new_node
                continue
    
            left_right = 0
            parent = None
            next_node = self.root
            while next_node != None:
                parent = next_node

                if next_node.value >= x:
                    left_right = 1
                    next_node = next_node.left
                else:
                    left_right = 0
                    next_node = next_node.right

            new_node.parent = parent
            if left_right:
                parent.left = new_node
            else:
                parent.right = new_node

    def addRandomNode(self):
        new_node = Node(randint(0,100))
        print("adding Node {}".format(new_node.value))
        if not self.root:
            self.root = new_node
            print("rooted")
            return
        else:
            next_node = self.root
            while next_node:
                left_or_right = randint(0,1)
                if left_or_right:
                    if next_node.left:
                        next_node = next_node.left
                        continue
                    else:
                        new_node.parent = next_node
                        next_node.left = new_node
                        if new_node.value > next_node.value:
                            new_node.value = next_node.value - 1
                        print("added left node")
                        print("********************************************")
                        return
                else:
                    if next_node.right:
                        next_node = next_node.right
                        continue
                    else:
                        new_node.parent = next_node
                        next_node.right = new_node
                        if new_node.value < next_node.value:
                            new_node.value = next_node.value + 1
                        print("added right node")
                        print("********************************************")
                        return

    def printTreeRecursive(self, node):
        print(node)
        if node.left:
            print("left")
            self.printTreeRecursive(node.left)
        if node.right:
            print('right')
            self.printTreeRecursive(node.right)
        else:
            return

    def printTreeIterative(self):
        next_node = self.root
        stack = Stack(10)
        stack.push(next_node)
        while not stack.stackEmpty():
            next_node = stack.array[stack.top]
            while next_node != None:
                stack.push(next_node.left)
                next_node = stack.array[stack.top]
            stack.pop()
            if not stack.stackEmpty():
                next_node = stack.pop()
                print(next_node, "PRINTING HERE")
                stack.push(next_node.right)
    
    def treeInsert(self, node):
        parent = None
        tree = self.root
        while tree != None:
            parent = tree
            if tree.value > node.value:
                tree = tree.left
            else:
                tree = tree.right
        node.parent = parent
        if parent.value > node.value:
            parent.left = node
        else:
            parent.right = node

    def recursiveTreeInsert(self, parent, node):
        if parent.value > node.value:
            if parent.left == None:
                parent.left = node
            else:
                self.recursiveTreeInsert(parent.left, node)
        else:
            if parent.right == None:
                parent.right = node
            else:
                self.recursiveTreeInsert(parent.right, node)

    def treeTransplant(self, node_1, node_2):
        if node_1.parent == None:
            self.root = node_2
        elif node_1 == node_1.parent.left:
            node_1.parent.left = node_2
        else:
            node_1.parent.right = node_2

        if node_2 != None:
            node_2.parent = node_1.parent
    
    def deleteNode(self, node):
        if node.left == None:
            self.treeTransplant(node, node.right)
        elif node.right == None:
            self.treeTransplant(node, node.left)
        else:
            successor = treeMin(node.right)
            if successor.parent != node:
                self.treeTransplant(successor, successor.right)
                successor.right = node.right
                successor.right.parent = successor

            self.treeTransplant(node, successor)
            successor.left = node.left
            successor.left.parent = successor
            

            



    def print2DUtil(self, root, space): 
        # Base case  
        if (root == None) : 
            return
  
        # Increase distance between levels  
        space += self.count
    
        # Process right child first  
        self.print2DUtil(root.right, space)  
    
        # Print current node after space  
        # count  
        print()  
        for i in range(self.count, space): 
            print(end = "-")  
        print(root.value)  
    
        # Process left child  
        self.print2DUtil(root.left, space) 




tree = Tree()
tree.createBinarySearchTree()
# for x in range(10):
#     tree.addRandomNode()

# tree.print2DUtil(tree.root, 0)
# print("*****************BEFORE*******************")
def sumChildren(node):
    left_child = node.left
    right_child = node.right
    sum = 0

    if left_child:
        sum += left_child.value
    if right_child:
        sum += right_child.value

    if (left_child or right_child) and sum > node.value:
        node.value = sum
    else:
        remainder = node.value - sum
        if left_child:
            left_child.value += remainder
        elif right_child:
            right_child.value += remainder

def traverseTree(node):
    if node != None:
        traverseTree(node.left)
        traverseTree(node.right)
        sumChildren(node)


# traverseTree(tree.root)
# tree.print2DUtil(tree.root, 0)
# print("*****************AFTER*******************")
# MULTIPLE ARRAY REP
# key = [None] * 8
# next = [None] * 8
# prev = [None] * 8

# next[1],key[1],prev[1] = 2, 4, 4
# next[2],key[2],prev[2] = None, 1, 1
# next[4],key[4],prev[4] = 1, 16, 6
# next[6],key[6],prev[6] = 4, 9, None
# print("******************************************************************")
# tree.print2DUtil(tree.root, 0)
# def treeWalk(node):
#     if node != None:
#         treeWalk(node.left)
#         print(node.value)
#         treeWalk(node.right)
# print("******************************************************************")
# treeWalk(tree.root)
# search = tree.node_values[-1]
# print('SEARCH VALUE: ', search)
def treeSearch(node, x):
    if node == None or node.value == x:
        return node
    if node.value > x:
        return treeSearch(node.left, x)
    else:
        return treeSearch(node.right, x)

def iterativeSearch(node, x):
    while node != None:
        if node.value == x:
            return node
        if node.value < x:
            node = node.right
        else:
            node = node.left

def treeMin(node):
    while node.left != None:
        node = node.left
    return node.value

def treeMax(node):
    while node.right != None:
        node = node.right
    return node.value

def treePredecesor(node):
    if node.left != None:
        print("going righto...")
        return treeMax(node.left)

    parent = node.parent
    while parent != None and node == parent.left:
        node = parent
        parent = parent.parent
    return parent

# node = iterativeSearch(tree.root, search)
# print(node)
# print("MIN: ", treeMin(tree.root))
# print("MAX: ", treeMax(tree.root))
# print("SEARCH PREDECESSOR: ", treePredecesor(node))
# insert_node = Node(26)
# tree.treeInsert(insert_node)
# print("***********************BEFORE INSERT******************************")
# tree.print2DUtil(tree.root, 0)
# print("***********************AFTER INSERT******************************")
# tree.deleteNode(insert_node)
# tree.print2DUtil(tree.root, 0)
# print("***********************AFTER DELETE******************************")

# stack = Stack(10)

# stack.push(tree.root)
# while not stack.stackEmpty():
#     next_node = stack.array[stack.top]
#     while next_node != None:
#         stack.push(next_node.left)
#         next_node = stack.array[stack.top]
#     stack.pop()
    
#     if not stack.stackEmpty():
#         next_node = stack.pop()
#         print(next_node.value, "VALUE")
#         stack.push(next_node.right)


# Dynamic Programming
rod_prices = [1,5,8,9,10,17,17,20,24, 30]

def cutRod(prices, lengths):
    if lengths == 0:
        return 0
    max_rev = -1
    for i in range(lengths):
        max_rev = max(max_rev, prices[i] + cutRod(prices, lengths - i - 1))

    return max_rev

def memoizedCutRod(prices, length):
    memo = [x for x in range(length + 1)]
    for i in range(length  + 1):
        memo[i] = -1
    return memoCutRodAux(prices, length, memo)

def memoCutRodAux(prices, length, memo):
    if memo[length] >= 0:
        return memo[length]

    if length <= 0:
        rev_max = 0
    else:
        rev_max = -1
        for i in range(length):
            rev_max = max(rev_max, prices[i] + memoCutRodAux(prices, length - i - 1, memo))

    memo[length] = rev_max

    return rev_max

def bottomUpCutRod(prices, length):
    temp = [0 for x in range(length + 1)]
    temp[0] = 0

    for i in range(1, length + 1):
        rev_max = -1
        for j in range(i):
            rev_max = max(rev_max, prices[j] + temp[i - j - 1])
        temp[i] = rev_max
    print(temp)
    return temp[length]

def extBottomUpCutRod(prices, length):
    temp = [0 for x in range(length + 1)]
    slices = [0 for x in range(length + 1)]

    for i in range(1, length + 1):
        rev_max = -1
        for j in range(i):
            previous_revenue = temp[i - j - 1]
            if rev_max < prices[j] + previous_revenue:
                rev_max = prices[j] + previous_revenue
                slices[i] = j + 1
        temp[i] = rev_max

    return temp[length], slices

def printCutRodSolution(prices, length):
    prices, slices = extBottomUpCutRod(prices, length)
    print(prices, 'max revenue')
    print(slices)
    while length > 0:
        print(slices[length], "length cut")
        length -= slices[length]
import sys

def maxtrixMultiplication(p, n):
    subs = [[0 for x in range(n)] for x in range(n)]
    mults = [[0 for x in range(n)] for x in range(n)]

    for i in range(1, n):
        subs[i][i] = 0

    for L in range(2, n):
        for i in range(1, n - L+1):
            ran = i + L - 1
            subs[i][ran] = sys.maxsize
            for k in range(i, ran):
                q = subs[i][k] + subs[k + 1][ran] + p[i-1] * p[k] * p[ran]
                if q < subs[i][ran]: 
                    subs[i][ran] = q
                    mults[i][ran] = k

    return subs[1][n - 1], mults
    
def printOptimalParens(s, i, j):
    if i == j:
        print("A")
    else:
        print("(")
        printOptimalParens(s, i, s[i][j])
        printOptimalParens(s, s[i][j] + 1, j)
        print(")")

def longestCommonSubSequence(x, y):
    m = len(x)
    n = len(y)

    c = [[0 for x in range(n + 1)] for x in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j ==0:
                continue

            if x[i - 1] == y[j - 1]:
                c[i][j] = c[i-1][j-1] + 1
            else:
                c[i][j] = max(c[i-1][j], c[i][j - 1])
    return c[m][n]
x, y = 'hellohowareyou', 'asdfasdfellowhow;jkasdf'
result = longestCommonSubSequence(x, y)

print(result)
