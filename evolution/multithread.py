import time
import threading
from collections import deque

def calc_square(numbers, return_val1, return_val_2):
    print("calculate square numbers")
    for n in numbers:
        time.sleep(1)
        print('square:',n*n)
        return_val_1.append(n)
        return_val_2.append(n*n)

def calc_cube(numbers, return_val_1, return_val_2):
    print("calculate cube of numbers")
    for n in numbers:
        time.sleep(1)
        print('cube:',n*n*n)
        return_val_1.append(n)
        return_val_2.append(n*n*n)

arr = [2,3,8,9]

t = time.time()
return_val_1 = deque()
return_val_2 = deque()

t1= threading.Thread(target=calc_square, args=(arr, return_val_1, return_val_2))
t2= threading.Thread(target=calc_cube, args=(arr, return_val_1, return_val_2))

t1.start()
t2.start()

t1.join()
t2.join()

print('return_val_1:', return_val_1)
print('return_val_2:', return_val_2)
print("done in : ",time.time()-t)
print("Hah... I am done with all my work now!")