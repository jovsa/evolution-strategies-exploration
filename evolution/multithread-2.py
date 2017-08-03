import Queue

my_queue = Queue(maxsize=0)
my_queue.put(1)
my_queue.put(2)
my_queue.put(3)
print (my_queue.get())
my_queue.task_done()