
import subprocess
import time
import sys

num_worker = int(sys.argv[1])
worker_list = [subprocess.Popen(['python', 'worker.py', '%d' % num, '%d' % num_worker], stderr=subprocess.STDOUT)
               for num in range(num_worker)]

start = time.time()
try:
    while True:
        time.sleep(1)
finally:
    print(time.time() - start)
    for worker in worker_list:
        worker.terminate()



