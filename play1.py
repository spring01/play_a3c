
import subprocess
import time
import sys

num_worker = int(sys.argv[1])
worker_list = [subprocess.Popen(['python', 'run_worker.py', '%d' % num, '%d' % num_worker], stderr=subprocess.STDOUT)
               for num in range(num_worker)]

try:
    while True:
        time.sleep(1)
finally:
    for worker in worker_list:
        worker.terminate()



