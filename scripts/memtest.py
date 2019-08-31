import psutil
import time
from threading import Thread


def threaded_function():
    df = np.random.random((10000, 5000))
    time.sleep(0.5)


thread = Thread(target=threaded_function)
thread.start()

thread.join()


def get_threads_cpu_percent(p, interval=0.1):
    return p.memory_info()


# total_percent = p.cpu_percent(interval)
# total_time = sum(p.cpu_times())
# return [total_percent * ((t.system_time + t.user_time)/total_time) for t in p.threads()]

# Example usage for process with process id 8008:
proc = psutil.Process(os.getpid())
thread = Thread(target=threaded_function)
thread.start()
mems = []
for i in range(200):
    t = get_threads_cpu_percent(proc)
    mems.append(t.rss)
    time.sleep(0.01)
