
from seq2struct.utils import parallelizer
import time
import random

def func_1(x, y):
    time.sleep(random.random() * 0.1)
    return x * 10 + y

def func_2(x, y):
    return x(y)

def test_parallelizer():
    p = parallelizer.CPUParallelizer(4)
    results = list(p.parallel_map(func_1, [(2, [3, 4, 5]), (3, [4, 5, 6]), (4, [])]))
    assert results == [23, 24, 25, 34, 35, 36], str(results)

    p = parallelizer.CPUParallelizer(1)
    from_parent = 1
    results = list(p.parallel_map(func_2, [(lambda x: x + from_parent, [3, 4, 5])]))
    assert results == [4, 5, 6], str(results)
