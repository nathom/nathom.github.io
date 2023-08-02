import time
from multiprocessing import Pool

import numpy as np


def basel(N: int) -> float:
    return sum(x ** (-2) for x in range(1, N))


def basel_2(N: int) -> float:
    s = 0.0
    for x in range(1, N):
        s += x ** (-2)
    return s


def basel_np(N: int) -> tuple[float, list[float]]:
    timings = []
    fstart = start = time.time()
    ones = np.ones(N - 1)
    timings.append(time.time() - start)
    print("Creating ones ", time.time() - start, time.time() - fstart)
    start = time.time()
    r = np.arange(1, N)
    timings.append(time.time() - start)
    print("Creating range took", time.time() - start, time.time() - fstart)
    start = time.time()
    squares = np.square(r)
    timings.append(time.time() - start)
    print("Squaring range took", time.time() - start, time.time() - fstart)
    start = time.time()
    div = np.divide(ones, squares, dtype=np.float64)
    timings.append(time.time() - start)
    print("Dividing ones/squares took", time.time() - start, time.time() - fstart)
    start = time.time()
    ret = float(np.sum(div))
    timings.append(time.time() - start)
    print("Final sum took", time.time() - start, time.time() - fstart)
    return ret, timings


# [N1, N2], inclusive
def basel_np_range(N1: int, N2: int):
    timings = []
    start = time.time()
    ones = np.ones(N2 - N1 + 1, dtype=np.float64)
    timings.append(time.time() - start)
    # print("Creating ones ", time.time() - start, time.time() - fstart)
    start = time.time()
    r = np.arange(N1, N2 + 1, dtype=np.float64)
    timings.append(time.time() - start)
    # print("Creating range took", time.time() - start, time.time() - fstart)
    start = time.time()
    squares = np.square(r, dtype=np.float64)
    timings.append(time.time() - start)
    # print("Squaring range took", time.time() - start, time.time() - fstart)
    start = time.time()
    div = np.divide(ones, squares, dtype=np.float64)
    timings.append(time.time() - start)
    # print("Dividing ones/squares took", time.time() - start, time.time() - fstart)
    start = time.time()
    ret = float(np.sum(div))
    timings.append(time.time() - start)
    # print("Final sum took", time.time() - start, time.time() - fstart)
    del ones
    del r
    del squares
    del div
    return ret, timings


def basel_np_range_untimed(N1: int, N2: int):
    ones = np.ones(N2 - N1 + 1, dtype=np.float64)
    r = np.arange(N1, N2 + 1, dtype=np.float64)
    squares = np.square(r, dtype=np.float64)
    div = np.divide(ones, squares, dtype=np.float64)
    ret = float(np.sum(div))
    del ones
    del r
    del squares
    del div
    return ret


def basel_chunks(N1: int, N2: int, chunk_size: int):
    s = 0.0
    num_chunks = (N2 - N1) // chunk_size
    times = [0.0] * 5
    for i in range(num_chunks):
        r, t = basel_np_range(N1 + i * chunk_size + 1, N1 + (i + 1) * chunk_size)
        s += r
        times = [x + y for x, y in zip(times, t)]

    return s, times


def basel_chunks_untimed(N1: int, N2: int, chunk_size: int):
    s = 0.0
    num_chunks = (N2 - N1) // chunk_size
    # times = [0.0] * 5
    for i in range(num_chunks):
        r = basel_np_range_untimed(N1 + i * chunk_size + 1, N1 + (i + 1) * chunk_size)
        s += r
        # times = [x + y for x, y in zip(times, t)]

    return s  # , times


def basel_multicore(N: int, chunk_size: int):
    num_cores = 10
    N_per_core = N // num_cores
    Ns = [
        # (N1, N2]
        (i * N_per_core, (i + 1) * N_per_core, chunk_size)
        for i in range(num_cores)
    ]
    print(Ns)
    with Pool(num_cores) as p:
        result = p.starmap(basel_chunks_untimed, Ns)
    print(result)
    return sum(result)


def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.6f} seconds.")
        return result

    return wrapper
