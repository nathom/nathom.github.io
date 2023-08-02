import time
from multiprocessing import Pool

import numpy as np


def basel(N: int) -> float:
    return sum(x ** (-2) for x in range(1, N))


def basel_less_pythonic(N: int) -> float:
    s = 0.0
    for x in range(1, N):
        s += x ** (-2)
    return s


def basel_np(N: int) -> tuple[float, list[float]]:
    times = []
    since_start = 0.0

    start = time.perf_counter()
    ones = np.ones(N - 1)
    end = time.perf_counter()
    step_time = end - start
    times.append(step_time)
    since_start += step_time

    start = time.perf_counter()
    r = np.arange(1, N)
    end = time.perf_counter()
    step_time = end - start
    times.append(step_time)
    since_start += step_time

    start = time.perf_counter()
    div = ones / r
    end = time.perf_counter()
    step_time = end - start
    times.append(step_time)
    since_start += step_time

    start = time.perf_counter()
    square = np.square(div)
    end = time.perf_counter()
    step_time = end - start
    times.append(step_time)
    since_start += step_time

    start = time.perf_counter()
    ret = np.sum(square)
    end = time.perf_counter()
    step_time = end - start
    times.append(step_time)
    since_start += step_time

    return ret, times


def basel_np_broadcast(N) -> float:
    return ((1 / np.arange(1, N)) ** 2).sum()


# [N1, N2], inclusive
def basel_np_range_timed(N1: int, N2: int):
    times = []
    since_start = 0.0

    start = time.perf_counter()
    ones = 1
    end = time.perf_counter()
    step_time = end - start
    times.append(step_time)
    since_start += step_time

    start = time.perf_counter()
    r = np.arange(N1, N2 + 1)
    end = time.perf_counter()
    step_time = end - start
    times.append(step_time)
    since_start += step_time

    start = time.perf_counter()
    div = ones / r
    end = time.perf_counter()
    step_time = end - start
    times.append(step_time)
    since_start += step_time

    start = time.perf_counter()
    square = np.square(div)
    end = time.perf_counter()
    step_time = end - start
    times.append(step_time)
    since_start += step_time

    start = time.perf_counter()
    ret = np.sum(square)
    end = time.perf_counter()
    step_time = end - start
    times.append(step_time)
    since_start += step_time

    return ret, times


def basel_np_broadcast_timed(N: int) -> tuple[float, list[float]]:
    times = []
    since_start = 0.0

    start = time.perf_counter()
    ones = 1
    end = time.perf_counter()
    step_time = end - start
    times.append(step_time)
    since_start += step_time

    start = time.perf_counter()
    r = np.arange(1, N)
    end = time.perf_counter()
    step_time = end - start
    times.append(step_time)
    since_start += step_time

    start = time.perf_counter()
    div = ones / r
    end = time.perf_counter()
    step_time = end - start
    times.append(step_time)
    since_start += step_time

    start = time.perf_counter()
    square = np.square(div)
    end = time.perf_counter()
    step_time = end - start
    times.append(step_time)
    since_start += step_time

    start = time.perf_counter()
    ret = np.sum(square)
    end = time.perf_counter()
    step_time = end - start
    times.append(step_time)
    since_start += step_time

    return ret, times


def basel_np_range_untimed(N1, N2):
    ones = 1
    r = np.arange(N1, N2 + 1)
    div = ones / r
    square = np.square(div)
    ret = np.sum(square)
    return ret


def basel_chunks(N1: int, N2: int, chunk_size: int):
    s = 0.0
    num_chunks = (N2 - N1) // chunk_size
    times = np.zeros(5)
    for i in range(num_chunks):
        r, t = basel_np_range_timed(N1 + i * chunk_size + 1, N1 + (i + 1) * chunk_size)
        s += r
        times += t

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
    with Pool(num_cores) as p:
        result = p.starmap(basel_chunks, Ns)
    s = sum(x for x, _ in result)
    times = sum(x for _, x in result)
    return s, times


def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.6f} seconds.")
        return result

    return wrapper
