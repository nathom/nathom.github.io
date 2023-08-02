import json
import time

import basel
import matplotlib.pyplot as plt
import numpy as np


def main():
    # plot_vary_chunks()
    # plot_multicore_cumulative()
    # plot_multicore_vs_chunks_total()
    run_all_timed()


def run_all_timed():
    N = int(1e11)
    chunk_size = 50000
    fs = [
        basel.basel_multicore,
        # basel.basel_chunks,
        # basel.basel,
        # basel.basel_less_pythonic,
        # basel.basel_np_broadcast,
        # basel.basel_np,
    ]
    args = [
        (N, chunk_size),
        # (1, N, chunk_size),
        # (N,),
        # (N,),
        # (N,),
        # (N,),
    ]
    timed_fs = [basel.time_function(f) for f in fs]
    for args, f in zip(args, timed_fs):
        result = f(*args)
        print(result)


def plot_multicore_vs_chunks_total():
    chunk_size = 50000
    Ns = np.logspace(7, 11, base=10, dtype=np.int64)
    totals_multicore = []
    totals_chunks = []
    #
    # for N in Ns:
    #     print(N)
    #     start = time.perf_counter()
    #     r, t = basel.basel_multicore(N, chunk_size)
    #     end = time.perf_counter()
    #     diff = end - start
    #     totals_multicore.append(diff)
    #     print(f"multicore: {diff}")
    #
    #     start = time.perf_counter()
    #     r = basel.basel_chunks_untimed(1, N, chunk_size)
    #     end = time.perf_counter()
    #     diff = end - start
    #     totals_chunks.append(diff)
    #     print(f"chunks: {diff}")

    # d = {"Multicore": totals_multicore, "Chunks": totals_chunks}
    with open("totals_multicore_chunks.json", "r") as f:
        d = json.load(f)
    #     json.dump(d, f)
    create_plot(Ns, d, "Multicore vs. Chunks Total Runtime", "N", log=False)


def plot_vary_chunks():
    N = int(1e9)
    num_chunks = np.linspace(500, int(250000), num=100, dtype=np.int64)
    times = []
    for chunk in num_chunks:
        print(chunk)
        r, t = basel.basel_chunks(1, N, chunk)
        times.append(t)
    for t in times:
        into_cumulative(t)
    with open("vary_chunks.json", "w") as f:
        json.dump(times, f)
    create_cumulative_plot(
        num_chunks,
        times,
        "Cumulative performance based on chunk size",
        "Chunk size",
        log=True,
    )


def plot_chunks_cumulative():
    Ns = np.logspace(7, 9, base=10, dtype=np.int64)
    times = []

    for N in Ns:
        print(N)
        r, t = basel.basel_chunks(1, N, 20000)
        times.append(t)

    for t in times:
        into_cumulative(t)
    with open("chunks_broadcast.json", "w") as f:
        json.dump(times, f)
    create_cumulative_plot(Ns, times, "Performance of chunked numpy", "N")


def plot_multicore_cumulative():
    Ns = np.logspace(7, 9, base=10, dtype=np.int64)
    times = []
    totals = []

    for N in Ns:
        print(N)
        start = time.perf_counter()
        r, t = basel.basel_multicore(N, 20000)
        end = time.perf_counter()
        times.append(t)
        totals.append(end - start)
        print(r, t)

    for t in times:
        into_cumulative(t)
    with open("multicore.json", "w") as f:
        json.dump({"cumulative": list(list(t) for t in times), "totals": totals}, f)
    create_cumulative_plot(Ns, times, "Performance of multicore", "N")


def plot_from_json():
    chunks = np.linspace(500, 1000000, dtype=np.int64)
    with open("vary_chunk_size.json") as f:
        times = json.load(f)
    create_cumulative_plot(chunks, times)


def slice(times, i):
    return [t[i] for t in times]


def into_cumulative(t):
    t[4] = sum(t)
    t[3] = sum(t[:-1])
    t[2] = sum(t[:-2])
    t[1] = sum(t[:-3])


def create_cumulative_plot(Ns, times: list[list[float]], title, xlabel, log=False):
    plt.plot(Ns, slice(times, 0), label="Allocate ones")
    plt.plot(Ns, slice(times, 1), label="Allocate range")
    plt.plot(Ns, slice(times, 2), label="Square range")
    plt.plot(Ns, slice(times, 3), label="Divide ones by squares")
    plt.plot(Ns, slice(times, 4), label="Final sum")
    title = plt.title(title)
    title.set_color("white")
    plt.xlabel(xlabel)
    plt.ylabel("Time (s)")
    if log:
        plt.xscale("log")
    # plt.xlim(1e7, 1e9)
    plt.legend()
    ax = plt.gca()
    ax.set_facecolor("none")

    # Change the color of the axis lines to blue
    ax.spines["left"].set_color("white")
    ax.spines["bottom"].set_color("white")
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")

    # Change the color of the tick parameters to green
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")

    # Change the color of the axis labels to red
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

    plt.savefig("perf.svg", format="svg", transparent=True)
    plt.show()


def create_plot(Ns, xs: dict, title, xlabel, log=False):
    for k, v in xs.items():
        plt.plot(Ns, v, label=k)
    title = plt.title(title)
    title.set_color("white")
    plt.xlabel(xlabel)
    plt.ylabel("Time (s)")
    if log:
        plt.xscale("log")
    # plt.xlim(1e7, 1e9)
    plt.legend()
    ax = plt.gca()
    ax.set_facecolor("none")

    # Change the color of the axis lines to blue
    ax.spines["left"].set_color("white")
    ax.spines["bottom"].set_color("white")
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")

    # Change the color of the tick parameters to green
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")

    # Change the color of the axis labels to red
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

    plt.savefig("plot.svg", format="svg", transparent=True)
    plt.show()


if __name__ == "__main__":
    main()
    # plot_from_json()
