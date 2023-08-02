import json

import matplotlib.pyplot as plt
import numpy as np
import pi


def main():
    Ns = np.logspace(7, 9, base=10, dtype=np.int64)
    # f = pi.time_function(pi.basel_chunks)
    # r, t = f(1000000000, 20000)
    # into_cumulative(t)
    # print(f"{r=} {t=}")
    # times = []
    # chunks = np.linspace(500, 1000000, dtype=np.int64)
    # for chunk_size in chunks:
    #     print(f"{chunk_size=}")
    #     r, t = pi.basel_chunks(1000000000, chunk_size)
    #     print(f"{r=} {t=}")
    #     times.append(t)
    # for t in times:
    #     into_cumulative(t)
    # with open('vary_chunk_size.json', 'w') as f:
    #     json.dump(times, f)
    # create_plot(chunks, times)
    N = 100000000000
    # f = pi.time_function(pi.basel_np)
    # r = f(N)
    # print("np", r)
    # f2 = pi.time_function(pi.basel_chunks)
    # r = f2(1, N, 40000)
    # print("chunks", r)
    f3 = pi.time_function(pi.basel_multicore)
    r = f3(N, 40000)
    print("multicore", r)
    # f4 = pi.time_function(pi.basel)
    # r = f4(N)
    # print("pythonic", r)
    # f5 = pi.time_function(pi.basel_2)
    # r = f5(N)
    # print("less pythonic", r)


def plot_from_json():
    chunks = np.linspace(500, 1000000, dtype=np.int64)
    with open("vary_chunk_size.json") as f:
        times = json.load(f)
    create_plot(chunks, times)


def slice(times, i):
    return [t[i] for t in times]


def into_cumulative(t):
    t[4] = sum(t)
    t[3] = sum(t[:-1])
    t[2] = sum(t[:-2])
    t[1] = sum(t[:-3])


def create_plot(Ns, times: list[list[float]]):
    plt.plot(Ns, slice(times, 0), label="Allocate ones")
    plt.plot(Ns, slice(times, 1), label="Allocate range")
    plt.plot(Ns, slice(times, 2), label="Square range")
    plt.plot(Ns, slice(times, 3), label="Divide ones by squares")
    plt.plot(Ns, slice(times, 4), label="Final sum")
    title = plt.title("Cumulative performance based on chunk size")
    title.set_color("white")
    plt.xlabel("Chunk size")
    plt.ylabel("Time (s)")
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


if __name__ == "__main__":
    main()
    # plot_from_json()
