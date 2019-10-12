

from abc import ABC, abstractmethod

import time
import queue
import multiprocessing
import tqdm

class Parallelizer(ABC):
    """
    A parallelizer is a general purpose task used to handle the situation of
        executing a function f(x, y) on a series of inputs x_i, y_i, where
        x is often quite large and thus difficult to transport to a given
        process, but there are typically fewer distinct values of x, allowing
        for each process to use at most one value of x.

    If it takes a constant amount of time `t_x` to ship x to a process and a constant
        amount of time `t_y` to ship y to a process then execute f(x, y) and return the
        result, then if we have `n` ys for a given x, then using `k` processes
        takes time `k * t_x + n/k * t_y`, which is minimized when `t_x = n * t_y / k^2`,
        or when `k = sqrt(n * t_y / t_x)`.

    If `ratio=None` is passed in, we empirically `ratio = t_x/t_y`
        with a 1-point estimate of `t_y` and `t_x`.
    """
    def __init__(self, max_nproc, ratio=None):
        """
        Create a Parallelizer

        Args
            nproc: the total number of processes to use
            batch_size: the size of a single chunk of elements to be sent to a process
        """
        assert isinstance(max_nproc, int) and max_nproc >= 1
        assert ratio is None or isinstance(ratio, float) and ratio > 0
        self.max_nproc = max_nproc
        self.ratio = ratio

    @abstractmethod
    def start_worker(self, f, x, input_queue, output_queue):
        """
        Create a worker, and transport f and x to it.
        """

    @abstractmethod
    def create_queue(self):
        """
        Create a queue.Queue that interacts well with the worker threads.
        """

    def _map_pooled(self, f, x, n_workers, ys, pbar):
        """
        A generator that yields (f(x, y) for y in ys)
        """
        input = self.create_queue()
        output = self.create_queue()
        for indexed in enumerate(ys):
            input.put(indexed)

        for _ in range(n_workers):
            self.start_worker(f, x, input, output)

        results = [None] * len(ys)
        filled = [False] * len(ys)
        to_yield = 0
        for _ in range(len(ys)):
            index, value = output.get()
            results[index] = value
            filled[index] = True
            pbar.update()
            while to_yield < len(ys) and filled[to_yield]:
                yield results[to_yield]
                to_yield += 1

    def _compute_ratio(self, f, x, y):
        start_x = time.time()
        input_queue = self.create_queue()
        output_queue = self.create_queue()
        input_queue.put((0, y))
        self.start_worker(f, x, input_queue, output_queue)
        output_queue.get()
        end_x = time.time()
        start_y = time.time()
        f(x, y)
        end_y = time.time()
        t_x = end_x - start_x
        t_y = end_y - start_y
        return t_x / t_y

    def parallel_map(self, f, grouped_args):
        """
        Run the function fn on each of the args, in parallel, and yields each of
            the results.

        Args
            fn: the function to run
            grouped_args: an iterable containing pairs (x, ys).
                x values are "large" in some way and will be transported as
                little as possible
        Yields
            (f(x, y) for x, ys in grouped_args for y in ys)
        """

        grouped_args = list(grouped_args)
        if not grouped_args:
            return
        total = sum(len(ys) for _, ys in grouped_args)
        pbar = tqdm.tqdm(total=total, smoothing=0, dynamic_ncols=True)

        if self.max_nproc == 1:
            for x, ys in grouped_args:
                for y in ys:
                    yield f(x, y)
                    pbar.update()
            pbar.close()
            return

        ratio = self.ratio
        if ratio is None:
            x, (y, *_) = grouped_args[0]
            ratio = self._compute_ratio(f, x, y)

        nworkers = []
        generators = []
        while grouped_args:
            x, ys = grouped_args.pop(0)
            k = min(max(1, int((len(ys) / ratio) ** 0.5)), self.max_nproc)
            while k + sum(nworkers) > self.max_nproc:
                nworkers.pop(0)
                yield from generators.pop(0)

            nworkers.append(k)
            generators.append(self._map_pooled(f, x, k, ys, pbar))

        for gen in generators:
            yield from gen

        pbar.close()

class CPUParallelizer(Parallelizer):
    def start_worker(self, f, x, input_queue, output_queue):
        worker = multiprocessing.Process(target=self.multi_processing_worker, args=(f, x, input_queue, output_queue))
        worker.start()

    def create_queue(self):
        return multiprocessing.Queue()

    @staticmethod
    def multi_processing_worker(f, x, input_queue, output_queue):
        while True:
            try:
                index, y = input_queue.get(False)
            except queue.Empty:
                return
            output_queue.put((index, f(x, y)))
