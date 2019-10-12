

from abc import ABC, abstractmethod

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
    def __init__(self, nproc, ratio):
        """
        Create a Parallelizer

        Args
            nproc: the total number of processes to use
            batch_size: the size of a single chunk of elements to be sent to a process
        """
        assert isinstance(nproc, int) and nproc >= 1
        assert isinstance(ratio, float) and ratio > 0
        self.nproc = nproc
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

    def map_pooled(self, f, x, n_workers, ys, pbar):
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
        for _ in range(len(ys)):
            index, value = output.get()
            results[index] = value
            pbar.update()

        yield from results


    @abstractmethod
    def _parallel_process_bar_class(self):
        """
        Return a class with an interface similar to tqdm.tqdm, whose update function
            can be shipped to individual processes.
        """

    def map_dispatched(self, f, grouped_args):
        """
        Run the function fn on each of the args, in parallel, and yields each of
            the results.

        Args
            fn: the function to run
            grouped_args: an iterable containing pairs (x, ys).
        """
        args = list(grouped_args)
        total = sum(len(ys) for _, ys in grouped_args)
        pbar = (self._parallel_process_bar_class() if self.nproc > 1 else tqdm.tqdm)(total=len(args), smoothing=0, dynamic_ncols=True)

        if self.nproc == 1:
            pbar = tqdm.tqdm(total=total)
            for x, ys in grouped_args:
                for y in ys:
                    yield f(x, y)
                    pbar.update()
            pbar.close()
            return

        nworkers = []
        generators = []
        while True:
            x, ys = grouped_args.pop(0)
            k = min(max(1, int((len(ys) / self.ratio) ** 0.5)), self.nproc)
            while k + sum(nworkers) > self.nproc:
                nworkers.pop(0)
                yield from generators.pop(0)

            nworkers.append(k)
            generators.append(self.map_pooled(f, x, k, ys, pbar))

        for gen in generators:
            yield from gen

        pbar.close()

class MultiProcessTqdm:
    def __init__(self, **kwargs):
        self.queue = multiprocessing.Manager().Queue()
        self.proc = multiprocessing.Process(target=listener, args=(self.queue,), kwargs=kwargs)
        self.proc.start()

    def close(self):
        self.queue.put(None)
        self.proc.join()

    @property
    def update(self):
        return UpdateCallback(self.queue)

def listener(pbar_queue, **kwargs):
    for _ in tqdm.tqdm(iter(pbar_queue.get, None), **kwargs):
        pass

class UpdateCallback:
    def __init__(self, queue):
        self.queue = queue
    def __call__(self):
        self.queue.put('update')


def chunked(items, size):
    size = max(1, size)
    elements = []
    for it in items:
        elements.append(it)
        if len(elements) == size:
            yield elements
            elements = []
    if elements:
        yield elements

def multi_processing_worker(f, x, input_queue, output_queue):
    while True:
        try:
            index, y = input_queue.get(False)
        except queue.Queue.Empty:
            return
        output_queue.put(index, f(x, y))

class CPUParallelizer(Parallelizer):
    def start_worker(self, f, x, input_queue, output_queue):
        worker = multiprocessing.Process(target=multi_processing_worker, args=(f, x, input_queue, output_queue))
        worker.start()
        return worker

    def create_queue(self):
        return multiprocessing.Queue()

    def _parallel_process_bar_class(self):
        return MultiProcessTqdm
