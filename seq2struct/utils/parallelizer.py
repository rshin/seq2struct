

from abc import ABC, abstractmethod

import multiprocessing
import tqdm

class Parallelizer(ABC):
    def __init__(self, nproc, batch_size):
        """
        Create a Parallelizer

        Args
            nproc: the total number of processes to use
            batch_size: the size of a single chunk of elements to be sent to a process
        """
        assert isinstance(nproc, int) and nproc >= 1
        assert isinstance(batch_size, int) and batch_size >= 1
        self.nproc = nproc
        self.batch_size = batch_size

    @abstractmethod
    def _map(self, fn, args):
        """
        Map the given function over the arguments, and yields each of the results.

        This function should behave functionally identically to the builtin `map`.

        This function should assume no batching, sending each of the arguments individually
            to a process. It also does not need to handle any progress-bar logic.
        """

    @abstractmethod
    def _parallel_process_bar_class(self):
        """
        Return a class with an interface similar to tqdm.tqdm, whose update function
            can be shipped to individual processes.
        """

    def map_batched(self, fn, args):
        """
        Run the function fn on each of the args, in parallel, and yields each of
            the results.

        This function should behave functionally identically to the builtin `map`

        Args
            fn: the function to run
            args: an iterable of arguments to run it on
        """
        args = list(args)

        pbar = (self._parallel_process_bar_class() if self.nproc > 1 else tqdm.tqdm)(total=len(args), smoothing=0, dynamic_ncols=True)
        batches = [(fn, pbar.update, chunk) for chunk in chunked(args, self.batch_size)]

        for processed_batch in (self._map if self.nproc > 1 else map)(Parallelizer.batch_process, batches):
            yield from processed_batch

        pbar.close()

    @staticmethod
    def batch_process(batch):
        """
        Processes the given batch.
        """
        fn, pbar, chunk = batch
        results = []
        for arg in chunk:
            results.append(fn(arg))
            pbar()
        return results

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

class CPUParallelizer(Parallelizer):
    def _map(self, fn, args):
        with multiprocessing.Pool(self.nproc) as pool:
            yield from pool.imap(fn, args)
    def _parallel_process_bar_class(self):
        return MultiProcessTqdm
