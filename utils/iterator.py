from queue import Queue
import SharedArray
import multiprocessing
import threading
from uuid import uuid4
import numpy as np
import config
from utils import augmentation
from utils.util import balance_per_class_indices


def _sldict(arr, sl):
    if isinstance(arr, dict):
        return {k: v[sl] for k, v in arr.items()}
    else:
        return arr[sl]


def load_shared(args):
    i, array_name, fname, kwargs = args
    array = SharedArray.attach(array_name)
    array[i] = augmentation.load_augment(fname, **kwargs)


class BatchIterator(object):
    def __init__(self, batch_size, shuffle=False, seed=42):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random = np.random.RandomState(seed)

    def __call__(self, X, y=None, transform=None, color_vec=None):
        self.tf = transform
        self.color_vec = color_vec
        if self.shuffle:
            self._shuffle_arrays([X, y] if y is not None else [X], self.random)
        self.X, self.y = X, y
        return self

    def __iter__(self):
        bs = self.batch_size
        for i in range((self.n_samples + bs - 1) // bs):
            sl = slice(i * bs, (i + 1) * bs)
            Xb = _sldict(self.X, sl)
            if self.y is not None:
                yb = _sldict(self.y, sl)
            else:
                yb = None
            yield self.transform(Xb, yb)

    @classmethod
    def _shuffle_arrays(cls, arrays, random):
        rstate = random.get_state()
        for array in arrays:
            if isinstance(array, dict):
                for v in list(array.values()):
                    random.set_state(rstate)
                    random.shuffle(v)
            else:
                random.set_state(rstate)
                random.shuffle(array)

    @property
    def n_samples(self):
        X = self.X
        if isinstance(X, dict):
            return len(list(X.values())[0])
        else:
            return len(X)

    def transform(self, Xb, yb):
        return Xb, yb

    def __getstate__(self):
        state = dict(self.__dict__)
        for attr in ('X', 'y',):
            if attr in state:
                del state[attr]
        return state


class QueueIterator(BatchIterator):
    """BatchIterator with seperate thread to do the image reading."""

    def __iter__(self):
        queue = Queue(maxsize=20)
        end_marker = object()

        def producer():
            for Xb, yb in super(QueueIterator, self).__iter__():
                queue.put((np.array(Xb), np.array(yb)))
            queue.put(end_marker)

        thread = threading.Thread(target=producer)
        thread.daemon = True
        thread.start()

        item = queue.get()
        while item is not end_marker:
            yield item
            queue.task_done()
            item = queue.get()


class SharedIterator(QueueIterator):
    def __init__(self, deterministic=False, *args, **kwargs):
        self.config = config
        self.deterministic = deterministic
        self.pool = multiprocessing.Pool()
        super(SharedIterator, self).__init__(*args, **kwargs)

    def transform(self, Xb, yb):

        shared_array_name = str(uuid4())
        try:
            shared_array = SharedArray.create(
                shared_array_name, [len(Xb), 3, config.img_width,
                                    config.img_height], dtype=np.float32)

            fnames, labels = super(SharedIterator, self).transform(Xb, yb)
            args = []

            for i, fname in enumerate(fnames):
                kwargs = {}
                kwargs['w'] = config.img_width
                kwargs['h'] = config.img_height
                if not self.deterministic:
                    kwargs['aug_params'] = config.aug_params
                    kwargs['sigma'] = config.sigma
                kwargs['transform'] = getattr(self, 'tf', None)
                kwargs['color_vec'] = getattr(self, 'color_vec', None)
                args.append((i, shared_array_name, fname, kwargs))

            self.pool.map(load_shared, args)
            Xb = np.array(shared_array, dtype=np.float32)

        finally:
            SharedArray.delete(shared_array_name)

        return Xb, labels


class ResampleIterator(SharedIterator):
    def __init__(self, *args, **kwargs):
        self.count = 0
        super(ResampleIterator, self).__init__(*args, **kwargs)

    def __call__(self, X, y=None, transform=None, color_vec=None):
        if y is not None:
            alpha = config.balance['balance_ratio'] ** self.count
            class_weights = config.balance['class_weights'] * alpha \
                            + config.balance['final_balance_weights'] * (1 - alpha)
            self.count += 1
            indices = balance_per_class_indices(y, weights=class_weights)
            X = X[indices]
            y = y[indices]
        return super(ResampleIterator, self).__call__(X, y)