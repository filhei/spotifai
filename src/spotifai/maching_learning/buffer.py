import torch


class Buffer:
    """Rolling buffer that simplifies reusing data over multiple training steps.
    The buffer has capacity for 'size' number of batches of data.
    Whenever 'add(x)' is called, the batch x is added to the buffer, by overwriting the oldest data if the buffer is full.

    Due to the contrastive learning training scheme this buffer is intended for, it operates on the assumption that
    an entire batch of data (x) contains data from a single contrastive class. The buffer collects multiple batches in order to have
    both positive and negative examples. The data in the buffer eventually needs to be split into two chunks to enable a model to train on positive examples.
    Some of the splitting logic is implemented in the buffer, which simplifies the logic in the training step.

    The structure of the buffer is a torch.tensor with shape (size * data_shape[0], *data_shape[1:]) - the batch dimension is
    increased by a multiple of 'size' and the rest of the dimension are kept the same.
    Incoming data will be divided and inserted in two locations, separated by half of the total capacity of the buffer. When the data is later retrieved, it can be chunked easily
    since every batch will have been divided evenly in the first and second half of the buffer, with matching indices in the resulting chunks.

    Args:
        size (int): number of batches the buffer can store.
        data_shape (tuple[int]): shape of each batch that will be added to the buffer (batch_dimension, ...).
        device (torch.device): device where buffer should exist.
        dtype (torch.dtype): data type of buffer data.

    Example:

        >>> buffer = Buffer(size=4, data_shape=(16, 2), device='cpu', dtype=torch.float) # creates a buffer with an underlying tensor of size (64, 2)
        >>> x = torch.rand(16, 2)

        >>> # fill buffer with data
        >>> buffer.add(x) # this adds the first 8 elements of x to buffer.data[0:8] and the last 8 elements to buffer.data[32:40]
        >>> buffer.add(x) # x is added to buffer.data[8:16] and buffer.data[40:48]
        >>> buffer.add(x) # x is added to buffer.data[16:24] and buffer.data[48:56]
        >>> buffer.add(x) # x is added to buffer.data[24:32] and buffer.data[56:64]

        >>> # adding more data starts overriding old entries
        >>> buffer.add(x) # x is added to buffer.data[0:8] and buffer.data[32:40]

        >>> # read and split data into two chunks
        >>> x_a, x_b = buffer.data.chunk(chunks=2, dim=0)
    """

    def __init__(self, size, data_shape, device, dtype):
        assert isinstance(data_shape, (tuple, list, torch.Size))

        dimensions = [x for x in data_shape]
        dimensions[0] = dimensions[0] * size

        self.count = 0
        self.size = size

        self._data = torch.empty(dimensions, device=device, dtype=dtype)

    @property
    def data(self):
        return self._data

    @property
    def has_data(self):
        return self.count >= self.size

    def to(self, device=None, dtype=None, empty_cuda_cache=False):
        self._data = self._data.to(device=device, dtype=dtype)

        # TODO: auto detect when to empty cache
        torch.cuda.empty_cache()

        return self

    def add(self, x):
        assert (x.shape[0] * self.size) == self._data.shape[0]
        assert x.shape[1:] == self._data.shape[1:]

        index = self.count % self.size

        # x will be split into two chunks (along batch dimension)
        m = len(x) // 2

        # buffer indices for first chunk
        index_a = m * index
        slice_a = [index_a, index_a + m]

        # buffer indices for second chunk
        index_b = index_a + m * self.size
        slice_b = [index_b, index_b + m]

        # combine buffer indices: [index * m, ..., (index + 1) * m, (index + self.size) * m, ..., (index + self.size + 1) * m]
        indices = torch.cat([torch.arange(*slice_a), torch.arange(*slice_b)])

        # copy data to buffer and implicitly split it into two chunks
        self._data[indices] = x.to(self._data.device, self._data.dtype)

        self.count += 1
