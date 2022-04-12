from torch.utils.data import IterableDataset


class FiniteWrapperDataset(IterableDataset):

    def __init__(self, ds: IterableDataset, size: int):
        self.ds = ds
        self.size = size

    def __iter__(self):
        ds_iter = iter(self.ds)
        for _ in range(self.size):
            yield next(ds_iter)
