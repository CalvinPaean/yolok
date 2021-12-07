

import bisect
from functools import wraps

from torch.utils.data.dataset import ConcatDataset as torchConcatDataset
from torch.utils.data.dataset import Dataset as torchDataset


class Dataset(torchDataset):
    """ This class is a subclass of the base :class:`torch.utils.data.Dataset`,
    that enables on the fly resizing of the ``input_dim``.

    Args:
        input_dimension (tuple): (width,height) tuple with default dimensions of the network
    """
    def __init__(self, input_dimension, mosaic=True):
        super().__init__()
        self.__input_dim = input_dimension[:2]
        self.enable_mosaic = mosaic

    @property
    def input_dim(self):
        """
        Dimension that can be used by transforms to set the correct image size, etc.
        This allows transforms to have a single source of truth
        for the input dimension of the network.

        Return:
            list: Tuple containing the current width,height
        """
        if hasattr(self, "_input_dim"):
            return self._input_dim
        return self.__input_dim


    @staticmethod
    def mosaic_getitem(getitem_fn):
        """
        Decorator method that needs to be used around the ``__getitem__`` method. |br|
        This decorator enables the closing mosaic

        Example:
            >>> class CustomSet(ln.data.Dataset):
            ...     def __len__(self):
            ...         return 10
            ...     @ln.data.Dataset.mosaic_getitem
            ...     def __getitem__(self, index):
            ...         return self.enable_mosaic
        """

        @wraps(getitem_fn)
        def wrapper(self, index):
            if not isinstance(index, int):
                self.enable_mosaic = index[0]
                index = index[1]

            ret_val = getitem_fn(self, index)

            return ret_val

        return wrapper