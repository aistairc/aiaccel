from __future__ import annotations
import copy
from typing import Any


class _buffer:
    def __init__(self, label: str) -> None:
        """
        Args:
            labal (str): A name of list.
        """
        self.label = label
        self.arr: list[Any] = []
        self._max_size = 65535

    def __call__(self, index: int) -> Any:
        return self.arr[index]

    def set_max_len(self, value: int) -> None:
        """ Set any max size of this label length.
        """
        self._max_size = int(value)

    def Add(self, value: Any) -> None:
        """ Append any data.
        """
        if self.Len == self._max_size:
            self.arr = [self.arr[-1]]
        self.arr.append(value)

    @property
    def Pre(self) -> Any:
        """ Refers to the previous value.
        """
        if self.Len >= 2:
            return self.arr[-2]
        else:
            return None

    @property
    def Now(self) -> Any:
        """ Refers to the current value.
        """
        return self.arr[-1]

    def Clear(self) -> None:
        """ Initialize list.
        """
        self.arr = []

    def Replace(self, arr: list[Any]) -> None:
        """ Replace to any list data.
        """
        self.Clear()
        self.arr = copy.deepcopy(arr)

    @property
    def Len(self) -> int:
        """ Get list length.
        """
        return len(self.arr)

    @property
    def Data(self) -> list[Any]:
        """ Get the list data itself.
        """
        return self.arr

    def Value(self, index: int) -> None:
        """ Get any data in list.

        Args:
            index (int): A index of list.
        """
        return self.arr[index]

    def Del(self, index: int) -> None:
        """ Delete any data in list.

        Args:
            index (int): A index of list.
        """
        self.arr.pop(index)

    @property
    def Is_Empty(self) -> bool:
        """ Itself is empty or not
        """
        if self.arr == []:
            return True
        else:
            return False

    def Duplicate(self, value: Any) -> int:
        """ Get index if exists duplicate data in list else -1.

        Args:
            value (Any): Check to see if the same value already exists.

        Returns:
            int: index value or -1.
        """
        for i in range(len(self.arr)):
            if self.arr[i] == value:
                return i
        return -1

    def delta(self) -> Any:
        """ Get numerical difference.
        """
        return self.Now - self.Pre

    def point_diff(self, idx_pre: int, idx_now: int) -> Any:
        """ Get the difference between any two points.

        Args:
            idx_pre (int): Any index value.
            idx_now (int): Any index value.

        Returns:
            any: difference between any two points.
        """
        return self.arr[idx_now] - self.arr[idx_pre]

    def has_difference(self, digit: int | None = None) -> bool:
        """Check there is a difference or not.

        Args:
            digit (int | None, optional): If this value is set, the value is
                rounded to the specified digit. Defaults to None.
        """
        if len(self.arr) >= 2:
            if digit is None:
                return (self.Pre != self.Now)
            else:
                return (round(self.Pre, digit) != round(self.Now, digit))
        else:
            return False


class Buffer:
    """ Buffer

    Args:
        labels (tuple) : Label names.

    Attributes:
        labels (list): A list of buffer data names.
        num_buff (int): A length of labels.
        d (dict): A dictionary for accessing arbitrary buffer data

    Example:
     ::

        # create buffer
        buff = Buffer(["data1", "data2", "data3"])

        # add data
        buff.d["data1"].Add(x)
        buff.d["data2"].Add(x)
        buff.d["data3"].Add(x)
        # or
        buff.Add("data1", x)
        buff.Add("data2", x)
        buff.Add("data3", x)
    """

    def __init__(self, *labels: Any) -> None:
        self.labels = labels[0]
        self.num_buff = len(self.labels)
        self.d = {}
        for i in range(self.num_buff):
            self.d[self.labels[i]] = _buffer(self.labels[i])

    def Add(self, label: str, value: Any) -> None:
        """ Add a data to any buffer.

        Args:
            label (str): A target buffer labele.
            value (any): Additional.
        """
        self.d[label].Add(value)

    def Del(self, label: str, index: int) -> None:
        """ Delete a any data in any buffer.

        Args:
            label (str): A target label.
            index (int): A Index to be deleted in the target buffer.
        """
        self.d[label].Del(index)

    def Clear(self, label: str) -> None:
        """ Delete all buffer data of the target.

        Args:
            label (str): A target label.
        """
        self.d[label].Clear()
