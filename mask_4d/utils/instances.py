# Modified by Rodrigo Marcuzzi from https://github.com/megvii-research/MOTR/blob/main/models/structures/instances.py
import itertools
from typing import Any, Dict, List, Union

import torch


class Tracks:
    """
    This class represents a list of tracks.
    It stores the attributes of tracks (e.g., id, labels, scores) as "fields".
    All fields must have the same ``__len__`` which is the number of tracks.

    All other (non-field) attributes of this class are considered private:
    they must start with '_' and are not modifiable by a user.

    Some basic usage:

    1. Set/get/check a field:

       .. code-block:: python

          tracks.gt_boxes = Boxes(...)
          print(tracks.pred_masks)  # a tensor of shape (N, H, W)
          print('gt_masks' in tracks)

    2. ``len(tracks)`` returns the number of tracks
    3. Indexing: ``tracks[indices]`` will apply the indexing on all the fields
       and returns a new :class:`Tracks`.
       Typically, ``indices`` is a integer vector of indices,
       or a binary mask of length ``num_instances``

       .. code-block:: python

          category_3_detections = tracks[tracks.pred_classes == 3]
          confident_detections = tracks[tracks.scores > 0.9]
    """

    def __init__(self, **kwargs: Any):
        """
        Args:
            kwargs: fields to add to this `Tracks`.
        """
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError(
                "Cannot find field '{}' in the given Tracks!".format(name)
            )
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of Tracks,
        and must agree with other existing fields in this object.
        """
        data_len = len(value)
        if len(self._fields):
            assert (
                len(self) == data_len
            ), "Adding a field of length {} to a Tracks of length {}".format(
                data_len, len(self)
            )
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "Tracks":
        """
        Returns:
            Tracks: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = Tracks()
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret

    def numpy(self):
        ret = Tracks()
        for k, v in self._fields.items():
            if hasattr(v, "numpy"):
                v = v.numpy()
            ret.set(k, v)
        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]):
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Tracks` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Tracks index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = Tracks()
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            # use __len__ because len() has to be int and is not friendly to tracing
            return v.__len__()
        raise NotImplementedError("Empty Tracks does not support __len__!")

    def __iter__(self):
        raise NotImplementedError("`Tracks` object is not iterable!")

    @staticmethod
    def cat(instance_lists: List["Tracks"]) -> "Tracks":
        """
        Args:
            instance_lists (list[Tracks])

        Returns:
            Tracks
        """
        assert all(isinstance(i, Tracks) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        ret = Tracks()
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError(
                    "Unsupported type {} for concatenation".format(type(v0))
                )
            ret.set(k, values)
        return ret

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "fields=[{}])".format(
            ", ".join((f"{k}: {v}" for k, v in self._fields.items()))
        )
        return s

    __repr__ = __str__
