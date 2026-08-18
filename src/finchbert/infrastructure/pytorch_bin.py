"""Load torch.save ZIP checkpoints as NumPy arrays without importing torch."""

from __future__ import annotations

import io
import pickle
import zipfile
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any

import numpy as np

_STORAGE_DTYPES: dict[str, np.dtype[Any]] = {
    "FloatStorage": np.dtype(np.float32),
    "DoubleStorage": np.dtype(np.float64),
    "HalfStorage": np.dtype(np.float16),
    "LongStorage": np.dtype(np.int64),
    "IntStorage": np.dtype(np.int32),
    "ShortStorage": np.dtype(np.int16),
    "CharStorage": np.dtype(np.int8),
    "ByteStorage": np.dtype(np.uint8),
    "BoolStorage": np.dtype(np.bool_),
}


def load_pytorch_bin(path: Path) -> dict[str, np.ndarray]:
    if not zipfile.is_zipfile(path):
        raise ValueError(f"{path} is not a zip-format pytorch checkpoint.")
    try:
        with zipfile.ZipFile(path) as archive:
            pickle_name = _data_pickle_name(archive)
            prefix = pickle_name[: -len("data.pkl")]
            with archive.open(pickle_name) as handle:
                loaded = _TorchUnpickler(handle, archive, prefix).load()
    except (OSError, pickle.UnpicklingError, KeyError) as exc:
        raise ValueError(f"failed to load pytorch checkpoint {path}.") from exc
    return _numpy_state_dict(loaded)


def write_pytorch_bin(path: Path, tensors: Mapping[str, np.ndarray]) -> None:
    prefix = "pytorch_model.bin/"
    entries: list[tuple[str, str, str, int, tuple[int, ...], tuple[int, ...]]] = []
    with zipfile.ZipFile(path, "w") as archive:
        for index, (name, array) in enumerate(tensors.items()):
            contiguous = np.ascontiguousarray(array)
            key = str(index)
            archive.writestr(f"{prefix}data/{key}", contiguous.tobytes())
            shape = tuple(int(dim) for dim in contiguous.shape)
            entries.append(
                (
                    name,
                    key,
                    _storage_name(contiguous.dtype),
                    int(contiguous.size),
                    shape,
                    _contiguous_strides(shape),
                )
            )
        buffer = io.BytesIO()
        _TorchPickler(buffer).dump(
            OrderedDict(
                (
                    name,
                    _PickleTensor(key, storage_name, numel, shape, stride),
                )
                for name, key, storage_name, numel, shape, stride in entries
            )
        )
        archive.writestr(f"{prefix}data.pkl", buffer.getvalue())


def _data_pickle_name(archive: zipfile.ZipFile) -> str:
    for name in archive.namelist():
        if name.endswith("data.pkl"):
            return name
    raise ValueError("pytorch checkpoint is missing data.pkl.")


def _numpy_state_dict(loaded: object) -> dict[str, np.ndarray]:
    if not isinstance(loaded, Mapping):
        raise ValueError("pytorch checkpoint must contain a state dict.")
    payload = loaded.get("state_dict", loaded)
    if not isinstance(payload, Mapping):
        raise ValueError("pytorch checkpoint must contain a state dict.")
    tensors = {
        name: np.asarray(value)
        for name, value in payload.items()
        if isinstance(name, str) and isinstance(value, np.ndarray)
    }
    if not tensors:
        raise ValueError("pytorch checkpoint has no tensor weights.")
    return tensors


def _storage_name(dtype: np.dtype[Any]) -> str:
    for name, storage_dtype in _STORAGE_DTYPES.items():
        if storage_dtype == dtype:
            return name
    raise ValueError(f"unsupported tensor dtype: {dtype}.")


def _int_tuple(values: object) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("tensor size and stride must be sequences.")
    return tuple(int(dim) for dim in values)


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides: list[int] = []
    step = 1
    for dim in reversed(shape):
        strides.append(step)
        step *= int(dim)
    return tuple(reversed(strides))


def _storage_dtype(storage_type: object) -> np.dtype[Any]:
    name = getattr(storage_type, "name", None)
    if not isinstance(name, str):
        name = getattr(storage_type, "__name__", None)
    if not isinstance(name, str):
        if isinstance(storage_type, str):
            name = storage_type
        else:
            raise ValueError("unsupported torch storage type.")
    dtype = _STORAGE_DTYPES.get(name)
    if dtype is None:
        raise ValueError(f"unsupported torch storage type: {name}.")
    return dtype


def _rebuild_tensor_v2(
    storage: np.ndarray,
    storage_offset: int,
    size: object,
    stride: object,
    requires_grad: object = None,
    backward_hooks: object = None,
    metadata: object = None,
) -> np.ndarray:
    del requires_grad, backward_hooks, metadata
    shape = _int_tuple(size)
    steps = _int_tuple(stride)
    if not shape:
        return np.array(storage[int(storage_offset)], dtype=storage.dtype)
    view = storage[int(storage_offset) :]
    itemsize = int(storage.dtype.itemsize)
    tensor = np.lib.stride_tricks.as_strided(
        view,
        shape=shape,
        strides=tuple(step * itemsize for step in steps),
    )
    return np.ascontiguousarray(tensor)


def _rebuild_tensor(
    storage: np.ndarray,
    storage_offset: int,
    size: object,
    stride: object,
) -> np.ndarray:
    return _rebuild_tensor_v2(storage, storage_offset, size, stride)


def _rebuild_parameter(
    tensor: np.ndarray,
    requires_grad: object = None,
    backward_hooks: object = None,
) -> np.ndarray:
    del requires_grad, backward_hooks
    return tensor


class _PickleTensor:
    def __init__(
        self,
        key: str,
        storage_name: str,
        numel: int,
        shape: tuple[int, ...],
        stride: tuple[int, ...],
    ) -> None:
        self._key = key
        self._storage_name = storage_name
        self._numel = numel
        self._shape = shape
        self._stride = stride

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (
            _rebuild_tensor_v2,
            (
                _StorageRef(self._key, self._storage_name, self._numel),
                0,
                self._shape,
                self._stride,
                False,
                None,
            ),
        )


class _StorageRef:
    def __init__(self, key: str, storage_name: str, numel: int) -> None:
        self.key = key
        self.storage_name = storage_name
        self.numel = numel


class _TorchPickler(pickle.Pickler):
    def persistent_id(self, obj: object) -> tuple[object, ...] | None:
        if isinstance(obj, _StorageRef):
            return ("storage", obj.storage_name, obj.key, "cpu", obj.numel)
        return None


class _TorchUnpickler(pickle.Unpickler):
    def __init__(
        self,
        file: IO[bytes],
        archive: zipfile.ZipFile,
        prefix: str,
    ) -> None:
        super().__init__(file)
        self._archive = archive
        self._prefix = prefix
        self._storages: dict[str, np.ndarray] = {}

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch._utils":
            if name == "_rebuild_tensor_v2":
                return _rebuild_tensor_v2
            if name == "_rebuild_tensor":
                return _rebuild_tensor
            if name == "_rebuild_parameter":
                return _rebuild_parameter
        if module == "finchbert.infrastructure.pytorch_bin":
            return super().find_class(module, name)
        if module == "torch" and name == "Size":
            return tuple
        if module.startswith("torch") and name.endswith("Storage"):
            return type(name, (), {"name": name})
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module in {"builtins", "collections", "copyreg", "copy_reg", "_codecs"}:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to unpickle {module}.{name}")

    def persistent_load(self, saved_id: object) -> np.ndarray:
        if not isinstance(saved_id, tuple) or not saved_id:
            raise pickle.UnpicklingError("invalid torch persistent id.")
        typename = saved_id[0]
        if isinstance(typename, bytes):
            typename = typename.decode("ascii")
        if typename != "storage":
            raise pickle.UnpicklingError(
                f"unsupported torch persistent type: {typename}."
            )
        storage_type, key, _location, numel, *_rest = saved_id[1:]
        key_text = key.decode("ascii") if isinstance(key, bytes) else str(key)
        storage = self._storages.get(key_text)
        if storage is None:
            raw = self._archive.read(f"{self._prefix}data/{key_text}")
            storage = np.array(
                np.frombuffer(
                    raw,
                    dtype=_storage_dtype(storage_type),
                    count=int(numel),
                ),
                copy=True,
            )
            self._storages[key_text] = storage
        return storage
