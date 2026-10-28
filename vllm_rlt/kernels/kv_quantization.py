"""PyTorch reference for affine INT4 KV storage and cross-loop residuals.

Inputs are logical ``[token, KV head, channel]`` tensors for one request/layer.
K is grouped across stored token rows per channel; V across channels per token.
Adjacent channels occupy the low and high nibbles of a byte, respectively.

Adapted from FlashLoop packing.py and cache.py at aeaadee10d75b7aa8c4a42a809f15c75d6d3bdc6.
Copyright (c) 2026 FlashLoop contributors. See LICENSES/FlashLoop-MIT.txt.

These eager reference operations validate tensor contents and may synchronize
CUDA. They do not allocate engine KV pages or provide graph-safe packed readers.
"""

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import torch

KVKind = Literal["key", "value"]
_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def _validate_format(kind: KVKind, group_size: int) -> None:
    if kind not in {"key", "value"}:
        raise ValueError("kind must be 'key' or 'value'")
    if isinstance(group_size, bool) or not isinstance(group_size, Integral) or group_size <= 0:
        raise ValueError("group_size must be a positive integer")


def _require_finite(values: torch.Tensor, name: str) -> None:
    if not torch.isfinite(values).all().item():
        raise ValueError(f"{name} must contain only finite values")


def _validate_values(values: torch.Tensor) -> None:
    if values.ndim != 3 or values.shape[1] <= 0 or values.shape[2] <= 0:
        raise ValueError("KV values must have shape [tokens, positive heads, positive channels]")
    if values.dtype not in _DTYPES:
        raise ValueError("KV values must use float16, bfloat16, or float32")
    _require_finite(values, "KV values")


def _metadata_shape(shape: tuple[int, int, int], kind: KVKind, group_size: int):
    tokens, heads, channels = shape
    if kind == "key":
        return ((tokens + group_size - 1) // group_size, heads, channels)
    return (tokens, heads, (channels + group_size - 1) // group_size)


@dataclass(frozen=True)
class PackedInt4KV:
    """Encoded tensor and FP16 affine parameters; treat its tensors as immutable.

    ``payload`` is uint8 ``[T, H, ceil(D/2)]``. For K, parameters are
    ``[ceil(T/G), H, D]``; for V, ``[T, H, ceil(D/G)]``. An odd last channel
    leaves the high nibble zero. Constant groups use code zero and scale one.
    """

    payload: torch.Tensor
    minimum: torch.Tensor
    scale: torch.Tensor
    original_shape: tuple[int, int, int]
    original_dtype: torch.dtype
    kind: KVKind
    group_size: int

    def __post_init__(self) -> None:
        _validate_format(self.kind, self.group_size)
        shape = self.original_shape
        if (
            len(shape) != 3
            or any(isinstance(n, bool) or not isinstance(n, Integral) for n in shape)
            or shape[0] < 0
            or shape[1] <= 0
            or shape[2] <= 0
        ):
            raise ValueError("invalid original KV shape")
        if self.original_dtype not in _DTYPES:
            raise ValueError("invalid original KV dtype")
        if self.payload.dtype != torch.uint8:
            raise ValueError("INT4 payload must use uint8")
        if tuple(self.payload.shape) != (*shape[:2], (shape[2] + 1) // 2):
            raise ValueError("INT4 payload shape does not match original_shape")
        metadata_shape = _metadata_shape(shape, self.kind, self.group_size)
        for name, tensor in (("minimum", self.minimum), ("scale", self.scale)):
            if tensor.dtype != torch.float16 or tuple(tensor.shape) != metadata_shape:
                raise ValueError(f"{name} must use float16 with shape {metadata_shape}")
            if tensor.device != self.payload.device:
                raise ValueError("payload and parameters must be on the same device")
            _require_finite(tensor, name)
        if (self.scale < 0).any().item():
            raise ValueError("scale must be nonnegative")

    @property
    def storage_bytes(self) -> int:
        """Encoded tensor bytes, including parameters and nibble padding.

        This excludes Python objects, allocator reservations and decode scratch.
        """
        return sum(t.numel() * t.element_size() for t in (self.payload, self.minimum, self.scale))

    @torch.no_grad()
    def decode(self) -> torch.Tensor:
        """Reconstruct with stored parameters in FP32, then cast to compute dtype."""
        channels = self.original_shape[2]
        codes = torch.empty(self.original_shape, dtype=torch.uint8, device=self.payload.device)
        codes[..., 0::2] = self.payload & 0x0F
        codes[..., 1::2] = (self.payload >> 4)[..., : channels // 2]
        axis = 0 if self.kind == "key" else 2
        length = self.original_shape[axis]
        minimum = self.minimum.repeat_interleave(self.group_size, dim=axis).narrow(axis, 0, length)
        scale = self.scale.repeat_interleave(self.group_size, dim=axis).narrow(axis, 0, length)
        output = (codes.float() * scale.float() + minimum.float()).to(self.original_dtype)
        _require_finite(output, "decoded KV")
        return output


@torch.no_grad()
def pack_kv(values: torch.Tensor, *, kind: KVKind, group_size: int = 64) -> PackedInt4KV:
    """Encode logical KV, retaining upstream FP32-code/FP16-parameter ordering.

    Codes are rounded to nearest, ties to even, using FP32 parameters before
    storing those parameters as FP16. Metadata overflow is rejected; FP16 scale
    underflow to zero is allowed and contributes to reconstruction error.
    ``decode`` additionally rejects overflow when casting back to compute dtype.
    """
    _validate_format(kind, group_size)
    _validate_values(values)
    shape = tuple(values.shape)
    work = values.float()
    axis = 0 if kind == "key" else 2
    codes = torch.empty_like(values, dtype=torch.uint8)
    minima, scales = [], []
    for start in range(0, shape[axis], group_size):
        length = min(group_size, shape[axis] - start)
        group = work.narrow(axis, start, length)
        minimum = group.amin(dim=axis, keepdim=True)
        span = group.amax(dim=axis, keepdim=True) - minimum
        scale = torch.where(span > 0, span / 15.0, torch.ones_like(span))
        codes.narrow(axis, start, length).copy_(
            torch.round((group - minimum) / scale).clamp(0, 15).to(torch.uint8)
        )
        minima.append(minimum)
        scales.append(scale)
    metadata_shape = _metadata_shape(shape, kind, group_size)
    minimum = (
        torch.cat(minima, dim=axis).to(torch.float16)
        if minima
        else torch.empty(metadata_shape, device=values.device, dtype=torch.float16)
    )
    scale = torch.cat(scales, dim=axis).to(torch.float16) if scales else torch.empty_like(minimum)
    payload = codes[..., 0::2].clone(memory_format=torch.contiguous_format)
    payload[..., : shape[2] // 2] |= codes[..., 1::2] << 4
    return PackedInt4KV(
        payload,
        minimum.contiguous(),
        scale.contiguous(),
        shape,
        values.dtype,
        kind,
        group_size,
    )


def _validate_positions(positions: torch.Tensor, rows: int, tokens: int, device) -> None:
    if positions.ndim != 1 or positions.dtype not in {torch.int32, torch.int64}:
        raise ValueError("positions must be a one-dimensional int32 or int64 tensor")
    if positions.device != device:
        raise ValueError("positions and KV must be on the same device")
    if positions.numel() != rows:
        raise ValueError("positions must identify every compact residual row")
    if rows and (
        (positions < 0).any().item()
        or (positions >= tokens).any().item()
        or (positions[1:] <= positions[:-1]).any().item()
    ):
        raise ValueError("positions must be strictly increasing and within the logical token range")


@dataclass(frozen=True)
class Int4Residual:
    """One depth's compact residual and its logical scatter map.

    ``positions=None`` denotes a dense update; an empty map denotes inheritance
    at every position. Positions affect scatter only: K groups follow compact
    row order, as in the pinned upstream cache. The caller owns the choice of
    encoding region and must supply its matching reconstructed predecessor.
    """

    data: PackedInt4KV
    positions: torch.Tensor | None
    num_tokens: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_tokens, bool)
            or not isinstance(self.num_tokens, Integral)
            or self.num_tokens < 0
        ):
            raise ValueError("num_tokens must be a nonnegative integer")
        rows = self.data.original_shape[0]
        if self.positions is None:
            if rows != self.num_tokens:
                raise ValueError("dense residual must cover every logical token")
        else:
            _validate_positions(self.positions, rows, self.num_tokens, self.data.payload.device)

    @property
    def storage_bytes(self) -> int:
        mapping_bytes = (
            0 if self.positions is None else self.positions.numel() * self.positions.element_size()
        )
        return self.data.storage_bytes + mapping_bytes

    @torch.no_grad()
    def reconstruct(self, previous: torch.Tensor) -> torch.Tensor:
        """Return a new full tensor; absent rows retain their previous values."""
        _validate_values(previous)
        expected_shape = (self.num_tokens, *self.data.original_shape[1:])
        if tuple(previous.shape) != expected_shape:
            raise ValueError("predecessor shape does not match residual logical extent")
        if (
            previous.dtype != self.data.original_dtype
            or previous.device != self.data.payload.device
        ):
            raise ValueError("predecessor dtype and device must match residual")
        delta = self.data.decode()
        if self.positions is None:
            result = previous + delta
        else:
            result = previous.clone()
            indices = self.positions.long()
            result.index_copy_(0, indices, previous.index_select(0, indices) + delta)
        _require_finite(result, "reconstructed KV")
        return result


@torch.no_grad()
def encode_residual(
    current: torch.Tensor,
    reconstructed_previous: torch.Tensor,
    *,
    kind: KVKind,
    positions: torch.Tensor | None = None,
    group_size: int = 64,
) -> Int4Residual:
    """Pack ``current - reconstructed_previous`` in the input compute dtype.

    With explicit positions, current contains only the compact active rows and
    the predecessor contains the full logical sequence. The returned map owns a
    copy. Inactive positions have no residual payload, not a zero INT4 code.
    Each BF16/FP16 subtraction and reconstruction addition rounds in that dtype,
    matching the pinned FlashLoop reference's per-loop arithmetic.
    """
    _validate_format(kind, group_size)
    _validate_values(current)
    _validate_values(reconstructed_previous)
    if (
        current.dtype != reconstructed_previous.dtype
        or current.device != reconstructed_previous.device
    ):
        raise ValueError("current and predecessor must have matching dtype and device")
    if current.shape[1:] != reconstructed_previous.shape[1:]:
        raise ValueError("current and predecessor must have matching heads and channels")
    tokens = reconstructed_previous.shape[0]
    if positions is None:
        if current.shape != reconstructed_previous.shape:
            raise ValueError("dense residual requires matching current and predecessor shapes")
        predecessor = reconstructed_previous
    else:
        _validate_positions(positions, current.shape[0], tokens, current.device)
        positions = positions.clone(memory_format=torch.contiguous_format)
        predecessor = reconstructed_previous.index_select(0, positions.long())
    data = pack_kv(current - predecessor, kind=kind, group_size=group_size)
    return Int4Residual(data, positions, tokens)
