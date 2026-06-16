# triton.language.clamp

## 1. Function Overview

Description: Clamps the tensor `x` to the range [min, max].

```python
triton.language.clamp(x, min, max, propagate_nan: constexpr = PropagateNan.NONE, _semantic=None)
```

## 2. Specification

### 2.1 Parameter Description

| Parameter Name | Type               | Description                                                      |
| -------------- | ------------------ | ---------------------------------------------------------------- |
| `x`            | `tensor`           | Input tensor data                                                |
| `min`          | `tensor`           | Lower bound (can be a tensor or scalar, broadcast to `x`'s shape) |
| `max`          | `tensor`           | Upper bound (can be a tensor or scalar, broadcast to `x`'s shape) |
| `propagate_nan`| `triton.language.core.constexpr` | Whether to propagate NaN from min or max                         |
| `_semantic`    | -                  | Reserved parameter, external calls not supported                 |

Return value:
`x`: Output tensor with the same shape as the input tensor `x`

### 2.2 OP Specification

#### 2.2.1 DataType Support

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | ×    | ×     | ×     | ×     | ×      | ×      | ×      | ×     | √    | √    | √    | √    | ×    |
| Ascend A2/A3 | ×    | ×     | ×     | ×     | ×      | ×      | ×      | ×     | √    | √    | ×    | √    | ×    |

#### 2.2.2 Shape Support

|        | Supported Dimension Range |
| ------ | ------------------------- |
| GPU    | Only supports 1~5D tensors |
| Ascend | Only supports 1~5D tensors |

Conclusion: In terms of shape, there is no difference between GPU and Ascend platforms; both support 1 to 5-dimensional tensors.

### 2.3 Special Limitations

> Missing capabilities compared to the community that cannot be implemented

Ascend lacks fp64 support compared to GPU.

### 2.4 Usage Example

The following example demonstrates clamping the input tensor `x`:

```python
@triton.jit
def tt_clamp_2d(in_ptr, out_ptr, min_ptr, max_ptr,
                   xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
                   XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
       xoffs = tl.program_id(0) * XB
       yoffs = tl.program_id(1) * YB
       xidx = tl.arange(0, XB) + xoffs
       yidx = tl.arange(0, YB) + yoffs
       idx = xidx[:, None] * ynumel + yidx[None, :]

       x = tl.load(in_ptr + idx)
       min_ = tl.load(min_ptr + idx)
       max_ = tl.load(max_ptr + idx)
       ret = tl.clamp(x, min_, max_)

       tl.store(out_ptr + idx, ret)
```