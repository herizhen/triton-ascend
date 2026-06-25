# triton.language.histogram

## 1. OP Overview

Description: Computes a histogram with `num_bins` bins based on the input, each bin has a width of 1, starting from 0.
Prototype:

```python
triton.language.histogram(
 input,
 num_bins,
 mask=None,
 _semantic=None,
 _generator=None
)
```

Can be called as a member function of a tensor, e.g., `x.histogram(...)`, which is equivalent to `histogram(x, ...)`.

## 2. OP Specification

### 2.1 Parameter Description

| Parameter Name | Type               | Description                                                        |
| -------------- | ------------------ | ------------------------------------------------------------------ |
| `input`        | `tensor`           | Input data, containing all numerical points for distribution statistics |
| `num_bins`     | `int`              | Defines how many equal-width intervals the entire data range is divided into |
| `mask`         | `int1` or `tensor<int1>`, optional | Specifies the data range to prevent out-of-bounds access |
| `_semantic`    | -                  | Reserved parameter, external calls not supported yet               |
| `_generator`   | -                  | Reserved parameter, external calls not supported yet               |

Return value:
Histogram represented as a tensor
Note: The current triton 3.2 version does not support `mask` yet; support will be added in future versions. The `input` range is limited to `[0, num_bins-1]`; full range support will be added in future versions.

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | ×    | ×     | √     | ×     | ×      | √      | ×      | ×     | ×    | ×    | ×    | ×    | ×    |
| Ascend A2/A3 | ×    | ×     | √    | ×     | ×      | √      | √      | √     | ×    | ×    | ×    | ×    | ×    |

#### 2.2.2 Shape Support

Currently only supports 1D

### 2.3 Special Limitations

> Capabilities missing compared to the community and cannot be implemented

### 2.4 Usage Example

The following example demonstrates the invocation of `histogram`:

```python
@triton.jit
def histogram_kernel(x_ptr, z_ptr, M: tl.constexpr, N: tl.constexpr):
    offset1 = tl.arange(0, M)
    offset2 = tl.arange(0, N)
    x = tl.load(x_ptr + offset1)
    z = tl.histogram(x, N)
    tl.store(z_ptr + offset2, z)

x = torch.randint(0, N, (M, ), device=device, dtype=torch.int32)
z = torch.empty(N, dtype=torch.int32, device=device)
histogram_kernel[(1, )](x, z, M=M, N=N)
```

## 3. Semantic GAP

> Capabilities missing compared to the community but can be developed and supported