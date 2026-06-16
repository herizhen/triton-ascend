# triton.language.device_print

## 1. Function Overview

`device_print` is used to print information from the device side during NPU runtime. Unlike `static_print`, this outputs information in real-time during kernel execution. The first parameter must be a `string`, and subsequent parameters must be `scalars` or `tensors`. **To use `device_print`, the environment variable `TRITON_DEVICE_PRINT` must be set to `True`.**

```python
triton.language.device_print(prefix, *args, hex=False, _semantic=None)
```

## 2. Specifications

### 2.1 Parameter Description

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prefix` | `str` | Required | Prefix string before the printed values |
| `args` | `tensor`/`scalar` | Required | Values to print, can be any tensor or scalar |
| `hex` | `bool` | `False` | Whether to print all values in hexadecimal format |
| `_semantic` | - | - | Reserved parameter, currently not supported for external calls |

### 2.2.1 Data Type Support

A3:

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ascend A2/A3 | ✓ | ✓ | ✓ | × | × | × | × | ✓ | ✓ | ✓ | × | ✓ | ✓ |

### 2.2.2 Shape Support

|        | Supported Dimension Range |
| ------ | ------------------------- |
| GPU    | Only supports 1~5D tensors |
| Ascend | Only supports 1~5D tensors |

### 2.3 Special Limitations

> Missing capabilities compared to the community that cannot be implemented

Ascend lacks support for uint8, uint16, uint32, uint64, and fp64 compared to GPU (hardware limitation).

**DevicePrint Functional Limitations**

**Phenomenon Description:**
`device_print` can only print result values involved in computations, and cannot print offset variables used purely for memory access.

**Root Cause:**
During the memory access analysis and optimization phase, the compiler optimizes away offsets used solely for address calculation. These intermediate variables are not retained in the final execution code.

**Example Scenario:**

```python
@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               of_ptr,
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):

    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    tl.device_print("offsets:", offsets)// ❌ Cannot print, already optimized away
```

Additionally, in certain cases, `device_print` may expand some auxiliary DMA code, causing underlying errors. Related functionality is still being optimized.

### 2.4 Usage

**Note**: The `prefix` string prefix must be included when using `device_print`; otherwise, a compilation error will occur. Currently, printing only the `prefix` string alone is not supported.

```python
import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr):
    idx = tl.arange(0,3)
    idy = tl.arange(0,4)
    offset = idx[:,None] * 4 + idy[None,:]
    val = tl.load(x_ptr + offset)
    # Print the value of the 2D tensor val
    tl.device_print("val:",val)
```