# triton.language.extra.ascend.libdevice.index_select_simd

## 1 Function Description

Performs parallel gather of multiple indices on non-tail dimensions and copies data zero-copy from global memory (GM) directly to the correct positions in the unified buffer (UB) in tile units. This operation is equivalent to a high-performance implementation of `torch.index_select`, suitable for scenarios such as embedding layer lookup and sparse index access.

**Syntax:**

- `triton.language.extra.ascend.libdevice.index_select_simd(src, dim, index, src_shape, src_offset, read_shape)`

**Functionality:**

- Reads data in batches from the source tensor along the specified dimension based on the index array
- Supports specifying the offset and size of the read region for flexible slicing
- Zero-copy efficient implementation, directly moving data from GM to UB
- Preserves element type and encoding unchanged

**Typical Use Cases:**

- Embedding lookup: Batch reading word vectors from a large vocabulary based on token IDs
- Sparse tensor operations: Accessing specific rows of a dense tensor based on sparse indices
- Dynamic routing and attention mechanisms: Selecting specific features based on dynamically computed indices

## 2 Parameter Specifications

### 2.1 Parameter Description

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| src | tensor/pointer | Yes | Source tensor pointer, data located in global memory (GM) |
| dim | int | Yes | Dimension along which to perform index_select, range [0, len(src_shape)-2], **does not support the tail axis** (last dimension) |
| index | tensor | Yes | 1D index array, located in UB, specifies the index positions to read |
| src_shape | Tuple[int] | Yes | Full shape of the source tensor |
| src_offset | Tuple[int] | Yes | Starting position for reading; can be set to -1 for the dim dimension (determined by index) |
| read_shape | Tuple[int] | Yes | Size of the data to read; must be -1 for the dim dimension (determined by index length) |

**Return Value:**

- **Type:** tensor (located in UB)
- **Shape:** Consistent with read_shape, where the size of the dim dimension equals the length of index
- **Data Type:** Same as the source tensor
- **Memory Location:** Unified Buffer (UB)

**Constraints:**

- `read_shape[dim]` must be -1
- `src_offset[dim]` can be set to -1 (ignored, as this dimension is determined by index)
- `len(src_shape) == len(src_offset) == len(read_shape)`
- `index` must be a 1D tensor
- `dim` cannot be the tail axis (last dimension), i.e., `dim < len(src_shape) - 1`
- For non-dim dimensions: `0 <= src_offset[i] < src_shape[i]`
- For non-dim dimensions: `src_offset[i] + read_shape[i] <= src_shape[i]` (automatically truncated if out of bounds)
- Index values in index must be within the range `[0, src_shape[dim])`

### 2.2 DataType Support Table

| Support | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float16 | float32 | bfloat16 | float8e4 | float8e5 | float64 | bool |
|---------|:----:|:-----:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:------:|:------:|:-------:|:--------:|:--------:|:------:|:----:|
| Ascend A2/A3 | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ | ✓ | × | × | × | ✓ |
| GPU Support | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

**Notes:**

- The data type of index must be int32 or int64
- This operation is not supported on GPU platforms (Ascend-specific intrinsic)

### 2.3 Shape Support Table

Supports any number of dimensions (1D to high-dimensional tensors), subject to the following conditions:

- index must be a 1D tensor
- The size of each dimension of the source tensor must comply with actual hardware memory limits
- The size of read_shape in non-dim dimensions must consider UB space limitations

**Common Shape Combinations:**

- 2D tensor: Suitable for embedding layer lookup, sparse matrix row selection
- 3D tensor: Suitable for batch embedding lookup, sequence feature extraction
- High-dimensional tensor: Suitable for complex multi-dimensional indexing operations

### 2.4 Special Constraints

1. **dim Constraint:** index_select is not supported on the tail axis (last dimension); dim must satisfy `dim < len(src_shape) - 1`
2. **Data Type Constraint:** uint16/uint32/uint64/float8/float64 data types are currently not supported
3. **Index Out of Bounds:** Out-of-bounds indices in index are not checked; users must ensure index validity

### 2.5 Usage

**Basic Usage (2D Embedding Lookup):**

```python
import triton
import triton.language as tl
import triton.language.extra.ascend.libdevice as libdevice

@triton.jit
def embedding_kernel(
    embed_ptr,      # [vocab_size, embed_dim]
    indices_ptr,    # [batch_size]
    output_ptr,     # [batch_size, embed_dim]
    vocab_size: tl.constexpr,
    embed_dim: tl.constexpr,
):
    pid = tl.program_id(0)

    # Load indices
    indices = tl.load(indices_ptr + pid * 16 + tl.arange(0, 16))

    # Use index_select to batch read embedding vectors
    embeddings = libdevice.index_select_simd(
        src=embed_ptr,
        dim=0,
        index=indices,
        src_shape=(vocab_size, embed_dim),
        src_offset=(-1, 0),
        read_shape=(-1, embed_dim)
    )

    # Store results
    offsets = tl.arange(0, 16)[:, None] * embed_dim + tl.arange(0, embed_dim)[None, :]
    tl.store(output_ptr + pid * 16 * embed_dim + offsets, embeddings)
```

**Relationship with torch.index_select:**

- `index_select_simd` is equivalent to `torch.index_select(src, dim, index)` combined with a slicing operation
- However, index_select_simd is implemented at the hardware level, offering better performance than the PyTorch implementation (approximately 0.6~1.5x AscendC performance)

**Difference from Regular Load:**

```python
## Regular load method (inefficient)
for i in range(len(indices)):
    idx = tl.load(indices_ptr + i)
    offsets = idx * stride + tl.arange(0, size)
    data = tl.load(src_ptr + offsets)
    # ... process data

## index_select method (efficient)
indices = tl.load(indices_ptr + tl.arange(0, len(indices)))
data = libdevice.index_select_simd(
    src=src_ptr,
    dim=0,
    index=indices,
    src_shape=(...),
    src_offset=(-1, 0),
    read_shape=(-1, size)
)
## Retrieve all data at once
```

## 3 Differences from GPU

New OP, no differences

## 4 Test Case Description

**Test File:**

- `ascend/examples/pytest_ut/test_index_select.py` - 2D tensor index_select test (multiple shape combinations)