# NPU High-Performance Programming Guide

## Merging Grid Kernels

### I. Principles for Automatic Grid Kernel Merging Optimization

In some scenarios, Triton operators are migrated from GPU to NPU. Due to architectural differences, Triton operators developed for GPU may have a large number of grid kernels. When executed on NPU, they cannot all be scheduled at once, and multiple rounds of dispatch lead to excessive dispatch latency, affecting operator performance. During the optimization of Triton operators for NPU, the number of grid kernels should be checked first. When the number of kernels is large, use the `TRITON_ALL_BLOCKS_PARALLEL` environment variable to improve operator execution performance.

## Instruction-Level Parallelism Optimization

### I. Core Principles of Instruction-Level Parallelism Optimization

When Triton operators execute on NPU, to improve performance, the NPU provides underlying parallel mechanisms such as multi-buffering and instruction-level parallelism, parallelizing "data load / data compute / data store" to enhance performance. However, in some scenarios, multi-buffering cannot be enabled, affecting parallelism and reducing operator execution performance. During performance optimization, if such issues exist, refer to the following points for troubleshooting and optimize according to the code examples:  
1. Data movement and computation have data dependencies, causing synchronization. The MTE movement can only be triggered after Vector operations are completed, resulting in low parallelism.  
2. Within the operator, there are no multiple data loads, or a single execution completes without Tiling. In such scenarios, multi-buffering cannot be enabled.  
3. Multi-buffering requires additional UB space. If UB space is insufficient during computation, multi-buffering cannot be enabled.

### II. Code Examples

- Example 1: Reduce Synchronization to Improve Parallelism

    During operator tuning, increasing instruction-level parallelism is an important method. In the following `tl.load` statement, when `N > M`, the loaded data can only fill part of the tensor memory space pointed to by `data`. For the unfilled portion, if the user does not specify an `other` value, GPU defaults to filling with 0. To reduce adaptation work for user migration, NPU maintains behavior consistent with GPU. NPU first uses Vector cores to set the entire memory space pointed to by `data` to the specified value (if the user does not specify `other`, it is also set to 0), and then uses MTE2 instructions to move data to the partial memory space pointed to by `data`. This creates a dependency between MTE2 and Vector, preventing efficient parallelism and affecting performance:

    ```diff
    @triton.jit
    def npu_vector_add_kernel(
        input,                          # [Tensor] input tensor (1 x col)
        output,                         # [Tensor] output tensor (1 x col)
        M: tl.constexpr,                # len of the vector
        BLOCK_SIZE: tl.constexpr
    ):
        N :tl.constexpr = BLOCK_SIZE
        idx = tl.arange(0, N)
        mask = idx < M
        data = tl.load(input + idx, mask = mask) # or specify other=-1, etc.
    ```

    To improve performance, when the loaded data only partially fills the pointed memory space, and the unfilled portion does not affect subsequent computation results, add `care_padding=False` in the `load` statement to remove the default padding, increasing parallelism and improving performance. The optimized version of the above operator is as follows:

    ```diff
    @triton.jit
    def npu_vector_add_kernel(
        input,                          # [Tensor] input tensor (1 x col)
        output,                         # [Tensor] output tensor (1 x col)
        M: tl.constexpr,                # len of the vector
        BLOCK_SIZE: tl.constexpr
    ):
        idx = tl.arange(0, N)
        mask = idx < M
    -   data = tl.load(input + idx, mask = mask) # or specify other=-1, etc.
    +   data = tl.load(input + idx, mask = mask, care_padding=False) # or specify other=-1, etc.
    ```

- Example 2: Use `for` Loops in Triton Operators to Add Tiling and Improve Parallelism

    In Triton operator programming, `mask` operations are often used in syntaxes like `load`, `store`, `where`, etc. During performance optimization, special attention should be paid to performance degradation caused by such operations. When the logic within a Triton operator is a single sequential execution (start -> data load -> compute -> data store -> end), instructions cannot be parallelized, resulting in low execution efficiency. By using `for` loops to add Tiling in the operator, the single processing amount is reduced, and multiple processing allows "data load / compute / data store" to be parallelized, reducing serial waiting time and improving overall performance. Additionally, using `for` loops to add Tiling also reduces the UB space consumed per single processing.  
    Note: Adding data Tiling also requires considering whether the mathematics after changing the data block is equivalent.

    ```diff
    @triton.jit
    def alloc_extend_kernel(
            pre_lens_ptr,
            seq_lens_ptr,
            free_page_ptr,
            out_indices,
            bs_upper: tl.constexpr,
            page_size: tl.constexpr,
            max_num_extend_tokens: tl.constexpr,
    +       BLOCK_SIZE: tl.constexpr = 1024,
    ):
        pid = tl.program_id(0)

        load_offset = tl.arange(0, bs_upper)
        seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
        pre_lens = tl.load(pre_lens_ptr + load_offset, mask=load_offset <= pid)
        extend_lens = seq_lens - pre_lens

        seq_len = tl.load(seq_lens_ptr + pid)
        pre_len = tl.load(pre_lens_ptr + pid)
        extend_len = seq_len - pre_len

        sum_extend_lens = tl.sum(extend_lens)
        output_start_loc = sum_extend_lens - extend_len

        num_pages_after = (seq_lens + page_size - 1) // page_size
        num_pages_before = (pre_lens + page_size - 1) // page_size
        num_new_pages = num_pages_after - num_pages_before

        num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
                pre_len + page_size - 1
        ) // page_size
        sum_num_new_pages = tl.sum(num_new_pages)
        new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

        # Part 2: fill the new full pages
        num_part2 = (
                seq_len // page_size * page_size
                - (pre_len + page_size - 1) // page_size * page_size
        )

    -   # load data at once
    -   offset_many_page = tl.arange(0, max_num_extend_tokens)
    -   page_start = tl.load(
    -       free_page_ptr + new_page_start_loc + offset_many_page // page_size,
    -       mask=offset_many_page < num_part2,
    -   )
    -   tl.store(
    -       out_indices + output_start_loc + offset_many_page,
    -       page_start * page_size + offset_many_page % page_size,
    -       mask=offset_many_page < num_part2,
    -   )

    +   # load data using loop
    +   num_loop = tl.cdiv(max_num_extend_tokens, BLOCK_SIZE)
    +   blk_offset = tl.arange(0, BLOCK_SIZE)
    +   for i in range(num_loop):
    +       offset_many_page = blk_offset + i * BLOCK_SIZE
    +       page_start = tl.load(
    +           free_page_ptr + new_page_start_loc + offset_many_page // page_size,
    +           mask=offset_many_page < num_part2,
    +       )
    +       tl.store(
    +           out_indices + output_start_loc + offset_many_page,
    +           page_start * page_size + offset_many_page % page_size,
    +           mask=offset_many_page < num_part2,
    +       )
    ```

## Data Type Optimization

### I. Core Principles of Data Type Optimization

Some operation types of the A2/A3 vector compute units do not support certain data types. In such scenarios, the corresponding vector operations degrade to scalar operations, affecting performance. If the overall operator precision is not affected, it is recommended to use supported data types to improve performance.  
The main operations involved are as follows:

| **OP Name** | **Unsupported Data Types** |
|---|---|
| Vector ADD | int64 |
| Vector CMP | int64/int32 |

### II. Code Examples

- Example Code for Vector Add Triton Operator

    In the following Triton operator, when the data types of the `x` and `y` input tensors are `int64`, the `x1 + y1` operation will be expanded into scalar operations, reducing performance. If precision is not affected, it is recommended to use the `int32` data type.

    ``` diff
    @triton.jit
    def npu_vector_add_kernel(
        x,                          # [Tensor] input tensor (1 x col)
        y,                          # [Tensor] input tensor (1 x col)
        z,                          # [Tensor] output tensor (1 x col)
        vector_len: tl.constexpr,   # len of the vector
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        offset = pid * BLOCK_SIZE + tl.arange(BLOCK_SIZE)
        len_mask = offset < vector_len
        x1 = tl.load(x + offset, mask=len_mask)
        y1 = tl.load(y + offset, mask=len_mask)
        z1 = x1 + y1
        tl.store(z + offset, z1, mask=len_mask)
    ```

- Example Code for Vector Cmp Triton Operator

    In the following Triton operator, when performing `mask` operations, a `Cmp` operation is used. `Cmp` does not support `int64`/`int32` data types, causing the `cols < N` operation to be expanded into scalar operations, reducing performance. If precision is not affected, it is recommended to use the `fp32` data type.  
    In Triton operator programming, `mask` operations are often used in syntaxes like `load`, `store`, `where`, etc. During performance optimization, special attention should be paid to performance degradation caused by such operations.

    ``` diff
    @triton.jit
    def npu_vector_cmp_kernel(
        X,                 # [Tensor] input tensor (row x col)
        Out,               # [Tensor] output tensor (row x col)
        Mean,              # [Vector] mean tensor (row, ) of X
        Rstd,              # [Vector] std tensor (row, ) of X
        stride_x_row,      # [Scalar] stride of row of x
        stride_out_row,    # [Scalar] stride of row of out, normally equals to stride_x_row
        M,                 # [Scalar] row number
        N,                 # [Scalar] col number
        eps,               # [Scalar] epsilon to avoid division by zeros
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr
    ):
        group_m = tl.program_id(0)
        group_n = tl.program_id(1)
        row = group_m

        # calculate index & offset
        Mean = Mean + group_n * M
        Rstd = Rstd + group_n * M
        X = X + row * stride_x_row + group_n * N
        Out = Out + row * stride_out_row + group_n * N

        cols = tl.arange(0, BLOCK_N)  # cols is int64
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)

        # calculate mean & rstd
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        # [Changed begin]
    -   xbar = tl.where(cols < N, X - mean, 0.0)
    +   cols_cmp = cols.to(tl.float32)
    +   xbar = tl.where(cols_cmp < N, x - mean, 0.0)
        # [Changed end]

        var = tl.sum(xbar * xbar, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        tl.store(Rstd + row, rstd)

        # calculate Out
        mask = cols < N
        out = (x - mean) * rstd
        tl.store(Out + cols, out, mask=mask)
    ```