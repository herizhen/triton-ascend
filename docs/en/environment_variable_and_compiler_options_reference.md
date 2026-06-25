# Environment Variables and Compilation Options

This document summarizes the behavior switches in Triton-Ascend that can be explicitly controlled by developers, including environment variables set before runtime and NPU compilation options passed via `triton.Config` or kernel launch meta-parameters during compilation.

## Environment Variables

### Environment Variable Usage Example

Environment variables must be set before running the Python program, for example:

```bash
export TRITON_DEBUG=1
python run_kernel.py
```

### Environment Variable Reference Table

The environment variable configuration is referenced in the table below:

| Category | Environment Variable | Default Value | Description | Configuration Notes | Change Notice |
|----------|---------------------|---------------|-------------|--------------------|---------------|
| **Debug & Logging** | TRITON_DEBUG | 0 or unset | Enables Triton's debug output functionality, printing detailed debug information at runtime. Useful for troubleshooting compilation or execution issues. When set to 1, Triton outputs more information about the compilation process, kernel generation, and execution. Some implementations may support finer-grained debug levels (e.g., 2, 3, etc.), depending on the Triton version and implementation. | 0: Disable DEBUG<br>1: Enable DEBUG | |
| **Debug & Logging** | MLIR_ENABLE_DUMP | 0 or unset | Dumps the IR of all kernels before each MLIR optimization pass. Use `MLIR_ENABLE_DUMP=kernelName` to dump the IR of a specific kernel only. | 0: Do not dump<br>1: Dump IR of all kernels<br>kernelName: Dump IR of a specific kernel | Triton cache may interfere with dumping. If `MLIR_ENABLE_DUMP=1` does not take effect, try clearing the Triton cache: `rm -r ~/.triton/cache` |
| **Debug & Logging** | LLVM_IR_ENABLE_DUMP | 0 or unset | Dumps the IR before each LLVM IR optimization pass. | 0: Do not dump<br>1: Dump IR | |
| **Debug & Logging** | TRITON_REPRODUCER_PATH | unset | Generates an MLIR reproducer file before each MLIR compilation stage. If a stage fails, `<reproducer_path>` will save the MLIR state before the failure. | <reproducer_path>: Save path | |
| **Debug & Logging** | TRITON_INTERPRET | 0 or unset | Uses the Triton interpreter instead of the GPU, supporting the insertion of Python breakpoints in kernel code. | 0: Breakpoints not supported<br>1: Breakpoints supported | |
| **Debug & Logging** | TRITON_ENABLE_LLVM_DEBUG | 0 or unset | Passes the `-debug` argument to LLVM, outputting a large amount of debug information. If the output is too verbose, use `TRITON_LLVM_DEBUG_ONLY` to limit the output scope. | 0: Do not pass<br>1: Pass | Another method to reduce output interference is to first run the program with `LLVM_IR_ENABLE_DUMP=1`, extract the intermediate representation (IR) before the target LLVM optimization pass, then run LLVM's `opt` tool separately, adding the `-debug-only=foo` argument via the command line to limit the debug scope. |
| **Debug & Logging** | TRITON_LLVM_DEBUG_ONLY | unset | Functions equivalently to LLVM's `-debug-only` command-line option. This parameter limits LLVM debug output to specific optimization passes or component names (defined via the `#define DEBUG_TYPE` macro in LLVM and Triton), effectively reducing redundant debug information output. Users can specify one or more comma-separated values, for example: `TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions"` or `TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions,regalloc"`. | Comma-separated values: Pass or component names | |
| **Debug & Logging** | USE_IR_LOC | 0 or unset | Controls whether location information (e.g., file name, line number) is included in the generated intermediate representation (IR). This information is helpful for debugging but may increase the size of the generated IR. When set to 1, the IR is re-parsed, mapping location information to line numbers of IR files with specific extensions (rather than Python source file line numbers). This establishes a direct mapping from IR to LLVM IR/PTX. When used with profiling tools, it enables fine-grained performance analysis of IR instructions. | 0: Do not include location info<br>1: Include location info | |
| **Debug & Logging** | TRITON_PRINT_AUTOTUNING | 0 or unset | After autotuning completes, outputs the best configuration and total time for each kernel. | 0: Do not output<br>1: Output | |
| **Debug & Logging** | MLIR_ENABLE_REMARK | 0 or unset | Enables the output of remark information during the MLIR compilation process, including performance warnings output as remarks. | 0: Disable<br>1: Enable | |
| **Debug & Logging** | TRITON_KERNEL_DUMP | 0 or unset | Enables or disables the dumping of Triton kernels. When enabled, Triton saves the generated kernel code (IR at each compilation stage and final PTX) to a specified directory. | 0: Disable<br>1: Enable | |
| **Debug & Logging** | TRITON_DUMP_DIR | Current working directory or unset | Specifies the directory for saving Triton kernel dump files. The directory where IR and PTX are saved when `TRITON_KERNEL_DUMP=1`. | "path": Save path | |
| **Debug & Logging** | TRITON_DEVICE_PRINT | 0 or unset | When set to `1` or `true` (`TRUE` will be converted to `true`), enables the `tl.device_print` functionality. Important note: This feature uses a GM buffer (whose pointer is passed to the kernel). | 0: Disable<br>1: Enable `tl.device_print` functionality | The GM buffer per thread is a maximum of 16KB; content exceeding this limit will be discarded. This value is currently fixed and will be adjustable via an environment variable in the future. |
| **Debug & Logging** | TRITON_MEMORY_DISPLAY | 0 or unset | Controls whether to generate a JSON file for memory usage. When `TRITON_MEMORY_DISPLAY=1`, saves the `memory_info_aic/aiv.json` file to the current directory. | 0: Disable<br>1: Enable | |
| **Compilation Control** | TRITON_ALWAYS_COMPILE | 0 or unset | Controls whether Triton forces recompilation of kernels on every run instead of using cached versions. By default, Triton caches previously compiled kernels (based on parameters and configuration) to improve performance. When set to 1, Triton ignores the cache and recompiles kernels every time, which is useful for debugging or testing new compiler features. | 0: Disable<br>1: Recompile all kernels on every run | |
| **Compilation Control** | DISABLE_LLVM_OPT | 0 or unset | When set to 1, disables optimization steps during LLVM compilation (LLVM optimizations for `make_llir` and `make_ptx`). When set to a string, it is parsed as a list of LLVM optimization flags to disable. For example, using `DISABLE_LLVM_OPT="disable-lsr"` disables loop strength reduction optimization (which can cause up to 10% performance fluctuation in some kernels with register pressure). | 0: LLVM optimizations are enabled<br>1: Disable optimization steps during LLVM compilation (LLVM optimizations for `make_llir` and `make_ptx`)<br><list>:"disable-lsr": Disable loop strength reduction</list> | |
| **Compilation Control** | MLIR_ENABLE_TIMING | 0 or unset | Enables or disables timing statistics during the MLIR compilation process. | 0: Disable<br>1: Enable | |
| **Compilation Control** | LLVM_ENABLE_TIMING | 0 or unset | Enables or disables timing statistics during the LLVM compilation process. | 0: Disable<br>1: Enable | |
| **Compilation Control** | TRITON_DEFAULT_FP_FUSION | 1 (enabled) | Controls whether floating-point operation fusion optimization is enabled by default, overriding the default floating-point fusion behavior (e.g., mul+add -> fma). | 0: Disable<br>1: Enable | |
| **Compilation Control** | TRITON_KERNEL_OVERRIDE | 0 or unset | Enables or disables the Triton kernel override functionality, allowing the default generated kernel code to be overridden with user-specified external files (IR/PTX, etc.) at the beginning of each compilation stage. | 0: Disable<br>1: Enable | |
| **Compilation Control** | TRITON_OVERRIDE_DIR | Current working directory or unset | Specifies the directory for finding Triton kernel override files. The directory from which IR/PTX files are loaded when `TRITON_KERNEL_OVERRIDE=1`. | "path": Save path | |
| **Compilation Control** | TRITON_ASCEND_COMPILE_SPEED_OPT | 0 or unset | Controls whether the JIT compiler skips subsequent compilation stages after a kernel compilation failure. Set to `1` to skip (default `0` continues trying). | 0: Continue trying<br>1: Skip | |
| **Compilation Control** | TRITON_COMPILE_ONLY | 0 or unset | Used with `remote_launch`, compiles only without running. | 0: Disable<br>1: Enable | |
| **Compilation Control** | TRITON_DISABLE_FFTS | 0 or unset | Whether to disable FFTS. | 0: Enable<br>1: Disable | |
| **Compilation Control** | TRITON_DISABLE_PRECOMPILE | 0 or unset | Whether to disable precompilation. | 0: Enable precompilation<br>1: Disable precompilation | |
| **Runtime & Scheduling** | TRITON_ALL_BLOCKS_PARALLEL | 0 or unset | Enables or disables automatically optimizing the number of logical cores based on the number of physical cores. Only effective when logical cores can run in parallel. When the number of logical cores is greater than the number of physical cores, enabling this optimization causes the compiler to automatically adjust the number of logical cores to the number of physical cores, reducing scheduling overhead. Enabling allows grid > 65535. Limitation: The logic of the Triton kernel must be insensitive to execution order to enable this option; otherwise, it may cause deadlocks. | 0: Disable<br>1: Enable | |
| **Runtime & Scheduling** | TRITON_ENABLE_TASKQUEUE | 0 or unset | Whether to enable task_queue. | 0: Disable<br>1: Enable | |
| **Runtime & Scheduling** | TRITON_ENABLE_SANITIZER | 0 or unset | Whether to enable the SANITIZER. | 0: Disable<br>1: Enable | |
| **Runtime & Scheduling** | ENABLE_PRINT_UB_BITS | 0 or unset | When enabled, retrieves the current UB usage for use by the inductor. | 0: Disable<br>1: Enable | |
| **Other** | TRITON_BENCH_METHOD | unset | When using Ascend NPU, switches `do_bench` in `testing.py` to `do_bench_npu` (requires `INDUCTOR_ASCEND_AGGRESSIVE_AUTOTUNE = 1`). When set to `default`, the original `do_bench` function is called even if an NPU is available. | "npu": Switch to `do_bench_npu` | |
| **Other** | TRITON_REMOTE_RUN_CONFIG_PATH | path | Specifies the configuration path for remote execution. | Directly provide the path | |

## Compilation Options

Compilation options control the compilation strategy for a single Triton kernel and can be passed via `triton.Config`, Autotune parameters, or kernel launch meta-parameters.

### Compilation Option Usage Example

For example, `multibuffer` can be passed directly during kernel launch:

```python
import triton
import triton.language as tl

@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr):
    ...

grid = (triton.cdiv(n_elements, 1024),)
kernel[grid](..., BLOCK_SIZE=1024, multibuffer=True)
```

### Compilation Option Reference Table

The compilation option configuration is referenced in the table below:

| Category | Compilation Option | Default/Allowed Values | Description | Configuration Method |
|----------|--------------------|------------------------|-------------|----------------------|
| **General Pipeline** | `multibuffer` | `True` (default), `False` | Enables or disables ping-pong/double buffer pipeline. Enabled by default. | `triton.Config` or launch meta-parameter |
| **CV Fusion** | `enable_auto_bind_sub_block` | `None`, `True`, `False` | Enables or disables automatic sub-block binding. | `triton.Config` or launch meta-parameter |
| **CV Fusion** | `enable_hivm_auto_cv_balance` | `None`, `True`, `False` | Enables or disables automatic CV balance. | `triton.Config` or Autotune parameter |
| **CV Fusion/Synchronization** | `sync_solver` | `None`, `True`, `False` | Enables or disables the HIVM synchronization solver. | `triton.Config` or launch meta-parameter |
| **Synchronization** | `unit_flag` | `None`, `True`, `False` | Cube move-out related synchronization optimization item. | `triton.Config` or Autotune parameter |
| **Synchronization** | `inject_barrier_all` | `None`, `True`, `False` | Enables or disables automatic barrier synchronization injection. | `triton.Config` or launch meta-parameter |
| **Synchronization** | `inject_block_all` | `None`, `True`, `False` | Enables or disables automatic block synchronization injection. | `triton.Config` or launch meta-parameter |
| **Multi-Buffer Scope** | `limit_auto_multi_buffer_only_for_local_buffer` | `None`, `True`, `False` | Limits automatic multi-buffering to only local buffers. | `triton.Config` or Autotune parameter |
| **Multi-Buffer Scope** | `limit_auto_multi_buffer_of_local_buffer` | `None`, `"no-limit"`, `"no-l0c"` | Configures the scope of automatic multi-buffering for local buffers. | `triton.Config` or Autotune parameter |
| **Workspace** | `set_workspace_multibuffer` | `None`, `2`, `4` | Configures the workspace multi-buffer level. | `triton.Config` or Autotune parameter |
| **CV Fusion Tiling** | `tile_mix_vector_loop` | `None`, `2`, `4`, `8` | Configures the number of partitions for the Vector loop. | `triton.Config` or Autotune parameter |
| **CV Fusion Tiling** | `tile_mix_cube_loop` | `None`, `2`, `4`, `8` | Configures the number of partitions for the Cube loop. | `triton.Config` or Autotune parameter |
| **CV Fusion/Synchronization** | `disable_auto_inject_block_sync` | `None`, `True`, `False` | Enables or disables automatic block sync injection. | `triton.Config` or launch meta-parameter |
| **Execution Stream** | `stream` | `None` or NPU stream identifier | Specifies the NPU stream. | launch meta-parameter |
| **Compilation Pass** | `enable_linearize` | Version-dependent | Enables or disables the linearization pass. | `triton.Config` or launch meta-parameter |
| **CV Fusion/Layout** | `enable_nd2nz_on_vector` | Default `False` | Enables or disables ND to NZ layout conversion on the Vector path. | `triton.Config` or launch meta-parameter |
| **Large Grid Optimization** | `auto_blockify_size` | Default `1` | Enables or disables the AutoBlockify pass. Ignored when `TRITON_ALL_BLOCKS_PARALLEL` is not set. | launch meta-parameter or `triton.Config` |