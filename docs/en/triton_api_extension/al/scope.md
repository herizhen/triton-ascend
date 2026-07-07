# al.scope Interface Documentation

## 1. Hardware Background

Ascend processors contain multiple types of compute units (e.g., Cube Unit for matrix operations and Vector Unit for vector/scalar operations). al.scope allows kernel developers to explicitly tell the Triton compiler which hardware unit a specific code region should target, enabling finer-grained performance tuning and resource utilization.

## 2. Interface Description

<table>
  <tr>
    <td>Python<br>with al.scope(core_mode: str):<br>    # Triton statements within this code block (e.g., tl.load, tl.store, arithmetic operations, etc.)<br>    # will be compiled and executed according to the specified core_mode.<br>    ...</td>
  </tr>
</table>

al.scope is a context manager in the `triton.language.extra.ascend` module, specifically designed to specify the execution mode of Ascend hardware for code blocks within Triton kernels.

### Parameters

<table>
  <tr>
    <td>Parameter Name</td>
    <td>Type</td>
    <td>Required</td>
    <td>Description</td>
    <td>Example Values</td>
  </tr>
  <tr>
    <td>core_mode</td>
    <td>str</td>
    <td>Yes</td>
    <td>Specifies the Ascend core type to be used by the code within this scope</td>
    <td>&quot;vector&quot;, &quot;cube&quot;</td>
  </tr>
</table>

### Common Values Description

<table>
  <tr>
    <td>Value</td>
    <td>Target Core</td>
    <td>Usage/Optimization Direction</td>
  </tr>
  <tr>
    <td>&quot;vector&quot;</td>
    <td>Vector Unit</td>
    <td>Suitable for element-wise operations, such as addition (+), multiplication (*), activation functions (ReLU, Sigmoid), data loading (tl.load), and storing (tl.store).</td>
  </tr>
  <tr>
    <td>&quot;cube&quot;</td>
    <td>Cube Unit</td>
    <td>Suitable for matrix computations, especially matrix multiplication (GEMM) and convolution operations. This is typically associated with operations like tl.dot.</td>
  </tr>
  <tr>
    <td>&quot;SIMT&quot;</td>
    <td>Single instruction multiple thread</td>
    <td>-</td>
  </tr>
  <tr>
    <td>&quot;SIMD&quot;</td>
    <td>Single instruction multiple data</td>
    <td>-</td>
  </tr>
</table>

## 3. Constraints

Each kernel has 1 scope for cube and vector, inside them they run in parallel and there are other syncing operations that declare the sync between both scopes.

- Parallel Execution: Operations within cube and vector scopes execute in parallel.
- Single Scope per Type: Each kernel supports one cube scope and one vector scope (?).
- Explicit Synchronization: Required for data dependencies between scopes using sync operations.

## 4. Usage Examples

<table>
  <tr>
    <td>Python<br>import os<br><br>os.environ[&quot;TORCH_DEVICE_BACKEND_AUTOLOAD&quot;] = &quot;0&quot;<br><br>import pytest<br><br>import triton<br><br>import triton.language as tl<br><br>import triton.language.extra.cann.extension as al<br><br>from triton.compiler.compiler import ASTSource<br><br>from triton.compiler.code_generator import ast_to_ttir<br><br>from triton._C.libtriton import ir<br><br>from triton._C.libtriton.ascend import ir as ascend_ir<br><br>class Options:<br><br>    num_warps = 4<br><br>    num_stages = 3<br><br>    num_ctas = 1<br><br>    cluster_dims = (1, 1, 1)<br><br>    enable_fp_fusion = True<br><br>    debug = False<br><br>def compile_kernel(kernel, signature, constants):<br><br>    src = ASTSource(kernel, signature, constants)<br><br>    context = ir.context()<br><br>    ir.load_dialects(context)<br><br>    ascend_ir.load_dialects(context)<br><br>    module = ast_to_ttir(kernel, src, context, Options(), {}, {})<br><br>    return str(module)<br><br>@triton.jit<br><br>def kernel_nested_scope(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):<br><br>    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)<br><br>    with al.scope(core_mode=&quot;vector&quot;):<br><br>        with al.scope(core_mode=&quot;vector&quot;):<br><br>            with al.scope(core_mode=&quot;cube&quot;):<br><br>                x = tl.load(x_ptr + i, mask=i &lt; n)<br><br>                y = tl.load(y_ptr + i, mask=i &lt; n)<br><br>                result = x + y<br><br>                tl.store(out_ptr + i, result, mask=i &lt; n)<br><br>@triton.jit<br><br>def kernel_scope_escape(x_ptr, out_ptr, n, BLOCK: tl.constexpr):<br><br>    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)<br><br>    with al.scope(core_mode=&quot;vector&quot;):<br><br>        x = tl.load(x_ptr + i, mask=i &lt; n)<br><br>    a = x + 1.0<br><br>    tl.store(out_ptr + i, a, mask=i &lt; n)<br><br>@triton.jit<br><br>def kernel_scope_cube(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):<br><br>    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)<br><br>    with al.scope(core_mode=&quot;cube&quot;):<br><br>        x = tl.load(x_ptr + i, mask=i &lt; n)<br><br>        y = tl.load(y_ptr + i, mask=i &lt; n)<br><br>        result = x + y<br><br>        tl.store(out_ptr + i, result, mask=i &lt; n)<br><br>@triton.jit<br><br>def kernel_scope_vector(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):<br><br>    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)<br><br>    with al.scope(core_mode=&quot;vector&quot;):<br><br>        x = tl.load(x_ptr + i, mask=i &lt; n)<br><br>        y = tl.load(y_ptr + i, mask=i &lt; n)<br><br>        result = x + y<br><br>        tl.store(out_ptr + i, result, mask=i &lt; n)<br><br>@triton.jit<br><br>def kernel_scope_disable_auto_sync(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):<br><br>    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)<br><br>    with al.scope(core_mode=&quot;vector&quot;, disable_auto_sync=True):<br><br>        x = tl.load(x_ptr + i, mask=i &lt; n)<br><br>        y = tl.load(y_ptr + i, mask=i &lt; n)<br><br>        result = x + y<br><br>        tl.store(out_ptr + i, result, mask=i &lt; n)<br><br>if __name__ == &quot;__main__&quot;:<br><br>    print(&quot;=&quot; * 60)<br><br>    print(&quot;Test 1: Nested Scopes&quot;)<br><br>    print(&quot;=&quot; * 60)<br><br>    mlir = compile_kernel(<br><br>        kernel_nested_scope, {&quot;x_ptr&quot;: &quot;*fp32&quot;, &quot;y_ptr&quot;: &quot;*fp32&quot;, &quot;out_ptr&quot;: &quot;*fp32&quot;, &quot;n&quot;: &quot;i32&quot;}, {&quot;BLOCK&quot;: 256}<br><br>    )<br><br>    print(f&quot;✅ Generated MLIR ({len(mlir)} chars):\n&quot;)<br><br>    print(mlir)<br><br>    print(&quot;\n&quot; + &quot;=&quot; * 60)<br><br>    print(&quot;Test 2: Scope Escape&quot;)<br><br>    print(&quot;=&quot; * 60)<br><br>    mlir = compile_kernel(kernel_scope_escape, {&quot;x_ptr&quot;: &quot;*fp32&quot;, &quot;out_ptr&quot;: &quot;*fp32&quot;, &quot;n&quot;: &quot;i32&quot;}, {&quot;BLOCK&quot;: 256})<br><br>    print(f&quot;✅ Generated MLIR ({len(mlir)} chars):\n&quot;)<br><br>    print(mlir)<br><br>    print(&quot;\n&quot; + &quot;=&quot; * 60)<br><br>    print(&quot;Test 3: Cube Core Mode&quot;)<br><br>    print(&quot;=&quot; * 60)<br><br>    mlir = compile_kernel(<br><br>        kernel_scope_cube, {&quot;x_ptr&quot;: &quot;*fp32&quot;, &quot;y_ptr&quot;: &quot;*fp32&quot;, &quot;out_ptr&quot;: &quot;*fp32&quot;, &quot;n&quot;: &quot;i32&quot;}, {&quot;BLOCK&quot;: 256}<br><br>    )<br><br>    print(f&quot;✅ Generated MLIR ({len(mlir)} chars):\n&quot;)<br><br>    print(mlir)<br><br>    print(&quot;\n&quot; + &quot;=&quot; * 60)<br><br>    print(&quot;Test 4: Vector Core Mode&quot;)<br><br>    print(&quot;=&quot; * 60)<br><br>    mlir = compile_kernel(<br><br>        kernel_scope_vector, {&quot;x_ptr&quot;: &quot;*fp32&quot;, &quot;y_ptr&quot;: &quot;*fp32&quot;, &quot;out_ptr&quot;: &quot;*fp32&quot;, &quot;n&quot;: &quot;i32&quot;}, {&quot;BLOCK&quot;: 256}<br><br>    )<br><br>    print(f&quot;✅ Generated MLIR ({len(mlir)} chars):\n&quot;)<br><br>    print(mlir)<br><br>    print(&quot;\n&quot; + &quot;=&quot; * 60)<br><br>    print(&quot;Test 5: Disable Auto Sync&quot;)<br><br>    print(&quot;=&quot; * 60)<br><br>    mlir = compile_kernel(<br><br>        kernel_scope_disable_auto_sync,<br><br>        {&quot;x_ptr&quot;: &quot;*fp32&quot;, &quot;y_ptr&quot;: &quot;*fp32&quot;, &quot;out_ptr&quot;: &quot;*fp32&quot;, &quot;n&quot;: &quot;i32&quot;},<br><br>        {&quot;BLOCK&quot;: 256},<br><br>    )<br><br>    print(f&quot;✅ Generated MLIR ({len(mlir)} chars):\n&quot;)<br><br>    print(mlir)<br></td>
  </tr>
</table>

## 5. Compilation Output

<table>
  <tr>
    <td>Plain Text<br>============================================================<br><br>Test 1: Nested Scopes<br><br>============================================================<br><br>✅ Generated MLIR (4155 chars):<br><br>#loc = loc(&quot;/home/linxin/triton-test/scope.py&quot;:34:0)<br><br>module {<br><br>  tt.func public @kernel_nested_scope(%arg0: !tt.ptr&lt;f32&gt; loc(&quot;/home/linxin/triton-test/scope.py&quot;:34:0), %arg1: !tt.ptr&lt;f32&gt; loc(&quot;/home/linxin/triton-test/scope.py&quot;:34:0), %arg2: !tt.ptr&lt;f32&gt; loc(&quot;/home/linxin/triton-test/scope.py&quot;:34:0), %arg3: i32 loc(&quot;/home/linxin/triton-test/scope.py&quot;:34:0)) attributes {noinline = false} {<br><br>    %0 = tt.get_program_id x : i32 loc(#loc1)<br><br>    %c256_i32 = arith.constant 256 : i32 loc(#loc2)<br><br>    %c256_i32_0 = arith.constant 256 : i32 loc(#loc2)<br><br>    %1 = arith.muli %0, %c256_i32_0 : i32 loc(#loc2)<br><br>    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor&lt;256xi32&gt; loc(#loc3)<br><br>    %3 = tt.splat %1 : i32 -&gt; tensor&lt;256xi32&gt; loc(#loc4)<br><br>    %4 = arith.addi %3, %2 : tensor&lt;256xi32&gt; loc(#loc4)<br><br>    %5:3 = scope.scope : () -&gt; (tensor&lt;256xf32&gt;, tensor&lt;256xf32&gt;, tensor&lt;256xf32&gt;) {<br><br>      %6:3 = scope.scope : () -&gt; (tensor&lt;256xf32&gt;, tensor&lt;256xf32&gt;, tensor&lt;256xf32&gt;) {<br><br>        %7:3 = scope.scope : () -&gt; (tensor&lt;256xf32&gt;, tensor&lt;256xf32&gt;, tensor&lt;256xf32&gt;) {<br><br>          %8 = tt.splat %arg3 : i32 -&gt; tensor&lt;256xi32&gt; loc(#loc8)<br><br>          %9 = arith.cmpi slt, %4, %8 : tensor&lt;256xi32&gt; loc(#loc8)<br><br>          %10 = tt.splat %arg0 : !tt.ptr&lt;f32&gt; -&gt; tensor&lt;256x!tt.ptr&lt;f32&gt;&gt; loc(#loc9)<br><br>          %11 = tt.addptr %10, %4 : tensor&lt;256x!tt.ptr&lt;f32&gt;&gt;, tensor&lt;256xi32&gt; loc(#loc9)<br><br>          %cst = arith.constant 0.000000e+00 : f32 loc(#loc10)<br><br>          %cst_1 = arith.constant dense&lt;0.000000e+00&gt; : tensor&lt;256xf32&gt; loc(#loc10)<br><br>          %12 = tt.load %11, %9, %cst_1 : tensor&lt;256x!tt.ptr&lt;f32&gt;&gt; loc(#loc10)<br><br>          %13 = tt.splat %arg3 : i32 -&gt; tensor&lt;256xi32&gt; loc(#loc11)<br><br>          %14 = arith.cmpi slt, %4, %13 : tensor&lt;256xi32&gt; loc(#loc11)<br><br>          %15 = tt.splat %arg1 : !tt.ptr&lt;f32&gt; -&gt; tensor&lt;256x!tt.ptr&lt;f32&gt;&gt; loc(#loc12)<br><br>          %16 = tt.addptr %15, %4 : tensor&lt;256x!tt.ptr&lt;f32&gt;&gt;, tensor&lt;256xi32&gt; loc(#loc12)<br><br>          %cst_2 = arith.constant 0.000000e+00 : f32 loc(#loc13)<br><br>          %cst_3 = arith.constant dense&lt;0.000000e+00&gt; : tensor&lt;256xf32&gt; loc(#loc13)<br><br>          %17 = tt.load %16, %14, %cst_3 :