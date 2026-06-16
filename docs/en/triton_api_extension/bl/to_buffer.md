# to_buffer

## 1. Hardware Background

Used to convert a `tl.tensor` tensor object into a `bl.buffer` buffer object dedicated to Ascend hardware. It is the core conversion interface between tensors and hardware memory buffers.

## 2. Interface Definition

<table>
  <tr>
    <td>Python<br>def to_buffer(<br>    tensor: tl.tensor,<br>    space: address_space = None,<br>    bind_buffer: buffer = None,<br>    _builder=None<br>) -&gt; buffer:</td>
  </tr>
</table>

## 3. Parameter Description

<table>
  <tr>
    <td>Parameter Name</td>
    <td>Type</td>
    <td>Required</td>
    <td>Description</td>
  </tr>
  <tr>
    <td>tensor</td>
    <td>tl.tensor</td>
    <td>Yes</td>
    <td>Input tensor to be converted into a buffer</td>
  </tr>
  <tr>
    <td>space</td>
    <td>bl.address_space</td>
    <td>No</td>
    <td>Specifies the Ascend hardware address space for the target buffer</td>
  </tr>
  <tr>
    <td>bind_buffer</td>
    <td>bl.buffer</td>
    <td>No</td>
    <td>Optional, directly binds the tensor to the specified target buffer</td>
  </tr>
  <tr>
    <td>_builder</td>
    <td>-</td>
    <td>Internal</td>
    <td>Automatically passed by the compiler; users do not need to use it</td>
  </tr>
</table>

## 4. Return Value

- Returns a `bl.buffer` object corresponding to the input tensor

- If the `bind_buffer` parameter is provided, directly returns the bound buffer itself

## 5. Constraints

- Interface constraints are consistent with `bl.allocate_local_buffer`

- The address space parameter must strictly match the memory regions supported by Ascend hardware (UB/L1/L0A/L0B/L0C)

## 6. Complete Usage Examples

### Basic Usage (Kernel Definition + Compilation Verification)

<table>
  <tr>
    <td>Python<br>import triton<br>import triton.language as tl<br>from triton.compiler import ASTSource<br>import triton.extension.buffer.language as bl<br>import triton.language.extra.cann.extension as al<br><br># Get the current hardware compilation target<br>target = triton.runtime.driver.active.get_current_target()<br><br>@triton.jit<br>def to_buffer_kernel():<br>    # 1. Basic conversion: no address space specified<br>    a = tl.full((32, 2, 4), 0, dtype=tl.int64)<br>    a_buf = bl.to_buffer(a)<br>    # 2. Convert and specify UB address space<br>    b = tl.full((32, 2, 4), 0, dtype=tl.int64)<br>    b_buf = bl.to_buffer(b, al.ascend_address_space.UB)<br>    # 3. Convert and specify L1 address space<br>    c = tl.full((32, 2, 4), 0, dtype=tl.int64)<br>    c_buf = bl.to_buffer(c, al.ascend_address_space.L1)<br>    # 4. Convert and specify L0A address space<br>    d = tl.full((32, 2, 4), 0, dtype=tl.int64)<br>    d_buf = bl.to_buffer(d, al.ascend_address_space.L0A)<br>    # 5. Convert and specify L0B address space<br>    e = tl.full((32, 2, 4), 0, dtype=tl.int64)<br>    e_buf = bl.to_buffer(e, al.ascend_address_space.L0B)<br>    # 6. Convert and specify L0C address space<br>    f = tl.full((32, 2, 4), 0, dtype=tl.int64)<br>    f_buf = bl.to_buffer(f, al.ascend_address_space.L0C)<br><br># Compilation test function<br>def test_to_buffer():<br>    src = ASTSource(<br>        fn=to_buffer_kernel,<br>        constants={},<br>        signature={},<br>    )<br>    # Compile the kernel (verify interface validity)<br>    triton.compile(src=src, target=target)<br>    print(&quot;✅ to_buffer interface compilation verification successful&quot;)<br><br>if __name__ == &quot;__main__&quot;:<br>    test_to_buffer()</td>
  </tr>
</table>

### Advanced Usage (Compilation + IR Print)

<table>
  <tr>
    <td>Python<br># Compile and print Triton IR (recommended for debugging)<br>def test_to_buffer_print_ir():<br>    src = ASTSource(<br>        fn=to_buffer_kernel,<br>        constants={},<br>        signature={},<br>    )<br>    # Enable IR printing<br>    compile_options = {&quot;dump_ir&quot;: True, &quot;optimization_level&quot;: 0}<br>    compiled_kernel = triton.compile(src=src, target=target, options=compile_options)<br>    print(&quot;\n📄 Kernel IR printing completed&quot;)<br><br>if __name__ == &quot;__main__&quot;:<br>    test_to_buffer_print_ir()</td>
  </tr>
</table>

## 7. Core Description

- This interface is the core conversion entry point between tensors and hardware buffers

- Supports manually specifying the full range of Ascend hardware address spaces (UB/L1/L0)

- Supports binding to existing buffers for fine-grained memory management

- Can only be used within kernel functions decorated with `@triton.jit`