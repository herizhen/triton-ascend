# Libdevice Developer Manual

## SIMT Compilation Example

Example of a triton kernel compiled using SIMT

```python
# Enable libdevice SIMT compilation
import os
os.environ['TRITON_ENABLE_LIBDEVICE_SIMT'] = '1'

import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice
import torch

@triton.jit
def triton_kernel(input, output, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base = tl.arange(0, XBLOCK_SUB)
    loops: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop in range(loops):
        x0 = offset + (loop * XBLOCK_SUB) + base
        x = tl.load(input + (x0), None)
        y = libdevice.abs(x)
        tl.store(output + (x0), y, None)

dtype, shape, ncore, xblock, xblock_sub = ['int32', (128, 4096), 512, 1024, 1024]
input = torch.randn(shape, dtype=dtype).npu()
output = torch.randn(shape, dtype=dtype).npu()
triton_kernel[ncore, 1, 1](input, output, xblock, xblock_sub, force_simt_only=True)
```

## 1. triton.language.extra.cann.libdevice.abs

### OP Overview

Computes the absolute value of the input argument.

Prototype:

```python
triton.language.extra.cann.libdevice.abs(x, _builder=None)
```

Input types:
- x: `int32`, `float32`

Return value: `tl.tensor`, returns the absolute value of the input argument.

Return type: `int32`, `float32`

Supported compilation modes: SIMT

## 2. triton.language.extra.cann.libdevice.acos

### OP Overview

Computes the arccosine of the input argument.

Prototype:

```python
triton.language.extra.cann.libdevice.acos(x, _builder=None)
```

Input types:
- x: `float32`

Return value: `tl.tensor`, returns the arccosine of the input argument, in the range \[0, π] radians.

Return type: `float32`

Supported compilation modes: SIMT, SIMD

## 3. triton.language.extra.cann.libdevice.acosh

### OP Overview

Computes the inverse hyperbolic cosine of the input argument.

Prototype:

```python
triton.language.extra.cann.libdevice.acosh(x, _builder=None)
```

Input types:
- x: `float32`

Return value: `tl.tensor`, returns the inverse hyperbolic cosine of the input argument, in the range \[0, +∞].

Return type: `float32`

Supported compilation modes: SIMT

## 4. triton.language.extra.cann.libdevice.add_rd

### OP Overview

Floating-point addition with round-towards-negative-infinity.

Prototype:

```python
triton.language.extra.cann.libdevice.add_rd(x, y, _builder=None)
```

Input types:
- x: `float32`
- y: `float32`

Return value: `tl.tensor`, returns the result of the addition rounded down.

Return type: `float32`

Supported compilation modes: SIMT

## 5. triton.language.extra.cann.libdevice.add_rn

### OP Overview

Floating-point addition with round-to-nearest-even.

Prototype:

```python
triton.language.extra.cann.libdevice.add_rn(x, y, _builder=None)
```

Input types:
- x: `float32`
- y: `float32`

Return value: `tl.tensor`, returns the result of the addition rounded to nearest even.

Return type: `float32`

Supported compilation modes: SIMT

## 6. triton.language.extra.cann.libdevice.add_ru

### OP Overview

Floating-point addition with round-towards-positive-infinity.

Prototype:

```python
triton.language.extra.cann.libdevice.add_ru(x, y, _builder=None)
```

Input types:
- x: `float32`
- y: `float32`

Return value: `tl.tensor`, returns the result of the addition rounded up.

Return type: `float32`

Supported compilation modes: SIMT

## 7. triton.language.extra.cann.libdevice.add_rz

### OP Overview

Floating-point addition with round-towards-zero.

Prototype:

```python
triton.language.extra.cann.libdevice.add_rz(x, y, _builder=None)
```

Input types:
- x: `float32`
- y: `float32`

Return value: `tl.tensor`, returns the result of the addition rounded towards zero.

Return type: `float32`

Supported compilation modes: SIMT

## 8. triton.language.extra.cann.libdevice.asin

### OP Overview

Computes the arcsine of the input argument.

Prototype:

```python
triton.language.extra.cann.libdevice.asin(x, _builder=None)
```

Input types:
- x: `float32`

Return value: `tl.tensor`, returns the arcsine of the input argument, in the range \[-π/2, π/2] radians.

Return type: `float32`

Supported compilation modes: SIMT

## 9. triton.language.extra.cann.libdevice.asinh

### OP Overview

Computes the inverse hyperbolic sine of the input argument.

Prototype:

```python
triton.language.extra.cann.libdevice.asinh(x, _builder=None)
```

Input types:
- x: `float32`

Return value: `tl.tensor`, returns the inverse hyperbolic sine of the input argument.

Return type: `float32`

Supported compilation modes: SIMT

## 10. triton.language.extra.cann.libdevice.atan

### OP Overview

Computes the arctangent of the input argument.

Prototype:

```python
triton.language.extra.cann.libdevice.atan(x, _builder=None)
```

Input types:
- x: `float32`

Return value: `tl.tensor`, returns the arctangent of the input argument, in the range \[-π/2, π/2] radians.

Return type: `float32`

Supported compilation modes: SIMT, SIMD

## 11. triton.language.extra.cann.libdevice.atan2

### OP Overview

Arctangent function, computes the arctangent of x / y.

Prototype:

```python
triton.language.extra.cann.libdevice.atan2(x, y, _builder=None)
```

Input types:
- x: `float32`
- y: `float32`

Return value: `tl.tensor`, returns the arctangent of x / y, in the range \[-π, π] radians.

Return type: `float32`

Supported compilation modes: SIMT, SIMD

## 12. triton.language.extra.cann.libdevice.atanh

### OP Overview

Inverse hyperbolic tangent function, computes the inverse hyperbolic tangent of the input argument.

Prototype:

```python
triton.language.extra.cann.libdevice.atanh(x, _builder=None)
```

Input types:
- x: `float32`

Return value: `tl.tensor`, returns the inverse hyperbolic tangent of the input argument, in the range \[-1, 1].

Return type: `float32`

Supported compilation modes: SIMT

## 13. triton.language.extra.cann.libdevice.brev

### OP Overview

Bit reversal function, reverses the bit order of a 32-bit integer.

Prototype:

```python
triton.language.extra.cann.libdevice.brev(x, _builder=None)
```

Input types:
- x: `int32`

Return value: `tl.tensor`, returns the 32-bit integer with reversed bits.

Return type: `int32`

Supported compilation modes: SIMT

## 14. triton.language.extra.cann.libdevice.byte_perm

### OP Overview

Byte permutation operation, selects bytes from two 32-bit integers to form a new integer. The byte order of input integers x and y is as follows.

```cpp
input[0] = x<7:0>     input[1] = x<15:8>
input[2] = x<23:16>   input[3] = x<31:24>
input[4] = y<7:0>     input[5] = y<15:8>
input[6] = y<23:16>   input[7] = y<31:24>
```

The byte selection parameter s is a 32-bit integer, and the mapping between its bits and byte selection is as follows.

```cpp
selector[0] = s<2:0>    selector[1] = s<6:4>
selector[2] = s<10:8>   selector[3] = s<14:12>
```

Prototype:

```python
triton.language.extra.cann.libdevice.byte_perm(x, y, s, _builder=None)
```

Input types:
- x: `int32`
- y: `int32`
- s: `int32`

Return value: `tl.tensor`, return value return\[n] := input\[selector\[n]], where n represents the n-th byte of the output integer.

Return type: `int32`

Supported compilation modes: SIMT

## 15. triton.language.extra.cann.libdevice.cbrt

### OP Overview

Computes the cube root of the input argument.

Prototype:

```python
triton.language.extra.cann.libdevice.cbrt(x, _builder=None)
```

Input types:
- x: `float32`

Return value: `tl.tensor`, returns the cube root of the input argument.

Return type: `float32`

Supported compilation modes: SIMT

## 16. triton.language.extra.cann.libdevice.ceil

### OP Overview

Rounds up, returns the smallest integer greater than or equal to x.

Prototype:

```python
triton.language.extra.cann.libdevice.ceil(x, _builder=None)
```

Input types:
- x: `float32`

Return value: `tl.tensor`, returns the result of rounding up.

Return type: `float32`

Supported compilation modes: SIMT

## 17. triton.language.extra.cann.libdevice.clz

### OP Overview

Counts the number of leading zeros in a 32-bit integer.

Prototype:

```python
triton.language.extra.cann.libdevice.clz(x, _builder=None)
```

Input types:
- x: `int32`

Return value: `tl.tensor`, returns the number of leading zeros in the input argument. Range \[0, 32].

Return type: `int32`

Supported compilation modes: SIMT

## 18. triton.language.extra.cann.libdevice.copysign

### OP Overview

Generates a floating-point number whose absolute value equals the absolute value of x and whose sign matches y.

Prototype:

```python
triton.language.extra.cann.libdevice.copysign(x, y, _builder=None)
```

Input types:
- x: `float32`
- y: `float32`

Return value: `tl.tensor`, returns a floating-point number whose absolute value equals the absolute value of x and whose sign matches y.

Return type: `float32`

Supported compilation modes: SIMT

## 19. triton.language.extra.cann.libdevice.cos

### OP Overview

Computes the cosine of the input argument (in radians).

Prototype:

```python
triton.language.extra.cann.libdevice.cos(x, _builder=None)
```

Input types:
- x: `float32`

Return value: `tl.tensor`, returns the cosine of the input argument.

Return type: `float32`

Supported compilation modes: SIMT

## 20. triton.language.extra.cann.libdevice.cosh

### OP Overview

Computes the hyperbolic cosine of the input argument.

Prototype:

```python
triton.language.extra.cann.libdevice.cosh(x, _builder=None)
```

Input types:
- x: `float32`

Return value: `tl.tensor`, returns the hyperbolic cosine of the input argument.

Return type: `float32`

Supported compilation modes: SIMT

## 21. triton.language.extra.cann.libdevice.cospi

### OP Overview

Computes cos(π × x).

Prototype:

```python
triton.language.extra.cann.libdevice.cospi(x, _builder=None)
```

Input types:
- x: `float32`

Return value: `tl.tensor`, returns the value of cos(π × x).

Return type: `float32`

Supported compilation modes: SIMT

## 22. triton.language.extra.cann.libdevice.cyl_bessel_i0

### OP Overview

Computes the modified Bessel function of the first kind of order zero for the input argument.

Prototype:

```python
triton.language.extra.cann.libdevice.cyl_bessel_i0(x, _builder=None)
```

Input types:
- x: `float32`

Return value: `tl.tensor`, returns the modified Bessel function of the first kind of order zero for the input argument.

Return type: `float32`

Supported compilation modes: SIMT

## 23. triton.language.extra.cann.libdevice.cyl_bessel_i1

### OP Overview

Computes the modified Bessel function of the first kind of order one for the input argument.

Prototype:

```python
triton.language.extra.cann.libdevice.cyl_bessel_i1(x, _builder=None)
```

Input types:
- x: `float32`

Return value: `tl.tensor`, returns the modified Bessel function of the first kind of order one for the input argument.

Return type: `float32`

Supported compilation modes: SIMT

## 24. triton.language.extra.cann.libdevice.div_rd

### OP Overview

Floating-point division with round-towards-negative-infinity.

Prototype:

```python
triton.language.extra.cann.libdevice.div_rd(x, y, _builder=None)
```

Input types:
- x: `float32`
- y: `float32`

Return value: `tl.tensor`, returns the result of the division.

Return type: `float32`

Supported compilation modes: SIMT

## 25. triton.language.extra.cann.libdevice.div_rn

### OP Overview

Floating-point division with round-to-nearest-even.

Prototype:

```python
triton.language.extra.cann.libdevice.div_rn(x, y, _builder=None)
```

Input types:
- x: `float32`
- y: `float32`

Return value: `tl.tensor`, returns the result of the division.

Return type: `float32`

Supported compilation modes: SIMT

## 26. triton.language.extra.cann.libdevice.div_ru

### OP Overview

Floating-point division with round-towards-positive-infinity.

Prototype:

```python
triton.language.extra.cann.libdevice.div_ru(x, y, _builder=None)
```

Input types:
- x: `float32`
- y: `float32`

Return value: `tl.tensor`, returns the result of the division.

Return type: `float32`

Supported compilation modes: SIMT

## 27. triton.language.extra.cann.libdevice.div_rz

### OP Overview

Floating-point division with round-towards-zero.

Prototype:

```python
triton.language.extra.cann.libdevice.div_rz(x, y, _builder=None)
```

Input types:
- x: `float32`
- y: `float32`

Return value: `tl.tensor`, returns the result of the division.

Return type: `float32`

Supported compilation modes: SIMT, SIMD

## 28. triton.language.extra.cann.libdevice.erf

### OP Overview

Computes the error function of the input argument.

Prototype:

```python
triton.language.extra.cann.libdevice.erf(x, _builder=None)
```

Input types:
- x: `float32`

Return value: `tl.tensor`, returns the error function of the input argument.

Return type: `float32`

Supported compilation modes: SIMT

## 29. triton.language.extra.cann.libdevice.erfc

### OP Overview

Computes the complementary error function of the input argument, i.e., 1 - erf(x).

Prototype:

```python
triton.language.extra.cann.libdevice.erfc(x, _builder=None)
```

Input types:
- x: `float32`

Return value: `tl.tensor`, returns the complementary error function of the input argument.

Return type: `float32`

Supported compilation modes: SIMT

## 30. triton.language.extra.cann.libdevice.erfcinv

### OP Overview

Inverse complementary error function, finds the value y satisfying x = erfc(y).

Prototype:

```python
triton.language.extra.cann.libdevice.erfcinv(x, _builder=None)
```

Input types:
- x: `float32`

Return value: `tl.tensor`, returns the inverse complementary error function of the input argument.

Return type: `float32`

Supported compilation modes: SIMT

## 31. triton.language.extra.cann.libdevice.erfcx

### OP Overview

Computes the scaled complementary error function of the input argument, i.e., exp(x²) × erfc(x).

Prototype:

```python
triton.language.extra.cann.libdevice.erfcx(x, _builder=None)
```

Input types:
- x: `float32`

Return value: `tl.tensor`, returns the scaled complementary error function of the input argument.

Return type: `float32`

Supported compilation modes: SIMT

## 32. triton.l