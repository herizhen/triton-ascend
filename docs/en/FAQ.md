# Triton-Ascend FAQ

## 1. Installation and Environment Configuration

**Q: How to correctly install Triton-Ascend? Does it support direct pip installation?**

A: You can install it directly using pip.

```Python
pip install triton-ascend
```

**Q: Can community Triton and Triton-Ascend coexist?**

A: For triton-ascend 3.2.0 and below, no. You need to uninstall the community Triton first, then install Triton-Ascend.<br>
For triton-ascend 3.2.1 and above, Triton-Ascend mitigates the installation overwrite issue by declaring Triton as an installation dependency.
When installing Triton-Ascend, the community Triton is installed first, and then Triton-Ascend overwrites the directory with the same name, thereby preventing subsequent installations of other software packages that depend on Triton from overwriting Triton-Ascend.
The reason x86 and arm use different versions of the community Triton installation package is that the community only started providing arm version packages from version 3.5 onwards: x86 depends on triton==3.2.0, arm depends on triton==3.5.0.

- Note: If you install a third-party component that depends on triton, or triton itself, after installing triton-ascend, it will overwrite the installed Triton-Ascend directory.
In this case, you need to uninstall the community Triton and Triton-Ascend first, then install Triton-Ascend again.

```Python
pip uninstall triton
pip uninstall triton-ascend
pip install triton-ascend
```

**Q: Can Triton-Ascend be used on non-Ascend hardware (e.g., CUDA, AMD)?**

A: No, Triton-Ascend can only be used in an Ascend NPU hardware environment.

## 2. Accuracy and Numerical Consistency Issues

**Q: The NPU execution results are inconsistent with PyTorch/CPU/GPU reference results. How to debug?**

A: For examples, please refer to [07_accuracy_comparison_example.md](../zh/examples/07_accuracy_comparison_example.md)
For debugging methods, please refer to [Interpreter Mode Debugging Method](./debug_guide/debugging.md#4-interpreter-mode)

## 3. Error Codes and Exception Handling

**Q: Why does the kernel compilation report MLIRCompilationError? How to locate the specific failing Pass?**

A: Please refer to [Compilation Error Debugging Method](./debug_guide/debugging.md#52-compilation-error-debugging-method)

## 4. Debugging and Logging

**Q: How to enable detailed log output? Where is TRITON_DEBUG=1 output?**

A: You can use TRITON_DEBUG=1 to obtain detailed debug dump files. Please refer to [Debug Dump Files](./debug_guide/debugging.md#32-debug-dump-files)

**Q: Can I print intermediate tensor values in the kernel? Is tl.device_print available?**

A: You can use tl.device_print to print tensors in the kernel. Please refer to [Print Debugging Method](debug_guide/debugging.md#51-print-debugging-method)

## 5. Development and Contribution

**Q: How to build and test Triton-Ascend locally?**

A: For local build and test methods, please refer to [Installing Triton-Ascend from Source](./installation_guide.md#installing-triton-ascend-from-source)

**Q: Which CI checks must a PR pass?**

A: The CI checks for a PR include: Code security and specification checks, open-source snippet checks, malicious code checks, compilation and build, and developer tests.

## 6. Performance Tuning

**Q: Is there a performance analysis tool (profiler) available?**

A: Yes, an integrated performance analysis tool (profiler) is available. Please refer to [Operator Performance Tuning Method](./debug_guide/profiling.md)