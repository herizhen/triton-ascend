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
When installing Triton-Ascend, the community Triton is installed first, then Triton-Ascend overwrites the directory with the same name, thereby preventing subsequent installation of other packages that depend on Triton from overwriting Triton-Ascend.
The reason x86 and arm use different versions of the community Triton installation package is that the community only started providing arm version packages from version 3.5 onwards: x86 depends on triton==3.2.0, arm depends on triton==3.5.0.

- Note: If, after installing triton-ascend, you install a third-party package that depends on triton, or triton itself, it will overwrite the installed Triton-Ascend directory.
In this case, you need to uninstall the community Triton and Triton-Ascend first, then reinstall Triton-Ascend.

```Python
pip uninstall triton
pip uninstall triton-ascend
pip install triton-ascend
```

**Q: Can Triton-Ascend be used on non-Ascend hardware (e.g., CUDA, AMD)?**

A: No, Triton-Ascend can only be used in an Ascend NPU hardware environment.

## 2. Accuracy and Numerical Consistency Issues

**Q: The NPU execution results are inconsistent with PyTorch/CPU/GPU reference results. How to troubleshoot?**

A: For examples, please refer to [07_accuracy_comparison_example.md](../zh/examples/07_accuracy_comparison_example.md)
For debugging methods, please refer to [Interpreter Mode Debugging Methods](./debug_guide/debugging.md#4-Interpreter-Mode)

## 3. Error Codes and Exception Handling

**Q: Why does the kernel compilation report MLIRCompilationError? How to locate the specific failing Pass?**

A: Please refer to [Compilation Error Debugging Methods](./debug_guide/debugging.md#52-Compilation-Error-Debugging-Methods)

## 4. Debugging and Logging

**Q: How to enable detailed log output? Where is TRITON_DEBUG=1 output?**

A: You can use TRITON_DEBUG=1 to obtain detailed debug dump files. Please refer to [Debug Dump Files](./debug_guide/debugging.md#32-Debug-Dump-Files)

**Q: Can intermediate tensor values be printed inside a kernel? Is tl.device_print available?**

A: You can use tl.device_print to print tensors inside a kernel. Please refer to [Print Debugging Methods](debug_guide/debugging.md#51-Print-Debugging-Methods)

## 5. Development and Contribution

**Q: How to build and test Triton-Ascend locally?**

A: For local build and test methods, please refer to [Installing Triton-Ascend from Source](./installation_guide.md#Installing-Triton-Ascend-from-Source)

**Q: What CI checks must a PR pass?**

A: PR CI checks include: code security and style checks, open-source snippet checks, malicious code checks, compilation and build, and developer tests.

## 6. Performance Tuning

**Q: Is there a performance analysis tool (profiler) available?**

A: Yes, an integrated performance analysis tool (profiler) is available. Please refer to [Operator Performance Tuning Methods](./debug_guide/profiling.md)