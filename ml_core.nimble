# Package
version       = "0.0.5"
author        = "jasagiri"
description   = "Core ML types: DType, Shape, TensorRef, OpSpec, Graph IR"
license       = "Apache-2.0"
srcDir        = "src"

# Dependencies
requires "nim >= 2.0.0"
requires "checksums >= 0.1.0"

# Tasks
task test, "Run tests":
  exec "nim c -r tests/test_dtype.nim"
  exec "nim c -r tests/test_shape.nim"
  exec "nim c -r tests/test_tensor.nim"
  exec "nim c -r tests/test_ops.nim"
  exec "nim c -r tests/test_ir.nim"
