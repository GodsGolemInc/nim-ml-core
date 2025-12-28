# Package
version       = "0.0.2"
author        = "jasagiri"
description   = "Core ML types: DType, Shape, TensorRef, OpSpec, Graph IR"
license       = "MIT"
srcDir        = "src"

# Dependencies
requires "nim >= 2.0.0"
requires "checksums >= 0.1.0"

# Tasks
task test, "Run tests":
  exec "nim c -r tests/test_dtype.nim"
  exec "nim c -r tests/test_shape.nim"
  exec "nim c -r tests/test_tensor.nim"
