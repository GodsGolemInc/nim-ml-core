# nim-ml-core

Core ML types and abstractions for the Nim ML framework.

## Features

- **DType** - Data type system with type promotion and casting rules
- **Shape** - N-dimensional shape handling with broadcasting support
- **TensorRef** - Content-addressed tensor references with SHA-256 hashing
- **OpSpec** - Operation specification with validation and shape inference
- **Graph IR** - Intermediate representation for computation graphs

## Installation

```bash
nimble install ml_core
```

Or add to your `.nimble` file:
```nim
requires "ml_core >= 0.0.4"
```

## Usage

```nim
import ml_core

# DType operations
let dt = DType.float32
echo dt.size  # 4
echo dt.isFloating  # true

# Shape operations
let shape = newShape(3, 4, 5)
echo shape.rank  # 3
echo shape.numel  # 60

# Broadcasting
let a = newShape(3, 1, 5)
let b = newShape(1, 4, 5)
let result = broadcast(a, b)  # Shape(3, 4, 5)

# TensorRef with content addressing
let data = newTensorData(DType.float32, newShape(2, 3))
let ref = newTensorRef(data)
echo ref.hash  # SHA-256 hash

# Graph IR
var builder = newGraphBuilder("my_graph")
let input = builder.addInput("x", DType.float32, newShape(10))
let output = builder.addOp(opRelu, @[input])
builder.addOutput(output)
let graph = builder.build()
```

## Modules

| Module | Description |
|--------|-------------|
| `dtype` | Data type definitions and operations |
| `shape` | N-dimensional shape with broadcasting |
| `tensor` | TensorData and TensorRef types |
| `ops` | Operation specifications |
| `ir` | Graph intermediate representation |

## Test Coverage

| Module | Tests | Status |
|--------|-------|--------|
| dtype | 27 | Pass |
| shape | 48 | Pass |
| tensor | 43 | Pass |
| ops | 57 | Pass |
| ir | 46 | Pass |
| **Total** | **221** | **100%** |

## Running Tests

```bash
nimble test
```

Or run individual test files:
```bash
nim c -r tests/test_dtype.nim
nim c -r tests/test_shape.nim
nim c -r tests/test_tensor.nim
nim c -r tests/test_ops.nim
nim c -r tests/test_ir.nim
```

## License

Apache-2.0
