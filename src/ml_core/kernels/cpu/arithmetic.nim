## CPU kernels for arithmetic operations
##
## Provides actual computation implementations for basic math operations.

import std/[math]
import ../../[dtype, shape, tensor]

# =============================================================================
# Unary Operations
# =============================================================================

proc negKernel*(input: TensorData): TensorData =
  ## Negate all elements: -x
  result = newTensorData(input.shape, input.dtype)
  let n = input.shape.size

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var dst = result.asFloat32
    for i in 0 ..< n:
      dst[i] = -src[i]
  of dtFloat64:
    let src = input.asFloat64
    var dst = result.asFloat64
    for i in 0 ..< n:
      dst[i] = -src[i]
  of dtInt32:
    let src = input.asInt32
    var dst = result.asInt32
    for i in 0 ..< n:
      dst[i] = -src[i]
  of dtInt64:
    let src = input.asInt64
    var dst = result.asInt64
    for i in 0 ..< n:
      dst[i] = -src[i]
  else:
    raise newException(ValueError, "Unsupported dtype for neg: " & $input.dtype)

proc absKernel*(input: TensorData): TensorData =
  ## Absolute value: |x|
  result = newTensorData(input.shape, input.dtype)
  let n = input.shape.size

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var dst = result.asFloat32
    for i in 0 ..< n:
      dst[i] = abs(src[i])
  of dtFloat64:
    let src = input.asFloat64
    var dst = result.asFloat64
    for i in 0 ..< n:
      dst[i] = abs(src[i])
  of dtInt32:
    let src = input.asInt32
    var dst = result.asInt32
    for i in 0 ..< n:
      dst[i] = abs(src[i])
  of dtInt64:
    let src = input.asInt64
    var dst = result.asInt64
    for i in 0 ..< n:
      dst[i] = abs(src[i])
  else:
    raise newException(ValueError, "Unsupported dtype for abs: " & $input.dtype)

proc sqrtKernel*(input: TensorData): TensorData =
  ## Square root: sqrt(x)
  result = newTensorData(input.shape, input.dtype)
  let n = input.shape.size

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var dst = result.asFloat32
    for i in 0 ..< n:
      dst[i] = sqrt(src[i])
  of dtFloat64:
    let src = input.asFloat64
    var dst = result.asFloat64
    for i in 0 ..< n:
      dst[i] = sqrt(src[i])
  else:
    raise newException(ValueError, "Unsupported dtype for sqrt: " & $input.dtype)

proc squareKernel*(input: TensorData): TensorData =
  ## Square: x^2
  result = newTensorData(input.shape, input.dtype)
  let n = input.shape.size

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var dst = result.asFloat32
    for i in 0 ..< n:
      dst[i] = src[i] * src[i]
  of dtFloat64:
    let src = input.asFloat64
    var dst = result.asFloat64
    for i in 0 ..< n:
      dst[i] = src[i] * src[i]
  else:
    raise newException(ValueError, "Unsupported dtype for square: " & $input.dtype)

proc expKernel*(input: TensorData): TensorData =
  ## Exponential: e^x
  result = newTensorData(input.shape, input.dtype)
  let n = input.shape.size

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var dst = result.asFloat32
    for i in 0 ..< n:
      dst[i] = exp(src[i])
  of dtFloat64:
    let src = input.asFloat64
    var dst = result.asFloat64
    for i in 0 ..< n:
      dst[i] = exp(src[i])
  else:
    raise newException(ValueError, "Unsupported dtype for exp: " & $input.dtype)

proc logKernel*(input: TensorData): TensorData =
  ## Natural logarithm: ln(x)
  result = newTensorData(input.shape, input.dtype)
  let n = input.shape.size

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var dst = result.asFloat32
    for i in 0 ..< n:
      dst[i] = ln(src[i])
  of dtFloat64:
    let src = input.asFloat64
    var dst = result.asFloat64
    for i in 0 ..< n:
      dst[i] = ln(src[i])
  else:
    raise newException(ValueError, "Unsupported dtype for log: " & $input.dtype)

proc signKernel*(input: TensorData): TensorData =
  ## Sign function: -1, 0, or 1
  result = newTensorData(input.shape, input.dtype)
  let n = input.shape.size

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var dst = result.asFloat32
    for i in 0 ..< n:
      dst[i] = if src[i] > 0: 1.0'f32 elif src[i] < 0: -1.0'f32 else: 0.0'f32
  of dtFloat64:
    let src = input.asFloat64
    var dst = result.asFloat64
    for i in 0 ..< n:
      dst[i] = if src[i] > 0: 1.0 elif src[i] < 0: -1.0 else: 0.0
  else:
    raise newException(ValueError, "Unsupported dtype for sign: " & $input.dtype)

# =============================================================================
# Binary Operations
# =============================================================================

proc addKernel*(a, b: TensorData): TensorData =
  ## Element-wise addition: a + b
  assert a.shape == b.shape, "Shapes must match for add"
  result = newTensorData(a.shape, a.dtype)
  let n = a.shape.size

  case a.dtype
  of dtFloat32:
    let srcA = a.asFloat32
    let srcB = b.asFloat32
    var dst = result.asFloat32
    for i in 0 ..< n:
      dst[i] = srcA[i] + srcB[i]
  of dtFloat64:
    let srcA = a.asFloat64
    let srcB = b.asFloat64
    var dst = result.asFloat64
    for i in 0 ..< n:
      dst[i] = srcA[i] + srcB[i]
  of dtInt32:
    let srcA = a.asInt32
    let srcB = b.asInt32
    var dst = result.asInt32
    for i in 0 ..< n:
      dst[i] = srcA[i] + srcB[i]
  of dtInt64:
    let srcA = a.asInt64
    let srcB = b.asInt64
    var dst = result.asInt64
    for i in 0 ..< n:
      dst[i] = srcA[i] + srcB[i]
  else:
    raise newException(ValueError, "Unsupported dtype for add: " & $a.dtype)

proc subKernel*(a, b: TensorData): TensorData =
  ## Element-wise subtraction: a - b
  assert a.shape == b.shape, "Shapes must match for sub"
  result = newTensorData(a.shape, a.dtype)
  let n = a.shape.size

  case a.dtype
  of dtFloat32:
    let srcA = a.asFloat32
    let srcB = b.asFloat32
    var dst = result.asFloat32
    for i in 0 ..< n:
      dst[i] = srcA[i] - srcB[i]
  of dtFloat64:
    let srcA = a.asFloat64
    let srcB = b.asFloat64
    var dst = result.asFloat64
    for i in 0 ..< n:
      dst[i] = srcA[i] - srcB[i]
  else:
    raise newException(ValueError, "Unsupported dtype for sub: " & $a.dtype)

proc mulKernel*(a, b: TensorData): TensorData =
  ## Element-wise multiplication: a * b
  assert a.shape == b.shape, "Shapes must match for mul"
  result = newTensorData(a.shape, a.dtype)
  let n = a.shape.size

  case a.dtype
  of dtFloat32:
    let srcA = a.asFloat32
    let srcB = b.asFloat32
    var dst = result.asFloat32
    for i in 0 ..< n:
      dst[i] = srcA[i] * srcB[i]
  of dtFloat64:
    let srcA = a.asFloat64
    let srcB = b.asFloat64
    var dst = result.asFloat64
    for i in 0 ..< n:
      dst[i] = srcA[i] * srcB[i]
  else:
    raise newException(ValueError, "Unsupported dtype for mul: " & $a.dtype)

proc divKernel*(a, b: TensorData): TensorData =
  ## Element-wise division: a / b
  assert a.shape == b.shape, "Shapes must match for div"
  result = newTensorData(a.shape, a.dtype)
  let n = a.shape.size

  case a.dtype
  of dtFloat32:
    let srcA = a.asFloat32
    let srcB = b.asFloat32
    var dst = result.asFloat32
    for i in 0 ..< n:
      dst[i] = srcA[i] / srcB[i]
  of dtFloat64:
    let srcA = a.asFloat64
    let srcB = b.asFloat64
    var dst = result.asFloat64
    for i in 0 ..< n:
      dst[i] = srcA[i] / srcB[i]
  else:
    raise newException(ValueError, "Unsupported dtype for div: " & $a.dtype)

proc scaleKernel*(input: TensorData, scalar: float64): TensorData =
  ## Multiply by scalar: x * s
  result = newTensorData(input.shape, input.dtype)
  let n = input.shape.size

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var dst = result.asFloat32
    let s = scalar.float32
    for i in 0 ..< n:
      dst[i] = src[i] * s
  of dtFloat64:
    let src = input.asFloat64
    var dst = result.asFloat64
    for i in 0 ..< n:
      dst[i] = src[i] * scalar
  else:
    raise newException(ValueError, "Unsupported dtype for scale: " & $input.dtype)

proc powKernel*(base, exp: TensorData): TensorData =
  ## Element-wise power: base^exp
  assert base.shape == exp.shape, "Shapes must match for pow"
  result = newTensorData(base.shape, base.dtype)
  let n = base.shape.size

  case base.dtype
  of dtFloat32:
    let srcBase = base.asFloat32
    let srcExp = exp.asFloat32
    var dst = result.asFloat32
    for i in 0 ..< n:
      dst[i] = pow(srcBase[i], srcExp[i])
  of dtFloat64:
    let srcBase = base.asFloat64
    let srcExp = exp.asFloat64
    var dst = result.asFloat64
    for i in 0 ..< n:
      dst[i] = pow(srcBase[i], srcExp[i])
  else:
    raise newException(ValueError, "Unsupported dtype for pow: " & $base.dtype)

proc maxKernel*(a, b: TensorData): TensorData =
  ## Element-wise maximum: max(a, b)
  assert a.shape == b.shape, "Shapes must match for max"
  result = newTensorData(a.shape, a.dtype)
  let n = a.shape.size

  case a.dtype
  of dtFloat32:
    let srcA = a.asFloat32
    let srcB = b.asFloat32
    var dst = result.asFloat32
    for i in 0 ..< n:
      dst[i] = max(srcA[i], srcB[i])
  of dtFloat64:
    let srcA = a.asFloat64
    let srcB = b.asFloat64
    var dst = result.asFloat64
    for i in 0 ..< n:
      dst[i] = max(srcA[i], srcB[i])
  else:
    raise newException(ValueError, "Unsupported dtype for max: " & $a.dtype)

proc minKernel*(a, b: TensorData): TensorData =
  ## Element-wise minimum: min(a, b)
  assert a.shape == b.shape, "Shapes must match for min"
  result = newTensorData(a.shape, a.dtype)
  let n = a.shape.size

  case a.dtype
  of dtFloat32:
    let srcA = a.asFloat32
    let srcB = b.asFloat32
    var dst = result.asFloat32
    for i in 0 ..< n:
      dst[i] = min(srcA[i], srcB[i])
  of dtFloat64:
    let srcA = a.asFloat64
    let srcB = b.asFloat64
    var dst = result.asFloat64
    for i in 0 ..< n:
      dst[i] = min(srcA[i], srcB[i])
  else:
    raise newException(ValueError, "Unsupported dtype for min: " & $a.dtype)
