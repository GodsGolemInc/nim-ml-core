## CPU kernels for comparison operations
##
## Returns boolean tensors (stored as uint8: 0=false, 1=true)

import ../../[dtype, shape, tensor]

# =============================================================================
# Comparison Operations (return bool tensor)
# =============================================================================

proc geKernel*(a, b: TensorData): TensorData =
  ## Greater than or equal: a >= b
  assert a.shape == b.shape, "Shapes must match for ge"
  result = newTensorData(a.shape, dtBool)
  let n = a.shape.size
  var dst = cast[ptr UncheckedArray[uint8]](addr result.data[0])

  case a.dtype
  of dtFloat32:
    let srcA = a.asFloat32
    let srcB = b.asFloat32
    for i in 0 ..< n:
      dst[i] = if srcA[i] >= srcB[i]: 1'u8 else: 0'u8
  of dtFloat64:
    let srcA = a.asFloat64
    let srcB = b.asFloat64
    for i in 0 ..< n:
      dst[i] = if srcA[i] >= srcB[i]: 1'u8 else: 0'u8
  of dtInt32:
    let srcA = a.asInt32
    let srcB = b.asInt32
    for i in 0 ..< n:
      dst[i] = if srcA[i] >= srcB[i]: 1'u8 else: 0'u8
  of dtInt64:
    let srcA = a.asInt64
    let srcB = b.asInt64
    for i in 0 ..< n:
      dst[i] = if srcA[i] >= srcB[i]: 1'u8 else: 0'u8
  else:
    raise newException(ValueError, "Unsupported dtype for ge: " & $a.dtype)

proc leKernel*(a, b: TensorData): TensorData =
  ## Less than or equal: a <= b
  assert a.shape == b.shape, "Shapes must match for le"
  result = newTensorData(a.shape, dtBool)
  let n = a.shape.size
  var dst = cast[ptr UncheckedArray[uint8]](addr result.data[0])

  case a.dtype
  of dtFloat32:
    let srcA = a.asFloat32
    let srcB = b.asFloat32
    for i in 0 ..< n:
      dst[i] = if srcA[i] <= srcB[i]: 1'u8 else: 0'u8
  of dtFloat64:
    let srcA = a.asFloat64
    let srcB = b.asFloat64
    for i in 0 ..< n:
      dst[i] = if srcA[i] <= srcB[i]: 1'u8 else: 0'u8
  else:
    raise newException(ValueError, "Unsupported dtype for le: " & $a.dtype)

proc gtKernel*(a, b: TensorData): TensorData =
  ## Greater than: a > b
  assert a.shape == b.shape, "Shapes must match for gt"
  result = newTensorData(a.shape, dtBool)
  let n = a.shape.size
  var dst = cast[ptr UncheckedArray[uint8]](addr result.data[0])

  case a.dtype
  of dtFloat32:
    let srcA = a.asFloat32
    let srcB = b.asFloat32
    for i in 0 ..< n:
      dst[i] = if srcA[i] > srcB[i]: 1'u8 else: 0'u8
  of dtFloat64:
    let srcA = a.asFloat64
    let srcB = b.asFloat64
    for i in 0 ..< n:
      dst[i] = if srcA[i] > srcB[i]: 1'u8 else: 0'u8
  else:
    raise newException(ValueError, "Unsupported dtype for gt: " & $a.dtype)

proc ltKernel*(a, b: TensorData): TensorData =
  ## Less than: a < b
  assert a.shape == b.shape, "Shapes must match for lt"
  result = newTensorData(a.shape, dtBool)
  let n = a.shape.size
  var dst = cast[ptr UncheckedArray[uint8]](addr result.data[0])

  case a.dtype
  of dtFloat32:
    let srcA = a.asFloat32
    let srcB = b.asFloat32
    for i in 0 ..< n:
      dst[i] = if srcA[i] < srcB[i]: 1'u8 else: 0'u8
  of dtFloat64:
    let srcA = a.asFloat64
    let srcB = b.asFloat64
    for i in 0 ..< n:
      dst[i] = if srcA[i] < srcB[i]: 1'u8 else: 0'u8
  else:
    raise newException(ValueError, "Unsupported dtype for lt: " & $a.dtype)

proc eqKernel*(a, b: TensorData): TensorData =
  ## Equal: a == b
  assert a.shape == b.shape, "Shapes must match for eq"
  result = newTensorData(a.shape, dtBool)
  let n = a.shape.size
  var dst = cast[ptr UncheckedArray[uint8]](addr result.data[0])

  case a.dtype
  of dtFloat32:
    let srcA = a.asFloat32
    let srcB = b.asFloat32
    for i in 0 ..< n:
      dst[i] = if srcA[i] == srcB[i]: 1'u8 else: 0'u8
  of dtFloat64:
    let srcA = a.asFloat64
    let srcB = b.asFloat64
    for i in 0 ..< n:
      dst[i] = if srcA[i] == srcB[i]: 1'u8 else: 0'u8
  else:
    raise newException(ValueError, "Unsupported dtype for eq: " & $a.dtype)

proc neKernel*(a, b: TensorData): TensorData =
  ## Not equal: a != b
  assert a.shape == b.shape, "Shapes must match for ne"
  result = newTensorData(a.shape, dtBool)
  let n = a.shape.size
  var dst = cast[ptr UncheckedArray[uint8]](addr result.data[0])

  case a.dtype
  of dtFloat32:
    let srcA = a.asFloat32
    let srcB = b.asFloat32
    for i in 0 ..< n:
      dst[i] = if srcA[i] != srcB[i]: 1'u8 else: 0'u8
  of dtFloat64:
    let srcA = a.asFloat64
    let srcB = b.asFloat64
    for i in 0 ..< n:
      dst[i] = if srcA[i] != srcB[i]: 1'u8 else: 0'u8
  else:
    raise newException(ValueError, "Unsupported dtype for ne: " & $a.dtype)

# =============================================================================
# Scalar Comparison Operations
# =============================================================================

proc geScalarKernel*(input: TensorData, scalar: float64): TensorData =
  ## Greater than or equal to scalar: x >= s
  result = newTensorData(input.shape, dtBool)
  let n = input.shape.size
  var dst = cast[ptr UncheckedArray[uint8]](addr result.data[0])

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    let s = scalar.float32
    for i in 0 ..< n:
      dst[i] = if src[i] >= s: 1'u8 else: 0'u8
  of dtFloat64:
    let src = input.asFloat64
    for i in 0 ..< n:
      dst[i] = if src[i] >= scalar: 1'u8 else: 0'u8
  else:
    raise newException(ValueError, "Unsupported dtype for geScalar: " & $input.dtype)

proc leScalarKernel*(input: TensorData, scalar: float64): TensorData =
  ## Less than or equal to scalar: x <= s
  result = newTensorData(input.shape, dtBool)
  let n = input.shape.size
  var dst = cast[ptr UncheckedArray[uint8]](addr result.data[0])

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    let s = scalar.float32
    for i in 0 ..< n:
      dst[i] = if src[i] <= s: 1'u8 else: 0'u8
  of dtFloat64:
    let src = input.asFloat64
    for i in 0 ..< n:
      dst[i] = if src[i] <= scalar: 1'u8 else: 0'u8
  else:
    raise newException(ValueError, "Unsupported dtype for leScalar: " & $input.dtype)
