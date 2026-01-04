## CPU kernels for selection operations
##
## Provides where, clamp, masked operations.

import ../../[dtype, shape, tensor]

# =============================================================================
# Conditional Selection
# =============================================================================

proc whereKernel*(condition: TensorData, x: TensorData, y: TensorData): TensorData =
  ## Element-wise selection: condition ? x : y
  ## condition should be boolean (dtBool)
  assert condition.shape == x.shape and x.shape == y.shape, "Shapes must match"
  assert x.dtype == y.dtype, "x and y must have same dtype"

  result = newTensorData(x.shape, x.dtype)
  let n = x.shape.size
  let cond = cast[ptr UncheckedArray[uint8]](addr condition.data[0])

  case x.dtype
  of dtFloat32:
    let srcX = x.asFloat32
    let srcY = y.asFloat32
    var dst = result.asFloat32
    for i in 0 ..< n:
      dst[i] = if cond[i] != 0: srcX[i] else: srcY[i]
  of dtFloat64:
    let srcX = x.asFloat64
    let srcY = y.asFloat64
    var dst = result.asFloat64
    for i in 0 ..< n:
      dst[i] = if cond[i] != 0: srcX[i] else: srcY[i]
  of dtInt32:
    let srcX = x.asInt32
    let srcY = y.asInt32
    var dst = result.asInt32
    for i in 0 ..< n:
      dst[i] = if cond[i] != 0: srcX[i] else: srcY[i]
  of dtInt64:
    let srcX = x.asInt64
    let srcY = y.asInt64
    var dst = result.asInt64
    for i in 0 ..< n:
      dst[i] = if cond[i] != 0: srcX[i] else: srcY[i]
  else:
    raise newException(ValueError, "Unsupported dtype for where: " & $x.dtype)

proc whereScalarKernel*(condition: TensorData, xScalar: float64, yScalar: float64, dtype: DType): TensorData =
  ## Element-wise selection with scalar values
  result = newTensorData(condition.shape, dtype)
  let n = condition.shape.size
  let cond = cast[ptr UncheckedArray[uint8]](addr condition.data[0])

  case dtype
  of dtFloat32:
    var dst = result.asFloat32
    let x = xScalar.float32
    let y = yScalar.float32
    for i in 0 ..< n:
      dst[i] = if cond[i] != 0: x else: y
  of dtFloat64:
    var dst = result.asFloat64
    for i in 0 ..< n:
      dst[i] = if cond[i] != 0: xScalar else: yScalar
  else:
    raise newException(ValueError, "Unsupported dtype for whereScalar: " & $dtype)

# =============================================================================
# Clamp Operations
# =============================================================================

proc clampKernel*(input: TensorData, minVal: float64, maxVal: float64): TensorData =
  ## Clamp values to [min, max] range
  result = newTensorData(input.shape, input.dtype)
  let n = input.shape.size

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var dst = result.asFloat32
    let minF = minVal.float32
    let maxF = maxVal.float32
    for i in 0 ..< n:
      dst[i] = max(minF, min(maxF, src[i]))
  of dtFloat64:
    let src = input.asFloat64
    var dst = result.asFloat64
    for i in 0 ..< n:
      dst[i] = max(minVal, min(maxVal, src[i]))
  of dtInt32:
    let src = input.asInt32
    var dst = result.asInt32
    let minI = minVal.int32
    let maxI = maxVal.int32
    for i in 0 ..< n:
      dst[i] = max(minI, min(maxI, src[i]))
  of dtInt64:
    let src = input.asInt64
    var dst = result.asInt64
    let minI = minVal.int64
    let maxI = maxVal.int64
    for i in 0 ..< n:
      dst[i] = max(minI, min(maxI, src[i]))
  else:
    raise newException(ValueError, "Unsupported dtype for clamp: " & $input.dtype)

proc clampMinKernel*(input: TensorData, minVal: float64): TensorData =
  ## Clamp to minimum value (no upper bound)
  result = newTensorData(input.shape, input.dtype)
  let n = input.shape.size

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var dst = result.asFloat32
    let minF = minVal.float32
    for i in 0 ..< n:
      dst[i] = max(minF, src[i])
  of dtFloat64:
    let src = input.asFloat64
    var dst = result.asFloat64
    for i in 0 ..< n:
      dst[i] = max(minVal, src[i])
  else:
    raise newException(ValueError, "Unsupported dtype for clampMin: " & $input.dtype)

proc clampMaxKernel*(input: TensorData, maxVal: float64): TensorData =
  ## Clamp to maximum value (no lower bound)
  result = newTensorData(input.shape, input.dtype)
  let n = input.shape.size

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var dst = result.asFloat32
    let maxF = maxVal.float32
    for i in 0 ..< n:
      dst[i] = min(maxF, src[i])
  of dtFloat64:
    let src = input.asFloat64
    var dst = result.asFloat64
    for i in 0 ..< n:
      dst[i] = min(maxVal, src[i])
  else:
    raise newException(ValueError, "Unsupported dtype for clampMax: " & $input.dtype)

# =============================================================================
# Masked Operations
# =============================================================================

proc maskedFillKernel*(input: TensorData, mask: TensorData, value: float64): TensorData =
  ## Fill elements where mask is true with value
  assert input.shape == mask.shape, "Shapes must match"
  result = newTensorData(input.shape, input.dtype)
  let n = input.shape.size
  let maskData = cast[ptr UncheckedArray[uint8]](addr mask.data[0])

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var dst = result.asFloat32
    let v = value.float32
    for i in 0 ..< n:
      dst[i] = if maskData[i] != 0: v else: src[i]
  of dtFloat64:
    let src = input.asFloat64
    var dst = result.asFloat64
    for i in 0 ..< n:
      dst[i] = if maskData[i] != 0: value else: src[i]
  else:
    raise newException(ValueError, "Unsupported dtype for maskedFill: " & $input.dtype)

proc applyMaskKernel*(input: TensorData, mask: TensorData): TensorData =
  ## Apply boolean mask: keep values where mask is true, zero elsewhere
  assert input.shape == mask.shape, "Shapes must match"
  result = newTensorData(input.shape, input.dtype)
  let n = input.shape.size
  let maskData = cast[ptr UncheckedArray[uint8]](addr mask.data[0])

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var dst = result.asFloat32
    for i in 0 ..< n:
      dst[i] = if maskData[i] != 0: src[i] else: 0.0'f32
  of dtFloat64:
    let src = input.asFloat64
    var dst = result.asFloat64
    for i in 0 ..< n:
      dst[i] = if maskData[i] != 0: src[i] else: 0.0
  else:
    raise newException(ValueError, "Unsupported dtype for applyMask: " & $input.dtype)
