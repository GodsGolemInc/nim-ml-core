## CPU kernels for reduction operations
##
## Provides sum, mean, max, min, norm operations.

import std/[math]
import ../../[dtype, shape, tensor]

# =============================================================================
# Full Reduction (to scalar)
# =============================================================================

proc sumKernel*(input: TensorData): TensorData =
  ## Sum all elements
  result = newTensorData(newShape(), input.dtype)  # Scalar output
  let n = input.shape.size

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var total: float32 = 0.0
    for i in 0 ..< n:
      total += src[i]
    result.asFloat32[0] = total
  of dtFloat64:
    let src = input.asFloat64
    var total: float64 = 0.0
    for i in 0 ..< n:
      total += src[i]
    result.asFloat64[0] = total
  of dtInt32:
    let src = input.asInt32
    var total: int32 = 0
    for i in 0 ..< n:
      total += src[i]
    result.asInt32[0] = total
  of dtInt64:
    let src = input.asInt64
    var total: int64 = 0
    for i in 0 ..< n:
      total += src[i]
    result.asInt64[0] = total
  else:
    raise newException(ValueError, "Unsupported dtype for sum: " & $input.dtype)

proc meanKernel*(input: TensorData): TensorData =
  ## Mean of all elements
  result = newTensorData(newShape(), input.dtype)  # Scalar output
  let n = input.shape.size

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var total: float32 = 0.0
    for i in 0 ..< n:
      total += src[i]
    result.asFloat32[0] = total / n.float32
  of dtFloat64:
    let src = input.asFloat64
    var total: float64 = 0.0
    for i in 0 ..< n:
      total += src[i]
    result.asFloat64[0] = total / n.float64
  else:
    raise newException(ValueError, "Unsupported dtype for mean: " & $input.dtype)

proc maxReduceKernel*(input: TensorData): TensorData =
  ## Maximum of all elements
  result = newTensorData(newShape(), input.dtype)  # Scalar output
  let n = input.shape.size
  if n == 0:
    raise newException(ValueError, "Cannot compute max of empty tensor")

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var maxVal = src[0]
    for i in 1 ..< n:
      if src[i] > maxVal:
        maxVal = src[i]
    result.asFloat32[0] = maxVal
  of dtFloat64:
    let src = input.asFloat64
    var maxVal = src[0]
    for i in 1 ..< n:
      if src[i] > maxVal:
        maxVal = src[i]
    result.asFloat64[0] = maxVal
  else:
    raise newException(ValueError, "Unsupported dtype for max: " & $input.dtype)

proc minReduceKernel*(input: TensorData): TensorData =
  ## Minimum of all elements
  result = newTensorData(newShape(), input.dtype)  # Scalar output
  let n = input.shape.size
  if n == 0:
    raise newException(ValueError, "Cannot compute min of empty tensor")

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var minVal = src[0]
    for i in 1 ..< n:
      if src[i] < minVal:
        minVal = src[i]
    result.asFloat32[0] = minVal
  of dtFloat64:
    let src = input.asFloat64
    var minVal = src[0]
    for i in 1 ..< n:
      if src[i] < minVal:
        minVal = src[i]
    result.asFloat64[0] = minVal
  else:
    raise newException(ValueError, "Unsupported dtype for min: " & $input.dtype)

proc prodKernel*(input: TensorData): TensorData =
  ## Product of all elements
  result = newTensorData(newShape(), input.dtype)  # Scalar output
  let n = input.shape.size

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var total: float32 = 1.0
    for i in 0 ..< n:
      total *= src[i]
    result.asFloat32[0] = total
  of dtFloat64:
    let src = input.asFloat64
    var total: float64 = 1.0
    for i in 0 ..< n:
      total *= src[i]
    result.asFloat64[0] = total
  else:
    raise newException(ValueError, "Unsupported dtype for prod: " & $input.dtype)

# =============================================================================
# Norm Operations
# =============================================================================

proc normL1Kernel*(input: TensorData): TensorData =
  ## L1 norm: sum(|x|)
  result = newTensorData(newShape(), input.dtype)
  let n = input.shape.size

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var total: float32 = 0.0
    for i in 0 ..< n:
      total += abs(src[i])
    result.asFloat32[0] = total
  of dtFloat64:
    let src = input.asFloat64
    var total: float64 = 0.0
    for i in 0 ..< n:
      total += abs(src[i])
    result.asFloat64[0] = total
  else:
    raise newException(ValueError, "Unsupported dtype for normL1: " & $input.dtype)

proc normL2Kernel*(input: TensorData): TensorData =
  ## L2 norm: sqrt(sum(x^2))
  result = newTensorData(newShape(), input.dtype)
  let n = input.shape.size

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var total: float32 = 0.0
    for i in 0 ..< n:
      total += src[i] * src[i]
    result.asFloat32[0] = sqrt(total)
  of dtFloat64:
    let src = input.asFloat64
    var total: float64 = 0.0
    for i in 0 ..< n:
      total += src[i] * src[i]
    result.asFloat64[0] = sqrt(total)
  else:
    raise newException(ValueError, "Unsupported dtype for normL2: " & $input.dtype)

proc normInfKernel*(input: TensorData): TensorData =
  ## Infinity norm: max(|x|)
  result = newTensorData(newShape(), input.dtype)
  let n = input.shape.size
  if n == 0:
    raise newException(ValueError, "Cannot compute norm of empty tensor")

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var maxVal = abs(src[0])
    for i in 1 ..< n:
      let v = abs(src[i])
      if v > maxVal:
        maxVal = v
    result.asFloat32[0] = maxVal
  of dtFloat64:
    let src = input.asFloat64
    var maxVal = abs(src[0])
    for i in 1 ..< n:
      let v = abs(src[i])
      if v > maxVal:
        maxVal = v
    result.asFloat64[0] = maxVal
  else:
    raise newException(ValueError, "Unsupported dtype for normInf: " & $input.dtype)

proc normKernel*(input: TensorData, p: float64 = 2.0): TensorData =
  ## Lp norm: (sum(|x|^p))^(1/p)
  if p == 1.0:
    return normL1Kernel(input)
  elif p == 2.0:
    return normL2Kernel(input)
  elif p == Inf:
    return normInfKernel(input)

  result = newTensorData(newShape(), input.dtype)
  let n = input.shape.size

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var total: float32 = 0.0
    for i in 0 ..< n:
      total += pow(abs(src[i]), p.float32)
    result.asFloat32[0] = pow(total, (1.0 / p).float32)
  of dtFloat64:
    let src = input.asFloat64
    var total: float64 = 0.0
    for i in 0 ..< n:
      total += pow(abs(src[i]), p)
    result.asFloat64[0] = pow(total, 1.0 / p)
  else:
    raise newException(ValueError, "Unsupported dtype for norm: " & $input.dtype)

# =============================================================================
# Axis-wise Reduction
# =============================================================================

proc sumAxisKernel*(input: TensorData, axis: int, keepdims: bool = false): TensorData =
  ## Sum along specified axis
  let inShape = input.shape
  let rank = inShape.rank

  # Normalize negative axis
  let actualAxis = if axis < 0: rank + axis else: axis
  assert actualAxis >= 0 and actualAxis < rank, "Invalid axis"

  # Compute output shape
  var outDims: seq[int] = @[]
  for i in 0 ..< rank:
    if i == actualAxis:
      if keepdims:
        outDims.add(1)
    else:
      outDims.add(inShape[i])

  let outShape = if outDims.len == 0: newShape() else: newShape(outDims)
  result = newTensorDataZeros(outShape, input.dtype)

  # Compute strides for iteration
  let axisSize = inShape[actualAxis]
  var outerSize = 1
  var innerSize = 1
  for i in 0 ..< actualAxis:
    outerSize *= inShape[i]
  for i in actualAxis + 1 ..< rank:
    innerSize *= inShape[i]

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var dst = result.asFloat32
    for outer in 0 ..< outerSize:
      for inner in 0 ..< innerSize:
        var total: float32 = 0.0
        for k in 0 ..< axisSize:
          let srcIdx = outer * axisSize * innerSize + k * innerSize + inner
          total += src[srcIdx]
        let dstIdx = outer * innerSize + inner
        dst[dstIdx] = total
  of dtFloat64:
    let src = input.asFloat64
    var dst = result.asFloat64
    for outer in 0 ..< outerSize:
      for inner in 0 ..< innerSize:
        var total: float64 = 0.0
        for k in 0 ..< axisSize:
          let srcIdx = outer * axisSize * innerSize + k * innerSize + inner
          total += src[srcIdx]
        let dstIdx = outer * innerSize + inner
        dst[dstIdx] = total
  else:
    raise newException(ValueError, "Unsupported dtype for sumAxis: " & $input.dtype)

proc meanAxisKernel*(input: TensorData, axis: int, keepdims: bool = false): TensorData =
  ## Mean along specified axis
  let axisSize = input.shape[if axis < 0: input.shape.rank + axis else: axis]
  result = sumAxisKernel(input, axis, keepdims)

  # Divide by axis size
  let n = result.shape.size
  case result.dtype
  of dtFloat32:
    var dst = result.asFloat32
    let divisor = axisSize.float32
    for i in 0 ..< n:
      dst[i] = dst[i] / divisor
  of dtFloat64:
    var dst = result.asFloat64
    let divisor = axisSize.float64
    for i in 0 ..< n:
      dst[i] = dst[i] / divisor
  else:
    discard
