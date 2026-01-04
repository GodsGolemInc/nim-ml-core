## CPU kernels with broadcasting support
##
## Provides broadcast-aware versions of arithmetic and comparison operations.

import ../../[dtype, shape, tensor]

# =============================================================================
# Broadcasting Utilities
# =============================================================================

proc computeBroadcastShape*(a, b: Shape): Shape =
  ## Compute the output shape after broadcasting two shapes
  let maxNdim = max(a.rank, b.rank)
  var resultDims: seq[int] = @[]

  for i in 0 ..< maxNdim:
    let dimA = if i < maxNdim - a.rank: 1 else: a[i - (maxNdim - a.rank)]
    let dimB = if i < maxNdim - b.rank: 1 else: b[i - (maxNdim - b.rank)]

    if dimA == dimB:
      resultDims.add(dimA)
    elif dimA == 1:
      resultDims.add(dimB)
    elif dimB == 1:
      resultDims.add(dimA)
    else:
      raise newException(ValueError,
        "Shapes " & $a & " and " & $b & " cannot be broadcast together")

  newShape(resultDims)

proc computeBroadcastStrides*(shape: Shape, targetShape: Shape): seq[int] =
  ## Compute strides for broadcasting shape to targetShape
  let ndim = targetShape.rank
  let shapeNdim = shape.rank
  let padLen = ndim - shapeNdim

  # First compute normal strides for the shape
  var baseStrides: seq[int] = @[]
  var stride = 1
  for i in countdown(shapeNdim - 1, 0):
    baseStrides.insert(stride, 0)
    stride *= shape[i]

  # Now create broadcast strides with padding
  result = newSeq[int](ndim)
  for i in 0 ..< ndim:
    if i < padLen:
      result[i] = 0  # Broadcast over this dimension
    else:
      let shapeIdx = i - padLen
      if shape[shapeIdx] == 1:
        result[i] = 0  # Broadcast over this dimension
      else:
        result[i] = baseStrides[shapeIdx]

proc computeOutputStrides*(shape: Shape): seq[int] =
  ## Compute contiguous strides for output shape
  let ndim = shape.rank
  result = newSeq[int](ndim)
  var stride = 1
  for i in countdown(ndim - 1, 0):
    result[i] = stride
    stride *= shape[i]

proc flatIndexFromCoords*(coords: seq[int], strides: seq[int]): int =
  ## Convert multi-dimensional coordinates to flat index
  result = 0
  for i in 0 ..< coords.len:
    result += coords[i] * strides[i]

proc coordsFromFlatIndex*(flatIdx: int, strides: seq[int], shape: Shape): seq[int] =
  ## Convert flat index to multi-dimensional coordinates
  result = newSeq[int](shape.rank)
  var remaining = flatIdx
  for i in 0 ..< shape.rank:
    result[i] = remaining div strides[i]
    remaining = remaining mod strides[i]

# =============================================================================
# Broadcast Binary Operations
# =============================================================================

proc broadcastAddKernel*(a, b: TensorData): TensorData =
  ## Element-wise addition with broadcasting
  assert a.dtype == b.dtype, "tensors must have same dtype"

  let outShape = computeBroadcastShape(a.shape, b.shape)
  let outStrides = computeOutputStrides(outShape)
  let aStrides = computeBroadcastStrides(a.shape, outShape)
  let bStrides = computeBroadcastStrides(b.shape, outShape)

  result = newTensorData(outShape, a.dtype)
  let n = outShape.size

  case a.dtype
  of dtFloat32:
    let srcA = a.asFloat32
    let srcB = b.asFloat32
    var dst = result.asFloat32

    for outIdx in 0 ..< n:
      let coords = coordsFromFlatIndex(outIdx, outStrides, outShape)
      let aIdx = flatIndexFromCoords(coords, aStrides)
      let bIdx = flatIndexFromCoords(coords, bStrides)
      dst[outIdx] = srcA[aIdx] + srcB[bIdx]

  of dtFloat64:
    let srcA = a.asFloat64
    let srcB = b.asFloat64
    var dst = result.asFloat64

    for outIdx in 0 ..< n:
      let coords = coordsFromFlatIndex(outIdx, outStrides, outShape)
      let aIdx = flatIndexFromCoords(coords, aStrides)
      let bIdx = flatIndexFromCoords(coords, bStrides)
      dst[outIdx] = srcA[aIdx] + srcB[bIdx]

  of dtInt32:
    let srcA = a.asInt32
    let srcB = b.asInt32
    var dst = result.asInt32

    for outIdx in 0 ..< n:
      let coords = coordsFromFlatIndex(outIdx, outStrides, outShape)
      let aIdx = flatIndexFromCoords(coords, aStrides)
      let bIdx = flatIndexFromCoords(coords, bStrides)
      dst[outIdx] = srcA[aIdx] + srcB[bIdx]

  else:
    raise newException(ValueError, "Unsupported dtype for broadcastAdd: " & $a.dtype)

proc broadcastSubKernel*(a, b: TensorData): TensorData =
  ## Element-wise subtraction with broadcasting
  assert a.dtype == b.dtype, "tensors must have same dtype"

  let outShape = computeBroadcastShape(a.shape, b.shape)
  let outStrides = computeOutputStrides(outShape)
  let aStrides = computeBroadcastStrides(a.shape, outShape)
  let bStrides = computeBroadcastStrides(b.shape, outShape)

  result = newTensorData(outShape, a.dtype)
  let n = outShape.size

  case a.dtype
  of dtFloat32:
    let srcA = a.asFloat32
    let srcB = b.asFloat32
    var dst = result.asFloat32

    for outIdx in 0 ..< n:
      let coords = coordsFromFlatIndex(outIdx, outStrides, outShape)
      let aIdx = flatIndexFromCoords(coords, aStrides)
      let bIdx = flatIndexFromCoords(coords, bStrides)
      dst[outIdx] = srcA[aIdx] - srcB[bIdx]

  of dtFloat64:
    let srcA = a.asFloat64
    let srcB = b.asFloat64
    var dst = result.asFloat64

    for outIdx in 0 ..< n:
      let coords = coordsFromFlatIndex(outIdx, outStrides, outShape)
      let aIdx = flatIndexFromCoords(coords, aStrides)
      let bIdx = flatIndexFromCoords(coords, bStrides)
      dst[outIdx] = srcA[aIdx] - srcB[bIdx]

  else:
    raise newException(ValueError, "Unsupported dtype for broadcastSub: " & $a.dtype)

proc broadcastMulKernel*(a, b: TensorData): TensorData =
  ## Element-wise multiplication with broadcasting
  assert a.dtype == b.dtype, "tensors must have same dtype"

  let outShape = computeBroadcastShape(a.shape, b.shape)
  let outStrides = computeOutputStrides(outShape)
  let aStrides = computeBroadcastStrides(a.shape, outShape)
  let bStrides = computeBroadcastStrides(b.shape, outShape)

  result = newTensorData(outShape, a.dtype)
  let n = outShape.size

  case a.dtype
  of dtFloat32:
    let srcA = a.asFloat32
    let srcB = b.asFloat32
    var dst = result.asFloat32

    for outIdx in 0 ..< n:
      let coords = coordsFromFlatIndex(outIdx, outStrides, outShape)
      let aIdx = flatIndexFromCoords(coords, aStrides)
      let bIdx = flatIndexFromCoords(coords, bStrides)
      dst[outIdx] = srcA[aIdx] * srcB[bIdx]

  of dtFloat64:
    let srcA = a.asFloat64
    let srcB = b.asFloat64
    var dst = result.asFloat64

    for outIdx in 0 ..< n:
      let coords = coordsFromFlatIndex(outIdx, outStrides, outShape)
      let aIdx = flatIndexFromCoords(coords, aStrides)
      let bIdx = flatIndexFromCoords(coords, bStrides)
      dst[outIdx] = srcA[aIdx] * srcB[bIdx]

  else:
    raise newException(ValueError, "Unsupported dtype for broadcastMul: " & $a.dtype)

proc broadcastDivKernel*(a, b: TensorData): TensorData =
  ## Element-wise division with broadcasting
  assert a.dtype == b.dtype, "tensors must have same dtype"

  let outShape = computeBroadcastShape(a.shape, b.shape)
  let outStrides = computeOutputStrides(outShape)
  let aStrides = computeBroadcastStrides(a.shape, outShape)
  let bStrides = computeBroadcastStrides(b.shape, outShape)

  result = newTensorData(outShape, a.dtype)
  let n = outShape.size

  case a.dtype
  of dtFloat32:
    let srcA = a.asFloat32
    let srcB = b.asFloat32
    var dst = result.asFloat32

    for outIdx in 0 ..< n:
      let coords = coordsFromFlatIndex(outIdx, outStrides, outShape)
      let aIdx = flatIndexFromCoords(coords, aStrides)
      let bIdx = flatIndexFromCoords(coords, bStrides)
      dst[outIdx] = srcA[aIdx] / srcB[bIdx]

  of dtFloat64:
    let srcA = a.asFloat64
    let srcB = b.asFloat64
    var dst = result.asFloat64

    for outIdx in 0 ..< n:
      let coords = coordsFromFlatIndex(outIdx, outStrides, outShape)
      let aIdx = flatIndexFromCoords(coords, aStrides)
      let bIdx = flatIndexFromCoords(coords, bStrides)
      dst[outIdx] = srcA[aIdx] / srcB[bIdx]

  else:
    raise newException(ValueError, "Unsupported dtype for broadcastDiv: " & $a.dtype)

proc broadcastMaxKernel*(a, b: TensorData): TensorData =
  ## Element-wise maximum with broadcasting
  assert a.dtype == b.dtype, "tensors must have same dtype"

  let outShape = computeBroadcastShape(a.shape, b.shape)
  let outStrides = computeOutputStrides(outShape)
  let aStrides = computeBroadcastStrides(a.shape, outShape)
  let bStrides = computeBroadcastStrides(b.shape, outShape)

  result = newTensorData(outShape, a.dtype)
  let n = outShape.size

  case a.dtype
  of dtFloat32:
    let srcA = a.asFloat32
    let srcB = b.asFloat32
    var dst = result.asFloat32

    for outIdx in 0 ..< n:
      let coords = coordsFromFlatIndex(outIdx, outStrides, outShape)
      let aIdx = flatIndexFromCoords(coords, aStrides)
      let bIdx = flatIndexFromCoords(coords, bStrides)
      dst[outIdx] = max(srcA[aIdx], srcB[bIdx])

  of dtFloat64:
    let srcA = a.asFloat64
    let srcB = b.asFloat64
    var dst = result.asFloat64

    for outIdx in 0 ..< n:
      let coords = coordsFromFlatIndex(outIdx, outStrides, outShape)
      let aIdx = flatIndexFromCoords(coords, aStrides)
      let bIdx = flatIndexFromCoords(coords, bStrides)
      dst[outIdx] = max(srcA[aIdx], srcB[bIdx])

  else:
    raise newException(ValueError, "Unsupported dtype for broadcastMax: " & $a.dtype)

proc broadcastMinKernel*(a, b: TensorData): TensorData =
  ## Element-wise minimum with broadcasting
  assert a.dtype == b.dtype, "tensors must have same dtype"

  let outShape = computeBroadcastShape(a.shape, b.shape)
  let outStrides = computeOutputStrides(outShape)
  let aStrides = computeBroadcastStrides(a.shape, outShape)
  let bStrides = computeBroadcastStrides(b.shape, outShape)

  result = newTensorData(outShape, a.dtype)
  let n = outShape.size

  case a.dtype
  of dtFloat32:
    let srcA = a.asFloat32
    let srcB = b.asFloat32
    var dst = result.asFloat32

    for outIdx in 0 ..< n:
      let coords = coordsFromFlatIndex(outIdx, outStrides, outShape)
      let aIdx = flatIndexFromCoords(coords, aStrides)
      let bIdx = flatIndexFromCoords(coords, bStrides)
      dst[outIdx] = min(srcA[aIdx], srcB[bIdx])

  of dtFloat64:
    let srcA = a.asFloat64
    let srcB = b.asFloat64
    var dst = result.asFloat64

    for outIdx in 0 ..< n:
      let coords = coordsFromFlatIndex(outIdx, outStrides, outShape)
      let aIdx = flatIndexFromCoords(coords, aStrides)
      let bIdx = flatIndexFromCoords(coords, bStrides)
      dst[outIdx] = min(srcA[aIdx], srcB[bIdx])

  else:
    raise newException(ValueError, "Unsupported dtype for broadcastMin: " & $a.dtype)

# =============================================================================
# Broadcast Comparison Operations
# =============================================================================

proc broadcastGeKernel*(a, b: TensorData): TensorData =
  ## Element-wise greater-than-or-equal with broadcasting
  assert a.dtype == b.dtype, "tensors must have same dtype"

  let outShape = computeBroadcastShape(a.shape, b.shape)
  let outStrides = computeOutputStrides(outShape)
  let aStrides = computeBroadcastStrides(a.shape, outShape)
  let bStrides = computeBroadcastStrides(b.shape, outShape)

  result = newTensorData(outShape, dtBool)
  let n = outShape.size
  var dst = cast[ptr UncheckedArray[uint8]](addr result.data[0])

  case a.dtype
  of dtFloat32:
    let srcA = a.asFloat32
    let srcB = b.asFloat32

    for outIdx in 0 ..< n:
      let coords = coordsFromFlatIndex(outIdx, outStrides, outShape)
      let aIdx = flatIndexFromCoords(coords, aStrides)
      let bIdx = flatIndexFromCoords(coords, bStrides)
      dst[outIdx] = if srcA[aIdx] >= srcB[bIdx]: 1'u8 else: 0'u8

  of dtFloat64:
    let srcA = a.asFloat64
    let srcB = b.asFloat64

    for outIdx in 0 ..< n:
      let coords = coordsFromFlatIndex(outIdx, outStrides, outShape)
      let aIdx = flatIndexFromCoords(coords, aStrides)
      let bIdx = flatIndexFromCoords(coords, bStrides)
      dst[outIdx] = if srcA[aIdx] >= srcB[bIdx]: 1'u8 else: 0'u8

  else:
    raise newException(ValueError, "Unsupported dtype for broadcastGe: " & $a.dtype)

proc broadcastLeKernel*(a, b: TensorData): TensorData =
  ## Element-wise less-than-or-equal with broadcasting
  assert a.dtype == b.dtype, "tensors must have same dtype"

  let outShape = computeBroadcastShape(a.shape, b.shape)
  let outStrides = computeOutputStrides(outShape)
  let aStrides = computeBroadcastStrides(a.shape, outShape)
  let bStrides = computeBroadcastStrides(b.shape, outShape)

  result = newTensorData(outShape, dtBool)
  let n = outShape.size
  var dst = cast[ptr UncheckedArray[uint8]](addr result.data[0])

  case a.dtype
  of dtFloat32:
    let srcA = a.asFloat32
    let srcB = b.asFloat32

    for outIdx in 0 ..< n:
      let coords = coordsFromFlatIndex(outIdx, outStrides, outShape)
      let aIdx = flatIndexFromCoords(coords, aStrides)
      let bIdx = flatIndexFromCoords(coords, bStrides)
      dst[outIdx] = if srcA[aIdx] <= srcB[bIdx]: 1'u8 else: 0'u8

  of dtFloat64:
    let srcA = a.asFloat64
    let srcB = b.asFloat64

    for outIdx in 0 ..< n:
      let coords = coordsFromFlatIndex(outIdx, outStrides, outShape)
      let aIdx = flatIndexFromCoords(coords, aStrides)
      let bIdx = flatIndexFromCoords(coords, bStrides)
      dst[outIdx] = if srcA[aIdx] <= srcB[bIdx]: 1'u8 else: 0'u8

  else:
    raise newException(ValueError, "Unsupported dtype for broadcastLe: " & $a.dtype)

proc broadcastGtKernel*(a, b: TensorData): TensorData =
  ## Element-wise greater-than with broadcasting
  assert a.dtype == b.dtype, "tensors must have same dtype"

  let outShape = computeBroadcastShape(a.shape, b.shape)
  let outStrides = computeOutputStrides(outShape)
  let aStrides = computeBroadcastStrides(a.shape, outShape)
  let bStrides = computeBroadcastStrides(b.shape, outShape)

  result = newTensorData(outShape, dtBool)
  let n = outShape.size
  var dst = cast[ptr UncheckedArray[uint8]](addr result.data[0])

  case a.dtype
  of dtFloat32:
    let srcA = a.asFloat32
    let srcB = b.asFloat32

    for outIdx in 0 ..< n:
      let coords = coordsFromFlatIndex(outIdx, outStrides, outShape)
      let aIdx = flatIndexFromCoords(coords, aStrides)
      let bIdx = flatIndexFromCoords(coords, bStrides)
      dst[outIdx] = if srcA[aIdx] > srcB[bIdx]: 1'u8 else: 0'u8

  of dtFloat64:
    let srcA = a.asFloat64
    let srcB = b.asFloat64

    for outIdx in 0 ..< n:
      let coords = coordsFromFlatIndex(outIdx, outStrides, outShape)
      let aIdx = flatIndexFromCoords(coords, aStrides)
      let bIdx = flatIndexFromCoords(coords, bStrides)
      dst[outIdx] = if srcA[aIdx] > srcB[bIdx]: 1'u8 else: 0'u8

  else:
    raise newException(ValueError, "Unsupported dtype for broadcastGt: " & $a.dtype)

proc broadcastLtKernel*(a, b: TensorData): TensorData =
  ## Element-wise less-than with broadcasting
  assert a.dtype == b.dtype, "tensors must have same dtype"

  let outShape = computeBroadcastShape(a.shape, b.shape)
  let outStrides = computeOutputStrides(outShape)
  let aStrides = computeBroadcastStrides(a.shape, outShape)
  let bStrides = computeBroadcastStrides(b.shape, outShape)

  result = newTensorData(outShape, dtBool)
  let n = outShape.size
  var dst = cast[ptr UncheckedArray[uint8]](addr result.data[0])

  case a.dtype
  of dtFloat32:
    let srcA = a.asFloat32
    let srcB = b.asFloat32

    for outIdx in 0 ..< n:
      let coords = coordsFromFlatIndex(outIdx, outStrides, outShape)
      let aIdx = flatIndexFromCoords(coords, aStrides)
      let bIdx = flatIndexFromCoords(coords, bStrides)
      dst[outIdx] = if srcA[aIdx] < srcB[bIdx]: 1'u8 else: 0'u8

  of dtFloat64:
    let srcA = a.asFloat64
    let srcB = b.asFloat64

    for outIdx in 0 ..< n:
      let coords = coordsFromFlatIndex(outIdx, outStrides, outShape)
      let aIdx = flatIndexFromCoords(coords, aStrides)
      let bIdx = flatIndexFromCoords(coords, bStrides)
      dst[outIdx] = if srcA[aIdx] < srcB[bIdx]: 1'u8 else: 0'u8

  else:
    raise newException(ValueError, "Unsupported dtype for broadcastLt: " & $a.dtype)

proc broadcastEqKernel*(a, b: TensorData): TensorData =
  ## Element-wise equality with broadcasting
  assert a.dtype == b.dtype, "tensors must have same dtype"

  let outShape = computeBroadcastShape(a.shape, b.shape)
  let outStrides = computeOutputStrides(outShape)
  let aStrides = computeBroadcastStrides(a.shape, outShape)
  let bStrides = computeBroadcastStrides(b.shape, outShape)

  result = newTensorData(outShape, dtBool)
  let n = outShape.size
  var dst = cast[ptr UncheckedArray[uint8]](addr result.data[0])

  case a.dtype
  of dtFloat32:
    let srcA = a.asFloat32
    let srcB = b.asFloat32

    for outIdx in 0 ..< n:
      let coords = coordsFromFlatIndex(outIdx, outStrides, outShape)
      let aIdx = flatIndexFromCoords(coords, aStrides)
      let bIdx = flatIndexFromCoords(coords, bStrides)
      dst[outIdx] = if srcA[aIdx] == srcB[bIdx]: 1'u8 else: 0'u8

  of dtFloat64:
    let srcA = a.asFloat64
    let srcB = b.asFloat64

    for outIdx in 0 ..< n:
      let coords = coordsFromFlatIndex(outIdx, outStrides, outShape)
      let aIdx = flatIndexFromCoords(coords, aStrides)
      let bIdx = flatIndexFromCoords(coords, bStrides)
      dst[outIdx] = if srcA[aIdx] == srcB[bIdx]: 1'u8 else: 0'u8

  else:
    raise newException(ValueError, "Unsupported dtype for broadcastEq: " & $a.dtype)

# =============================================================================
# Broadcast Where Operation
# =============================================================================

proc broadcastWhereKernel*(condition: TensorData, x: TensorData, y: TensorData): TensorData =
  ## Element-wise selection with broadcasting: condition ? x : y
  assert x.dtype == y.dtype, "x and y must have same dtype"
  assert condition.dtype == dtBool, "condition must be boolean"

  # First broadcast condition with x, then with y
  let tempShape = computeBroadcastShape(condition.shape, x.shape)
  let outShape = computeBroadcastShape(tempShape, y.shape)

  let outStrides = computeOutputStrides(outShape)
  let condStrides = computeBroadcastStrides(condition.shape, outShape)
  let xStrides = computeBroadcastStrides(x.shape, outShape)
  let yStrides = computeBroadcastStrides(y.shape, outShape)

  result = newTensorData(outShape, x.dtype)
  let n = outShape.size
  let condData = cast[ptr UncheckedArray[uint8]](addr condition.data[0])

  case x.dtype
  of dtFloat32:
    let srcX = x.asFloat32
    let srcY = y.asFloat32
    var dst = result.asFloat32

    for outIdx in 0 ..< n:
      let coords = coordsFromFlatIndex(outIdx, outStrides, outShape)
      let condIdx = flatIndexFromCoords(coords, condStrides)
      let xIdx = flatIndexFromCoords(coords, xStrides)
      let yIdx = flatIndexFromCoords(coords, yStrides)
      dst[outIdx] = if condData[condIdx] != 0: srcX[xIdx] else: srcY[yIdx]

  of dtFloat64:
    let srcX = x.asFloat64
    let srcY = y.asFloat64
    var dst = result.asFloat64

    for outIdx in 0 ..< n:
      let coords = coordsFromFlatIndex(outIdx, outStrides, outShape)
      let condIdx = flatIndexFromCoords(coords, condStrides)
      let xIdx = flatIndexFromCoords(coords, xStrides)
      let yIdx = flatIndexFromCoords(coords, yStrides)
      dst[outIdx] = if condData[condIdx] != 0: srcX[xIdx] else: srcY[yIdx]

  else:
    raise newException(ValueError, "Unsupported dtype for broadcastWhere: " & $x.dtype)
