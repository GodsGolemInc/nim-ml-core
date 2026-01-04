## CPU kernels for matrix multiplication operations
##
## Provides mm, bmm, matmul, mv, dot, outer.

import ../../[dtype, shape, tensor]

# Forward declarations
proc dotKernel*(a: TensorData, b: TensorData): TensorData

# =============================================================================
# Basic Matrix Multiplication
# =============================================================================

proc mmKernel*(a: TensorData, b: TensorData): TensorData =
  ## Matrix multiplication for 2D tensors: C = A @ B
  ## A: (M, K), B: (K, N) -> C: (M, N)
  assert a.shape.rank == 2 and b.shape.rank == 2, "mm requires 2D tensors"
  assert a.dtype == b.dtype, "tensors must have same dtype"
  assert a.shape[1] == b.shape[0], "inner dimensions must match"

  let m = a.shape[0]
  let k = a.shape[1]
  let n = b.shape[1]

  result = newTensorData(newShape(m, n), a.dtype)

  case a.dtype
  of dtFloat32:
    let srcA = a.asFloat32
    let srcB = b.asFloat32
    var dst = result.asFloat32

    for i in 0 ..< m:
      for j in 0 ..< n:
        var sum: float32 = 0.0
        for l in 0 ..< k:
          sum += srcA[i * k + l] * srcB[l * n + j]
        dst[i * n + j] = sum

  of dtFloat64:
    let srcA = a.asFloat64
    let srcB = b.asFloat64
    var dst = result.asFloat64

    for i in 0 ..< m:
      for j in 0 ..< n:
        var sum: float64 = 0.0
        for l in 0 ..< k:
          sum += srcA[i * k + l] * srcB[l * n + j]
        dst[i * n + j] = sum

  else:
    raise newException(ValueError, "Unsupported dtype for mm: " & $a.dtype)

proc bmmKernel*(a: TensorData, b: TensorData): TensorData =
  ## Batched matrix multiplication for 3D tensors: C = A @ B
  ## A: (B, M, K), B: (B, K, N) -> C: (B, M, N)
  assert a.shape.rank == 3 and b.shape.rank == 3, "bmm requires 3D tensors"
  assert a.dtype == b.dtype, "tensors must have same dtype"
  assert a.shape[0] == b.shape[0], "batch dimensions must match"
  assert a.shape[2] == b.shape[1], "inner dimensions must match"

  let batch = a.shape[0]
  let m = a.shape[1]
  let k = a.shape[2]
  let n = b.shape[2]

  result = newTensorData(newShape(batch, m, n), a.dtype)

  case a.dtype
  of dtFloat32:
    let srcA = a.asFloat32
    let srcB = b.asFloat32
    var dst = result.asFloat32

    for bi in 0 ..< batch:
      for i in 0 ..< m:
        for j in 0 ..< n:
          var sum: float32 = 0.0
          for l in 0 ..< k:
            sum += srcA[bi * m * k + i * k + l] * srcB[bi * k * n + l * n + j]
          dst[bi * m * n + i * n + j] = sum

  of dtFloat64:
    let srcA = a.asFloat64
    let srcB = b.asFloat64
    var dst = result.asFloat64

    for bi in 0 ..< batch:
      for i in 0 ..< m:
        for j in 0 ..< n:
          var sum: float64 = 0.0
          for l in 0 ..< k:
            sum += srcA[bi * m * k + i * k + l] * srcB[bi * k * n + l * n + j]
          dst[bi * m * n + i * n + j] = sum

  else:
    raise newException(ValueError, "Unsupported dtype for bmm: " & $a.dtype)

# =============================================================================
# General Matrix Multiplication (supports broadcasting)
# =============================================================================

proc matmulKernel*(a: TensorData, b: TensorData): TensorData =
  ## General matrix multiplication with broadcasting support
  ## Handles 1D, 2D, and higher dimensional inputs
  assert a.dtype == b.dtype, "tensors must have same dtype"

  let ndimA = a.shape.rank
  let ndimB = b.shape.rank

  # Handle 1D cases
  if ndimA == 1 and ndimB == 1:
    # Dot product
    return dotKernel(a, b)

  if ndimA == 1:
    # (K,) @ (..., K, N) -> (..., N)
    # Treat as (1, K) @ (..., K, N) -> (..., 1, N), then squeeze
    let aExpanded = newTensorData(newShape(1, a.shape[0]), a.dtype)
    copyMem(addr aExpanded.data[0], addr a.data[0], a.byteSize)
    let result2d = matmulKernel(aExpanded, b)
    # Squeeze the second-to-last dimension
    var outDims: seq[int] = @[]
    for i in 0 ..< result2d.shape.rank - 2:
      outDims.add(result2d.shape[i])
    outDims.add(result2d.shape[result2d.shape.rank - 1])
    result = newTensorData(newShape(outDims), result2d.dtype)
    copyMem(addr result.data[0], addr result2d.data[0], result2d.byteSize)
    return result

  if ndimB == 1:
    # (..., M, K) @ (K,) -> (..., M)
    # Treat as (..., M, K) @ (K, 1) -> (..., M, 1), then squeeze
    let bExpanded = newTensorData(newShape(b.shape[0], 1), b.dtype)
    copyMem(addr bExpanded.data[0], addr b.data[0], b.byteSize)
    let result2d = matmulKernel(a, bExpanded)
    # Squeeze the last dimension
    var outDims: seq[int] = @[]
    for i in 0 ..< result2d.shape.rank - 1:
      outDims.add(result2d.shape[i])
    result = newTensorData(newShape(outDims), result2d.dtype)
    copyMem(addr result.data[0], addr result2d.data[0], result2d.byteSize)
    return result

  # Both are at least 2D
  if ndimA == 2 and ndimB == 2:
    return mmKernel(a, b)

  if ndimA == 3 and ndimB == 3 and a.shape[0] == b.shape[0]:
    return bmmKernel(a, b)

  # General case with broadcasting for batch dimensions
  assert a.shape[ndimA - 1] == b.shape[ndimB - 2],
    "matrix dimensions incompatible for matmul"

  let m = a.shape[ndimA - 2]
  let k = a.shape[ndimA - 1]
  let n = b.shape[ndimB - 1]

  # Compute broadcast batch shape
  let batchDimsA = if ndimA > 2: ndimA - 2 else: 0
  let batchDimsB = if ndimB > 2: ndimB - 2 else: 0
  let maxBatchDims = max(batchDimsA, batchDimsB)

  var outBatchDims: seq[int] = @[]
  for i in 0 ..< maxBatchDims:
    let dimA = if i < maxBatchDims - batchDimsA: 1 else: a.shape[i - (maxBatchDims - batchDimsA)]
    let dimB = if i < maxBatchDims - batchDimsB: 1 else: b.shape[i - (maxBatchDims - batchDimsB)]
    if dimA == 1:
      outBatchDims.add(dimB)
    elif dimB == 1:
      outBatchDims.add(dimA)
    else:
      assert dimA == dimB, "batch dimensions must be broadcastable"
      outBatchDims.add(dimA)

  var outDims = outBatchDims
  outDims.add(m)
  outDims.add(n)

  result = newTensorData(newShape(outDims), a.dtype)

  # Total number of matrix multiplications
  var totalBatch = 1
  for d in outBatchDims:
    totalBatch *= d

  case a.dtype
  of dtFloat32:
    let srcA = a.asFloat32
    let srcB = b.asFloat32
    var dst = result.asFloat32

    for batchIdx in 0 ..< totalBatch:
      # Compute batch coordinates
      var batchCoords: seq[int] = newSeq[int](maxBatchDims)
      var remaining = batchIdx
      for i in countdown(maxBatchDims - 1, 0):
        batchCoords[i] = remaining mod outBatchDims[i]
        remaining = remaining div outBatchDims[i]

      # Compute offsets in A and B
      var offsetA = 0
      var strideA = m * k
      for i in 0 ..< batchDimsA:
        let dimIdx = i + (maxBatchDims - batchDimsA)
        let coord = batchCoords[dimIdx] mod a.shape[i]
        for j in (i + 1) ..< batchDimsA:
          strideA *= a.shape[j]
        offsetA += coord * strideA
        strideA = m * k

      var offsetB = 0
      var strideB = k * n
      for i in 0 ..< batchDimsB:
        let dimIdx = i + (maxBatchDims - batchDimsB)
        let coord = batchCoords[dimIdx] mod b.shape[i]
        for j in (i + 1) ..< batchDimsB:
          strideB *= b.shape[j]
        offsetB += coord * strideB
        strideB = k * n

      let offsetOut = batchIdx * m * n

      # Perform matrix multiplication for this batch
      for i in 0 ..< m:
        for j in 0 ..< n:
          var sum: float32 = 0.0
          for l in 0 ..< k:
            sum += srcA[offsetA + i * k + l] * srcB[offsetB + l * n + j]
          dst[offsetOut + i * n + j] = sum

  of dtFloat64:
    let srcA = a.asFloat64
    let srcB = b.asFloat64
    var dst = result.asFloat64

    for batchIdx in 0 ..< totalBatch:
      var batchCoords: seq[int] = newSeq[int](maxBatchDims)
      var remaining = batchIdx
      for i in countdown(maxBatchDims - 1, 0):
        batchCoords[i] = remaining mod outBatchDims[i]
        remaining = remaining div outBatchDims[i]

      var offsetA = 0
      var strideA = m * k
      for i in 0 ..< batchDimsA:
        let dimIdx = i + (maxBatchDims - batchDimsA)
        let coord = batchCoords[dimIdx] mod a.shape[i]
        for j in (i + 1) ..< batchDimsA:
          strideA *= a.shape[j]
        offsetA += coord * strideA
        strideA = m * k

      var offsetB = 0
      var strideB = k * n
      for i in 0 ..< batchDimsB:
        let dimIdx = i + (maxBatchDims - batchDimsB)
        let coord = batchCoords[dimIdx] mod b.shape[i]
        for j in (i + 1) ..< batchDimsB:
          strideB *= b.shape[j]
        offsetB += coord * strideB
        strideB = k * n

      let offsetOut = batchIdx * m * n

      for i in 0 ..< m:
        for j in 0 ..< n:
          var sum: float64 = 0.0
          for l in 0 ..< k:
            sum += srcA[offsetA + i * k + l] * srcB[offsetB + l * n + j]
          dst[offsetOut + i * n + j] = sum

  else:
    raise newException(ValueError, "Unsupported dtype for matmul: " & $a.dtype)

# =============================================================================
# Vector Operations
# =============================================================================

proc mvKernel*(matrix: TensorData, vector: TensorData): TensorData =
  ## Matrix-vector multiplication: y = A @ x
  ## A: (M, N), x: (N,) -> y: (M,)
  assert matrix.shape.rank == 2 and vector.shape.rank == 1, "mv requires 2D matrix and 1D vector"
  assert matrix.dtype == vector.dtype, "tensors must have same dtype"
  assert matrix.shape[1] == vector.shape[0], "dimensions must match"

  let m = matrix.shape[0]
  let n = matrix.shape[1]

  result = newTensorData(newShape(m), matrix.dtype)

  case matrix.dtype
  of dtFloat32:
    let srcA = matrix.asFloat32
    let srcX = vector.asFloat32
    var dst = result.asFloat32

    for i in 0 ..< m:
      var sum: float32 = 0.0
      for j in 0 ..< n:
        sum += srcA[i * n + j] * srcX[j]
      dst[i] = sum

  of dtFloat64:
    let srcA = matrix.asFloat64
    let srcX = vector.asFloat64
    var dst = result.asFloat64

    for i in 0 ..< m:
      var sum: float64 = 0.0
      for j in 0 ..< n:
        sum += srcA[i * n + j] * srcX[j]
      dst[i] = sum

  else:
    raise newException(ValueError, "Unsupported dtype for mv: " & $matrix.dtype)

proc dotKernel*(a: TensorData, b: TensorData): TensorData =
  ## Dot product of two 1D tensors
  assert a.shape.rank == 1 and b.shape.rank == 1, "dot requires 1D tensors"
  assert a.dtype == b.dtype, "tensors must have same dtype"
  assert a.shape[0] == b.shape[0], "tensors must have same length"

  let n = a.shape[0]
  result = newTensorData(newShape(), a.dtype)  # Scalar

  case a.dtype
  of dtFloat32:
    let srcA = a.asFloat32
    let srcB = b.asFloat32
    var sum: float32 = 0.0
    for i in 0 ..< n:
      sum += srcA[i] * srcB[i]
    result.asFloat32[0] = sum

  of dtFloat64:
    let srcA = a.asFloat64
    let srcB = b.asFloat64
    var sum: float64 = 0.0
    for i in 0 ..< n:
      sum += srcA[i] * srcB[i]
    result.asFloat64[0] = sum

  else:
    raise newException(ValueError, "Unsupported dtype for dot: " & $a.dtype)

proc outerKernel*(a: TensorData, b: TensorData): TensorData =
  ## Outer product of two 1D tensors: C[i,j] = a[i] * b[j]
  assert a.shape.rank == 1 and b.shape.rank == 1, "outer requires 1D tensors"
  assert a.dtype == b.dtype, "tensors must have same dtype"

  let m = a.shape[0]
  let n = b.shape[0]

  result = newTensorData(newShape(m, n), a.dtype)

  case a.dtype
  of dtFloat32:
    let srcA = a.asFloat32
    let srcB = b.asFloat32
    var dst = result.asFloat32

    for i in 0 ..< m:
      for j in 0 ..< n:
        dst[i * n + j] = srcA[i] * srcB[j]

  of dtFloat64:
    let srcA = a.asFloat64
    let srcB = b.asFloat64
    var dst = result.asFloat64

    for i in 0 ..< m:
      for j in 0 ..< n:
        dst[i * n + j] = srcA[i] * srcB[j]

  else:
    raise newException(ValueError, "Unsupported dtype for outer: " & $a.dtype)

# =============================================================================
# Linear Algebra Operations
# =============================================================================

proc traceKernel*(input: TensorData): TensorData =
  ## Compute trace (sum of diagonal elements) of a matrix
  assert input.shape.rank == 2, "trace requires 2D tensor"

  let m = input.shape[0]
  let n = input.shape[1]
  let diagLen = min(m, n)

  result = newTensorData(newShape(), input.dtype)  # Scalar

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var sum: float32 = 0.0
    for i in 0 ..< diagLen:
      sum += src[i * n + i]
    result.asFloat32[0] = sum

  of dtFloat64:
    let src = input.asFloat64
    var sum: float64 = 0.0
    for i in 0 ..< diagLen:
      sum += src[i * n + i]
    result.asFloat64[0] = sum

  else:
    raise newException(ValueError, "Unsupported dtype for trace: " & $input.dtype)

proc diagKernel*(input: TensorData, diagonal: int = 0): TensorData =
  ## Extract diagonal from matrix or construct diagonal matrix from 1D tensor
  if input.shape.rank == 1:
    # Construct diagonal matrix
    let n = input.shape[0]
    let size = n + abs(diagonal)
    result = newTensorData(newShape(size, size), input.dtype)

    let elemSize = dtypeSize(input.dtype)
    for i in 0 ..< n:
      let row = if diagonal >= 0: i else: i - diagonal
      let col = if diagonal >= 0: i + diagonal else: i
      copyMem(addr result.data[(row * size + col) * elemSize],
              addr input.data[i * elemSize], elemSize)

  elif input.shape.rank == 2:
    # Extract diagonal
    let m = input.shape[0]
    let n = input.shape[1]

    let startRow = if diagonal >= 0: 0 else: -diagonal
    let startCol = if diagonal >= 0: diagonal else: 0
    let diagLen = min(m - startRow, n - startCol)

    if diagLen <= 0:
      result = newTensorData(newShape(0), input.dtype)
      return

    result = newTensorData(newShape(diagLen), input.dtype)
    let elemSize = dtypeSize(input.dtype)

    for i in 0 ..< diagLen:
      let row = startRow + i
      let col = startCol + i
      copyMem(addr result.data[i * elemSize],
              addr input.data[(row * n + col) * elemSize], elemSize)

  else:
    raise newException(ValueError, "diag requires 1D or 2D tensor")

proc triuKernel*(input: TensorData, diagonal: int = 0): TensorData =
  ## Upper triangular part of a matrix
  assert input.shape.rank == 2, "triu requires 2D tensor"

  let m = input.shape[0]
  let n = input.shape[1]

  result = newTensorData(input.shape, input.dtype)

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var dst = result.asFloat32
    for i in 0 ..< m:
      for j in 0 ..< n:
        if j >= i + diagonal:
          dst[i * n + j] = src[i * n + j]
        else:
          dst[i * n + j] = 0.0'f32

  of dtFloat64:
    let src = input.asFloat64
    var dst = result.asFloat64
    for i in 0 ..< m:
      for j in 0 ..< n:
        if j >= i + diagonal:
          dst[i * n + j] = src[i * n + j]
        else:
          dst[i * n + j] = 0.0

  else:
    raise newException(ValueError, "Unsupported dtype for triu: " & $input.dtype)

proc trilKernel*(input: TensorData, diagonal: int = 0): TensorData =
  ## Lower triangular part of a matrix
  assert input.shape.rank == 2, "tril requires 2D tensor"

  let m = input.shape[0]
  let n = input.shape[1]

  result = newTensorData(input.shape, input.dtype)

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var dst = result.asFloat32
    for i in 0 ..< m:
      for j in 0 ..< n:
        if j <= i + diagonal:
          dst[i * n + j] = src[i * n + j]
        else:
          dst[i * n + j] = 0.0'f32

  of dtFloat64:
    let src = input.asFloat64
    var dst = result.asFloat64
    for i in 0 ..< m:
      for j in 0 ..< n:
        if j <= i + diagonal:
          dst[i * n + j] = src[i * n + j]
        else:
          dst[i * n + j] = 0.0

  else:
    raise newException(ValueError, "Unsupported dtype for tril: " & $input.dtype)
