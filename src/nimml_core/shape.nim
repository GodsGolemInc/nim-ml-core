## Shape module for multi-dimensional tensor shapes
##
## Provides Shape type with broadcasting semantics, stride computation,
## and contiguity checks.

import std/[sequtils, strutils, algorithm]

type
  Shape* = object
    ## Multi-dimensional shape for tensors
    dims*: seq[int]

  MemoryLayout* = enum
    ## Memory layout for tensors
    mlRowMajor    # C-style, last dimension varies fastest
    mlColumnMajor # Fortran-style, first dimension varies fastest

  ShapeError* = object of CatchableError

proc newShape*(dims: varargs[int]): Shape =
  ## Create a new shape from dimensions
  for d in dims:
    if d < 0:
      raise newException(ShapeError, "Shape dimensions must be non-negative")
  Shape(dims: @dims)

proc newShape*(dims: seq[int]): Shape =
  ## Create a new shape from a sequence of dimensions
  for d in dims:
    if d < 0:
      raise newException(ShapeError, "Shape dimensions must be non-negative")
  Shape(dims: dims)

proc rank*(s: Shape): int =
  ## Returns the number of dimensions (rank) of the shape
  s.dims.len

proc size*(s: Shape): int =
  ## Returns the total number of elements
  if s.dims.len == 0:
    return 1  # Scalar
  result = 1
  for d in s.dims:
    result *= d

proc `[]`*(s: Shape, i: int): int =
  ## Get dimension at index (supports negative indexing)
  let idx = if i < 0: s.dims.len + i else: i
  if idx < 0 or idx >= s.dims.len:
    raise newException(IndexDefect, "Shape index out of bounds: " & $i)
  s.dims[idx]

proc `[]`*(s: Shape, slice: HSlice[int, int]): seq[int] =
  ## Get a slice of dimensions
  s.dims[slice]

proc len*(s: Shape): int =
  ## Returns the number of dimensions (alias for rank)
  s.dims.len

proc `==`*(a, b: Shape): bool =
  ## Check if two shapes are equal
  a.dims == b.dims

proc `$`*(s: Shape): string =
  ## String representation of shape
  "(" & s.dims.join(", ") & ")"

proc strides*(s: Shape, layout: MemoryLayout = mlRowMajor): seq[int] =
  ## Compute strides for the shape in the given memory layout.
  ## Strides are in number of elements (not bytes).
  if s.dims.len == 0:
    return @[]

  result = newSeq[int](s.dims.len)

  case layout
  of mlRowMajor:
    # Last dimension has stride 1
    result[^1] = 1
    for i in countdown(s.dims.len - 2, 0):
      result[i] = result[i + 1] * s.dims[i + 1]
  of mlColumnMajor:
    # First dimension has stride 1
    result[0] = 1
    for i in 1 ..< s.dims.len:
      result[i] = result[i - 1] * s.dims[i - 1]

proc isContiguous*(s: Shape, actualStrides: seq[int],
                   layout: MemoryLayout = mlRowMajor): bool =
  ## Check if tensor with given strides is contiguous in memory
  if s.dims.len == 0:
    return true
  if actualStrides.len != s.dims.len:
    return false

  let expectedStrides = s.strides(layout)
  for i in 0 ..< s.dims.len:
    # Allow stride mismatch for dimensions of size 1
    if s.dims[i] > 1 and actualStrides[i] != expectedStrides[i]:
      return false
  true

proc isScalar*(s: Shape): bool =
  ## Check if shape represents a scalar (0-dimensional)
  s.dims.len == 0

proc isVector*(s: Shape): bool =
  ## Check if shape represents a vector (1-dimensional)
  s.dims.len == 1

proc isMatrix*(s: Shape): bool =
  ## Check if shape represents a matrix (2-dimensional)
  s.dims.len == 2

# Broadcasting

proc broadcastable*(a, b: Shape): bool =
  ## Check if two shapes can be broadcast together.
  ## Broadcasting rules:
  ## 1. Shapes are aligned from the right
  ## 2. Dimensions are compatible if equal or one is 1
  let maxRank = max(a.rank, b.rank)

  for i in 1 .. maxRank:
    let dimA = if i <= a.rank: a.dims[a.rank - i] else: 1
    let dimB = if i <= b.rank: b.dims[b.rank - i] else: 1

    if dimA != dimB and dimA != 1 and dimB != 1:
      return false

  true

proc broadcast*(a, b: Shape): Shape =
  ## Compute the result shape of broadcasting a and b.
  ## Raises ShapeError if shapes are not broadcastable.
  if not broadcastable(a, b):
    raise newException(ShapeError,
      "Shapes " & $a & " and " & $b & " are not broadcastable")

  let maxRank = max(a.rank, b.rank)
  var resultDims = newSeq[int](maxRank)

  for i in 1 .. maxRank:
    let dimA = if i <= a.rank: a.dims[a.rank - i] else: 1
    let dimB = if i <= b.rank: b.dims[b.rank - i] else: 1
    resultDims[maxRank - i] = max(dimA, dimB)

  newShape(resultDims)

proc broadcastTo*(s: Shape, target: Shape): Shape =
  ## Broadcast shape s to target shape.
  ## Raises ShapeError if not possible.
  if not broadcastable(s, target):
    raise newException(ShapeError,
      "Cannot broadcast " & $s & " to " & $target)

  # Verify target is the correct broadcast result
  let broadcasted = broadcast(s, target)
  if broadcasted != target:
    raise newException(ShapeError,
      "Cannot broadcast " & $s & " to " & $target & ", result would be " & $broadcasted)

  target

# Shape manipulation

proc squeeze*(s: Shape, dim: int = -1): Shape =
  ## Remove dimensions of size 1.
  ## If dim is specified, only remove that dimension if it's size 1.
  if dim >= 0:
    let idx = if dim < 0: s.dims.len + dim else: dim
    if idx < 0 or idx >= s.dims.len:
      raise newException(ShapeError, "Dimension out of range: " & $dim)
    if s.dims[idx] != 1:
      return s  # Don't squeeze if not size 1
    var newDims = s.dims
    newDims.delete(idx)
    return newShape(newDims)
  else:
    var newDims: seq[int] = @[]
    for d in s.dims:
      if d != 1:
        newDims.add(d)
    if newDims.len == 0:
      return newShape()  # Scalar
    return newShape(newDims)

proc unsqueeze*(s: Shape, dim: int): Shape =
  ## Add a dimension of size 1 at the specified position.
  var idx = dim
  if idx < 0:
    idx = s.dims.len + 1 + idx
  if idx < 0 or idx > s.dims.len:
    raise newException(ShapeError, "Dimension out of range: " & $dim)

  var newDims = s.dims
  newDims.insert(1, idx)
  newShape(newDims)

proc reshape*(s: Shape, newDims: varargs[int]): Shape =
  ## Reshape to new dimensions.
  ## One dimension can be -1 to infer from others.
  var dims = @newDims
  var inferIdx = -1
  var knownSize = 1

  for i, d in dims:
    if d == -1:
      if inferIdx >= 0:
        raise newException(ShapeError, "Can only have one inferred dimension (-1)")
      inferIdx = i
    elif d < 0:
      raise newException(ShapeError, "Invalid dimension: " & $d)
    else:
      knownSize *= d

  if inferIdx >= 0:
    if s.size mod knownSize != 0:
      raise newException(ShapeError,
        "Cannot reshape " & $s & " to shape with inferred dimension")
    dims[inferIdx] = s.size div knownSize

  let newShape = newShape(dims)
  if newShape.size != s.size:
    raise newException(ShapeError,
      "Cannot reshape " & $s & " (size " & $s.size & ") to " & $newShape & " (size " & $newShape.size & ")")

  newShape

proc transpose*(s: Shape, perm: seq[int]): Shape =
  ## Transpose dimensions according to permutation.
  if perm.len != s.dims.len:
    raise newException(ShapeError,
      "Permutation length must match number of dimensions")

  # Verify permutation is valid
  var seen = newSeq[bool](s.dims.len)
  for p in perm:
    if p < 0 or p >= s.dims.len:
      raise newException(ShapeError, "Invalid permutation index: " & $p)
    if seen[p]:
      raise newException(ShapeError, "Duplicate in permutation: " & $p)
    seen[p] = true

  var newDims = newSeq[int](s.dims.len)
  for i, p in perm:
    newDims[i] = s.dims[p]

  newShape(newDims)

proc transpose*(s: Shape): Shape =
  ## Transpose a matrix (swap last two dimensions).
  if s.dims.len < 2:
    raise newException(ShapeError, "Cannot transpose shape with less than 2 dimensions")

  var perm = toSeq(0 ..< s.dims.len)
  swap(perm[^1], perm[^2])
  s.transpose(perm)

proc flatten*(s: Shape, startDim: int = 0, endDim: int = -1): Shape =
  ## Flatten dimensions from startDim to endDim into a single dimension.
  let start = if startDim < 0: s.dims.len + startDim else: startDim
  var `end` = if endDim < 0: s.dims.len + endDim else: endDim

  if start < 0 or start >= s.dims.len:
    raise newException(ShapeError, "Start dimension out of range")
  if `end` < 0 or `end` >= s.dims.len:
    raise newException(ShapeError, "End dimension out of range")
  if start > `end`:
    raise newException(ShapeError, "Start must be <= end")

  var flattenedSize = 1
  for i in start .. `end`:
    flattenedSize *= s.dims[i]

  var newDims: seq[int] = @[]
  for i in 0 ..< start:
    newDims.add(s.dims[i])
  newDims.add(flattenedSize)
  for i in (`end` + 1) ..< s.dims.len:
    newDims.add(s.dims[i])

  newShape(newDims)

# Utility functions

proc matmulShape*(a, b: Shape): Shape =
  ## Compute result shape for matrix multiplication.
  ## Supports batched matmul with broadcasting.
  if a.rank < 1 or b.rank < 1:
    raise newException(ShapeError, "Matmul requires at least 1D tensors")

  # Get matrix dimensions
  let (m, k1) = if a.rank == 1: (1, a.dims[0]) else: (a.dims[^2], a.dims[^1])
  let (k2, n) = if b.rank == 1: (b.dims[0], 1) else: (b.dims[^2], b.dims[^1])

  if k1 != k2:
    raise newException(ShapeError,
      "Matmul dimension mismatch: " & $k1 & " vs " & $k2)

  # Handle batch dimensions
  var batchDimsA = if a.rank > 2: a.dims[0 ..< a.rank - 2] else: @[]
  var batchDimsB = if b.rank > 2: b.dims[0 ..< b.rank - 2] else: @[]

  var resultBatch: seq[int]
  if batchDimsA.len > 0 or batchDimsB.len > 0:
    let batchA = if batchDimsA.len > 0: newShape(batchDimsA) else: newShape(1)
    let batchB = if batchDimsB.len > 0: newShape(batchDimsB) else: newShape(1)
    resultBatch = broadcast(batchA, batchB).dims
  else:
    resultBatch = @[]

  # Build result shape
  var resultDims = resultBatch
  if a.rank > 1:
    resultDims.add(m)
  if b.rank > 1:
    resultDims.add(n)

  if resultDims.len == 0:
    return newShape()  # Scalar result (1x1 @ 1x1 style)

  newShape(resultDims)

proc convOutputShape*(inputShape: Shape, kernelSize: (int, int),
                      stride: (int, int) = (1, 1),
                      padding: (int, int) = (0, 0),
                      dilation: (int, int) = (1, 1)): (int, int) =
  ## Compute output height and width for 2D convolution.
  ## Input shape should be (N, C, H, W) or (C, H, W) or (H, W).
  let h = inputShape.dims[^2]
  let w = inputShape.dims[^1]

  let outH = (h + 2 * padding[0] - dilation[0] * (kernelSize[0] - 1) - 1) div stride[0] + 1
  let outW = (w + 2 * padding[1] - dilation[1] * (kernelSize[1] - 1) - 1) div stride[1] + 1

  (outH, outW)
