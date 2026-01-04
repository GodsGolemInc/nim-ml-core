## CPU kernels for shape manipulation operations
##
## Provides reshape, transpose, permute, squeeze, unsqueeze, flatten, view.

import ../../[dtype, shape, tensor]

# =============================================================================
# Reshape Operations
# =============================================================================

proc reshapeKernel*(input: TensorData, newShape: Shape): TensorData =
  ## Reshape tensor to new shape (must have same total size)
  assert input.shape.size == newShape.size,
    "Cannot reshape tensor of size " & $input.shape.size & " to shape " & $newShape

  # Create new tensor with same data but different shape
  result = newTensorData(newShape, input.dtype)
  copyMem(addr result.data[0], addr input.data[0], input.byteSize)

proc viewKernel*(input: TensorData, newShape: Shape): TensorData =
  ## View tensor with new shape (alias for reshape, same semantics)
  reshapeKernel(input, newShape)

proc flattenKernel*(input: TensorData, startDim: int = 0, endDim: int = -1): TensorData =
  ## Flatten dimensions from startDim to endDim into a single dimension
  let ndim = input.shape.rank
  let actualEndDim = if endDim < 0: ndim + endDim else: endDim

  assert startDim >= 0 and startDim < ndim, "startDim out of range"
  assert actualEndDim >= startDim and actualEndDim < ndim, "endDim out of range"

  var newDims: seq[int] = @[]

  # Dimensions before startDim
  for i in 0 ..< startDim:
    newDims.add(input.shape[i])

  # Flattened dimension
  var flatSize = 1
  for i in startDim .. actualEndDim:
    flatSize *= input.shape[i]
  newDims.add(flatSize)

  # Dimensions after endDim
  for i in (actualEndDim + 1) ..< ndim:
    newDims.add(input.shape[i])

  reshapeKernel(input, newShape(newDims))

# =============================================================================
# Squeeze/Unsqueeze Operations
# =============================================================================

proc squeezeKernel*(input: TensorData, dim: int = -1): TensorData =
  ## Remove dimensions of size 1
  ## If dim is specified, only squeeze that dimension
  ## If dim is -1, squeeze all dimensions of size 1
  var newDims: seq[int] = @[]

  if dim == -1:
    # Squeeze all dimensions of size 1
    for i in 0 ..< input.shape.rank:
      if input.shape[i] != 1:
        newDims.add(input.shape[i])
    if newDims.len == 0:
      newDims.add(1)  # Keep at least one dimension
  else:
    let actualDim = if dim < 0: input.shape.rank + dim else: dim
    assert actualDim >= 0 and actualDim < input.shape.rank, "dim out of range"

    for i in 0 ..< input.shape.rank:
      if i == actualDim:
        if input.shape[i] != 1:
          newDims.add(input.shape[i])  # Can't squeeze non-1 dimension
      else:
        newDims.add(input.shape[i])

  reshapeKernel(input, newShape(newDims))

proc unsqueezeKernel*(input: TensorData, dim: int): TensorData =
  ## Insert a dimension of size 1 at the specified position
  let actualDim = if dim < 0: input.shape.rank + dim + 1 else: dim
  assert actualDim >= 0 and actualDim <= input.shape.rank, "dim out of range"

  var newDims: seq[int] = @[]
  for i in 0 ..< input.shape.rank:
    if i == actualDim:
      newDims.add(1)
    newDims.add(input.shape[i])

  if actualDim == input.shape.rank:
    newDims.add(1)

  reshapeKernel(input, newShape(newDims))

# =============================================================================
# Transpose Operations
# =============================================================================

proc transposeKernel*(input: TensorData, dim0: int, dim1: int): TensorData =
  ## Swap two dimensions
  let ndim = input.shape.rank
  let d0 = if dim0 < 0: ndim + dim0 else: dim0
  let d1 = if dim1 < 0: ndim + dim1 else: dim1

  assert d0 >= 0 and d0 < ndim, "dim0 out of range"
  assert d1 >= 0 and d1 < ndim, "dim1 out of range"

  if d0 == d1:
    return input.clone()

  # Build new shape
  var newDims: seq[int] = @[]
  for i in 0 ..< ndim:
    if i == d0:
      newDims.add(input.shape[d1])
    elif i == d1:
      newDims.add(input.shape[d0])
    else:
      newDims.add(input.shape[i])

  result = newTensorData(newShape(newDims), input.dtype)

  # Compute strides for input and output
  var inStrides: seq[int] = @[]
  var outStrides: seq[int] = @[]
  var stride = 1
  for i in countdown(ndim - 1, 0):
    inStrides.insert(stride, 0)
    stride *= input.shape[i]

  stride = 1
  for i in countdown(ndim - 1, 0):
    outStrides.insert(stride, 0)
    stride *= newDims[i]

  # Permutation mapping
  var perm: seq[int] = @[]
  for i in 0 ..< ndim:
    if i == d0:
      perm.add(d1)
    elif i == d1:
      perm.add(d0)
    else:
      perm.add(i)

  let n = input.shape.size
  let elemSize = dtypeSize(input.dtype)

  # Copy elements with transposition
  for outIdx in 0 ..< n:
    # Convert flat output index to multi-dimensional
    var outCoords: seq[int] = newSeq[int](ndim)
    var remaining = outIdx
    for i in 0 ..< ndim:
      outCoords[i] = remaining div outStrides[i]
      remaining = remaining mod outStrides[i]

    # Map to input coordinates
    var inCoords: seq[int] = newSeq[int](ndim)
    for i in 0 ..< ndim:
      inCoords[perm[i]] = outCoords[i]

    # Convert input coordinates to flat index
    var inIdx = 0
    for i in 0 ..< ndim:
      inIdx += inCoords[i] * inStrides[i]

    # Copy element
    copyMem(addr result.data[outIdx * elemSize], addr input.data[inIdx * elemSize], elemSize)

proc transpose2DKernel*(input: TensorData): TensorData =
  ## Transpose a 2D tensor (matrix transpose)
  assert input.shape.rank == 2, "transpose2D requires 2D tensor"
  transposeKernel(input, 0, 1)

proc permuteKernel*(input: TensorData, dims: seq[int]): TensorData =
  ## Permute dimensions according to the given order
  let ndim = input.shape.rank
  assert dims.len == ndim, "permutation must have same length as number of dimensions"

  # Validate permutation
  var seen: seq[bool] = newSeq[bool](ndim)
  for d in dims:
    let actualD = if d < 0: ndim + d else: d
    assert actualD >= 0 and actualD < ndim, "dimension out of range"
    assert not seen[actualD], "duplicate dimension in permutation"
    seen[actualD] = true

  # Normalize dims
  var perm: seq[int] = @[]
  for d in dims:
    perm.add(if d < 0: ndim + d else: d)

  # Build new shape
  var newDims: seq[int] = @[]
  for i in 0 ..< ndim:
    newDims.add(input.shape[perm[i]])

  result = newTensorData(newShape(newDims), input.dtype)

  # Compute strides
  var inStrides: seq[int] = @[]
  var outStrides: seq[int] = @[]
  var stride = 1
  for i in countdown(ndim - 1, 0):
    inStrides.insert(stride, 0)
    stride *= input.shape[i]

  stride = 1
  for i in countdown(ndim - 1, 0):
    outStrides.insert(stride, 0)
    stride *= newDims[i]

  let n = input.shape.size
  let elemSize = dtypeSize(input.dtype)

  # Copy elements with permutation
  for outIdx in 0 ..< n:
    var outCoords: seq[int] = newSeq[int](ndim)
    var remaining = outIdx
    for i in 0 ..< ndim:
      outCoords[i] = remaining div outStrides[i]
      remaining = remaining mod outStrides[i]

    var inCoords: seq[int] = newSeq[int](ndim)
    for i in 0 ..< ndim:
      inCoords[perm[i]] = outCoords[i]

    var inIdx = 0
    for i in 0 ..< ndim:
      inIdx += inCoords[i] * inStrides[i]

    copyMem(addr result.data[outIdx * elemSize], addr input.data[inIdx * elemSize], elemSize)

# =============================================================================
# Expand/Repeat Operations
# =============================================================================

proc expandKernel*(input: TensorData, newShape: Shape): TensorData =
  ## Expand tensor to new shape by broadcasting
  ## Dimensions of size 1 can be expanded to any size
  let inNdim = input.shape.rank
  let outNdim = newShape.rank

  assert outNdim >= inNdim, "expanded shape must have at least as many dimensions"

  # Pad input shape with leading 1s to match output ndim
  var paddedInShape: seq[int] = @[]
  for i in 0 ..< (outNdim - inNdim):
    paddedInShape.add(1)
  for i in 0 ..< inNdim:
    paddedInShape.add(input.shape[i])

  # Validate expansion is possible
  for i in 0 ..< outNdim:
    assert paddedInShape[i] == 1 or paddedInShape[i] == newShape[i],
      "cannot expand dimension " & $i & " from " & $paddedInShape[i] & " to " & $newShape[i]

  result = newTensorData(newShape, input.dtype)

  # Compute strides
  var inStrides: seq[int] = @[]
  var stride = 1
  for i in countdown(inNdim - 1, 0):
    inStrides.insert(stride, 0)
    stride *= input.shape[i]

  # Pad with zeros for expanded dimensions
  var paddedInStrides: seq[int] = @[]
  for i in 0 ..< (outNdim - inNdim):
    paddedInStrides.add(0)
  for i in 0 ..< inNdim:
    paddedInStrides.add(inStrides[i])

  # Set stride to 0 for broadcasted dimensions
  for i in 0 ..< outNdim:
    if paddedInShape[i] == 1 and newShape[i] > 1:
      paddedInStrides[i] = 0

  var outStrides: seq[int] = @[]
  stride = 1
  for i in countdown(outNdim - 1, 0):
    outStrides.insert(stride, 0)
    stride *= newShape[i]

  let n = newShape.size
  let elemSize = dtypeSize(input.dtype)

  for outIdx in 0 ..< n:
    var outCoords: seq[int] = newSeq[int](outNdim)
    var remaining = outIdx
    for i in 0 ..< outNdim:
      outCoords[i] = remaining div outStrides[i]
      remaining = remaining mod outStrides[i]

    var inIdx = 0
    for i in 0 ..< outNdim:
      inIdx += outCoords[i] * paddedInStrides[i]

    copyMem(addr result.data[outIdx * elemSize], addr input.data[inIdx * elemSize], elemSize)

proc repeatKernel*(input: TensorData, repeats: seq[int]): TensorData =
  ## Repeat tensor along each dimension
  let ndim = input.shape.rank
  assert repeats.len == ndim, "repeats must match number of dimensions"

  var newDims: seq[int] = @[]
  for i in 0 ..< ndim:
    newDims.add(input.shape[i] * repeats[i])

  result = newTensorData(newShape(newDims), input.dtype)

  var inStrides: seq[int] = @[]
  var outStrides: seq[int] = @[]
  var stride = 1
  for i in countdown(ndim - 1, 0):
    inStrides.insert(stride, 0)
    stride *= input.shape[i]

  stride = 1
  for i in countdown(ndim - 1, 0):
    outStrides.insert(stride, 0)
    stride *= newDims[i]

  let n = result.shape.size
  let elemSize = dtypeSize(input.dtype)

  for outIdx in 0 ..< n:
    var outCoords: seq[int] = newSeq[int](ndim)
    var remaining = outIdx
    for i in 0 ..< ndim:
      outCoords[i] = remaining div outStrides[i]
      remaining = remaining mod outStrides[i]

    var inIdx = 0
    for i in 0 ..< ndim:
      let inCoord = outCoords[i] mod input.shape[i]
      inIdx += inCoord * inStrides[i]

    copyMem(addr result.data[outIdx * elemSize], addr input.data[inIdx * elemSize], elemSize)

# =============================================================================
# Contiguous Operations
# =============================================================================

proc contiguousKernel*(input: TensorData): TensorData =
  ## Ensure tensor is contiguous in memory (row-major order)
  ## For already contiguous tensors, returns a clone
  result = input.clone()
