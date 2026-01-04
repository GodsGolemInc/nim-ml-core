## CPU kernels for concatenation and split operations
##
## Provides cat, stack, split, chunk, unbind.

import ../../[dtype, shape, tensor]

# =============================================================================
# Concatenation Operations
# =============================================================================

proc catKernel*(tensors: seq[TensorData], dim: int = 0): TensorData =
  ## Concatenate tensors along the given dimension
  assert tensors.len > 0, "need at least one tensor to concatenate"

  let first = tensors[0]
  let ndim = first.shape.rank
  let actualDim = if dim < 0: ndim + dim else: dim

  assert actualDim >= 0 and actualDim < ndim, "dim out of range"

  # Validate all tensors have compatible shapes
  for i in 1 ..< tensors.len:
    assert tensors[i].shape.rank == ndim, "all tensors must have same number of dimensions"
    assert tensors[i].dtype == first.dtype, "all tensors must have same dtype"
    for d in 0 ..< ndim:
      if d != actualDim:
        assert tensors[i].shape[d] == first.shape[d],
          "tensors must have same shape except in concat dimension"

  # Compute output shape
  var newDims: seq[int] = @[]
  var catDimSize = 0
  for t in tensors:
    catDimSize += t.shape[actualDim]

  for d in 0 ..< ndim:
    if d == actualDim:
      newDims.add(catDimSize)
    else:
      newDims.add(first.shape[d])

  result = newTensorData(newShape(newDims), first.dtype)

  # Compute sizes for copying
  # Elements per slice along concat dimension
  var elementsPerSlice = 1
  for d in (actualDim + 1) ..< ndim:
    elementsPerSlice *= first.shape[d]

  # Number of slices (product of dimensions before concat dim)
  var numSlices = 1
  for d in 0 ..< actualDim:
    numSlices *= first.shape[d]

  let elemSize = dtypeSize(first.dtype)
  var outOffset = 0

  for sliceIdx in 0 ..< numSlices:
    for t in tensors:
      let sliceElements = t.shape[actualDim] * elementsPerSlice
      let sliceBytes = sliceElements * elemSize
      let inOffset = sliceIdx * sliceElements * elemSize

      copyMem(addr result.data[outOffset], addr t.data[inOffset], sliceBytes)
      outOffset += sliceBytes

proc stackKernel*(tensors: seq[TensorData], dim: int = 0): TensorData =
  ## Stack tensors along a new dimension
  ## All tensors must have the same shape
  assert tensors.len > 0, "need at least one tensor to stack"

  let first = tensors[0]
  let ndim = first.shape.rank
  let actualDim = if dim < 0: ndim + dim + 1 else: dim

  assert actualDim >= 0 and actualDim <= ndim, "dim out of range"

  # Validate all tensors have same shape
  for i in 1 ..< tensors.len:
    assert tensors[i].shape == first.shape, "all tensors must have same shape"
    assert tensors[i].dtype == first.dtype, "all tensors must have same dtype"

  # Build output shape with new dimension
  var newDims: seq[int] = @[]
  for d in 0 ..< actualDim:
    newDims.add(first.shape[d])
  newDims.add(tensors.len)
  for d in actualDim ..< ndim:
    newDims.add(first.shape[d])

  result = newTensorData(newShape(newDims), first.dtype)

  # For stacking, each tensor contributes one slice in the new dimension
  let sliceSize = first.byteSize

  for i, t in tensors:
    copyMem(addr result.data[i * sliceSize], addr t.data[0], sliceSize)

# =============================================================================
# Split Operations
# =============================================================================

proc splitKernel*(input: TensorData, splitSize: int, dim: int = 0): seq[TensorData] =
  ## Split tensor into chunks of given size along dimension
  ## Last chunk may be smaller
  let ndim = input.shape.rank
  let actualDim = if dim < 0: ndim + dim else: dim

  assert actualDim >= 0 and actualDim < ndim, "dim out of range"
  assert splitSize > 0, "split size must be positive"

  let dimSize = input.shape[actualDim]
  let numChunks = (dimSize + splitSize - 1) div splitSize

  result = @[]

  var elementsPerSlice = 1
  for d in (actualDim + 1) ..< ndim:
    elementsPerSlice *= input.shape[d]

  var numSlices = 1
  for d in 0 ..< actualDim:
    numSlices *= input.shape[d]

  let elemSize = dtypeSize(input.dtype)
  var offset = 0

  for chunkIdx in 0 ..< numChunks:
    let chunkStart = chunkIdx * splitSize
    let chunkEnd = min(chunkStart + splitSize, dimSize)
    let chunkSize = chunkEnd - chunkStart

    # Build chunk shape
    var chunkDims: seq[int] = @[]
    for d in 0 ..< ndim:
      if d == actualDim:
        chunkDims.add(chunkSize)
      else:
        chunkDims.add(input.shape[d])

    let chunk = newTensorData(newShape(chunkDims), input.dtype)
    let chunkElements = chunk.shape.size

    for sliceIdx in 0 ..< numSlices:
      let srcOffset = sliceIdx * dimSize * elementsPerSlice * elemSize +
                      chunkStart * elementsPerSlice * elemSize
      let dstOffset = sliceIdx * chunkSize * elementsPerSlice * elemSize
      let copySize = chunkSize * elementsPerSlice * elemSize

      copyMem(addr chunk.data[dstOffset], addr input.data[srcOffset], copySize)

    result.add(chunk)

proc chunkKernel*(input: TensorData, numChunks: int, dim: int = 0): seq[TensorData] =
  ## Split tensor into specified number of chunks along dimension
  let ndim = input.shape.rank
  let actualDim = if dim < 0: ndim + dim else: dim

  assert actualDim >= 0 and actualDim < ndim, "dim out of range"
  assert numChunks > 0, "number of chunks must be positive"

  let dimSize = input.shape[actualDim]
  let chunkSize = (dimSize + numChunks - 1) div numChunks

  splitKernel(input, chunkSize, actualDim)

proc unbindKernel*(input: TensorData, dim: int = 0): seq[TensorData] =
  ## Remove a dimension and return a sequence of tensors
  ## Similar to split with split_size=1, but removes the dimension
  let ndim = input.shape.rank
  let actualDim = if dim < 0: ndim + dim else: dim

  assert actualDim >= 0 and actualDim < ndim, "dim out of range"

  let dimSize = input.shape[actualDim]
  result = @[]

  # Build output shape (without the unbind dimension)
  var outDims: seq[int] = @[]
  for d in 0 ..< ndim:
    if d != actualDim:
      outDims.add(input.shape[d])

  let outShape = if outDims.len > 0: newShape(outDims) else: newShape(1)

  var elementsPerSlice = 1
  for d in (actualDim + 1) ..< ndim:
    elementsPerSlice *= input.shape[d]

  var numSlices = 1
  for d in 0 ..< actualDim:
    numSlices *= input.shape[d]

  let elemSize = dtypeSize(input.dtype)
  let sliceBytes = elementsPerSlice * elemSize

  for i in 0 ..< dimSize:
    let tensor = newTensorData(outShape, input.dtype)

    for sliceIdx in 0 ..< numSlices:
      let srcOffset = sliceIdx * dimSize * elementsPerSlice * elemSize +
                      i * elementsPerSlice * elemSize
      let dstOffset = sliceIdx * elementsPerSlice * elemSize

      copyMem(addr tensor.data[dstOffset], addr input.data[srcOffset], sliceBytes)

    result.add(tensor)

# =============================================================================
# Narrow/Slice Operations
# =============================================================================

proc narrowKernel*(input: TensorData, dim: int, start: int, length: int): TensorData =
  ## Return a narrowed version of input tensor
  ## Selects length elements starting from start along dim
  let ndim = input.shape.rank
  let actualDim = if dim < 0: ndim + dim else: dim

  assert actualDim >= 0 and actualDim < ndim, "dim out of range"
  assert start >= 0 and start < input.shape[actualDim], "start out of range"
  assert length > 0 and start + length <= input.shape[actualDim], "length out of range"

  var outDims: seq[int] = @[]
  for d in 0 ..< ndim:
    if d == actualDim:
      outDims.add(length)
    else:
      outDims.add(input.shape[d])

  result = newTensorData(newShape(outDims), input.dtype)

  var elementsPerSlice = 1
  for d in (actualDim + 1) ..< ndim:
    elementsPerSlice *= input.shape[d]

  var numSlices = 1
  for d in 0 ..< actualDim:
    numSlices *= input.shape[d]

  let elemSize = dtypeSize(input.dtype)
  let copyBytes = length * elementsPerSlice * elemSize

  for sliceIdx in 0 ..< numSlices:
    let srcOffset = sliceIdx * input.shape[actualDim] * elementsPerSlice * elemSize +
                    start * elementsPerSlice * elemSize
    let dstOffset = sliceIdx * length * elementsPerSlice * elemSize

    copyMem(addr result.data[dstOffset], addr input.data[srcOffset], copyBytes)

proc selectKernel*(input: TensorData, dim: int, index: int): TensorData =
  ## Select a single index along a dimension (removes that dimension)
  let ndim = input.shape.rank
  let actualDim = if dim < 0: ndim + dim else: dim
  let actualIndex = if index < 0: input.shape[actualDim] + index else: index

  assert actualDim >= 0 and actualDim < ndim, "dim out of range"
  assert actualIndex >= 0 and actualIndex < input.shape[actualDim], "index out of range"

  var outDims: seq[int] = @[]
  for d in 0 ..< ndim:
    if d != actualDim:
      outDims.add(input.shape[d])

  let outShape = if outDims.len > 0: newShape(outDims) else: newShape()

  result = newTensorData(outShape, input.dtype)

  var elementsPerSlice = 1
  for d in (actualDim + 1) ..< ndim:
    elementsPerSlice *= input.shape[d]

  var numSlices = 1
  for d in 0 ..< actualDim:
    numSlices *= input.shape[d]

  let elemSize = dtypeSize(input.dtype)
  let sliceBytes = elementsPerSlice * elemSize

  for sliceIdx in 0 ..< numSlices:
    let srcOffset = sliceIdx * input.shape[actualDim] * elementsPerSlice * elemSize +
                    actualIndex * elementsPerSlice * elemSize
    let dstOffset = sliceIdx * elementsPerSlice * elemSize

    copyMem(addr result.data[dstOffset], addr input.data[srcOffset], sliceBytes)
