## CPU kernels for indexing operations
##
## Provides gather, scatter, index_select, masked_select, take.

import ../../[dtype, shape, tensor]

# =============================================================================
# Index Select Operations
# =============================================================================

proc indexSelectKernel*(input: TensorData, dim: int, indices: TensorData): TensorData =
  ## Select elements along dimension using index tensor
  ## indices must be 1D int64 tensor
  assert indices.dtype == dtInt64, "indices must be int64"

  let ndim = input.shape.rank
  let actualDim = if dim < 0: ndim + dim else: dim
  assert actualDim >= 0 and actualDim < ndim, "dim out of range"

  let numIndices = indices.shape.size
  let idxArr = indices.asInt64

  # Build output shape
  var outDims: seq[int] = @[]
  for d in 0 ..< ndim:
    if d == actualDim:
      outDims.add(numIndices)
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
  let sliceBytes = elementsPerSlice * elemSize

  for sliceIdx in 0 ..< numSlices:
    for i in 0 ..< numIndices:
      let srcIndex = idxArr[i]
      assert srcIndex >= 0 and srcIndex < input.shape[actualDim], "index out of range"

      let srcOffset = sliceIdx * input.shape[actualDim] * elementsPerSlice * elemSize +
                      srcIndex * elementsPerSlice * elemSize
      let dstOffset = sliceIdx * numIndices * elementsPerSlice * elemSize +
                      i * elementsPerSlice * elemSize

      copyMem(addr result.data[dstOffset], addr input.data[srcOffset], sliceBytes)

# =============================================================================
# Gather/Scatter Operations
# =============================================================================

proc gatherKernel*(input: TensorData, dim: int, indices: TensorData): TensorData =
  ## Gather values along an axis specified by dim
  ## output[i][j][k] = input[index[i][j][k]][j][k] for dim=0
  ## output[i][j][k] = input[i][index[i][j][k]][k] for dim=1
  assert indices.dtype == dtInt64, "indices must be int64"

  let ndim = input.shape.rank
  let actualDim = if dim < 0: ndim + dim else: dim
  assert actualDim >= 0 and actualDim < ndim, "dim out of range"
  assert indices.shape.rank == ndim, "indices must have same number of dimensions as input"

  # Validate shapes match except in gather dimension
  for d in 0 ..< ndim:
    if d != actualDim:
      assert indices.shape[d] == input.shape[d],
        "indices shape must match input shape except in gather dimension"

  result = newTensorData(indices.shape, input.dtype)

  let idxArr = indices.asInt64
  let n = indices.shape.size
  let elemSize = dtypeSize(input.dtype)

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
    stride *= indices.shape[i]

  for outIdx in 0 ..< n:
    # Convert flat index to coordinates
    var coords: seq[int] = newSeq[int](ndim)
    var remaining = outIdx
    for d in 0 ..< ndim:
      coords[d] = remaining div outStrides[d]
      remaining = remaining mod outStrides[d]

    # Get the index value at this position
    let gatherIndex = idxArr[outIdx]
    assert gatherIndex >= 0 and gatherIndex < input.shape[actualDim], "gather index out of range"

    # Replace the gather dimension coordinate with the index
    coords[actualDim] = gatherIndex.int

    # Convert back to flat input index
    var inIdx = 0
    for d in 0 ..< ndim:
      inIdx += coords[d] * inStrides[d]

    copyMem(addr result.data[outIdx * elemSize], addr input.data[inIdx * elemSize], elemSize)

proc scatterKernel*(input: TensorData, dim: int, indices: TensorData, src: TensorData): TensorData =
  ## Scatter values from src into input at positions specified by indices
  ## Inverse of gather
  assert indices.dtype == dtInt64, "indices must be int64"
  assert src.dtype == input.dtype, "src must have same dtype as input"

  let ndim = input.shape.rank
  let actualDim = if dim < 0: ndim + dim else: dim
  assert actualDim >= 0 and actualDim < ndim, "dim out of range"
  assert indices.shape == src.shape, "indices and src must have same shape"

  result = input.clone()

  let idxArr = indices.asInt64
  let n = indices.shape.size
  let elemSize = dtypeSize(input.dtype)

  # Compute strides
  var outStrides: seq[int] = @[]
  var srcStrides: seq[int] = @[]
  var stride = 1
  for i in countdown(ndim - 1, 0):
    outStrides.insert(stride, 0)
    stride *= result.shape[i]

  stride = 1
  for i in countdown(ndim - 1, 0):
    srcStrides.insert(stride, 0)
    stride *= src.shape[i]

  for srcIdx in 0 ..< n:
    # Convert flat index to coordinates
    var coords: seq[int] = newSeq[int](ndim)
    var remaining = srcIdx
    for d in 0 ..< ndim:
      coords[d] = remaining div srcStrides[d]
      remaining = remaining mod srcStrides[d]

    # Get the index value at this position
    let scatterIndex = idxArr[srcIdx]
    assert scatterIndex >= 0 and scatterIndex < result.shape[actualDim], "scatter index out of range"

    # Replace the scatter dimension coordinate with the index
    coords[actualDim] = scatterIndex.int

    # Convert back to flat output index
    var outIdx = 0
    for d in 0 ..< ndim:
      outIdx += coords[d] * outStrides[d]

    copyMem(addr result.data[outIdx * elemSize], addr src.data[srcIdx * elemSize], elemSize)

proc scatterFillKernel*(input: TensorData, dim: int, indices: TensorData, value: float64): TensorData =
  ## Scatter a scalar value into input at positions specified by indices
  assert indices.dtype == dtInt64, "indices must be int64"

  let ndim = input.shape.rank
  let actualDim = if dim < 0: ndim + dim else: dim
  assert actualDim >= 0 and actualDim < ndim, "dim out of range"

  result = input.clone()

  let idxArr = indices.asInt64
  let n = indices.shape.size

  # Compute strides
  var outStrides: seq[int] = @[]
  var idxStrides: seq[int] = @[]
  var stride = 1
  for i in countdown(ndim - 1, 0):
    outStrides.insert(stride, 0)
    stride *= result.shape[i]

  stride = 1
  for i in countdown(indices.shape.rank - 1, 0):
    idxStrides.insert(stride, 0)
    stride *= indices.shape[i]

  case result.dtype
  of dtFloat32:
    var dst = result.asFloat32
    let v = value.float32
    for srcIdx in 0 ..< n:
      var coords: seq[int] = newSeq[int](ndim)
      var remaining = srcIdx
      for d in 0 ..< indices.shape.rank:
        coords[d] = remaining div idxStrides[d]
        remaining = remaining mod idxStrides[d]

      let scatterIndex = idxArr[srcIdx]
      coords[actualDim] = scatterIndex.int

      var outIdx = 0
      for d in 0 ..< ndim:
        outIdx += coords[d] * outStrides[d]

      dst[outIdx] = v

  of dtFloat64:
    var dst = result.asFloat64
    for srcIdx in 0 ..< n:
      var coords: seq[int] = newSeq[int](ndim)
      var remaining = srcIdx
      for d in 0 ..< indices.shape.rank:
        coords[d] = remaining div idxStrides[d]
        remaining = remaining mod idxStrides[d]

      let scatterIndex = idxArr[srcIdx]
      coords[actualDim] = scatterIndex.int

      var outIdx = 0
      for d in 0 ..< ndim:
        outIdx += coords[d] * outStrides[d]

      dst[outIdx] = value

  else:
    raise newException(ValueError, "Unsupported dtype for scatterFill: " & $result.dtype)

# =============================================================================
# Take/Put Operations
# =============================================================================

proc takeKernel*(input: TensorData, indices: TensorData): TensorData =
  ## Take elements from flattened input at given indices
  ## Returns 1D tensor
  assert indices.dtype == dtInt64, "indices must be int64"

  let n = input.shape.size
  let numIndices = indices.shape.size
  let idxArr = indices.asInt64

  result = newTensorData(indices.shape, input.dtype)
  let elemSize = dtypeSize(input.dtype)

  for i in 0 ..< numIndices:
    let idx = idxArr[i]
    assert idx >= 0 and idx < n, "index out of range"
    copyMem(addr result.data[i * elemSize], addr input.data[idx * elemSize], elemSize)

proc putKernel*(input: TensorData, indices: TensorData, values: TensorData): TensorData =
  ## Put values into flattened input at given indices
  assert indices.dtype == dtInt64, "indices must be int64"
  assert values.dtype == input.dtype, "values must have same dtype as input"
  assert indices.shape.size == values.shape.size, "indices and values must have same size"

  let n = input.shape.size
  let numIndices = indices.shape.size
  let idxArr = indices.asInt64

  result = input.clone()
  let elemSize = dtypeSize(input.dtype)

  for i in 0 ..< numIndices:
    let idx = idxArr[i]
    assert idx >= 0 and idx < n, "index out of range"
    copyMem(addr result.data[idx * elemSize], addr values.data[i * elemSize], elemSize)

# =============================================================================
# Masked Operations
# =============================================================================

proc maskedSelectKernel*(input: TensorData, mask: TensorData): TensorData =
  ## Select elements where mask is true
  ## Returns 1D tensor
  assert mask.dtype == dtBool, "mask must be boolean"
  assert mask.shape == input.shape, "mask must have same shape as input"

  let n = input.shape.size
  let maskArr = cast[ptr UncheckedArray[uint8]](addr mask.data[0])
  let elemSize = dtypeSize(input.dtype)

  # Count true elements
  var count = 0
  for i in 0 ..< n:
    if maskArr[i] != 0:
      count += 1

  result = newTensorData(newShape(count), input.dtype)

  var outIdx = 0
  for i in 0 ..< n:
    if maskArr[i] != 0:
      copyMem(addr result.data[outIdx * elemSize], addr input.data[i * elemSize], elemSize)
      outIdx += 1

proc maskedScatterKernel*(input: TensorData, mask: TensorData, source: TensorData): TensorData =
  ## Scatter source values into input at positions where mask is true
  assert mask.dtype == dtBool, "mask must be boolean"
  assert mask.shape == input.shape, "mask must have same shape as input"
  assert source.dtype == input.dtype, "source must have same dtype as input"

  let n = input.shape.size
  let maskArr = cast[ptr UncheckedArray[uint8]](addr mask.data[0])
  let elemSize = dtypeSize(input.dtype)

  result = input.clone()

  var srcIdx = 0
  for i in 0 ..< n:
    if maskArr[i] != 0:
      assert srcIdx < source.shape.size, "source has fewer elements than mask has true values"
      copyMem(addr result.data[i * elemSize], addr source.data[srcIdx * elemSize], elemSize)
      srcIdx += 1

# =============================================================================
# Nonzero
# =============================================================================

proc nonzeroKernel*(input: TensorData): TensorData =
  ## Return indices of non-zero elements
  ## Returns 2D tensor of shape (num_nonzero, ndim)
  let n = input.shape.size
  let ndim = input.shape.rank

  # Compute strides for coordinate calculation
  var strides: seq[int] = @[]
  var stride = 1
  for i in countdown(ndim - 1, 0):
    strides.insert(stride, 0)
    stride *= input.shape[i]

  # Count non-zero elements
  var count = 0
  case input.dtype
  of dtFloat32:
    let arr = input.asFloat32
    for i in 0 ..< n:
      if arr[i] != 0.0'f32:
        count += 1
  of dtFloat64:
    let arr = input.asFloat64
    for i in 0 ..< n:
      if arr[i] != 0.0:
        count += 1
  of dtInt32:
    let arr = input.asInt32
    for i in 0 ..< n:
      if arr[i] != 0'i32:
        count += 1
  of dtInt64:
    let arr = input.asInt64
    for i in 0 ..< n:
      if arr[i] != 0'i64:
        count += 1
  of dtBool:
    let arr = cast[ptr UncheckedArray[uint8]](addr input.data[0])
    for i in 0 ..< n:
      if arr[i] != 0'u8:
        count += 1
  else:
    raise newException(ValueError, "Unsupported dtype for nonzero: " & $input.dtype)

  result = newTensorData(newShape(count, ndim), dtInt64)
  var dst = result.asInt64

  var outIdx = 0
  case input.dtype
  of dtFloat32:
    let arr = input.asFloat32
    for i in 0 ..< n:
      if arr[i] != 0.0'f32:
        # Convert flat index to coordinates
        var remaining = i
        for d in 0 ..< ndim:
          dst[outIdx * ndim + d] = (remaining div strides[d]).int64
          remaining = remaining mod strides[d]
        outIdx += 1
  of dtFloat64:
    let arr = input.asFloat64
    for i in 0 ..< n:
      if arr[i] != 0.0:
        var remaining = i
        for d in 0 ..< ndim:
          dst[outIdx * ndim + d] = (remaining div strides[d]).int64
          remaining = remaining mod strides[d]
        outIdx += 1
  of dtInt32:
    let arr = input.asInt32
    for i in 0 ..< n:
      if arr[i] != 0'i32:
        var remaining = i
        for d in 0 ..< ndim:
          dst[outIdx * ndim + d] = (remaining div strides[d]).int64
          remaining = remaining mod strides[d]
        outIdx += 1
  of dtInt64:
    let arr = input.asInt64
    for i in 0 ..< n:
      if arr[i] != 0'i64:
        var remaining = i
        for d in 0 ..< ndim:
          dst[outIdx * ndim + d] = (remaining div strides[d]).int64
          remaining = remaining mod strides[d]
        outIdx += 1
  of dtBool:
    let arr = cast[ptr UncheckedArray[uint8]](addr input.data[0])
    for i in 0 ..< n:
      if arr[i] != 0'u8:
        var remaining = i
        for d in 0 ..< ndim:
          dst[outIdx * ndim + d] = (remaining div strides[d]).int64
          remaining = remaining mod strides[d]
        outIdx += 1
  else:
    discard
