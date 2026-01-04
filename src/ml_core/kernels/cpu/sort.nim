## CPU kernels for sort and selection operations
##
## Provides sort, kthvalue, topk operations critical for pruning.

import std/algorithm
import ../../[dtype, shape, tensor]

# =============================================================================
# Sort Operations
# =============================================================================

proc sortKernel*(input: TensorData, descending: bool = false): tuple[values: TensorData, indices: TensorData] =
  ## Sort tensor elements
  ## Returns sorted values and original indices
  let n = input.shape.size
  result.values = newTensorData(input.shape, input.dtype)
  result.indices = newTensorData(input.shape, dtInt64)

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var pairs: seq[tuple[value: float32, index: int]] = @[]
    for i in 0 ..< n:
      pairs.add((src[i], i))

    if descending:
      pairs.sort(proc(a, b: tuple[value: float32, index: int]): int =
        if a.value > b.value: -1
        elif a.value < b.value: 1
        else: 0
      )
    else:
      pairs.sort(proc(a, b: tuple[value: float32, index: int]): int =
        if a.value < b.value: -1
        elif a.value > b.value: 1
        else: 0
      )

    var dstValues = result.values.asFloat32
    var dstIndices = result.indices.asInt64
    for i in 0 ..< n:
      dstValues[i] = pairs[i].value
      dstIndices[i] = pairs[i].index.int64

  of dtFloat64:
    let src = input.asFloat64
    var pairs: seq[tuple[value: float64, index: int]] = @[]
    for i in 0 ..< n:
      pairs.add((src[i], i))

    if descending:
      pairs.sort(proc(a, b: tuple[value: float64, index: int]): int =
        if a.value > b.value: -1
        elif a.value < b.value: 1
        else: 0
      )
    else:
      pairs.sort(proc(a, b: tuple[value: float64, index: int]): int =
        if a.value < b.value: -1
        elif a.value > b.value: 1
        else: 0
      )

    var dstValues = result.values.asFloat64
    var dstIndices = result.indices.asInt64
    for i in 0 ..< n:
      dstValues[i] = pairs[i].value
      dstIndices[i] = pairs[i].index.int64

  else:
    raise newException(ValueError, "Unsupported dtype for sort: " & $input.dtype)

proc argsortKernel*(input: TensorData, descending: bool = false): TensorData =
  ## Return indices that would sort the tensor
  let (_, indices) = sortKernel(input, descending)
  return indices

# =============================================================================
# Selection Operations
# =============================================================================

proc kthvalueKernel*(input: TensorData, k: int): tuple[value: TensorData, index: TensorData] =
  ## Find the k-th smallest value (1-indexed like PyTorch)
  ## k=1 returns minimum, k=n returns maximum
  let n = input.shape.size
  assert k >= 1 and k <= n, "k must be in range [1, n]"

  result.value = newTensorData(newShape(), input.dtype)  # Scalar
  result.index = newTensorData(newShape(), dtInt64)       # Scalar

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var pairs: seq[tuple[value: float32, index: int]] = @[]
    for i in 0 ..< n:
      pairs.add((src[i], i))

    # Sort ascending
    pairs.sort(proc(a, b: tuple[value: float32, index: int]): int =
      if a.value < b.value: -1
      elif a.value > b.value: 1
      else: 0
    )

    result.value.asFloat32[0] = pairs[k - 1].value
    result.index.asInt64[0] = pairs[k - 1].index.int64

  of dtFloat64:
    let src = input.asFloat64
    var pairs: seq[tuple[value: float64, index: int]] = @[]
    for i in 0 ..< n:
      pairs.add((src[i], i))

    pairs.sort(proc(a, b: tuple[value: float64, index: int]): int =
      if a.value < b.value: -1
      elif a.value > b.value: 1
      else: 0
    )

    result.value.asFloat64[0] = pairs[k - 1].value
    result.index.asInt64[0] = pairs[k - 1].index.int64

  else:
    raise newException(ValueError, "Unsupported dtype for kthvalue: " & $input.dtype)

proc topkKernel*(input: TensorData, k: int, largest: bool = true): tuple[values: TensorData, indices: TensorData] =
  ## Return top k values and their indices
  ## If largest=true, returns k largest values; otherwise k smallest
  let n = input.shape.size
  assert k >= 1 and k <= n, "k must be in range [1, n]"

  result.values = newTensorData(newShape(k), input.dtype)
  result.indices = newTensorData(newShape(k), dtInt64)

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var pairs: seq[tuple[value: float32, index: int]] = @[]
    for i in 0 ..< n:
      pairs.add((src[i], i))

    if largest:
      pairs.sort(proc(a, b: tuple[value: float32, index: int]): int =
        if a.value > b.value: -1
        elif a.value < b.value: 1
        else: 0
      )
    else:
      pairs.sort(proc(a, b: tuple[value: float32, index: int]): int =
        if a.value < b.value: -1
        elif a.value > b.value: 1
        else: 0
      )

    var dstValues = result.values.asFloat32
    var dstIndices = result.indices.asInt64
    for i in 0 ..< k:
      dstValues[i] = pairs[i].value
      dstIndices[i] = pairs[i].index.int64

  of dtFloat64:
    let src = input.asFloat64
    var pairs: seq[tuple[value: float64, index: int]] = @[]
    for i in 0 ..< n:
      pairs.add((src[i], i))

    if largest:
      pairs.sort(proc(a, b: tuple[value: float64, index: int]): int =
        if a.value > b.value: -1
        elif a.value < b.value: 1
        else: 0
      )
    else:
      pairs.sort(proc(a, b: tuple[value: float64, index: int]): int =
        if a.value < b.value: -1
        elif a.value > b.value: 1
        else: 0
      )

    var dstValues = result.values.asFloat64
    var dstIndices = result.indices.asInt64
    for i in 0 ..< k:
      dstValues[i] = pairs[i].value
      dstIndices[i] = pairs[i].index.int64

  else:
    raise newException(ValueError, "Unsupported dtype for topk: " & $input.dtype)

proc medianKernel*(input: TensorData): TensorData =
  ## Compute median of all elements
  let n = input.shape.size
  assert n > 0, "Cannot compute median of empty tensor"

  result = newTensorData(newShape(), input.dtype)  # Scalar

  case input.dtype
  of dtFloat32:
    let src = input.asFloat32
    var values: seq[float32] = @[]
    for i in 0 ..< n:
      values.add(src[i])
    values.sort()

    if n mod 2 == 1:
      result.asFloat32[0] = values[n div 2]
    else:
      result.asFloat32[0] = (values[n div 2 - 1] + values[n div 2]) / 2.0

  of dtFloat64:
    let src = input.asFloat64
    var values: seq[float64] = @[]
    for i in 0 ..< n:
      values.add(src[i])
    values.sort()

    if n mod 2 == 1:
      result.asFloat64[0] = values[n div 2]
    else:
      result.asFloat64[0] = (values[n div 2 - 1] + values[n div 2]) / 2.0

  else:
    raise newException(ValueError, "Unsupported dtype for median: " & $input.dtype)
