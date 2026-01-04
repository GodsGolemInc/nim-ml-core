## Extended Kernel Tests
##
## Tests for reshape, concat, index, matmul, and broadcast kernels.

import unittest
import ml_core
import ml_core/kernels/cpu/[reshape, concat, index, matmul, broadcast]

# =============================================================================
# Reshape Kernels
# =============================================================================

suite "Reshape Kernels":
  test "reshapeKernel basic":
    let data = newTensorData(newShape(2, 3), dtFloat32)
    var arr = data.asFloat32
    for i in 0..<6:
      arr[i] = (i + 1).float32

    let reshaped = reshapeKernel(data, newShape(3, 2))
    check reshaped.shape.dims == @[3, 2]
    check reshaped.shape.size == 6

    let resArr = reshaped.asFloat32
    check resArr[0] == 1.0'f32
    check resArr[5] == 6.0'f32

  test "reshapeKernel to 1D":
    let data = newTensorData(newShape(2, 3), dtFloat32)
    let reshaped = reshapeKernel(data, newShape(6))
    check reshaped.shape.dims == @[6]

  test "viewKernel":
    let data = newTensorData(newShape(4, 4), dtFloat32)
    let viewed = viewKernel(data, newShape(2, 8))
    check viewed.shape.dims == @[2, 8]

  test "flattenKernel full":
    let data = newTensorData(newShape(2, 3, 4), dtFloat32)
    let flat = flattenKernel(data, 0, -1)
    check flat.shape.dims == @[24]

  test "flattenKernel partial":
    let data = newTensorData(newShape(2, 3, 4), dtFloat32)
    let flat = flattenKernel(data, 1, 2)
    check flat.shape.dims == @[2, 12]

  test "squeezeKernel all":
    let data = newTensorData(newShape(1, 3, 1, 4, 1), dtFloat32)
    let squeezed = squeezeKernel(data)
    check squeezed.shape.dims == @[3, 4]

  test "squeezeKernel specific dim":
    let data = newTensorData(newShape(1, 3, 1, 4), dtFloat32)
    let squeezed = squeezeKernel(data, 0)
    check squeezed.shape.dims == @[3, 1, 4]

  test "unsqueezeKernel":
    let data = newTensorData(newShape(3, 4), dtFloat32)
    let unsqueezed = unsqueezeKernel(data, 0)
    check unsqueezed.shape.dims == @[1, 3, 4]

  test "unsqueezeKernel middle":
    let data = newTensorData(newShape(3, 4), dtFloat32)
    let unsqueezed = unsqueezeKernel(data, 1)
    check unsqueezed.shape.dims == @[3, 1, 4]

  test "transposeKernel 2D":
    let data = newTensorData(newShape(2, 3), dtFloat32)
    var arr = data.asFloat32
    for i in 0..<6:
      arr[i] = (i + 1).float32  # [1,2,3,4,5,6] -> [[1,2,3],[4,5,6]]

    let transposed = transposeKernel(data, 0, 1)
    check transposed.shape.dims == @[3, 2]

    let tArr = transposed.asFloat32
    # [[1,4],[2,5],[3,6]]
    check tArr[0] == 1.0'f32
    check tArr[1] == 4.0'f32
    check tArr[2] == 2.0'f32

  test "transpose2DKernel":
    let data = newTensorData(newShape(2, 3), dtFloat32)
    let transposed = transpose2DKernel(data)
    check transposed.shape.dims == @[3, 2]

  test "permuteKernel":
    let data = newTensorData(newShape(2, 3, 4), dtFloat32)
    let permuted = permuteKernel(data, @[2, 0, 1])
    check permuted.shape.dims == @[4, 2, 3]

  test "expandKernel":
    let data = newTensorData(newShape(1, 3), dtFloat32)
    var arr = data.asFloat32
    arr[0] = 1.0'f32
    arr[1] = 2.0'f32
    arr[2] = 3.0'f32

    let expanded = expandKernel(data, newShape(4, 3))
    check expanded.shape.dims == @[4, 3]

    let expArr = expanded.asFloat32
    check expArr[0] == 1.0'f32
    check expArr[3] == 1.0'f32  # Second row starts same
    check expArr[6] == 1.0'f32  # Third row
    check expArr[9] == 1.0'f32  # Fourth row

  test "repeatKernel":
    let data = newTensorData(newShape(2, 3), dtFloat32)
    let repeated = repeatKernel(data, @[2, 3])
    check repeated.shape.dims == @[4, 9]

  test "contiguousKernel":
    let data = newTensorData(newShape(2, 3), dtFloat32)
    let contig = contiguousKernel(data)
    check contig.shape == data.shape

# =============================================================================
# Concat Kernels
# =============================================================================

suite "Concat Kernels":
  test "catKernel dim 0":
    let a = newTensorData(newShape(2, 3), dtFloat32)
    let b = newTensorData(newShape(3, 3), dtFloat32)
    let cat = catKernel(@[a, b], 0)
    check cat.shape.dims == @[5, 3]

  test "catKernel dim 1":
    let a = newTensorData(newShape(2, 3), dtFloat32)
    let b = newTensorData(newShape(2, 4), dtFloat32)
    let cat = catKernel(@[a, b], 1)
    check cat.shape.dims == @[2, 7]

  test "catKernel multiple tensors":
    let a = newTensorData(newShape(2, 3), dtFloat32)
    let b = newTensorData(newShape(2, 3), dtFloat32)
    let c = newTensorData(newShape(2, 3), dtFloat32)
    let cat = catKernel(@[a, b, c], 0)
    check cat.shape.dims == @[6, 3]

  test "stackKernel dim 0":
    let a = newTensorData(newShape(2, 3), dtFloat32)
    let b = newTensorData(newShape(2, 3), dtFloat32)
    let stacked = stackKernel(@[a, b], 0)
    check stacked.shape.dims == @[2, 2, 3]

  test "stackKernel dim 1":
    let a = newTensorData(newShape(2, 3), dtFloat32)
    let b = newTensorData(newShape(2, 3), dtFloat32)
    let stacked = stackKernel(@[a, b], 1)
    check stacked.shape.dims == @[2, 2, 3]

  test "splitKernel":
    let data = newTensorData(newShape(6, 4), dtFloat32)
    let splits = splitKernel(data, 2, 0)
    check splits.len == 3
    check splits[0].shape.dims == @[2, 4]
    check splits[1].shape.dims == @[2, 4]
    check splits[2].shape.dims == @[2, 4]

  test "splitKernel uneven":
    let data = newTensorData(newShape(5, 4), dtFloat32)
    let splits = splitKernel(data, 2, 0)
    check splits.len == 3
    check splits[0].shape.dims == @[2, 4]
    check splits[1].shape.dims == @[2, 4]
    check splits[2].shape.dims == @[1, 4]

  test "chunkKernel":
    let data = newTensorData(newShape(6, 4), dtFloat32)
    let chunks = chunkKernel(data, 3, 0)
    check chunks.len == 3
    check chunks[0].shape.dims == @[2, 4]

  test "unbindKernel":
    let data = newTensorData(newShape(3, 4), dtFloat32)
    let unbound = unbindKernel(data, 0)
    check unbound.len == 3
    check unbound[0].shape.dims == @[4]

  test "narrowKernel":
    let data = newTensorData(newShape(4, 6), dtFloat32)
    let narrow = narrowKernel(data, 1, 1, 3)
    check narrow.shape.dims == @[4, 3]

  test "selectKernel":
    let data = newTensorData(newShape(3, 4, 5), dtFloat32)
    let selected = selectKernel(data, 0, 1)
    check selected.shape.dims == @[4, 5]

# =============================================================================
# Index Kernels
# =============================================================================

suite "Index Kernels":
  test "indexSelectKernel":
    let data = newTensorData(newShape(4, 3), dtFloat32)
    var arr = data.asFloat32
    for i in 0..<12:
      arr[i] = i.float32

    let indices = newTensorData(newShape(2), dtInt64)
    var idxArr = indices.asInt64
    idxArr[0] = 0
    idxArr[1] = 2

    let selected = indexSelectKernel(data, 0, indices)
    check selected.shape.dims == @[2, 3]

    let selArr = selected.asFloat32
    check selArr[0] == 0.0'f32  # Row 0
    check selArr[3] == 6.0'f32  # Row 2

  test "gatherKernel":
    let data = newTensorData(newShape(3, 4), dtFloat32)
    var arr = data.asFloat32
    for i in 0..<12:
      arr[i] = i.float32

    let indices = newTensorData(newShape(3, 2), dtInt64)
    var idxArr = indices.asInt64
    idxArr[0] = 0; idxArr[1] = 1
    idxArr[2] = 2; idxArr[3] = 3
    idxArr[4] = 0; idxArr[5] = 1

    let gathered = gatherKernel(data, 1, indices)
    check gathered.shape.dims == @[3, 2]

  test "scatterKernel":
    let input = newTensorData(newShape(3, 4), dtFloat32)
    let indices = newTensorData(newShape(3, 2), dtInt64)
    var idxArr = indices.asInt64
    idxArr[0] = 0; idxArr[1] = 1
    idxArr[2] = 2; idxArr[3] = 0
    idxArr[4] = 1; idxArr[5] = 2

    let src = newTensorData(newShape(3, 2), dtFloat32)
    var srcArr = src.asFloat32
    for i in 0..<6:
      srcArr[i] = 100.0'f32 + i.float32

    let scattered = scatterKernel(input, 1, indices, src)
    check scattered.shape.dims == @[3, 4]

  test "takeKernel":
    let data = newTensorData(newShape(3, 4), dtFloat32)
    var arr = data.asFloat32
    for i in 0..<12:
      arr[i] = i.float32

    let indices = newTensorData(newShape(3), dtInt64)
    var idxArr = indices.asInt64
    idxArr[0] = 0
    idxArr[1] = 5
    idxArr[2] = 11

    let taken = takeKernel(data, indices)
    check taken.shape.dims == @[3]

    let takenArr = taken.asFloat32
    check takenArr[0] == 0.0'f32
    check takenArr[1] == 5.0'f32
    check takenArr[2] == 11.0'f32

  test "putKernel":
    let input = newTensorData(newShape(12), dtFloat32)
    let indices = newTensorData(newShape(3), dtInt64)
    var idxArr = indices.asInt64
    idxArr[0] = 0
    idxArr[1] = 5
    idxArr[2] = 11

    let values = newTensorData(newShape(3), dtFloat32)
    var valArr = values.asFloat32
    valArr[0] = 100.0'f32
    valArr[1] = 200.0'f32
    valArr[2] = 300.0'f32

    let result = putKernel(input, indices, values)
    let resArr = result.asFloat32
    check resArr[0] == 100.0'f32
    check resArr[5] == 200.0'f32
    check resArr[11] == 300.0'f32

  test "maskedSelectKernel":
    let data = newTensorData(newShape(4), dtFloat32)
    var arr = data.asFloat32
    arr[0] = 1.0; arr[1] = 2.0; arr[2] = 3.0; arr[3] = 4.0

    let mask = newTensorData(newShape(4), dtBool)
    var maskArr = cast[ptr UncheckedArray[uint8]](addr mask.data[0])
    maskArr[0] = 1; maskArr[1] = 0; maskArr[2] = 1; maskArr[3] = 0

    let selected = maskedSelectKernel(data, mask)
    check selected.shape.size == 2

    let selArr = selected.asFloat32
    check selArr[0] == 1.0'f32
    check selArr[1] == 3.0'f32

  test "nonzeroKernel":
    let data = newTensorData(newShape(2, 3), dtFloat32)
    var arr = data.asFloat32
    arr[0] = 0.0; arr[1] = 1.0; arr[2] = 0.0
    arr[3] = 2.0; arr[4] = 0.0; arr[5] = 3.0

    let nz = nonzeroKernel(data)
    check nz.shape.dims[1] == 2  # 2D tensor

    let nzArr = nz.asInt64
    check nz.shape.dims[0] == 3  # 3 nonzero elements

# =============================================================================
# Matmul Kernels
# =============================================================================

suite "Matmul Kernels":
  test "mmKernel basic":
    let a = newTensorData(newShape(2, 3), dtFloat32)
    let b = newTensorData(newShape(3, 4), dtFloat32)

    var arrA = a.asFloat32
    var arrB = b.asFloat32
    for i in 0..<6: arrA[i] = 1.0'f32
    for i in 0..<12: arrB[i] = 1.0'f32

    let c = mmKernel(a, b)
    check c.shape.dims == @[2, 4]

    let arrC = c.asFloat32
    check arrC[0] == 3.0'f32  # 1*1 + 1*1 + 1*1

  test "mmKernel identity":
    let a = newTensorData(newShape(2, 2), dtFloat32)
    let identity = newTensorData(newShape(2, 2), dtFloat32)

    var arrA = a.asFloat32
    arrA[0] = 1.0; arrA[1] = 2.0; arrA[2] = 3.0; arrA[3] = 4.0

    var arrI = identity.asFloat32
    arrI[0] = 1.0; arrI[1] = 0.0; arrI[2] = 0.0; arrI[3] = 1.0

    let c = mmKernel(a, identity)
    let arrC = c.asFloat32
    check arrC[0] == 1.0'f32
    check arrC[1] == 2.0'f32
    check arrC[2] == 3.0'f32
    check arrC[3] == 4.0'f32

  test "bmmKernel":
    let a = newTensorData(newShape(2, 3, 4), dtFloat32)
    let b = newTensorData(newShape(2, 4, 5), dtFloat32)

    var arrA = a.asFloat32
    var arrB = b.asFloat32
    for i in 0..<24: arrA[i] = 1.0'f32
    for i in 0..<40: arrB[i] = 1.0'f32

    let c = bmmKernel(a, b)
    check c.shape.dims == @[2, 3, 5]

  test "mvKernel":
    let mat = newTensorData(newShape(2, 3), dtFloat32)
    let vec = newTensorData(newShape(3), dtFloat32)

    var arrM = mat.asFloat32
    var arrV = vec.asFloat32
    for i in 0..<6: arrM[i] = 1.0'f32
    for i in 0..<3: arrV[i] = 2.0'f32

    let result = mvKernel(mat, vec)
    check result.shape.dims == @[2]

    let arrR = result.asFloat32
    check arrR[0] == 6.0'f32
    check arrR[1] == 6.0'f32

  test "dotKernel":
    let a = newTensorData(newShape(4), dtFloat32)
    let b = newTensorData(newShape(4), dtFloat32)

    var arrA = a.asFloat32
    var arrB = b.asFloat32
    arrA[0] = 1.0; arrA[1] = 2.0; arrA[2] = 3.0; arrA[3] = 4.0
    arrB[0] = 1.0; arrB[1] = 1.0; arrB[2] = 1.0; arrB[3] = 1.0

    let dot = dotKernel(a, b)
    check dot.shape.size == 1
    let dotVal = dot.asFloat32
    check dotVal[0] == 10.0'f32

  test "outerKernel":
    let a = newTensorData(newShape(3), dtFloat32)
    let b = newTensorData(newShape(4), dtFloat32)

    var arrA = a.asFloat32
    var arrB = b.asFloat32
    for i in 0..<3: arrA[i] = (i + 1).float32
    for i in 0..<4: arrB[i] = 1.0'f32

    let outer = outerKernel(a, b)
    check outer.shape.dims == @[3, 4]

    let arrO = outer.asFloat32
    check arrO[0] == 1.0'f32
    check arrO[4] == 2.0'f32
    check arrO[8] == 3.0'f32

  test "traceKernel":
    let mat = newTensorData(newShape(3, 3), dtFloat32)
    var arr = mat.asFloat32
    arr[0] = 1.0; arr[1] = 2.0; arr[2] = 3.0
    arr[3] = 4.0; arr[4] = 5.0; arr[5] = 6.0
    arr[6] = 7.0; arr[7] = 8.0; arr[8] = 9.0

    let trace = traceKernel(mat)
    let traceVal = trace.asFloat32
    check traceVal[0] == 15.0'f32  # 1 + 5 + 9

  test "diagKernel extract":
    let mat = newTensorData(newShape(3, 3), dtFloat32)
    var arr = mat.asFloat32
    arr[0] = 1.0; arr[1] = 2.0; arr[2] = 3.0
    arr[3] = 4.0; arr[4] = 5.0; arr[5] = 6.0
    arr[6] = 7.0; arr[7] = 8.0; arr[8] = 9.0

    let diag = diagKernel(mat)
    check diag.shape.dims == @[3]
    let diagArr = diag.asFloat32
    check diagArr[0] == 1.0'f32
    check diagArr[1] == 5.0'f32
    check diagArr[2] == 9.0'f32

  test "diagKernel create":
    let vec = newTensorData(newShape(3), dtFloat32)
    var arr = vec.asFloat32
    arr[0] = 1.0; arr[1] = 2.0; arr[2] = 3.0

    let mat = diagKernel(vec)
    check mat.shape.dims == @[3, 3]

  test "triuKernel":
    let mat = newTensorData(newShape(3, 3), dtFloat32)
    var arr = mat.asFloat32
    for i in 0..<9: arr[i] = (i + 1).float32

    let triu = triuKernel(mat, 0)
    let triuArr = triu.asFloat32
    check triuArr[0] == 1.0'f32  # (0,0)
    check triuArr[3] == 0.0'f32  # (1,0) below diag
    check triuArr[6] == 0.0'f32  # (2,0) below diag

  test "trilKernel":
    let mat = newTensorData(newShape(3, 3), dtFloat32)
    var arr = mat.asFloat32
    for i in 0..<9: arr[i] = (i + 1).float32

    let tril = trilKernel(mat, 0)
    let trilArr = tril.asFloat32
    check trilArr[0] == 1.0'f32  # (0,0)
    check trilArr[1] == 0.0'f32  # (0,1) above diag
    check trilArr[2] == 0.0'f32  # (0,2) above diag

# =============================================================================
# Broadcast Kernels
# =============================================================================

suite "Broadcast Kernels":
  test "computeBroadcastShape same":
    let s1 = newShape(2, 3)
    let s2 = newShape(2, 3)
    let result = computeBroadcastShape(s1, s2)
    check result.dims == @[2, 3]

  test "computeBroadcastShape expand":
    let s1 = newShape(1, 3)
    let s2 = newShape(2, 3)
    let result = computeBroadcastShape(s1, s2)
    check result.dims == @[2, 3]

  test "computeBroadcastShape different ndims":
    let s1 = newShape(3)
    let s2 = newShape(2, 3)
    let result = computeBroadcastShape(s1, s2)
    check result.dims == @[2, 3]

  test "broadcastAddKernel":
    let a = newTensorData(newShape(2, 3), dtFloat32)
    let b = newTensorData(newShape(1, 3), dtFloat32)

    var arrA = a.asFloat32
    var arrB = b.asFloat32
    for i in 0..<6: arrA[i] = 1.0'f32
    for i in 0..<3: arrB[i] = 10.0'f32

    let result = broadcastAddKernel(a, b)
    check result.shape.dims == @[2, 3]

    let resArr = result.asFloat32
    for i in 0..<6:
      check resArr[i] == 11.0'f32

  test "broadcastSubKernel":
    let a = newTensorData(newShape(2, 3), dtFloat32)
    let b = newTensorData(newShape(3), dtFloat32)

    var arrA = a.asFloat32
    var arrB = b.asFloat32
    for i in 0..<6: arrA[i] = 10.0'f32
    for i in 0..<3: arrB[i] = 3.0'f32

    let result = broadcastSubKernel(a, b)
    let resArr = result.asFloat32
    for i in 0..<6:
      check resArr[i] == 7.0'f32

  test "broadcastMulKernel":
    let a = newTensorData(newShape(2, 3), dtFloat32)
    let b = newTensorData(newShape(1, 3), dtFloat32)

    var arrA = a.asFloat32
    var arrB = b.asFloat32
    for i in 0..<6: arrA[i] = 2.0'f32
    arrB[0] = 1.0; arrB[1] = 2.0; arrB[2] = 3.0

    let result = broadcastMulKernel(a, b)
    let resArr = result.asFloat32
    check resArr[0] == 2.0'f32
    check resArr[1] == 4.0'f32
    check resArr[2] == 6.0'f32

  test "broadcastDivKernel":
    let a = newTensorData(newShape(2, 3), dtFloat32)
    let b = newTensorData(newShape(3), dtFloat32)

    var arrA = a.asFloat32
    var arrB = b.asFloat32
    for i in 0..<6: arrA[i] = 12.0'f32
    arrB[0] = 1.0; arrB[1] = 2.0; arrB[2] = 3.0

    let result = broadcastDivKernel(a, b)
    let resArr = result.asFloat32
    check resArr[0] == 12.0'f32
    check resArr[1] == 6.0'f32
    check resArr[2] == 4.0'f32

  test "broadcastMaxKernel":
    let a = newTensorData(newShape(2, 3), dtFloat32)
    let b = newTensorData(newShape(3), dtFloat32)

    var arrA = a.asFloat32
    var arrB = b.asFloat32
    arrA[0] = 1.0; arrA[1] = 5.0; arrA[2] = 3.0
    arrA[3] = 4.0; arrA[4] = 2.0; arrA[5] = 6.0
    arrB[0] = 3.0; arrB[1] = 3.0; arrB[2] = 3.0

    let result = broadcastMaxKernel(a, b)
    let resArr = result.asFloat32
    check resArr[0] == 3.0'f32
    check resArr[1] == 5.0'f32
    check resArr[4] == 3.0'f32

  test "broadcastGeKernel":
    let a = newTensorData(newShape(3), dtFloat32)
    let b = newTensorData(newShape(3), dtFloat32)

    var arrA = a.asFloat32
    var arrB = b.asFloat32
    arrA[0] = 1.0; arrA[1] = 2.0; arrA[2] = 3.0
    arrB[0] = 2.0; arrB[1] = 2.0; arrB[2] = 2.0

    let result = broadcastGeKernel(a, b)
    check result.dtype == dtBool
    let resArr = cast[ptr UncheckedArray[uint8]](addr result.data[0])
    check resArr[0] == 0  # 1 >= 2 false
    check resArr[1] == 1  # 2 >= 2 true
    check resArr[2] == 1  # 3 >= 2 true

  test "broadcastWhereKernel":
    let cond = newTensorData(newShape(3), dtBool)
    var condArr = cast[ptr UncheckedArray[uint8]](addr cond.data[0])
    condArr[0] = 1; condArr[1] = 0; condArr[2] = 1

    let a = newTensorData(newShape(3), dtFloat32)
    let b = newTensorData(newShape(3), dtFloat32)
    var arrA = a.asFloat32
    var arrB = b.asFloat32
    arrA[0] = 10.0; arrA[1] = 20.0; arrA[2] = 30.0
    arrB[0] = 1.0; arrB[1] = 2.0; arrB[2] = 3.0

    let result = broadcastWhereKernel(cond, a, b)
    let resArr = result.asFloat32
    check resArr[0] == 10.0'f32  # cond true -> a
    check resArr[1] == 2.0'f32   # cond false -> b
    check resArr[2] == 30.0'f32  # cond true -> a
