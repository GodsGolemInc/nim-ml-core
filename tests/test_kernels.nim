## Tests for CPU kernels
##
## Covers arithmetic, comparison, reduction, sort, and selection operations.

import unittest
import std/math
import ../src/ml_core/dtype
import ../src/ml_core/shape
import ../src/ml_core/tensor
import ../src/ml_core/kernels

# =============================================================================
# Test Helpers
# =============================================================================

proc createFloat32Tensor*(data: seq[float32], shape: Shape): TensorData =
  result = newTensorData(shape, dtFloat32)
  var arr = result.asFloat32
  for i, v in data:
    arr[i] = v

proc createFloat64Tensor*(data: seq[float64], shape: Shape): TensorData =
  result = newTensorData(shape, dtFloat64)
  var arr = result.asFloat64
  for i, v in data:
    arr[i] = v

proc createInt32Tensor*(data: seq[int32], shape: Shape): TensorData =
  result = newTensorData(shape, dtInt32)
  var arr = result.asInt32
  for i, v in data:
    arr[i] = v

proc createBoolTensor*(data: seq[bool], shape: Shape): TensorData =
  result = newTensorData(shape, dtBool)
  var arr = cast[ptr UncheckedArray[uint8]](addr result.data[0])
  for i, v in data:
    arr[i] = if v: 1'u8 else: 0'u8

proc almostEqual(a, b: float32, tol: float32 = 1e-5): bool =
  abs(a - b) < tol

proc almostEqual(a, b: float64, tol: float64 = 1e-10): bool =
  abs(a - b) < tol

# =============================================================================
# Arithmetic Kernel Tests
# =============================================================================

suite "Arithmetic Kernels - Unary":
  test "negKernel float32":
    let input = createFloat32Tensor(@[1.0'f32, -2.0, 3.0, 0.0], newShape(4))
    let result = negKernel(input)
    let arr = result.asFloat32
    check arr[0] == -1.0'f32
    check arr[1] == 2.0'f32
    check arr[2] == -3.0'f32
    check arr[3] == 0.0'f32

  test "absKernel float32":
    let input = createFloat32Tensor(@[-1.0'f32, 2.0, -3.0, 0.0], newShape(4))
    let result = absKernel(input)
    let arr = result.asFloat32
    check arr[0] == 1.0'f32
    check arr[1] == 2.0'f32
    check arr[2] == 3.0'f32
    check arr[3] == 0.0'f32

  test "sqrtKernel float32":
    let input = createFloat32Tensor(@[4.0'f32, 9.0, 16.0, 1.0], newShape(4))
    let result = sqrtKernel(input)
    let arr = result.asFloat32
    check almostEqual(arr[0], 2.0'f32)
    check almostEqual(arr[1], 3.0'f32)
    check almostEqual(arr[2], 4.0'f32)
    check almostEqual(arr[3], 1.0'f32)

  test "squareKernel float32":
    let input = createFloat32Tensor(@[2.0'f32, -3.0, 0.0, 4.0], newShape(4))
    let result = squareKernel(input)
    let arr = result.asFloat32
    check almostEqual(arr[0], 4.0'f32)
    check almostEqual(arr[1], 9.0'f32)
    check almostEqual(arr[2], 0.0'f32)
    check almostEqual(arr[3], 16.0'f32)

  test "expKernel float32":
    let input = createFloat32Tensor(@[0.0'f32, 1.0], newShape(2))
    let result = expKernel(input)
    let arr = result.asFloat32
    check almostEqual(arr[0], 1.0'f32)
    check almostEqual(arr[1], E.float32, 1e-4)

  test "logKernel float32":
    let input = createFloat32Tensor(@[1.0'f32, E.float32], newShape(2))
    let result = logKernel(input)
    let arr = result.asFloat32
    check almostEqual(arr[0], 0.0'f32)
    check almostEqual(arr[1], 1.0'f32, 1e-4)

  test "signKernel float32":
    let input = createFloat32Tensor(@[-5.0'f32, 0.0, 3.0], newShape(3))
    let result = signKernel(input)
    let arr = result.asFloat32
    check arr[0] == -1.0'f32
    check arr[1] == 0.0'f32
    check arr[2] == 1.0'f32

suite "Arithmetic Kernels - Binary":
  test "addKernel float32":
    let a = createFloat32Tensor(@[1.0'f32, 2.0, 3.0], newShape(3))
    let b = createFloat32Tensor(@[4.0'f32, 5.0, 6.0], newShape(3))
    let result = addKernel(a, b)
    let arr = result.asFloat32
    check arr[0] == 5.0'f32
    check arr[1] == 7.0'f32
    check arr[2] == 9.0'f32

  test "subKernel float32":
    let a = createFloat32Tensor(@[5.0'f32, 8.0, 3.0], newShape(3))
    let b = createFloat32Tensor(@[2.0'f32, 3.0, 4.0], newShape(3))
    let result = subKernel(a, b)
    let arr = result.asFloat32
    check arr[0] == 3.0'f32
    check arr[1] == 5.0'f32
    check arr[2] == -1.0'f32

  test "mulKernel float32":
    let a = createFloat32Tensor(@[2.0'f32, 3.0, 4.0], newShape(3))
    let b = createFloat32Tensor(@[5.0'f32, 2.0, 0.5], newShape(3))
    let result = mulKernel(a, b)
    let arr = result.asFloat32
    check arr[0] == 10.0'f32
    check arr[1] == 6.0'f32
    check arr[2] == 2.0'f32

  test "divKernel float32":
    let a = createFloat32Tensor(@[10.0'f32, 9.0, 8.0], newShape(3))
    let b = createFloat32Tensor(@[2.0'f32, 3.0, 4.0], newShape(3))
    let result = divKernel(a, b)
    let arr = result.asFloat32
    check arr[0] == 5.0'f32
    check arr[1] == 3.0'f32
    check arr[2] == 2.0'f32

  test "powKernel float32":
    let a = createFloat32Tensor(@[2.0'f32, 3.0, 4.0], newShape(3))
    let b = createFloat32Tensor(@[3.0'f32, 2.0, 0.5], newShape(3))
    let result = powKernel(a, b)
    let arr = result.asFloat32
    check almostEqual(arr[0], 8.0'f32)
    check almostEqual(arr[1], 9.0'f32)
    check almostEqual(arr[2], 2.0'f32)

  test "maxKernel float32":
    let a = createFloat32Tensor(@[1.0'f32, 5.0, 3.0], newShape(3))
    let b = createFloat32Tensor(@[2.0'f32, 3.0, 4.0], newShape(3))
    let result = maxKernel(a, b)
    let arr = result.asFloat32
    check arr[0] == 2.0'f32
    check arr[1] == 5.0'f32
    check arr[2] == 4.0'f32

  test "minKernel float32":
    let a = createFloat32Tensor(@[1.0'f32, 5.0, 3.0], newShape(3))
    let b = createFloat32Tensor(@[2.0'f32, 3.0, 4.0], newShape(3))
    let result = minKernel(a, b)
    let arr = result.asFloat32
    check arr[0] == 1.0'f32
    check arr[1] == 3.0'f32
    check arr[2] == 3.0'f32

suite "Arithmetic Kernels - Scalar":
  test "scaleKernel float32":
    let input = createFloat32Tensor(@[1.0'f32, 2.0, 3.0], newShape(3))
    let result = scaleKernel(input, 2.5)
    let arr = result.asFloat32
    check arr[0] == 2.5'f32
    check arr[1] == 5.0'f32
    check arr[2] == 7.5'f32

# =============================================================================
# Comparison Kernel Tests
# =============================================================================

suite "Comparison Kernels":
  test "geKernel float32":
    let a = createFloat32Tensor(@[1.0'f32, 2.0, 3.0, 3.0], newShape(4))
    let b = createFloat32Tensor(@[2.0'f32, 2.0, 2.0, 4.0], newShape(4))
    let result = geKernel(a, b)
    let arr = cast[ptr UncheckedArray[uint8]](addr result.data[0])
    check arr[0] == 0'u8  # 1 >= 2 = false
    check arr[1] == 1'u8  # 2 >= 2 = true
    check arr[2] == 1'u8  # 3 >= 2 = true
    check arr[3] == 0'u8  # 3 >= 4 = false

  test "leKernel float32":
    let a = createFloat32Tensor(@[1.0'f32, 2.0, 3.0, 3.0], newShape(4))
    let b = createFloat32Tensor(@[2.0'f32, 2.0, 2.0, 4.0], newShape(4))
    let result = leKernel(a, b)
    let arr = cast[ptr UncheckedArray[uint8]](addr result.data[0])
    check arr[0] == 1'u8  # 1 <= 2 = true
    check arr[1] == 1'u8  # 2 <= 2 = true
    check arr[2] == 0'u8  # 3 <= 2 = false
    check arr[3] == 1'u8  # 3 <= 4 = true

  test "gtKernel float32":
    let a = createFloat32Tensor(@[1.0'f32, 2.0, 3.0], newShape(3))
    let b = createFloat32Tensor(@[2.0'f32, 2.0, 2.0], newShape(3))
    let result = gtKernel(a, b)
    let arr = cast[ptr UncheckedArray[uint8]](addr result.data[0])
    check arr[0] == 0'u8  # 1 > 2 = false
    check arr[1] == 0'u8  # 2 > 2 = false
    check arr[2] == 1'u8  # 3 > 2 = true

  test "ltKernel float32":
    let a = createFloat32Tensor(@[1.0'f32, 2.0, 3.0], newShape(3))
    let b = createFloat32Tensor(@[2.0'f32, 2.0, 2.0], newShape(3))
    let result = ltKernel(a, b)
    let arr = cast[ptr UncheckedArray[uint8]](addr result.data[0])
    check arr[0] == 1'u8  # 1 < 2 = true
    check arr[1] == 0'u8  # 2 < 2 = false
    check arr[2] == 0'u8  # 3 < 2 = false

  test "eqKernel float32":
    let a = createFloat32Tensor(@[1.0'f32, 2.0, 3.0], newShape(3))
    let b = createFloat32Tensor(@[1.0'f32, 3.0, 3.0], newShape(3))
    let result = eqKernel(a, b)
    let arr = cast[ptr UncheckedArray[uint8]](addr result.data[0])
    check arr[0] == 1'u8  # 1 == 1 = true
    check arr[1] == 0'u8  # 2 == 3 = false
    check arr[2] == 1'u8  # 3 == 3 = true

  test "neKernel float32":
    let a = createFloat32Tensor(@[1.0'f32, 2.0, 3.0], newShape(3))
    let b = createFloat32Tensor(@[1.0'f32, 3.0, 3.0], newShape(3))
    let result = neKernel(a, b)
    let arr = cast[ptr UncheckedArray[uint8]](addr result.data[0])
    check arr[0] == 0'u8  # 1 != 1 = false
    check arr[1] == 1'u8  # 2 != 3 = true
    check arr[2] == 0'u8  # 3 != 3 = false

  test "geScalarKernel float32":
    let input = createFloat32Tensor(@[1.0'f32, 2.0, 3.0, 4.0], newShape(4))
    let result = geScalarKernel(input, 2.5)
    let arr = cast[ptr UncheckedArray[uint8]](addr result.data[0])
    check arr[0] == 0'u8  # 1 >= 2.5 = false
    check arr[1] == 0'u8  # 2 >= 2.5 = false
    check arr[2] == 1'u8  # 3 >= 2.5 = true
    check arr[3] == 1'u8  # 4 >= 2.5 = true

  test "leScalarKernel float32":
    let input = createFloat32Tensor(@[1.0'f32, 2.0, 3.0, 4.0], newShape(4))
    let result = leScalarKernel(input, 2.5)
    let arr = cast[ptr UncheckedArray[uint8]](addr result.data[0])
    check arr[0] == 1'u8  # 1 <= 2.5 = true
    check arr[1] == 1'u8  # 2 <= 2.5 = true
    check arr[2] == 0'u8  # 3 <= 2.5 = false
    check arr[3] == 0'u8  # 4 <= 2.5 = false

# =============================================================================
# Reduction Kernel Tests
# =============================================================================

suite "Reduction Kernels":
  test "sumKernel float32":
    let input = createFloat32Tensor(@[1.0'f32, 2.0, 3.0, 4.0], newShape(4))
    let result = sumKernel(input)
    check almostEqual(result.asFloat32[0], 10.0'f32)

  test "meanKernel float32":
    let input = createFloat32Tensor(@[1.0'f32, 2.0, 3.0, 4.0], newShape(4))
    let result = meanKernel(input)
    check almostEqual(result.asFloat32[0], 2.5'f32)

  test "maxReduceKernel float32":
    let input = createFloat32Tensor(@[3.0'f32, 1.0, 4.0, 1.0, 5.0], newShape(5))
    let result = maxReduceKernel(input)
    check result.asFloat32[0] == 5.0'f32

  test "minReduceKernel float32":
    let input = createFloat32Tensor(@[3.0'f32, 1.0, 4.0, 1.0, 5.0], newShape(5))
    let result = minReduceKernel(input)
    check result.asFloat32[0] == 1.0'f32

  test "prodKernel float32":
    let input = createFloat32Tensor(@[1.0'f32, 2.0, 3.0, 4.0], newShape(4))
    let result = prodKernel(input)
    check almostEqual(result.asFloat32[0], 24.0'f32)

suite "Norm Kernels":
  test "normL1Kernel float32":
    let input = createFloat32Tensor(@[-1.0'f32, 2.0, -3.0, 4.0], newShape(4))
    let result = normL1Kernel(input)
    check almostEqual(result.asFloat32[0], 10.0'f32)

  test "normL2Kernel float32":
    let input = createFloat32Tensor(@[3.0'f32, 4.0], newShape(2))
    let result = normL2Kernel(input)
    check almostEqual(result.asFloat32[0], 5.0'f32)

  test "normKernel float32 p=3":
    let input = createFloat32Tensor(@[1.0'f32, 2.0, 2.0], newShape(3))
    let result = normKernel(input, 3.0)
    # (1^3 + 2^3 + 2^3)^(1/3) = (1 + 8 + 8)^(1/3) = 17^(1/3) ≈ 2.571
    check almostEqual(result.asFloat32[0], pow(17.0, 1.0/3.0).float32, 1e-4)

  test "normInfKernel float32":
    let input = createFloat32Tensor(@[-5.0'f32, 3.0, -1.0, 4.0], newShape(4))
    let result = normInfKernel(input)
    check result.asFloat32[0] == 5.0'f32

# =============================================================================
# Sort Kernel Tests
# =============================================================================

suite "Sort Kernels":
  test "sortKernel ascending float32":
    let input = createFloat32Tensor(@[3.0'f32, 1.0, 4.0, 1.0, 5.0], newShape(5))
    let (values, indices) = sortKernel(input, descending = false)
    let valArr = values.asFloat32
    let idxArr = indices.asInt64
    check valArr[0] == 1.0'f32
    check valArr[1] == 1.0'f32
    check valArr[2] == 3.0'f32
    check valArr[3] == 4.0'f32
    check valArr[4] == 5.0'f32
    # Indices should point to original positions
    check idxArr[0] == 1 or idxArr[0] == 3  # Either 1.0 at pos 1 or 3

  test "sortKernel descending float32":
    let input = createFloat32Tensor(@[3.0'f32, 1.0, 4.0, 1.0, 5.0], newShape(5))
    let (values, indices) = sortKernel(input, descending = true)
    let valArr = values.asFloat32
    check valArr[0] == 5.0'f32
    check valArr[1] == 4.0'f32
    check valArr[2] == 3.0'f32
    check valArr[3] == 1.0'f32
    check valArr[4] == 1.0'f32

  test "argsortKernel float32":
    let input = createFloat32Tensor(@[3.0'f32, 1.0, 2.0], newShape(3))
    let indices = argsortKernel(input, descending = false)
    let arr = indices.asInt64
    check arr[0] == 1  # 1.0 is smallest, at index 1
    check arr[1] == 2  # 2.0 is middle, at index 2
    check arr[2] == 0  # 3.0 is largest, at index 0

  test "kthvalueKernel float32":
    let input = createFloat32Tensor(@[5.0'f32, 2.0, 8.0, 1.0, 9.0], newShape(5))
    # Find 3rd smallest (sorted: 1, 2, 5, 8, 9)
    let (value, index) = kthvalueKernel(input, 3)
    check value.asFloat32[0] == 5.0'f32
    check index.asInt64[0] == 0  # 5.0 is at original index 0

  test "topkKernel largest float32":
    let input = createFloat32Tensor(@[5.0'f32, 2.0, 8.0, 1.0, 9.0], newShape(5))
    let (values, indices) = topkKernel(input, 3, largest = true)
    let valArr = values.asFloat32
    let idxArr = indices.asInt64
    check valArr[0] == 9.0'f32
    check valArr[1] == 8.0'f32
    check valArr[2] == 5.0'f32
    check idxArr[0] == 4  # 9.0 at index 4
    check idxArr[1] == 2  # 8.0 at index 2
    check idxArr[2] == 0  # 5.0 at index 0

  test "topkKernel smallest float32":
    let input = createFloat32Tensor(@[5.0'f32, 2.0, 8.0, 1.0, 9.0], newShape(5))
    let (values, indices) = topkKernel(input, 2, largest = false)
    let valArr = values.asFloat32
    let idxArr = indices.asInt64
    check valArr[0] == 1.0'f32
    check valArr[1] == 2.0'f32
    check idxArr[0] == 3  # 1.0 at index 3
    check idxArr[1] == 1  # 2.0 at index 1

  test "medianKernel odd count float32":
    let input = createFloat32Tensor(@[3.0'f32, 1.0, 2.0, 5.0, 4.0], newShape(5))
    let result = medianKernel(input)
    check result.asFloat32[0] == 3.0'f32  # sorted: 1,2,3,4,5 -> median = 3

  test "medianKernel even count float32":
    let input = createFloat32Tensor(@[4.0'f32, 1.0, 3.0, 2.0], newShape(4))
    let result = medianKernel(input)
    check result.asFloat32[0] == 2.5'f32  # sorted: 1,2,3,4 -> median = (2+3)/2 = 2.5

# =============================================================================
# Selection Kernel Tests
# =============================================================================

suite "Selection Kernels":
  test "whereKernel float32":
    let condition = createBoolTensor(@[true, false, true, false], newShape(4))
    let x = createFloat32Tensor(@[1.0'f32, 2.0, 3.0, 4.0], newShape(4))
    let y = createFloat32Tensor(@[10.0'f32, 20.0, 30.0, 40.0], newShape(4))
    let result = whereKernel(condition, x, y)
    let arr = result.asFloat32
    check arr[0] == 1.0'f32   # true: x[0]
    check arr[1] == 20.0'f32  # false: y[1]
    check arr[2] == 3.0'f32   # true: x[2]
    check arr[3] == 40.0'f32  # false: y[3]

  test "whereScalarKernel float32":
    let condition = createBoolTensor(@[true, false, true], newShape(3))
    let result = whereScalarKernel(condition, 100.0, 0.0, dtFloat32)
    let arr = result.asFloat32
    check arr[0] == 100.0'f32
    check arr[1] == 0.0'f32
    check arr[2] == 100.0'f32

  test "clampKernel float32":
    let input = createFloat32Tensor(@[-2.0'f32, 0.5, 1.5, 3.0], newShape(4))
    let result = clampKernel(input, 0.0, 2.0)
    let arr = result.asFloat32
    check arr[0] == 0.0'f32   # clamped to min
    check arr[1] == 0.5'f32   # unchanged
    check arr[2] == 1.5'f32   # unchanged
    check arr[3] == 2.0'f32   # clamped to max

  test "clampMinKernel float32":
    let input = createFloat32Tensor(@[-2.0'f32, 0.5, 1.5, 3.0], newShape(4))
    let result = clampMinKernel(input, 0.0)
    let arr = result.asFloat32
    check arr[0] == 0.0'f32   # clamped
    check arr[1] == 0.5'f32
    check arr[2] == 1.5'f32
    check arr[3] == 3.0'f32

  test "clampMaxKernel float32":
    let input = createFloat32Tensor(@[-2.0'f32, 0.5, 1.5, 3.0], newShape(4))
    let result = clampMaxKernel(input, 1.0)
    let arr = result.asFloat32
    check arr[0] == -2.0'f32
    check arr[1] == 0.5'f32
    check arr[2] == 1.0'f32   # clamped
    check arr[3] == 1.0'f32   # clamped

  test "maskedFillKernel float32":
    let input = createFloat32Tensor(@[1.0'f32, 2.0, 3.0, 4.0], newShape(4))
    let mask = createBoolTensor(@[false, true, false, true], newShape(4))
    let result = maskedFillKernel(input, mask, 0.0)
    let arr = result.asFloat32
    check arr[0] == 1.0'f32   # not masked
    check arr[1] == 0.0'f32   # masked -> 0
    check arr[2] == 3.0'f32   # not masked
    check arr[3] == 0.0'f32   # masked -> 0

  test "applyMaskKernel float32":
    let input = createFloat32Tensor(@[1.0'f32, 2.0, 3.0, 4.0], newShape(4))
    let mask = createBoolTensor(@[true, false, true, false], newShape(4))
    let result = applyMaskKernel(input, mask)
    let arr = result.asFloat32
    check arr[0] == 1.0'f32   # kept
    check arr[1] == 0.0'f32   # zeroed
    check arr[2] == 3.0'f32   # kept
    check arr[3] == 0.0'f32   # zeroed

# =============================================================================
# Float64 and Int32 Coverage Tests
# =============================================================================

suite "Multi-dtype Coverage":
  test "addKernel float64":
    let a = createFloat64Tensor(@[1.0, 2.0, 3.0], newShape(3))
    let b = createFloat64Tensor(@[4.0, 5.0, 6.0], newShape(3))
    let result = addKernel(a, b)
    let arr = result.asFloat64
    check arr[0] == 5.0
    check arr[1] == 7.0
    check arr[2] == 9.0

  test "sumKernel float64":
    let input = createFloat64Tensor(@[1.0, 2.0, 3.0, 4.0], newShape(4))
    let result = sumKernel(input)
    check almostEqual(result.asFloat64[0], 10.0)

  test "addKernel int32":
    let a = createInt32Tensor(@[1'i32, 2, 3], newShape(3))
    let b = createInt32Tensor(@[4'i32, 5, 6], newShape(3))
    let result = addKernel(a, b)
    let arr = result.asInt32
    check arr[0] == 5'i32
    check arr[1] == 7'i32
    check arr[2] == 9'i32

  test "geKernel int32":
    let a = createInt32Tensor(@[1'i32, 2, 3, 3], newShape(4))
    let b = createInt32Tensor(@[2'i32, 2, 2, 4], newShape(4))
    let result = geKernel(a, b)
    let arr = cast[ptr UncheckedArray[uint8]](addr result.data[0])
    check arr[0] == 0'u8
    check arr[1] == 1'u8
    check arr[2] == 1'u8
    check arr[3] == 0'u8

  test "clampKernel int32":
    let input = createInt32Tensor(@[-5'i32, 0, 5, 10], newShape(4))
    let result = clampKernel(input, 0.0, 7.0)
    let arr = result.asInt32
    check arr[0] == 0'i32
    check arr[1] == 0'i32
    check arr[2] == 5'i32
    check arr[3] == 7'i32

  test "sortKernel float64":
    let input = createFloat64Tensor(@[3.0, 1.0, 2.0], newShape(3))
    let (values, indices) = sortKernel(input, descending = false)
    let valArr = values.asFloat64
    check valArr[0] == 1.0
    check valArr[1] == 2.0
    check valArr[2] == 3.0
