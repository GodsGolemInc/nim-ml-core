## Tests for dtype module

import unittest
import ../src/ml_core/dtype

suite "DType":
  test "dtypeSize returns correct sizes":
    check dtypeSize(dtFloat16) == 2
    check dtypeSize(dtBFloat16) == 2
    check dtypeSize(dtFloat32) == 4
    check dtypeSize(dtFloat64) == 8
    check dtypeSize(dtInt8) == 1
    check dtypeSize(dtInt16) == 2
    check dtypeSize(dtInt32) == 4
    check dtypeSize(dtInt64) == 8
    check dtypeSize(dtUInt8) == 1
    check dtypeSize(dtBool) == 1
    check dtypeSize(dtComplex64) == 8
    check dtypeSize(dtComplex128) == 16

  test "category classification":
    check category(dtFloat32) == dcFloat
    check category(dtInt32) == dcInteger
    check category(dtUInt32) == dcUnsigned
    check category(dtBool) == dcBool
    check category(dtComplex64) == dcComplex

  test "isFloating":
    check isFloating(dtFloat16) == true
    check isFloating(dtFloat32) == true
    check isFloating(dtInt32) == false
    check isFloating(dtComplex64) == false

  test "isInteger":
    check isInteger(dtInt32) == true
    check isInteger(dtUInt32) == true
    check isInteger(dtFloat32) == false

  test "isSigned":
    check isSigned(dtInt32) == true
    check isSigned(dtFloat32) == true
    check isSigned(dtUInt32) == false

  test "isComplex":
    check isComplex(dtComplex64) == true
    check isComplex(dtComplex128) == true
    check isComplex(dtFloat64) == false

suite "Type Promotion":
  test "same type returns itself":
    check promote(dtFloat32, dtFloat32) == dtFloat32
    check promote(dtInt64, dtInt64) == dtInt64

  test "float promotion":
    check promote(dtFloat16, dtFloat32) == dtFloat32
    check promote(dtFloat32, dtFloat64) == dtFloat64
    check promote(dtBFloat16, dtFloat32) == dtFloat32

  test "float + integer = float":
    check promote(dtFloat32, dtInt32) == dtFloat32
    check promote(dtInt32, dtFloat32) == dtFloat32

  test "int64 + float32 = float64":
    check promote(dtFloat32, dtInt64) == dtFloat64
    check promote(dtInt64, dtFloat32) == dtFloat64

  test "integer promotion":
    check promote(dtInt8, dtInt32) == dtInt32
    check promote(dtInt16, dtInt64) == dtInt64

  test "signed + unsigned = signed with larger size":
    check promote(dtInt8, dtUInt8) == dtInt16
    check promote(dtInt16, dtUInt16) == dtInt32
    check promote(dtInt32, dtUInt32) == dtInt64

  test "complex wins":
    check promote(dtComplex64, dtFloat64) == dtComplex64
    check promote(dtFloat32, dtComplex128) == dtComplex128
    check promote(dtComplex64, dtComplex128) == dtComplex128

  test "bool promotes to anything":
    check promote(dtBool, dtInt32) == dtInt32
    check promote(dtBool, dtFloat32) == dtFloat32
    check promote(dtFloat64, dtBool) == dtFloat64

suite "Type Casting":
  test "canCast always returns true":
    check canCast(dtFloat32, dtInt32) == true
    check canCast(dtComplex128, dtBool) == true

  test "mayLosePrecision":
    check mayLosePrecision(dtFloat32, dtFloat32) == false
    check mayLosePrecision(dtFloat64, dtFloat32) == true
    check mayLosePrecision(dtFloat32, dtInt32) == true
    check mayLosePrecision(dtInt32, dtFloat32) == false
    check mayLosePrecision(dtComplex64, dtFloat32) == true

  test "defaultDType":
    check defaultDType() == dtFloat32

  test "toNimType":
    check toNimType(dtFloat32) == "float32"
    check toNimType(dtInt64) == "int64"
    check toNimType(dtBool) == "bool"

suite "DType Additional Coverage":
  test "string representation":
    check $dtFloat32 == "float32"
    check $dtInt64 == "int64"
    check $dtBool == "bool"
    check $dtComplex128 == "complex128"

  test "dtypeSize all types":
    check dtypeSize(dtUInt16) == 2
    check dtypeSize(dtUInt32) == 4
    check dtypeSize(dtUInt64) == 8

  test "category all types":
    check category(dtFloat16) == dcFloat
    check category(dtBFloat16) == dcFloat
    check category(dtFloat64) == dcFloat
    check category(dtInt8) == dcInteger
    check category(dtInt16) == dcInteger
    check category(dtUInt8) == dcUnsigned
    check category(dtUInt16) == dcUnsigned
    check category(dtUInt64) == dcUnsigned
    check category(dtComplex128) == dcComplex

  test "isFloating all float types":
    check isFloating(dtBFloat16) == true
    check isFloating(dtFloat64) == true

  test "isSigned complex types":
    check isSigned(dtComplex64) == true
    check isSigned(dtComplex128) == true

  test "mayLosePrecision float16 to bfloat16":
    check mayLosePrecision(dtFloat16, dtBFloat16) == true

  test "toNimType all types":
    check toNimType(dtFloat16) == "float16"
    check toNimType(dtBFloat16) == "bfloat16"
    check toNimType(dtFloat64) == "float64"
    check toNimType(dtInt8) == "int8"
    check toNimType(dtInt16) == "int16"
    check toNimType(dtInt32) == "int32"
    check toNimType(dtUInt8) == "uint8"
    check toNimType(dtUInt16) == "uint16"
    check toNimType(dtUInt32) == "uint32"
    check toNimType(dtUInt64) == "uint64"
    check toNimType(dtComplex64) == "Complex32"
    check toNimType(dtComplex128) == "Complex64"

  test "promote unsigned same category":
    check promote(dtUInt8, dtUInt32) == dtUInt32
    check promote(dtUInt32, dtUInt8) == dtUInt32

  test "promote uint64 with float":
    check promote(dtUInt64, dtFloat32) == dtFloat64
