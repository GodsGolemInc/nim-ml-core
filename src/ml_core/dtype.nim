## Data types for ML tensors
##
## Defines 15 data types covering floating-point, integer, boolean, and complex types.
## Provides type promotion rules for mixed-precision operations.

type
  DType* = enum
    ## Supported data types for tensors
    dtFloat16
    dtBFloat16
    dtFloat32
    dtFloat64
    dtInt8
    dtInt16
    dtInt32
    dtInt64
    dtUInt8
    dtUInt16
    dtUInt32
    dtUInt64
    dtBool
    dtComplex64
    dtComplex128

  DTypeCategory* = enum
    ## Category of data types for promotion rules
    dcFloat
    dcInteger
    dcUnsigned
    dcBool
    dcComplex

const DTypeNames*: array[DType, string] = [
  dtFloat16: "float16",
  dtBFloat16: "bfloat16",
  dtFloat32: "float32",
  dtFloat64: "float64",
  dtInt8: "int8",
  dtInt16: "int16",
  dtInt32: "int32",
  dtInt64: "int64",
  dtUInt8: "uint8",
  dtUInt16: "uint16",
  dtUInt32: "uint32",
  dtUInt64: "uint64",
  dtBool: "bool",
  dtComplex64: "complex64",
  dtComplex128: "complex128",
]

proc `$`*(dtype: DType): string =
  ## String representation of DType
  DTypeNames[dtype]

proc dtypeSize*(dtype: DType): int =
  ## Returns the size in bytes of a data type
  result = case dtype
  of dtFloat16, dtBFloat16: 2
  of dtFloat32: 4
  of dtFloat64: 8
  of dtInt8, dtUInt8, dtBool: 1
  of dtInt16, dtUInt16: 2
  of dtInt32, dtUInt32: 4
  of dtInt64, dtUInt64: 8
  of dtComplex64: 8
  of dtComplex128: 16

proc category*(dtype: DType): DTypeCategory =
  ## Returns the category of a data type
  case dtype
  of dtFloat16, dtBFloat16, dtFloat32, dtFloat64: dcFloat
  of dtInt8, dtInt16, dtInt32, dtInt64: dcInteger
  of dtUInt8, dtUInt16, dtUInt32, dtUInt64: dcUnsigned
  of dtBool: dcBool
  of dtComplex64, dtComplex128: dcComplex

proc isFloating*(dtype: DType): bool =
  ## Check if dtype is a floating-point type
  dtype.category == dcFloat

proc isInteger*(dtype: DType): bool =
  ## Check if dtype is an integer type (signed or unsigned)
  dtype.category in {dcInteger, dcUnsigned}

proc isSigned*(dtype: DType): bool =
  ## Check if dtype is a signed type
  dtype.category in {dcFloat, dcInteger, dcComplex}

proc isComplex*(dtype: DType): bool =
  ## Check if dtype is a complex type
  dtype.category == dcComplex

# Type promotion priority (higher = more priority in promotion)
const dtypePriority: array[DType, int] = [
  dtFloat16: 10,
  dtBFloat16: 11,
  dtFloat32: 20,
  dtFloat64: 30,
  dtInt8: 1,
  dtInt16: 2,
  dtInt32: 3,
  dtInt64: 4,
  dtUInt8: 1,
  dtUInt16: 2,
  dtUInt32: 3,
  dtUInt64: 4,
  dtBool: 0,
  dtComplex64: 40,
  dtComplex128: 50,
]

proc promote*(a, b: DType): DType =
  ## Compute the promoted type for mixed-precision operations.
  ## Rules:
  ## - Complex > Float > Integer > Bool
  ## - Within category, larger size wins
  ## - Float + Integer -> Float
  ## - Complex + any -> Complex

  if a == b:
    return a

  let catA = a.category
  let catB = b.category

  # Complex always wins
  if catA == dcComplex or catB == dcComplex:
    if a == dtComplex128 or b == dtComplex128:
      return dtComplex128
    return dtComplex64

  # Float wins over integer/bool
  if catA == dcFloat or catB == dcFloat:
    # Both float - return higher precision
    if catA == dcFloat and catB == dcFloat:
      if dtypePriority[a] >= dtypePriority[b]:
        return a
      return b
    # One float, one integer - return the float (or promote if needed)
    if catA == dcFloat:
      # If integer is 64-bit, promote to float64
      if b in {dtInt64, dtUInt64}:
        return dtFloat64
      return a
    else:
      if a in {dtInt64, dtUInt64}:
        return dtFloat64
      return b

  # Integer promotion
  if catA in {dcInteger, dcUnsigned} and catB in {dcInteger, dcUnsigned}:
    # Mixed signed/unsigned - promote to signed with larger size
    if catA != catB:
      let sizeA = dtypeSize(a)
      let sizeB = dtypeSize(b)
      let maxSize = max(sizeA, sizeB)
      # Need signed type that can hold both
      case maxSize
      of 1: return dtInt16  # int8 + uint8 needs int16
      of 2: return dtInt32
      of 4: return dtInt64
      else: return dtInt64
    # Same category - return larger
    if dtypePriority[a] >= dtypePriority[b]:
      return a
    return b

  # Bool promotes to anything
  if catA == dcBool:
    return b
  if catB == dcBool:
    return a

  # Default: return larger priority
  if dtypePriority[a] >= dtypePriority[b]:
    return a
  return b

proc canCast*(src, dst: DType): bool =
  ## Check if casting from src to dst is allowed
  ## All casts are allowed but some may lose precision
  true

proc mayLosePrecision*(src, dst: DType): bool =
  ## Check if casting from src to dst may lose precision
  if src == dst:
    return false

  let srcSize = dtypeSize(src)
  let dstSize = dtypeSize(dst)

  # Casting to smaller size loses precision
  if dstSize < srcSize:
    return true

  # Float to integer loses fractional part
  if src.isFloating and dst.isInteger:
    return true

  # Complex to non-complex loses imaginary part
  if src.isComplex and not dst.isComplex:
    return true

  # bfloat16 has fewer mantissa bits than float16
  if src == dtFloat16 and dst == dtBFloat16:
    return true

  false

proc defaultDType*(): DType =
  ## Returns the default data type for tensors
  dtFloat32

proc toNimType*(dtype: DType): string =
  ## Returns the corresponding Nim type name
  case dtype
  of dtFloat16: "float16"  # Would need custom type
  of dtBFloat16: "bfloat16"  # Would need custom type
  of dtFloat32: "float32"
  of dtFloat64: "float64"
  of dtInt8: "int8"
  of dtInt16: "int16"
  of dtInt32: "int32"
  of dtInt64: "int64"
  of dtUInt8: "uint8"
  of dtUInt16: "uint16"
  of dtUInt32: "uint32"
  of dtUInt64: "uint64"
  of dtBool: "bool"
  of dtComplex64: "Complex32"
  of dtComplex128: "Complex64"
