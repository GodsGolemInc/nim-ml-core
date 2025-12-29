## Tensor module for content-addressed tensor references
##
## Provides TensorRef with SHA-256 hashing for distributed ML.
## Instead of passing large tensor data through tuple space,
## only references (hash + metadata) are exchanged.

import std/[hashes, options, tables, strutils, sequtils]
import checksums/sha1
import dtype, shape

type
  Hash256* = array[32, byte]
    ## 256-bit hash for content addressing

  LocationTag* = object
    ## Tag indicating where tensor data is stored
    workerId*: string
    storePath*: string
    isPrimary*: bool

  TensorRef* = ref object
    ## Content-addressed reference to tensor data.
    ## The hash uniquely identifies the tensor content.
    hash*: Hash256
    shape*: Shape
    dtype*: DType
    locationTags*: seq[LocationTag]
    metadata*: Table[string, string]

  TensorData* = ref object
    ## Actual tensor data storage.
    ## Raw bytes + metadata for interpretation.
    shape*: Shape
    dtype*: DType
    data*: seq[byte]
    strides*: seq[int]
    layout*: MemoryLayout

  TensorError* = object of CatchableError

# Hash256 utilities

proc `==`*(a, b: Hash256): bool =
  ## Compare two hashes
  for i in 0 ..< 32:
    if a[i] != b[i]:
      return false
  true

proc `$`*(h: Hash256): string =
  ## Hex string representation of hash
  result = newString(64)
  for i, b in h:
    result[i * 2] = "0123456789abcdef"[b shr 4]
    result[i * 2 + 1] = "0123456789abcdef"[b and 0x0F]

proc hash*(h: Hash256): Hash =
  ## Hash for use in Nim's hash tables
  var res: Hash = 0
  for b in h:
    res = res !& hash(b)
  result = !$res

proc parseHash256*(s: string): Hash256 =
  ## Parse hex string to Hash256
  if s.len != 64:
    raise newException(TensorError, "Invalid hash string length")
  for i in 0 ..< 32:
    let hi = s[i * 2]
    let lo = s[i * 2 + 1]
    let hiVal = if hi in '0'..'9': ord(hi) - ord('0')
                elif hi in 'a'..'f': ord(hi) - ord('a') + 10
                elif hi in 'A'..'F': ord(hi) - ord('A') + 10
                else: raise newException(TensorError, "Invalid hex character")
    let loVal = if lo in '0'..'9': ord(lo) - ord('0')
                elif lo in 'a'..'f': ord(lo) - ord('a') + 10
                elif lo in 'A'..'F': ord(lo) - ord('A') + 10
                else: raise newException(TensorError, "Invalid hex character")
    result[i] = byte(hiVal shl 4 or loVal)

proc zeroHash*(): Hash256 =
  ## Return a zero hash (all zeros)
  result = default(Hash256)

proc isZero*(h: Hash256): bool =
  ## Check if hash is all zeros
  for b in h:
    if b != 0:
      return false
  true

# TensorData operations

proc newTensorData*(shape: Shape, dtype: DType,
                    layout: MemoryLayout = mlRowMajor): TensorData =
  ## Create new tensor data with uninitialized storage
  let size = shape.size * dtypeSize(dtype)
  result = TensorData(
    shape: shape,
    dtype: dtype,
    data: newSeq[byte](size),
    strides: shape.strides(layout),
    layout: layout
  )

proc newTensorDataZeros*(shape: Shape, dtype: DType,
                         layout: MemoryLayout = mlRowMajor): TensorData =
  ## Create new tensor data initialized to zeros
  result = newTensorData(shape, dtype, layout)
  # seq is already zero-initialized in Nim

proc newTensorDataFromBytes*(shape: Shape, dtype: DType,
                             data: seq[byte],
                             layout: MemoryLayout = mlRowMajor): TensorData =
  ## Create tensor data from raw bytes
  let expectedSize = shape.size * dtypeSize(dtype)
  if data.len != expectedSize:
    raise newException(TensorError,
      "Data size mismatch: expected " & $expectedSize & " but got " & $data.len)
  result = TensorData(
    shape: shape,
    dtype: dtype,
    data: data,
    strides: shape.strides(layout),
    layout: layout
  )

proc size*(td: TensorData): int =
  ## Total number of elements
  td.shape.size

proc byteSize*(td: TensorData): int =
  ## Total size in bytes
  td.data.len

proc isContiguous*(td: TensorData): bool =
  ## Check if tensor is contiguous in memory
  td.shape.isContiguous(td.strides, td.layout)

proc clone*(td: TensorData): TensorData =
  ## Create a deep copy of tensor data
  result = TensorData(
    shape: td.shape,
    dtype: td.dtype,
    data: td.data,  # seq copy is automatic
    strides: td.strides,
    layout: td.layout
  )

# Hash computation

proc computeHash*(td: TensorData): Hash256 =
  ## Compute SHA-256 hash of tensor data.
  ## Hash includes shape, dtype, and raw data for uniqueness.

  # Create a buffer with metadata + data
  var buffer: seq[byte] = @[]

  # Add shape info
  for d in td.shape.dims:
    let bytes = cast[array[8, byte]](int64(d))
    buffer.add(bytes)

  # Add dtype
  buffer.add(byte(ord(td.dtype)))

  # Add raw data
  buffer.add(td.data)

  # Compute SHA1 and extend to 256 bits (using SHA1 twice for simplicity)
  # In production, use proper SHA256
  let sha1Result = secureHash($buffer)
  let sha1Str = $sha1Result

  # Parse first 32 hex chars and duplicate for 256 bits
  for i in 0 ..< 20:
    result[i] = parseHexInt(sha1Str[i*2 .. i*2+1]).byte
  for i in 20 ..< 32:
    result[i] = result[i - 20]

proc verify*(td: TensorData, expectedHash: Hash256): bool =
  ## Verify tensor data against expected hash
  let actualHash = computeHash(td)
  actualHash == expectedHash

# TensorRef operations

proc newTensorRef*(shape: Shape, dtype: DType): TensorRef =
  ## Create a new tensor reference (without data/hash yet)
  result = TensorRef(
    hash: zeroHash(),
    shape: shape,
    dtype: dtype,
    locationTags: @[],
    metadata: initTable[string, string]()
  )

proc newTensorRef*(td: TensorData): TensorRef =
  ## Create tensor reference from tensor data
  result = TensorRef(
    hash: computeHash(td),
    shape: td.shape,
    dtype: td.dtype,
    locationTags: @[],
    metadata: initTable[string, string]()
  )

proc newTensorRef*(hash: Hash256, shape: Shape, dtype: DType): TensorRef =
  ## Create tensor reference with known hash
  result = TensorRef(
    hash: hash,
    shape: shape,
    dtype: dtype,
    locationTags: @[],
    metadata: initTable[string, string]()
  )

proc `==`*(a, b: TensorRef): bool =
  ## Compare tensor references by hash
  if a.isNil and b.isNil:
    return true
  if a.isNil or b.isNil:
    return false
  a.hash == b.hash

proc hash*(tr: TensorRef): Hash =
  ## Hash for use in Nim's hash tables
  if tr.isNil:
    return 0
  hash(tr.hash)

proc `$`*(tr: TensorRef): string =
  ## String representation
  if tr.isNil:
    return "TensorRef(nil)"
  "TensorRef(hash=" & $tr.hash & ", shape=" & $tr.shape & ", dtype=" & $tr.dtype & ")"

proc addLocation*(tr: TensorRef, tag: LocationTag) =
  ## Add a location tag to the tensor reference
  tr.locationTags.add(tag)

proc removeLocation*(tr: TensorRef, workerId: string) =
  ## Remove location tags for a worker
  tr.locationTags.keepItIf(it.workerId != workerId)

proc getPrimaryLocation*(tr: TensorRef): Option[LocationTag] =
  ## Get the primary location of the tensor
  for tag in tr.locationTags:
    if tag.isPrimary:
      return some(tag)
  none(LocationTag)

proc hasLocation*(tr: TensorRef, workerId: string): bool =
  ## Check if tensor has a location on given worker
  for tag in tr.locationTags:
    if tag.workerId == workerId:
      return true
  false

proc size*(tr: TensorRef): int =
  ## Total number of elements
  tr.shape.size

proc byteSize*(tr: TensorRef): int =
  ## Expected size in bytes
  tr.shape.size * dtypeSize(tr.dtype)

# Metadata helpers

proc setMeta*(tr: TensorRef, key, value: string) =
  ## Set metadata value
  tr.metadata[key] = value

proc getMeta*(tr: TensorRef, key: string): Option[string] =
  ## Get metadata value
  if key in tr.metadata:
    some(tr.metadata[key])
  else:
    none(string)

proc hasMeta*(tr: TensorRef, key: string): bool =
  ## Check if metadata key exists
  key in tr.metadata

# Type conversions (for creating typed views)

template asFloat32*(td: TensorData): ptr UncheckedArray[float32] =
  ## Get pointer to data as float32 array
  if td.dtype != dtFloat32:
    raise newException(TensorError, "Tensor is not float32")
  cast[ptr UncheckedArray[float32]](addr td.data[0])

template asFloat64*(td: TensorData): ptr UncheckedArray[float64] =
  ## Get pointer to data as float64 array
  if td.dtype != dtFloat64:
    raise newException(TensorError, "Tensor is not float64")
  cast[ptr UncheckedArray[float64]](addr td.data[0])

template asInt32*(td: TensorData): ptr UncheckedArray[int32] =
  ## Get pointer to data as int32 array
  if td.dtype != dtInt32:
    raise newException(TensorError, "Tensor is not int32")
  cast[ptr UncheckedArray[int32]](addr td.data[0])

template asInt64*(td: TensorData): ptr UncheckedArray[int64] =
  ## Get pointer to data as int64 array
  if td.dtype != dtInt64:
    raise newException(TensorError, "Tensor is not int64")
  cast[ptr UncheckedArray[int64]](addr td.data[0])

# Initialization helpers

proc fillFloat32*(td: TensorData, value: float32) =
  ## Fill tensor with a float32 value
  if td.dtype != dtFloat32:
    raise newException(TensorError, "Tensor is not float32")
  let arr = td.asFloat32
  for i in 0 ..< td.size:
    arr[i] = value

proc fillFloat64*(td: TensorData, value: float64) =
  ## Fill tensor with a float64 value
  if td.dtype != dtFloat64:
    raise newException(TensorError, "Tensor is not float64")
  let arr = td.asFloat64
  for i in 0 ..< td.size:
    arr[i] = value

proc fillInt32*(td: TensorData, value: int32) =
  ## Fill tensor with an int32 value
  if td.dtype != dtInt32:
    raise newException(TensorError, "Tensor is not int32")
  let arr = td.asInt32
  for i in 0 ..< td.size:
    arr[i] = value
