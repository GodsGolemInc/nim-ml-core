## Tests for tensor module

import unittest
import std/[options, tables, strutils]
import ../src/nimml_core/dtype
import ../src/nimml_core/shape
import ../src/nimml_core/tensor

suite "Hash256":
  test "zero hash":
    let h = zeroHash()
    check h.isZero == true

  test "hash string conversion":
    var h: Hash256
    for i in 0 ..< 32:
      h[i] = byte(i)
    let s = $h
    check s.len == 64
    check s[0..1] == "00"
    check s[62..63] == "1f"

  test "parse hash":
    let s = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
    let h = parseHash256(s)
    for i in 0 ..< 32:
      check h[i] == byte(i)

  test "hash equality":
    var a, b: Hash256
    for i in 0 ..< 32:
      a[i] = byte(i)
      b[i] = byte(i)
    check a == b

    b[0] = 255
    check a != b

suite "TensorData":
  test "create tensor data":
    let td = newTensorData(newShape(2, 3), dtFloat32)
    check td.shape == newShape(2, 3)
    check td.dtype == dtFloat32
    check td.size == 6
    check td.byteSize == 24  # 6 * 4 bytes

  test "create zeros":
    let td = newTensorDataZeros(newShape(10), dtInt32)
    check td.size == 10
    check td.byteSize == 40
    # All bytes should be zero
    for b in td.data:
      check b == 0

  test "create from bytes":
    var data = newSeq[byte](12)  # 3 * 4 bytes for float32
    let td = newTensorDataFromBytes(newShape(3), dtFloat32, data)
    check td.size == 3
    check td.byteSize == 12

  test "size mismatch raises":
    expect TensorError:
      var data = newSeq[byte](10)
      discard newTensorDataFromBytes(newShape(3), dtFloat32, data)

  test "clone":
    let td1 = newTensorData(newShape(2, 2), dtFloat32)
    td1.fillFloat32(3.14)
    let td2 = td1.clone()
    check td2.shape == td1.shape
    check td2.dtype == td1.dtype
    # Modify original shouldn't affect clone
    td1.fillFloat32(0.0)
    check td2.asFloat32[0] == 3.14'f32

  test "isContiguous":
    let td = newTensorData(newShape(2, 3), dtFloat32, mlRowMajor)
    check td.isContiguous == true

  test "fill float32":
    let td = newTensorData(newShape(5), dtFloat32)
    td.fillFloat32(2.5)
    let arr = td.asFloat32
    for i in 0 ..< 5:
      check arr[i] == 2.5'f32

  test "fill int32":
    let td = newTensorData(newShape(3), dtInt32)
    td.fillInt32(42)
    let arr = td.asInt32
    for i in 0 ..< 3:
      check arr[i] == 42'i32

suite "Hash Computation":
  test "compute hash":
    let td = newTensorData(newShape(2, 3), dtFloat32)
    td.fillFloat32(1.0)
    let h = computeHash(td)
    check h.isZero == false

  test "same data same hash":
    let td1 = newTensorData(newShape(2, 3), dtFloat32)
    td1.fillFloat32(1.0)
    let td2 = newTensorData(newShape(2, 3), dtFloat32)
    td2.fillFloat32(1.0)

    check computeHash(td1) == computeHash(td2)

  test "different data different hash":
    let td1 = newTensorData(newShape(2, 3), dtFloat32)
    td1.fillFloat32(1.0)
    let td2 = newTensorData(newShape(2, 3), dtFloat32)
    td2.fillFloat32(2.0)

    check computeHash(td1) != computeHash(td2)

  test "different shape different hash":
    let td1 = newTensorData(newShape(6), dtFloat32)
    td1.fillFloat32(1.0)
    let td2 = newTensorData(newShape(2, 3), dtFloat32)
    td2.fillFloat32(1.0)

    check computeHash(td1) != computeHash(td2)

  test "verify hash":
    let td = newTensorData(newShape(5), dtFloat32)
    td.fillFloat32(3.14)
    let h = computeHash(td)

    check td.verify(h) == true

    # Modify data
    td.fillFloat32(2.71)
    check td.verify(h) == false

suite "TensorRef":
  test "create from shape":
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    check tr.shape == newShape(2, 3)
    check tr.dtype == dtFloat32
    check tr.hash.isZero == true

  test "create from tensor data":
    let td = newTensorData(newShape(2, 3), dtFloat32)
    td.fillFloat32(1.0)
    let tr = newTensorRef(td)

    check tr.shape == td.shape
    check tr.dtype == td.dtype
    check tr.hash.isZero == false

  test "tensor ref equality":
    let td = newTensorData(newShape(2, 3), dtFloat32)
    td.fillFloat32(1.0)
    let tr1 = newTensorRef(td)
    let tr2 = newTensorRef(td)

    check tr1 == tr2

  test "nil tensor ref":
    let tr1: TensorRef = nil
    let tr2: TensorRef = nil
    check tr1 == tr2

    let tr3 = newTensorRef(newShape(1), dtFloat32)
    check tr1 != tr3

  test "size and byte size":
    let tr = newTensorRef(newShape(2, 3, 4), dtFloat64)
    check tr.size == 24
    check tr.byteSize == 192  # 24 * 8 bytes

  test "string representation":
    let td = newTensorData(newShape(2, 3), dtFloat32)
    let tr = newTensorRef(td)
    let s = $tr
    check "TensorRef" in s
    check "shape=(2, 3)" in s
    check "dtype=float32" in s

suite "Location Tags":
  test "add location":
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    tr.addLocation(LocationTag(workerId: "worker-1", storePath: "/data/t1", isPrimary: true))
    tr.addLocation(LocationTag(workerId: "worker-2", storePath: "/data/t1", isPrimary: false))

    check tr.locationTags.len == 2

  test "has location":
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    tr.addLocation(LocationTag(workerId: "worker-1", storePath: "/data/t1", isPrimary: true))

    check tr.hasLocation("worker-1") == true
    check tr.hasLocation("worker-2") == false

  test "get primary location":
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    tr.addLocation(LocationTag(workerId: "worker-1", storePath: "/data/t1", isPrimary: false))
    tr.addLocation(LocationTag(workerId: "worker-2", storePath: "/data/t2", isPrimary: true))

    let primary = tr.getPrimaryLocation()
    check primary.isSome
    check primary.get.workerId == "worker-2"

  test "remove location":
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    tr.addLocation(LocationTag(workerId: "worker-1", storePath: "/data/t1", isPrimary: true))
    tr.addLocation(LocationTag(workerId: "worker-2", storePath: "/data/t2", isPrimary: false))

    tr.removeLocation("worker-1")
    check tr.locationTags.len == 1
    check tr.hasLocation("worker-1") == false
    check tr.hasLocation("worker-2") == true

suite "Metadata":
  test "set and get metadata":
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    tr.setMeta("name", "weights")
    tr.setMeta("layer", "fc1")

    check tr.getMeta("name") == some("weights")
    check tr.getMeta("layer") == some("fc1")
    check tr.getMeta("unknown") == none(string)

  test "has metadata":
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    tr.setMeta("name", "bias")

    check tr.hasMeta("name") == true
    check tr.hasMeta("other") == false

suite "Hash256 Additional Coverage":
  test "parse hash invalid length":
    expect TensorError:
      discard parseHash256("0102030405")  # Too short

  test "parse hash invalid character":
    expect TensorError:
      discard parseHash256("zz0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e")

  test "hash for hash table":
    var h: Hash256
    for i in 0 ..< 32:
      h[i] = byte(i)
    let hashVal = hash(h)
    check hashVal != 0

  test "uppercase hex parsing":
    let s = "000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F"
    let h = parseHash256(s)
    check h[10] == 0x0A
    check h[15] == 0x0F

suite "TensorData Additional Coverage":
  test "column major strides":
    let td = newTensorData(newShape(2, 3), dtFloat32, mlColumnMajor)
    check td.layout == mlColumnMajor
    check td.strides == @[1, 2]

  test "fill float64":
    let td = newTensorData(newShape(3), dtFloat64)
    td.fillFloat64(2.5)
    let arr = td.asFloat64
    for i in 0 ..< 3:
      check arr[i] == 2.5

  test "fill wrong dtype raises":
    let td = newTensorData(newShape(3), dtInt32)
    expect TensorError:
      td.fillFloat32(1.0)

  test "as wrong dtype raises":
    let td = newTensorData(newShape(3), dtFloat32)
    expect TensorError:
      discard td.asFloat64

    let td2 = newTensorData(newShape(3), dtFloat64)
    expect TensorError:
      discard td2.asInt32

    let td3 = newTensorData(newShape(3), dtInt32)
    expect TensorError:
      discard td3.asInt64

  test "asInt64":
    let td = newTensorData(newShape(3), dtInt64)
    let arr = td.asInt64
    arr[0] = 100'i64
    check arr[0] == 100'i64

suite "TensorRef Additional Coverage":
  test "create with known hash":
    var h: Hash256
    for i in 0 ..< 32:
      h[i] = byte(i)
    let tr = newTensorRef(h, newShape(2, 3), dtFloat32)
    check tr.hash == h
    check tr.hash.isZero == false

  test "hash for hash table":
    let td = newTensorData(newShape(2, 3), dtFloat32)
    let tr = newTensorRef(td)
    let hashVal = hash(tr)
    check hashVal != 0

  test "hash nil tensorref":
    let tr: TensorRef = nil
    check hash(tr) == 0

  test "nil tensorref string":
    let tr: TensorRef = nil
    check $tr == "TensorRef(nil)"

  test "get primary location none":
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    tr.addLocation(LocationTag(workerId: "w1", storePath: "/data", isPrimary: false))
    check tr.getPrimaryLocation().isNone
