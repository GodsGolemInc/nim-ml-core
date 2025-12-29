## Tests for shape module

import unittest
import ../src/ml_core/shape

suite "Shape Basics":
  test "newShape creates shape":
    let s = newShape(2, 3, 4)
    check s.dims == @[2, 3, 4]
    check s.rank == 3

  test "newShape from seq":
    let s = newShape(@[2, 3, 4])
    check s.dims == @[2, 3, 4]

  test "size computes total elements":
    check newShape(2, 3, 4).size == 24
    check newShape(10).size == 10
    check newShape().size == 1  # Scalar

  test "indexing":
    let s = newShape(2, 3, 4)
    check s[0] == 2
    check s[1] == 3
    check s[-1] == 4
    check s[-2] == 3

  test "equality":
    check newShape(2, 3) == newShape(2, 3)
    check newShape(2, 3) != newShape(3, 2)

  test "string representation":
    check $newShape(2, 3, 4) == "(2, 3, 4)"
    check $newShape() == "()"

  test "isScalar, isVector, isMatrix":
    check newShape().isScalar == true
    check newShape(5).isVector == true
    check newShape(2, 3).isMatrix == true
    check newShape(2, 3, 4).isMatrix == false

suite "Strides":
  test "row major strides":
    let s = newShape(2, 3, 4)
    check s.strides(mlRowMajor) == @[12, 4, 1]

  test "column major strides":
    let s = newShape(2, 3, 4)
    check s.strides(mlColumnMajor) == @[1, 2, 6]

  test "isContiguous":
    let s = newShape(2, 3, 4)
    check s.isContiguous(@[12, 4, 1], mlRowMajor) == true
    check s.isContiguous(@[1, 2, 6], mlColumnMajor) == true
    check s.isContiguous(@[12, 4, 1], mlColumnMajor) == false

suite "Broadcasting":
  test "broadcastable":
    check broadcastable(newShape(3, 4), newShape(4)) == true
    check broadcastable(newShape(2, 3, 4), newShape(3, 4)) == true
    check broadcastable(newShape(2, 3, 4), newShape(1, 4)) == true
    check broadcastable(newShape(2, 3, 4), newShape(2, 1, 4)) == true
    check broadcastable(newShape(3, 4), newShape(5)) == false

  test "broadcast":
    check broadcast(newShape(3, 4), newShape(4)) == newShape(3, 4)
    check broadcast(newShape(2, 1), newShape(1, 3)) == newShape(2, 3)
    check broadcast(newShape(5, 1, 4), newShape(3, 1)) == newShape(5, 3, 4)

  test "broadcastTo":
    check broadcastTo(newShape(1, 3), newShape(2, 3)) == newShape(2, 3)
    check broadcastTo(newShape(4), newShape(3, 4)) == newShape(3, 4)

suite "Shape Manipulation":
  test "squeeze removes size-1 dims":
    check squeeze(newShape(1, 3, 1, 4)) == newShape(3, 4)
    check squeeze(newShape(1, 1, 1)) == newShape()

  test "squeeze specific dim":
    check squeeze(newShape(1, 3, 1, 4), 0) == newShape(3, 1, 4)
    check squeeze(newShape(1, 3, 1, 4), 2) == newShape(1, 3, 4)

  test "unsqueeze adds size-1 dim":
    check unsqueeze(newShape(3, 4), 0) == newShape(1, 3, 4)
    check unsqueeze(newShape(3, 4), 1) == newShape(3, 1, 4)
    check unsqueeze(newShape(3, 4), -1) == newShape(3, 4, 1)

  test "reshape":
    check reshape(newShape(2, 3, 4), 6, 4) == newShape(6, 4)
    check reshape(newShape(2, 3, 4), 24) == newShape(24)
    check reshape(newShape(2, 3, 4), -1, 4) == newShape(6, 4)
    check reshape(newShape(2, 3, 4), 2, -1) == newShape(2, 12)

  test "transpose 2D":
    check transpose(newShape(3, 4)) == newShape(4, 3)

  test "transpose with permutation":
    check transpose(newShape(2, 3, 4), @[2, 0, 1]) == newShape(4, 2, 3)
    check transpose(newShape(2, 3, 4), @[0, 2, 1]) == newShape(2, 4, 3)

  test "flatten":
    check flatten(newShape(2, 3, 4)) == newShape(24)
    check flatten(newShape(2, 3, 4), 1, 2) == newShape(2, 12)
    check flatten(newShape(2, 3, 4, 5), 1, -2) == newShape(2, 12, 5)

suite "Matrix Operations":
  test "matmulShape 2D":
    check matmulShape(newShape(2, 3), newShape(3, 4)) == newShape(2, 4)
    check matmulShape(newShape(5, 3), newShape(3, 7)) == newShape(5, 7)

  test "matmulShape with batch":
    check matmulShape(newShape(10, 2, 3), newShape(3, 4)) == newShape(10, 2, 4)
    check matmulShape(newShape(10, 2, 3), newShape(10, 3, 4)) == newShape(10, 2, 4)

  test "matmulShape with broadcast":
    check matmulShape(newShape(1, 2, 3), newShape(5, 3, 4)) == newShape(5, 2, 4)

  test "matmulShape 1D":
    check matmulShape(newShape(3), newShape(3, 4)) == newShape(4)
    check matmulShape(newShape(2, 3), newShape(3)) == newShape(2)

  test "convOutputShape":
    let (h, w) = convOutputShape(newShape(1, 3, 28, 28), (3, 3))
    check h == 26
    check w == 26

    let (h2, w2) = convOutputShape(newShape(1, 3, 28, 28), (3, 3), padding = (1, 1))
    check h2 == 28
    check w2 == 28

    let (h3, w3) = convOutputShape(newShape(1, 3, 28, 28), (3, 3), stride = (2, 2))
    check h3 == 13
    check w3 == 13

suite "Error Handling":
  test "negative dimension raises":
    expect ShapeError:
      discard newShape(-1, 3)

  test "reshape size mismatch raises":
    expect ShapeError:
      discard reshape(newShape(2, 3), 5, 2)

  test "matmul dimension mismatch raises":
    expect ShapeError:
      discard matmulShape(newShape(2, 3), newShape(4, 5))

  test "non-broadcastable raises":
    expect ShapeError:
      discard broadcast(newShape(3, 4), newShape(5))

  test "invalid permutation raises":
    expect ShapeError:
      discard transpose(newShape(2, 3, 4), @[0, 1])  # Wrong length

    expect ShapeError:
      discard transpose(newShape(2, 3, 4), @[0, 1, 1])  # Duplicate

suite "Shape Additional Coverage":
  test "slice indexing":
    let s = newShape(2, 3, 4, 5)
    check s[1..2] == @[3, 4]
    check s[0..0] == @[2]

  test "len alias":
    let s = newShape(2, 3, 4)
    check s.len == 3
    check s.len == s.rank

  test "empty shape strides":
    let s = newShape()
    check s.strides(mlRowMajor) == newSeq[int]()
    check s.strides(mlColumnMajor) == newSeq[int]()

  test "isContiguous mismatched length":
    let s = newShape(2, 3)
    check s.isContiguous(@[6, 1, 1], mlRowMajor) == false

  test "isContiguous size-1 dimension":
    let s = newShape(1, 3, 4)
    # Stride for size-1 dim doesn't matter
    check s.isContiguous(@[999, 4, 1], mlRowMajor) == true

  test "squeeze non-1 dim unchanged":
    let s = newShape(2, 3, 4)
    check squeeze(s, 1) == newShape(2, 3, 4)  # 3 is not 1, unchanged

  test "unsqueeze negative dim":
    let s = newShape(3, 4)
    check unsqueeze(s, -2) == newShape(3, 1, 4)

  test "unsqueeze out of range raises":
    let s = newShape(3, 4)
    expect ShapeError:
      discard unsqueeze(s, -10)

  test "flatten negative indices":
    let s = newShape(2, 3, 4, 5)
    check flatten(s, -3, -2) == newShape(2, 12, 5)

  test "flatten out of range raises":
    let s = newShape(2, 3, 4)
    expect ShapeError:
      discard flatten(s, 5, 2)
    expect ShapeError:
      discard flatten(s, 0, 10)
    expect ShapeError:
      discard flatten(s, 2, 1)  # start > end

  test "transpose with less than 2 dims raises":
    expect ShapeError:
      discard transpose(newShape(5))

  test "transpose permutation out of range raises":
    expect ShapeError:
      discard transpose(newShape(2, 3), @[0, 5])

  test "broadcastTo mismatch raises":
    expect ShapeError:
      discard broadcastTo(newShape(3, 4), newShape(2, 3))

  test "reshape multiple inferred raises":
    expect ShapeError:
      discard reshape(newShape(24), -1, -1)

  test "reshape inferred not divisible raises":
    expect ShapeError:
      discard reshape(newShape(24), -1, 7)

  test "index out of bounds raises":
    let s = newShape(2, 3)
    expect IndexDefect:
      discard s[10]
    expect IndexDefect:
      discard s[-10]

  test "matmulShape 1D vectors":
    check matmulShape(newShape(3), newShape(3)) == newShape()  # dot product

  test "convOutputShape with dilation":
    let (h, w) = convOutputShape(newShape(1, 3, 28, 28), (3, 3), dilation = (2, 2))
    check h == 24
    check w == 24
