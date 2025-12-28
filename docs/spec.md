# nim-ml-core Specification

## Overview

ML Coreは異種間分散機械学習フレームワークの基盤となる型定義を提供する。
Linda（ninda）やExecutorに依存しない純粋な計算意味論を定義。

---

## Module Structure

```
nim-ml-core/
├── src/
│   ├── nimml_core.nim           # エントリポイント
│   └── nimml_core/
│       ├── dtype.nim            # データ型定義
│       ├── shape.nim            # Shape操作
│       ├── tensor.nim           # TensorRef, TensorData
│       ├── ops.nim              # 演算定義 (OpSpec)
│       └── ir.nim               # Graph IR
└── nimml_core.nimble
```

---

## 1. DType Module (`nimml_core/dtype.nim`)

### Purpose

テンソルのデータ型を定義する。

### Types

```nim
type
  DType* = enum
    dtFloat16     # 16-bit floating point (half precision)
    dtBFloat16    # Brain floating point
    dtFloat32     # 32-bit floating point (single precision)
    dtFloat64     # 64-bit floating point (double precision)
    dtInt8        # 8-bit signed integer
    dtInt16       # 16-bit signed integer
    dtInt32       # 32-bit signed integer
    dtInt64       # 64-bit signed integer
    dtUInt8       # 8-bit unsigned integer
    dtUInt16      # 16-bit unsigned integer
    dtUInt32      # 32-bit unsigned integer
    dtUInt64      # 64-bit unsigned integer
    dtBool        # Boolean
    dtComplex64   # Complex with float32 real/imag
    dtComplex128  # Complex with float64 real/imag

  DTypeInfo* = object
    dtype*: DType
    size*: int            # bytes per element
    alignment*: int       # byte alignment
    isFloating*: bool
    isComplex*: bool
    isSigned*: bool
```

### API

```nim
proc sizeOf*(dtype: DType): int
  ## Returns the size in bytes of a single element.

proc alignmentOf*(dtype: DType): int
  ## Returns the alignment requirement in bytes.

proc info*(dtype: DType): DTypeInfo
  ## Returns detailed information about the dtype.

proc canCast*(from, to: DType): bool
  ## Returns true if casting from one dtype to another is allowed.

proc promotedType*(a, b: DType): DType
  ## Returns the promoted type when combining two dtypes (e.g., int32 + float32 = float32).

proc isFloating*(dtype: DType): bool
proc isInteger*(dtype: DType): bool
proc isSigned*(dtype: DType): bool
proc isComplex*(dtype: DType): bool
```

---

## 2. Shape Module (`nimml_core/shape.nim`)

### Purpose

テンソルの形状とストライドを管理する。

### Types

```nim
type
  Shape* = object
    dims*: seq[int]

  Stride* = object
    strides*: seq[int]

  TensorLayout* = enum
    tlRowMajor      # C-style, last dim contiguous
    tlColMajor      # Fortran-style, first dim contiguous

  ShapeInfo* = object
    shape*: Shape
    stride*: Stride
    layout*: TensorLayout
    offset*: int
    isContiguous*: bool
```

### API

```nim
proc newShape*(dims: varargs[int]): Shape
proc newShape*(dims: seq[int]): Shape

proc rank*(s: Shape): int
  ## Number of dimensions.

proc size*(s: Shape): int
  ## Total number of elements.

proc `[]`*(s: Shape, i: int): int
  ## Access dimension size by index.

proc `==`*(a, b: Shape): bool

proc broadcast*(a, b: Shape): Shape
  ## Compute broadcast shape. Raises if incompatible.

proc canBroadcast*(a, b: Shape): bool
  ## Check if shapes are broadcastable.

proc transpose*(s: Shape, axes: seq[int]): Shape
  ## Transpose with given axis permutation.

proc reshape*(s: Shape, newShape: Shape): bool
  ## Check if reshape is valid (same total size).

proc squeeze*(s: Shape, axis: int): Shape
  ## Remove dimension of size 1.

proc unsqueeze*(s: Shape, axis: int): Shape
  ## Add dimension of size 1.

proc computeStrides*(s: Shape, layout: TensorLayout = tlRowMajor): Stride
  ## Compute strides for given shape and layout.

proc isContiguous*(shape: Shape, stride: Stride, layout: TensorLayout): bool
  ## Check if memory layout is contiguous.

proc flatIndex*(shape: Shape, stride: Stride, indices: seq[int]): int
  ## Convert multi-dimensional index to flat index.

proc multiIndex*(shape: Shape, flatIndex: int): seq[int]
  ## Convert flat index to multi-dimensional index.
```

---

## 3. Tensor Module (`nimml_core/tensor.nim`)

### Purpose

テンソルの参照（TensorRef）とデータ（TensorData）を定義する。
TensorRefはContent-Addressed（ハッシュベース）で、分散環境での参照渡しに使用。

### Types

```nim
type
  Hash256* = array[32, byte]

  TensorRef* = object
    hash*: Hash256            # Content hash (SHA-256)
    shape*: Shape
    dtype*: DType
    layout*: TensorLayout
    locationTags*: seq[string]  # e.g., ["node-1", "gpu:0"]
    createdAt*: Time
    metadata*: Table[string, string]

  TensorData* = object
    tensorRef*: TensorRef
    data*: seq[byte]          # Raw bytes

  TensorSlice* = object
    tensorRef*: TensorRef
    offset*: int              # Byte offset
    shape*: Shape             # Slice shape
    stride*: Stride           # Slice stride

  TensorView* = object
    ## Non-owning view into tensor data
    dataPtr*: pointer
    shape*: Shape
    stride*: Stride
    dtype*: DType
```

### API

```nim
# TensorRef creation
proc computeHash*(data: openArray[byte]): Hash256
  ## Compute SHA-256 hash of data.

proc newTensorRef*(data: openArray[byte], shape: Shape, dtype: DType,
                   layout: TensorLayout = tlRowMajor): TensorRef
  ## Create TensorRef from data.

proc newTensorRef*(shape: Shape, dtype: DType): TensorRef
  ## Create TensorRef for uninitialized tensor (hash is zeros).

# TensorRef utilities
proc sizeInBytes*(t: TensorRef): int
  ## Total size in bytes.

proc numElements*(t: TensorRef): int
  ## Total number of elements.

proc isValid*(t: TensorRef): bool
  ## Check if TensorRef is valid (non-zero hash).

proc hashHex*(t: TensorRef): string
  ## Return hash as hex string.

proc `==`*(a, b: TensorRef): bool
  ## Compare by hash (content equality).

# TensorData operations
proc newTensorData*(tensorRef: TensorRef, data: seq[byte]): TensorData

proc verify*(td: TensorData): bool
  ## Verify that data matches the hash in TensorRef.

proc slice*(td: TensorData, start, length: int): TensorSlice
  ## Create a slice view.

# Type-safe accessors (generic)
proc getData*[T](td: TensorData): seq[T]
  ## Get typed data. Raises if dtype doesn't match T.

proc setData*[T](td: var TensorData, data: seq[T])
  ## Set typed data.
```

### Serialization

```nim
proc serialize*(t: TensorRef): seq[byte]
  ## Serialize TensorRef to bytes (for network transmission).

proc deserializeTensorRef*(data: openArray[byte]): TensorRef
  ## Deserialize TensorRef from bytes.

proc serialize*(td: TensorData): seq[byte]
  ## Serialize TensorData to bytes.

proc deserializeTensorData*(data: openArray[byte]): TensorData
  ## Deserialize TensorData from bytes.
```

---

## 4. Ops Module (`nimml_core/ops.nim`)

### Purpose

演算の種類とスペックを定義する。

### Types

```nim
type
  OpKind* = enum
    # Unary
    opNeg, opAbs, opExp, opLog, opSqrt, opSin, opCos, opTan
    opSigmoid, opTanh, opRelu, opGelu, opSilu, opSoftmax
    opCast, opCopy

    # Binary
    opAdd, opSub, opMul, opDiv, opPow, opMod
    opMax, opMin
    opEq, opNe, opLt, opLe, opGt, opGe
    opAnd, opOr, opXor

    # Reduction
    opSum, opMean, opMax, opMin, opProd
    opArgMax, opArgMin

    # Matrix
    opMatMul, opBatchMatMul
    opTranspose, opReshape, opConcat, opSplit
    opGather, opScatter

    # NN specific
    opConv2d, opConv3d
    opMaxPool2d, opAvgPool2d
    opBatchNorm, opLayerNorm
    opDropout
    opEmbedding

    # Collective (distributed)
    opAllReduce, opAllGather, opReduceScatter, opBroadcast

  OpCategory* = enum
    ocUnary
    ocBinary
    ocReduction
    ocMatrix
    ocNN
    ocCollective

  OpSpec* = object
    kind*: OpKind
    inputs*: seq[TensorRef]
    output*: TensorRef
    attrs*: Table[string, JsonNode]  # Operation-specific attributes

  OpSignature* = object
    kind*: OpKind
    inputDtypes*: seq[DType]
    outputDtype*: DType
    inputShapes*: seq[Shape]
    outputShape*: Shape
```

### Attributes by OpKind

```nim
# opMatMul
#   transA: bool
#   transB: bool

# opConv2d
#   kernelSize: [int, int]
#   stride: [int, int]
#   padding: [int, int]
#   dilation: [int, int]
#   groups: int

# opSoftmax
#   axis: int

# opAllReduce
#   reduceOp: "sum" | "mean" | "max" | "min"

# opDropout
#   rate: float
#   training: bool
```

### API

```nim
proc category*(kind: OpKind): OpCategory
  ## Get the category of an operation.

proc isCommutative*(kind: OpKind): bool
  ## Check if operation is commutative (a op b = b op a).

proc isAssociative*(kind: OpKind): bool
  ## Check if operation is associative ((a op b) op c = a op (b op c)).

proc hasGradient*(kind: OpKind): bool
  ## Check if operation has a defined gradient.

proc inferOutputShape*(op: OpSpec): Shape
  ## Infer output shape from inputs and attributes.

proc inferOutputDtype*(op: OpSpec): DType
  ## Infer output dtype from inputs and attributes.

proc validate*(op: OpSpec): bool
  ## Validate operation (shape/dtype compatibility).

proc newOpSpec*(kind: OpKind, inputs: seq[TensorRef],
                attrs: Table[string, JsonNode] = initTable()): OpSpec
  ## Create OpSpec with inferred output.
```

---

## 5. IR Module (`nimml_core/ir.nim`)

### Purpose

計算グラフのIntermediate Representation（中間表現）を定義する。

### Types

```nim
type
  NodeId* = distinct int

  Node* = object
    id*: NodeId
    op*: OpSpec
    name*: string
    inputs*: seq[NodeId]      # Input node IDs
    consumers*: seq[NodeId]   # Consumer node IDs

  Graph* = ref object
    nodes*: seq[Node]
    inputs*: seq[NodeId]      # Graph input nodes
    outputs*: seq[NodeId]     # Graph output nodes
    name*: string
    metadata*: Table[string, string]

  SubGraph* = object
    graph*: Graph
    inputMapping*: Table[NodeId, NodeId]  # Parent -> SubGraph
    outputMapping*: Table[NodeId, NodeId] # SubGraph -> Parent

  ExecutionOrder* = seq[NodeId]
```

### API

```nim
# Graph construction
proc newGraph*(name: string = ""): Graph

proc addNode*(g: Graph, op: OpSpec, name: string = ""): NodeId
  ## Add a node to the graph.

proc addInput*(g: Graph, tensorRef: TensorRef, name: string = ""): NodeId
  ## Add an input node.

proc setOutputs*(g: Graph, outputs: seq[NodeId])
  ## Set the output nodes.

proc connect*(g: Graph, from, to: NodeId)
  ## Connect two nodes.

# Graph queries
proc getNode*(g: Graph, id: NodeId): Node
proc getInputs*(g: Graph): seq[Node]
proc getOutputs*(g: Graph): seq[Node]
proc predecessors*(g: Graph, id: NodeId): seq[NodeId]
proc successors*(g: Graph, id: NodeId): seq[NodeId]

# Graph analysis
proc topologicalSort*(g: Graph): ExecutionOrder
  ## Get execution order via topological sort.

proc findCycles*(g: Graph): seq[seq[NodeId]]
  ## Find cycles in the graph (should be empty for valid DAG).

proc isValid*(g: Graph): bool
  ## Validate graph structure.

# Graph transformation
proc fuse*(g: Graph, nodes: seq[NodeId], fusedOp: OpKind): NodeId
  ## Fuse multiple nodes into one.

proc inline*(g: Graph, subgraph: SubGraph, at: NodeId): void
  ## Inline a subgraph at a given node.

proc prune*(g: Graph, keepOutputs: seq[NodeId]): Graph
  ## Remove nodes not needed for given outputs.

proc clone*(g: Graph): Graph
  ## Deep copy the graph.

# Serialization
proc toJson*(g: Graph): JsonNode
proc fromJson*(j: JsonNode): Graph
proc serialize*(g: Graph): seq[byte]
proc deserializeGraph*(data: openArray[byte]): Graph
```

### Graph Visualization

```nim
proc toDot*(g: Graph): string
  ## Export graph to Graphviz DOT format.

proc summary*(g: Graph): string
  ## Human-readable summary of the graph.
```

---

## Dependencies

```nim
# nimml_core.nimble
requires "nim >= 2.0.0"
requires "nimcrypto >= 0.6.0"   # SHA-256 for TensorRef
```

---

## Error Handling

```nim
type
  CoreError* = object of CatchableError

  ShapeError* = object of CoreError
  DTypeError* = object of CoreError
  TensorError* = object of CoreError
  OpError* = object of CoreError
  GraphError* = object of CoreError

  ShapeMismatchError* = object of ShapeError
  BroadcastError* = object of ShapeError

  DTypeMismatchError* = object of DTypeError
  InvalidCastError* = object of DTypeError

  InvalidTensorRefError* = object of TensorError
  HashMismatchError* = object of TensorError

  InvalidOpError* = object of OpError
  UnsupportedOpError* = object of OpError

  CycleDetectedError* = object of GraphError
  DisconnectedNodeError* = object of GraphError
```

---

## Usage Example

```nim
import nimml_core

# Create shapes
let inputShape = newShape(32, 784)  # batch=32, features=784
let weightShape = newShape(784, 256)
let outputShape = newShape(32, 256)

# Create tensor refs
let input = newTensorRef(inputShape, dtFloat32)
let weight = newTensorRef(weightShape, dtFloat32)
let bias = newTensorRef(newShape(256), dtFloat32)

# Build computation graph
var g = newGraph("simple_nn")
let inputNode = g.addInput(input, "input")
let weightNode = g.addInput(weight, "weight")
let biasNode = g.addInput(bias, "bias")

# MatMul
let matmulOp = newOpSpec(opMatMul, @[input, weight])
let matmulNode = g.addNode(matmulOp, "matmul")
g.connect(inputNode, matmulNode)
g.connect(weightNode, matmulNode)

# Add bias
let addOp = newOpSpec(opAdd, @[matmulOp.output, bias])
let addNode = g.addNode(addOp, "add_bias")
g.connect(matmulNode, addNode)
g.connect(biasNode, addNode)

# ReLU
let reluOp = newOpSpec(opRelu, @[addOp.output])
let reluNode = g.addNode(reluOp, "relu")
g.connect(addNode, reluNode)

g.setOutputs(@[reluNode])

# Get execution order
let order = g.topologicalSort()
echo g.summary()
```
