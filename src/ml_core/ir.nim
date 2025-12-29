## IR module for computation graph representation
##
## Provides Graph as a directed acyclic graph (DAG) representation
## with topological sorting, validation, and serialization.

import std/[options, tables, sets, json, sequtils, strutils, strformat, algorithm]
import dtype, shape, tensor, ops

type
  NodeId* = string
    ## Unique identifier for a node in the graph

  NodeKind* = enum
    ## Kind of node in the graph
    nkInput     # External input
    nkConst     # Constant value
    nkOp        # Operation
    nkOutput    # Graph output

  Node* = ref object
    ## A node in the computation graph
    id*: NodeId
    kind*: NodeKind
    opSpec*: Option[OpSpec]     # For nkOp nodes
    inputs*: seq[NodeId]
    outputs*: seq[NodeId]
    tensorRef*: TensorRef       # Output tensor reference
    metadata*: Table[string, string]

  Graph* = ref object
    ## Computation graph as a DAG
    name*: string
    nodes*: OrderedTable[NodeId, Node]
    inputs*: seq[NodeId]       # Input node IDs
    outputs*: seq[NodeId]      # Output node IDs
    metadata*: Table[string, string]

  SubGraph* = ref object
    ## A subgraph for modular composition
    name*: string
    graph*: Graph
    inputMapping*: Table[NodeId, NodeId]
    outputMapping*: Table[NodeId, NodeId]

  GraphError* = object of CatchableError

  GraphValidationError* = object of GraphError

# Node creation

proc newInputNode*(id: NodeId, tensorRef: TensorRef): Node =
  ## Create an input node
  Node(
    id: id,
    kind: nkInput,
    opSpec: none(OpSpec),
    inputs: @[],
    outputs: @[],
    tensorRef: tensorRef,
    metadata: initTable[string, string]()
  )

proc newConstNode*(id: NodeId, tensorRef: TensorRef): Node =
  ## Create a constant node
  Node(
    id: id,
    kind: nkConst,
    opSpec: none(OpSpec),
    inputs: @[],
    outputs: @[],
    tensorRef: tensorRef,
    metadata: initTable[string, string]()
  )

proc newOpNode*(id: NodeId, opSpec: OpSpec, inputs: seq[NodeId],
                tensorRef: TensorRef): Node =
  ## Create an operation node
  Node(
    id: id,
    kind: nkOp,
    opSpec: some(opSpec),
    inputs: inputs,
    outputs: @[],
    tensorRef: tensorRef,
    metadata: initTable[string, string]()
  )

proc newOutputNode*(id: NodeId, inputId: NodeId, tensorRef: TensorRef): Node =
  ## Create an output node
  Node(
    id: id,
    kind: nkOutput,
    opSpec: none(OpSpec),
    inputs: @[inputId],
    outputs: @[],
    tensorRef: tensorRef,
    metadata: initTable[string, string]()
  )

# Graph creation

proc newGraph*(name: string = ""): Graph =
  ## Create a new empty graph
  Graph(
    name: name,
    nodes: initOrderedTable[NodeId, Node](),
    inputs: @[],
    outputs: @[],
    metadata: initTable[string, string]()
  )

proc addNode*(g: Graph, node: Node) =
  ## Add a node to the graph
  if node.id in g.nodes:
    raise newException(GraphError, "Node with id '" & node.id & "' already exists")
  g.nodes[node.id] = node

  # Update outputs of input nodes
  for inputId in node.inputs:
    if inputId in g.nodes:
      g.nodes[inputId].outputs.add(node.id)

  # Track input/output nodes
  if node.kind == nkInput:
    g.inputs.add(node.id)
  elif node.kind == nkOutput:
    g.outputs.add(node.id)

proc getNode*(g: Graph, id: NodeId): Option[Node] =
  ## Get a node by ID
  if id in g.nodes:
    some(g.nodes[id])
  else:
    none(Node)

proc hasNode*(g: Graph, id: NodeId): bool =
  ## Check if node exists
  id in g.nodes

proc removeNode*(g: Graph, id: NodeId) =
  ## Remove a node from the graph
  if id notin g.nodes:
    return

  let node = g.nodes[id]

  # Remove from parent outputs
  for inputId in node.inputs:
    if inputId in g.nodes:
      g.nodes[inputId].outputs.keepItIf(it != id)

  # Remove from child inputs
  for outputId in node.outputs:
    if outputId in g.nodes:
      g.nodes[outputId].inputs.keepItIf(it != id)

  # Remove from inputs/outputs lists
  g.inputs.keepItIf(it != id)
  g.outputs.keepItIf(it != id)

  # Remove the node
  g.nodes.del(id)

proc nodeCount*(g: Graph): int =
  ## Get the number of nodes
  g.nodes.len

proc opCount*(g: Graph): int =
  ## Get the number of operation nodes
  result = 0
  for node in g.nodes.values:
    if node.kind == nkOp:
      inc result

# Graph traversal

proc getInputs*(g: Graph, id: NodeId): seq[Node] =
  ## Get input nodes for a node
  if id notin g.nodes:
    return @[]
  result = @[]
  for inputId in g.nodes[id].inputs:
    if inputId in g.nodes:
      result.add(g.nodes[inputId])

proc getOutputs*(g: Graph, id: NodeId): seq[Node] =
  ## Get output nodes for a node
  if id notin g.nodes:
    return @[]
  result = @[]
  for outputId in g.nodes[id].outputs:
    if outputId in g.nodes:
      result.add(g.nodes[outputId])

proc predecessors*(g: Graph, id: NodeId): seq[NodeId] =
  ## Get all predecessor node IDs (recursive)
  var visited: HashSet[NodeId]
  var stack: seq[NodeId] = @[id]
  result = @[]

  while stack.len > 0:
    let current = stack.pop()
    if current in visited:
      continue
    visited.incl(current)

    if current != id:
      result.add(current)

    if current in g.nodes:
      for inputId in g.nodes[current].inputs:
        if inputId notin visited:
          stack.add(inputId)

proc successors*(g: Graph, id: NodeId): seq[NodeId] =
  ## Get all successor node IDs (recursive)
  var visited: HashSet[NodeId]
  var stack: seq[NodeId] = @[id]
  result = @[]

  while stack.len > 0:
    let current = stack.pop()
    if current in visited:
      continue
    visited.incl(current)

    if current != id:
      result.add(current)

    if current in g.nodes:
      for outputId in g.nodes[current].outputs:
        if outputId notin visited:
          stack.add(outputId)

# Topological sort

proc topologicalSort*(g: Graph): seq[NodeId] =
  ## Get nodes in topological order (Kahn's algorithm)
  var inDegree: Table[NodeId, int]
  var queue: seq[NodeId] = @[]
  result = @[]

  # Calculate in-degrees
  for id, node in g.nodes:
    inDegree[id] = node.inputs.len
    if node.inputs.len == 0:
      queue.add(id)

  # Process nodes
  while queue.len > 0:
    let current = queue[0]
    queue.delete(0)
    result.add(current)

    if current in g.nodes:
      for outputId in g.nodes[current].outputs:
        inDegree[outputId] -= 1
        if inDegree[outputId] == 0:
          queue.add(outputId)

  # Check for cycles
  if result.len != g.nodes.len:
    raise newException(GraphValidationError, "Graph contains cycles")

proc reverseTopologicalSort*(g: Graph): seq[NodeId] =
  ## Get nodes in reverse topological order (for backward pass)
  result = g.topologicalSort()
  result.reverse()

# Validation

proc hasCycles*(g: Graph): bool =
  ## Check if graph has cycles
  try:
    discard g.topologicalSort()
    false
  except GraphValidationError:
    true

proc validate*(g: Graph): bool =
  ## Validate the graph structure
  # Check for cycles
  if g.hasCycles:
    return false

  # Check that all referenced nodes exist
  for id, node in g.nodes:
    for inputId in node.inputs:
      if inputId notin g.nodes:
        return false
    for outputId in node.outputs:
      if outputId notin g.nodes:
        return false

  # Check that input nodes have no inputs
  for inputId in g.inputs:
    if inputId in g.nodes and g.nodes[inputId].inputs.len > 0:
      return false

  # Check that output nodes have inputs
  for outputId in g.outputs:
    if outputId in g.nodes and g.nodes[outputId].inputs.len == 0:
      return false

  true

proc validateInputs*(g: Graph): seq[string] =
  ## Validate and return list of validation errors
  result = @[]

  if g.hasCycles:
    result.add("Graph contains cycles")

  for id, node in g.nodes:
    for inputId in node.inputs:
      if inputId notin g.nodes:
        result.add(fmt"Node '{id}' references non-existent input '{inputId}'")

  for inputId in g.inputs:
    if inputId in g.nodes and g.nodes[inputId].inputs.len > 0:
      result.add(fmt"Input node '{inputId}' has inputs")

# Graph transformations

proc clone*(g: Graph): Graph =
  ## Create a deep copy of the graph
  result = newGraph(g.name)
  result.metadata = g.metadata

  for id, node in g.nodes:
    let newNode = Node(
      id: node.id,
      kind: node.kind,
      opSpec: node.opSpec,
      inputs: node.inputs,
      outputs: node.outputs,
      tensorRef: node.tensorRef,
      metadata: node.metadata
    )
    result.nodes[id] = newNode

  result.inputs = g.inputs
  result.outputs = g.outputs

proc subgraph*(g: Graph, nodeIds: seq[NodeId]): Graph =
  ## Extract a subgraph containing only the specified nodes
  result = newGraph(g.name & "_subgraph")

  let nodeSet = nodeIds.toHashSet()

  for id in nodeIds:
    if id in g.nodes:
      let node = g.nodes[id]
      let newNode = Node(
        id: node.id,
        kind: node.kind,
        opSpec: node.opSpec,
        inputs: node.inputs.filterIt(it in nodeSet),
        outputs: node.outputs.filterIt(it in nodeSet),
        tensorRef: node.tensorRef,
        metadata: node.metadata
      )
      result.nodes[id] = newNode

      if node.kind == nkInput:
        result.inputs.add(id)
      elif node.kind == nkOutput:
        result.outputs.add(id)

proc merge*(g1, g2: Graph, prefix: string = ""): Graph =
  ## Merge two graphs (g2 nodes are prefixed)
  result = g1.clone()

  for id, node in g2.nodes:
    let newId = if prefix.len > 0: prefix & "_" & id else: id
    let newNode = Node(
      id: newId,
      kind: node.kind,
      opSpec: node.opSpec,
      inputs: node.inputs.mapIt(if prefix.len > 0: prefix & "_" & it else: it),
      outputs: node.outputs.mapIt(if prefix.len > 0: prefix & "_" & it else: it),
      tensorRef: node.tensorRef,
      metadata: node.metadata
    )
    result.nodes[newId] = newNode

# Serialization

proc toJson*(node: Node): JsonNode =
  ## Convert node to JSON
  result = %*{
    "id": node.id,
    "kind": $node.kind,
    "inputs": node.inputs,
    "outputs": node.outputs
  }

  if node.opSpec.isSome:
    let spec = node.opSpec.get
    result["op"] = %*{
      "kind": $spec.kind,
      "attrs": spec.attrs
    }

  if not node.tensorRef.isNil:
    result["tensor"] = %*{
      "shape": node.tensorRef.shape.dims,
      "dtype": $node.tensorRef.dtype
    }

  if node.metadata.len > 0:
    result["metadata"] = %node.metadata

proc toJson*(g: Graph): JsonNode =
  ## Convert graph to JSON
  var nodes = newJArray()
  for id in g.topologicalSort():
    nodes.add(g.nodes[id].toJson())

  result = %*{
    "name": g.name,
    "nodes": nodes,
    "inputs": g.inputs,
    "outputs": g.outputs
  }

  if g.metadata.len > 0:
    result["metadata"] = %g.metadata

proc toDot*(g: Graph, rankdir: string = "TB"): string =
  ## Convert graph to Graphviz DOT format
  var lines: seq[string] = @[]
  lines.add(fmt"digraph {g.name} {{")
  lines.add(fmt"  rankdir={rankdir};")
  lines.add("  node [shape=box];")

  # Add nodes
  for id, node in g.nodes:
    let label = case node.kind
      of nkInput: fmt"INPUT\\n{id}"
      of nkConst: fmt"CONST\\n{id}"
      of nkOutput: fmt"OUTPUT\\n{id}"
      of nkOp:
        if node.opSpec.isSome:
          fmt"{node.opSpec.get.kind}\\n{id}"
        else:
          fmt"OP\\n{id}"

    let shape = case node.kind
      of nkInput: "ellipse"
      of nkConst: "box"
      of nkOutput: "ellipse"
      of nkOp: "box"

    let color = case node.kind
      of nkInput: "green"
      of nkConst: "gray"
      of nkOutput: "red"
      of nkOp: "blue"

    lines.add(fmt"""  "{id}" [label="{label}", shape={shape}, color={color}];""")

  # Add edges
  for id, node in g.nodes:
    for inputId in node.inputs:
      lines.add(fmt"""  "{inputId}" -> "{id}";""")

  lines.add("}")
  result = lines.join("\n")

proc `$`*(g: Graph): string =
  ## String representation of graph
  result = fmt"Graph({g.name}, nodes={g.nodeCount}, ops={g.opCount})"

# Builder pattern

type
  GraphBuilder* = ref object
    ## Builder for constructing graphs
    graph*: Graph
    nodeCounter: int

proc newGraphBuilder*(name: string = ""): GraphBuilder =
  ## Create a new graph builder
  GraphBuilder(
    graph: newGraph(name),
    nodeCounter: 0
  )

proc genNodeId(b: GraphBuilder, prefix: string = "node"): NodeId =
  ## Generate a unique node ID
  result = fmt"{prefix}_{b.nodeCounter}"
  inc b.nodeCounter

proc addInput*(b: GraphBuilder, tensorRef: TensorRef,
               name: string = ""): NodeId =
  ## Add an input node
  let id = if name.len > 0: name else: b.genNodeId("input")
  let node = newInputNode(id, tensorRef)
  b.graph.addNode(node)
  id

proc addConst*(b: GraphBuilder, tensorRef: TensorRef,
               name: string = ""): NodeId =
  ## Add a constant node
  let id = if name.len > 0: name else: b.genNodeId("const")
  let node = newConstNode(id, tensorRef)
  b.graph.addNode(node)
  id

proc addOp*(b: GraphBuilder, opKind: OpKind, inputs: seq[NodeId],
            outputShape: Shape, outputDtype: DType = dtFloat32,
            attrs: JsonNode = newJObject(),
            name: string = ""): NodeId =
  ## Add an operation node
  let id = if name.len > 0: name else: b.genNodeId($opKind)

  # Create input tensor refs from input nodes
  var inputRefs: seq[TensorRef] = @[]
  for inputId in inputs:
    if inputId in b.graph.nodes:
      inputRefs.add(b.graph.nodes[inputId].tensorRef)

  var opSpec = newOpSpec(opKind, inputRefs, attrs)
  opSpec.setId(id)

  let outputRef = newTensorRef(outputShape, outputDtype)
  let node = newOpNode(id, opSpec, inputs, outputRef)
  b.graph.addNode(node)
  id

proc addOutput*(b: GraphBuilder, inputId: NodeId,
                name: string = ""): NodeId =
  ## Add an output node
  let id = if name.len > 0: name else: b.genNodeId("output")

  var tensorRef: TensorRef = nil
  if inputId in b.graph.nodes:
    tensorRef = b.graph.nodes[inputId].tensorRef

  let node = newOutputNode(id, inputId, tensorRef)
  b.graph.addNode(node)
  id

proc build*(b: GraphBuilder): Graph =
  ## Finalize and return the graph
  b.graph
