## Tests for IR module

import unittest
import std/[json, options, strutils, tables]
import ../src/ml_core/dtype
import ../src/ml_core/shape
import ../src/ml_core/tensor
import ../src/ml_core/ops
import ../src/ml_core/ir

suite "Node Creation":
  test "input node":
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    let node = newInputNode("x", tr)
    check node.id == "x"
    check node.kind == nkInput
    check node.inputs.len == 0
    check node.tensorRef == tr

  test "const node":
    let tr = newTensorRef(newShape(3, 3), dtFloat32)
    let node = newConstNode("w", tr)
    check node.id == "w"
    check node.kind == nkConst

  test "op node":
    let tr1 = newTensorRef(newShape(2, 3), dtFloat32)
    let tr2 = newTensorRef(newShape(3, 4), dtFloat32)
    let outRef = newTensorRef(newShape(2, 4), dtFloat32)
    let opSpec = newOpSpec(opMatMul, tr1, tr2)
    let node = newOpNode("matmul_0", opSpec, @["x", "w"], outRef)

    check node.id == "matmul_0"
    check node.kind == nkOp
    check node.inputs == @["x", "w"]
    check node.opSpec.isSome
    check node.opSpec.get.kind == opMatMul

  test "output node":
    let tr = newTensorRef(newShape(2, 4), dtFloat32)
    let node = newOutputNode("out", "matmul_0", tr)
    check node.id == "out"
    check node.kind == nkOutput
    check node.inputs == @["matmul_0"]

suite "Graph Creation":
  test "empty graph":
    let g = newGraph("test")
    check g.name == "test"
    check g.nodeCount == 0
    check g.opCount == 0

  test "add nodes":
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)

    let inputNode = newInputNode("x", tr)
    g.addNode(inputNode)

    check g.nodeCount == 1
    check g.hasNode("x")
    check g.inputs == @["x"]

  test "node connections":
    let g = newGraph("test")
    let tr1 = newTensorRef(newShape(2, 3), dtFloat32)
    let tr2 = newTensorRef(newShape(2, 3), dtFloat32)

    g.addNode(newInputNode("x", tr1))
    g.addNode(newOpNode("neg", newOpSpec(opNeg, tr1), @["x"], tr2))

    check g.nodes["x"].outputs == @["neg"]
    check g.nodes["neg"].inputs == @["x"]

  test "duplicate node raises":
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)

    g.addNode(newInputNode("x", tr))

    expect GraphError:
      g.addNode(newInputNode("x", tr))

  test "remove node":
    let g = newGraph("test")
    let tr1 = newTensorRef(newShape(2, 3), dtFloat32)
    let tr2 = newTensorRef(newShape(2, 3), dtFloat32)

    g.addNode(newInputNode("x", tr1))
    g.addNode(newOpNode("neg", newOpSpec(opNeg, tr1), @["x"], tr2))

    g.removeNode("neg")

    check g.nodeCount == 1
    check not g.hasNode("neg")
    check g.nodes["x"].outputs.len == 0

suite "Graph Traversal":
  setup:
    # Build a simple graph: x -> neg -> relu -> out
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    g.addNode(newInputNode("x", tr))
    g.addNode(newOpNode("neg", newOpSpec(opNeg, tr), @["x"], tr))
    g.addNode(newOpNode("relu", newOpSpec(opRelu, tr), @["neg"], tr))
    g.addNode(newOutputNode("out", "relu", tr))

  test "get inputs":
    let inputs = g.getInputs("neg")
    check inputs.len == 1
    check inputs[0].id == "x"

  test "get outputs":
    let outputs = g.getOutputs("neg")
    check outputs.len == 1
    check outputs[0].id == "relu"

  test "predecessors":
    let preds = g.predecessors("out")
    check "x" in preds
    check "neg" in preds
    check "relu" in preds

  test "successors":
    let succs = g.successors("x")
    check "neg" in succs
    check "relu" in succs
    check "out" in succs

suite "Topological Sort":
  test "linear graph":
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    g.addNode(newInputNode("x", tr))
    g.addNode(newOpNode("a", newOpSpec(opNeg, tr), @["x"], tr))
    g.addNode(newOpNode("b", newOpSpec(opRelu, tr), @["a"], tr))
    g.addNode(newOutputNode("out", "b", tr))

    let order = g.topologicalSort()
    check order.find("x") < order.find("a")
    check order.find("a") < order.find("b")
    check order.find("b") < order.find("out")

  test "diamond graph":
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    g.addNode(newInputNode("x", tr))
    g.addNode(newOpNode("a", newOpSpec(opNeg, tr), @["x"], tr))
    g.addNode(newOpNode("b", newOpSpec(opRelu, tr), @["x"], tr))
    g.addNode(newOpNode("c", newOpSpec(opAdd, tr, tr), @["a", "b"], tr))

    let order = g.topologicalSort()
    check order.find("x") < order.find("a")
    check order.find("x") < order.find("b")
    check order.find("a") < order.find("c")
    check order.find("b") < order.find("c")

  test "reverse topological sort":
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    g.addNode(newInputNode("x", tr))
    g.addNode(newOpNode("a", newOpSpec(opNeg, tr), @["x"], tr))
    g.addNode(newOutputNode("out", "a", tr))

    let order = g.reverseTopologicalSort()
    check order.find("out") < order.find("a")
    check order.find("a") < order.find("x")

suite "Graph Validation":
  test "valid graph":
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    g.addNode(newInputNode("x", tr))
    g.addNode(newOpNode("a", newOpSpec(opNeg, tr), @["x"], tr))
    g.addNode(newOutputNode("out", "a", tr))

    check g.validate == true
    check g.hasCycles == false

  test "cycle detection":
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)

    # Create a cycle manually
    let nodeA = Node(
      id: "a", kind: nkOp,
      inputs: @["b"], outputs: @["b"],
      tensorRef: tr
    )
    let nodeB = Node(
      id: "b", kind: nkOp,
      inputs: @["a"], outputs: @["a"],
      tensorRef: tr
    )
    g.nodes["a"] = nodeA
    g.nodes["b"] = nodeB

    check g.hasCycles == true

  test "validation errors":
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    g.addNode(newOpNode("a", newOpSpec(opNeg, tr), @["nonexistent"], tr))

    let errors = g.validateInputs()
    check errors.len > 0

suite "Graph Transformations":
  test "clone":
    let g = newGraph("original")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    g.addNode(newInputNode("x", tr))
    g.addNode(newOpNode("a", newOpSpec(opNeg, tr), @["x"], tr))

    let g2 = g.clone()
    check g2.name == "original"
    check g2.nodeCount == 2
    check g2.hasNode("x")
    check g2.hasNode("a")

    # Modify original shouldn't affect clone
    g.removeNode("a")
    check g2.hasNode("a")

  test "subgraph":
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    g.addNode(newInputNode("x", tr))
    g.addNode(newOpNode("a", newOpSpec(opNeg, tr), @["x"], tr))
    g.addNode(newOpNode("b", newOpSpec(opRelu, tr), @["a"], tr))
    g.addNode(newOpNode("c", newOpSpec(opSigmoid, tr), @["b"], tr))

    let sub = g.subgraph(@["a", "b"])
    check sub.nodeCount == 2
    check sub.hasNode("a")
    check sub.hasNode("b")
    check not sub.hasNode("x")
    check not sub.hasNode("c")

suite "Serialization":
  test "toJson":
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    g.addNode(newInputNode("x", tr))
    g.addNode(newOpNode("neg", newOpSpec(opNeg, tr), @["x"], tr))

    let j = g.toJson()
    check j["name"].getStr == "test"
    check j["nodes"].len == 2
    check j["inputs"].len == 1

  test "toDot":
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    g.addNode(newInputNode("x", tr))
    g.addNode(newOpNode("neg", newOpSpec(opNeg, tr), @["x"], tr))

    let dot = g.toDot()
    check "digraph" in dot
    check "x" in dot
    check "neg" in dot
    check "->" in dot

suite "GraphBuilder":
  test "build simple graph":
    let b = newGraphBuilder("mlp")

    let x = b.addInput(newTensorRef(newShape(2, 784), dtFloat32), "input")
    let w1 = b.addConst(newTensorRef(newShape(784, 256), dtFloat32), "weight1")
    let h = b.addOp(opMatMul, @[x, w1], newShape(2, 256), name = "hidden")
    let act = b.addOp(opRelu, @[h], newShape(2, 256), name = "activation")
    discard b.addOutput(act, "output")

    let g = b.build()
    check g.name == "mlp"
    check g.nodeCount == 5
    check g.opCount == 2
    check g.inputs == @["input"]
    check g.outputs == @["output"]

  test "auto-generated node ids":
    let b = newGraphBuilder()

    let x = b.addInput(newTensorRef(newShape(10), dtFloat32))
    let y = b.addOp(opNeg, @[x], newShape(10))

    check x.startsWith("input_")
    check y.startsWith("neg_")

suite "Graph Additional Coverage":
  test "getNode existing":
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    g.addNode(newInputNode("x", tr))

    let node = g.getNode("x")
    check node.isSome
    check node.get.id == "x"

  test "getNode non-existing":
    let g = newGraph("test")
    let node = g.getNode("nonexistent")
    check node.isNone

  test "getInputs non-existing node":
    let g = newGraph("test")
    let inputs = g.getInputs("nonexistent")
    check inputs.len == 0

  test "getOutputs non-existing node":
    let g = newGraph("test")
    let outputs = g.getOutputs("nonexistent")
    check outputs.len == 0

  test "predecessors non-existing node":
    let g = newGraph("test")
    let preds = g.predecessors("nonexistent")
    check preds.len == 0

  test "successors non-existing node":
    let g = newGraph("test")
    let succs = g.successors("nonexistent")
    check succs.len == 0

  test "remove non-existing node":
    let g = newGraph("test")
    g.removeNode("nonexistent")  # Should not raise
    check g.nodeCount == 0

  test "graph string representation":
    let g = newGraph("mymodel")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    g.addNode(newInputNode("x", tr))
    g.addNode(newOpNode("neg", newOpSpec(opNeg, tr), @["x"], tr))

    let s = $g
    check "Graph" in s
    check "mymodel" in s
    check "nodes=2" in s
    check "ops=1" in s

  test "merge graphs":
    let g1 = newGraph("g1")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    g1.addNode(newInputNode("x", tr))
    g1.addNode(newOpNode("a", newOpSpec(opNeg, tr), @["x"], tr))

    let g2 = newGraph("g2")
    g2.addNode(newInputNode("y", tr))
    g2.addNode(newOpNode("b", newOpSpec(opRelu, tr), @["y"], tr))

    let merged = merge(g1, g2, "g2")
    check merged.hasNode("x")
    check merged.hasNode("a")
    check merged.hasNode("g2_y")
    check merged.hasNode("g2_b")

  test "merge without prefix":
    let g1 = newGraph("g1")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    g1.addNode(newInputNode("x", tr))

    let g2 = newGraph("g2")
    g2.addNode(newInputNode("y", tr))

    let merged = merge(g1, g2)
    check merged.hasNode("x")
    check merged.hasNode("y")

  test "graph metadata":
    let g = newGraph("test")
    g.metadata["version"] = "1.0"
    g.metadata["author"] = "test"

    let j = g.toJson()
    check j["metadata"]["version"].getStr == "1.0"

suite "Node Additional Coverage":
  test "node metadata":
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    let node = newInputNode("x", tr)
    node.metadata["desc"] = "input tensor"

    let j = node.toJson()
    check j["metadata"]["desc"].getStr == "input tensor"

  test "node toJson with nil tensorRef":
    var node = Node(
      id: "test",
      kind: nkOp,
      opSpec: none(OpSpec),
      inputs: @[],
      outputs: @[],
      tensorRef: nil,
      metadata: initTable[string, string]()
    )

    let j = node.toJson()
    check j.hasKey("id")
    check not j.hasKey("tensor")

  test "node toJson with opSpec":
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    let spec = newOpSpec(opNeg, tr)
    let node = newOpNode("neg", spec, @["x"], tr)

    let j = node.toJson()
    check j.hasKey("op")
    check j["op"]["kind"].getStr == "neg"

suite "Validation Additional Coverage":
  test "validation input node with inputs":
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)

    # First add a valid node that can be referenced
    let srcNode = newInputNode("src", tr)
    g.addNode(srcNode)

    # Manually create invalid input node with inputs (input nodes shouldn't have inputs)
    let inputNode = Node(
      id: "x",
      kind: nkInput,
      inputs: @["src"],
      outputs: @[],
      tensorRef: tr,
      metadata: initTable[string, string]()
    )
    g.nodes["x"] = inputNode
    g.inputs.add("x")

    check g.validate == false

    let errors = g.validateInputs()
    check errors.len > 0
    var foundError = false
    for e in errors:
      if "Input node 'x' has inputs" in e:
        foundError = true
        break
    check foundError

  test "validation output node without inputs":
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)

    # Manually create invalid output node without inputs
    let outputNode = Node(
      id: "out",
      kind: nkOutput,
      inputs: @[],
      outputs: @[],
      tensorRef: tr,
      metadata: initTable[string, string]()
    )
    g.nodes["out"] = outputNode
    g.outputs.add("out")

    check g.validate == false

  test "validation missing input reference":
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)

    # Create node that references non-existent input
    let node = Node(
      id: "a",
      kind: nkOp,
      inputs: @["nonexistent"],
      outputs: @[],
      tensorRef: tr,
      metadata: initTable[string, string]()
    )
    g.nodes["a"] = node

    # This should fail because the input doesn't exist
    let errors = g.validateInputs()
    check errors.len > 0
    var foundError = false
    for e in errors:
      if "nonexistent" in e:
        foundError = true
        break
    check foundError

suite "Graph toDot Additional Coverage":
  test "toDot with LR direction":
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    g.addNode(newInputNode("x", tr))
    g.addNode(newConstNode("c", tr))
    g.addNode(newOpNode("add", newOpSpec(opAdd, tr, tr), @["x", "c"], tr))
    g.addNode(newOutputNode("out", "add", tr))

    let dot = g.toDot("LR")
    check "rankdir=LR" in dot
    check "INPUT" in dot
    check "CONST" in dot
    check "OUTPUT" in dot

  test "toDot with op node without opSpec":
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)

    let node = Node(
      id: "op",
      kind: nkOp,
      opSpec: none(OpSpec),
      inputs: @[],
      outputs: @[],
      tensorRef: tr,
      metadata: initTable[string, string]()
    )
    g.nodes["op"] = node

    let dot = g.toDot()
    check "OP" in dot  # Falls back to "OP" label

suite "GraphBuilder Additional Coverage":
  test "addOp with input not in graph":
    let b = newGraphBuilder("test")
    # Add op with non-existent input - should still work but inputRefs will be empty
    let opId = b.addOp(opNeg, @["nonexistent"], newShape(10))
    check b.graph.hasNode(opId)

  test "addOutput with input not in graph":
    let b = newGraphBuilder("test")
    let outId = b.addOutput("nonexistent")
    check b.graph.hasNode(outId)
    check b.graph.nodes[outId].tensorRef.isNil
