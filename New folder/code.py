import torch
import torch.nn as nn
from collections import defaultdict
from graphviz import Digraph
from torchviz import make_dot

class Node:
    def __init__(self, operation, inputs=[]):
        self.operation = operation
        self.inputs = inputs
        self.outputs = []

    def __repr__(self):
        return f"Node({self.operation}, inputs={self.inputs})"

class ComputationGraph:
    def __init__(self):
        self.nodes = []
        self.edges = defaultdict(list)
        self.tensor_to_node = {}

    def add_node(self, operation, inputs=[]):
        node_inputs = [self.tensor_to_node.get(inp, inp) for inp in inputs]
        node = Node(operation, node_inputs)
        self.nodes.append(node)
        for inp in node_inputs:
            self.edges[inp].append(node)
            if isinstance(inp, Node):
                inp.outputs.append(node)
        output = node.operation(*inputs)
        self.tensor_to_node[output] = node
        return output

    def remove_node(self, node):
        if node in self.nodes:
            self.nodes.remove(node)
            if node in self.edges:
                del self.edges[node]
            for key in self.edges:
                if node in self.edges[key]:
                    self.edges[key].remove(node)
            for output in node.outputs:
                if node in output.inputs:
                    output.inputs.remove(node)
        self.tensor_to_node = {k: v for k, v in self.tensor_to_node.items() if v != node}

    def visualize(self):
        dot = Digraph()
        for node in self.nodes:
            dot.node(str(id(node)), repr(node))
            for inp in node.inputs:
                dot.edge(str(id(inp)), str(id(node)))
        return dot

class AddOperation(nn.Module):
    def forward(self, x, y):
        return x + y

class MulOperation(nn.Module):
    def forward(self, x, y):
        return x * y

# Creating the computation graph
graph = ComputationGraph()
  #Enter input for computation in x and y
x  
y 

add_op = AddOperation()
mul_op = MulOperation()

add_result = graph.add_node(add_op, [x, y])
mul_result = graph.add_node(mul_op, [add_result, y])

# Visualizing the initial computation graph
dot = graph.visualize()
dot.render('/content/simple_graph', format='png')
dot.view('/content/simple_graph.png')

class Optimizer:
    @staticmethod
    def fuse_operations(graph):
        fused_nodes = []
        for node in graph.nodes:
            if isinstance(node.operation, AddOperation) and node.inputs:
                for inp in node.inputs:
                    if isinstance(inp, Node) and isinstance(inp.operation, MulOperation):
                        fused_op = nn.Sequential(inp.operation, node.operation)
                        fused_node = Node(fused_op, inp.inputs)
                        fused_nodes.append(fused_node)
                        graph.remove_node(node)
                        graph.remove_node(inp)
                        graph.add_node(fused_op, inp.inputs)
        return graph

# Optimizing the computation graph
optimizer = Optimizer()
optimized_graph = optimizer.fuse_operations(graph)

# Visualizing the optimized computation graph
dot_optimized = optimized_graph.visualize()
dot_optimized.render('/content/optimized_graph', format='png')
dot_optimized.view('/content/optimized_graph.png')

class ExecutionEngine:
    def __init__(self, graph):
        self.graph = graph

    def execute(self, input_tensors):
        results = {}
        for node in self.graph.nodes:
            inputs = [results[inp] if inp in results else inp for inp in node.inputs]
            results[node] = node.operation(*inputs)
        return results

# Executing the optimized computation graph
engine = ExecutionEngine(optimized_graph)
output = engine.execute({x: x, y: y})

print("Output:", output)
