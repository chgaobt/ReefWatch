"""FAKE `xir` for PC tests of the board's vart backend. NOT the Vitis AI library; never ship it.

Mimics the part of the xir 3.5 Python API that reefwatch_board.backends uses, with the values the
real library returned for the same file: `Graph.deserialize(path)` reads specs/<basename>.json
(written by make_specs.py with the real xir inside the Vitis AI docker) after checking the file's
md5, so a different build with the same name is refused instead of silently mis-described.

Put this directory first on PYTHONPATH (or see tests/test_board.py) together with the fake `vart`.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = True
__version__ = "fake-3.5"
sys.stderr.write("[fake_vitis] FAKE xir loaded from %s (test shim, no Vitis AI)\n" % HERE)


class Tensor:
    def __init__(self, d):
        self.name = d["name"]
        self.dims = list(d["dims"])
        self.dtype = d["dtype"]
        self.ndim = len(self.dims)
        self._attrs = dict(d.get("attrs", {}))

    def has_attr(self, k):
        return k in self._attrs

    def get_attr(self, k):
        if k not in self._attrs:            # the real binding aborts the process here
            raise KeyError("fake xir: tensor %s has no attr %r" % (self.name, k))
        return self._attrs[k]

    def get_attrs(self):
        return dict(self._attrs)

    def get_element_num(self):
        n = 1
        for v in self.dims:
            n *= v
        return n

    def get_data_size(self):
        return self.get_element_num() * (1 if "int8" in self.dtype else 4)

    def __repr__(self):
        return "<fake xir.Tensor %s %s %s>" % (self.name, self.dims, self.dtype)


class Subgraph:
    def __init__(self, d=None, children=None, graph=None):
        d = d or {"name": "root", "attrs": {}, "inputs": [], "outputs": [], "n_ops": 0}
        self._d = d
        self._children = children or []
        self._graph = graph
        self.inputs = [Tensor(t) for t in d["inputs"]]
        self.outputs = [Tensor(t) for t in d["outputs"]]

    def get_name(self):
        return self._d["name"]

    def has_attr(self, k):
        return k in self._d["attrs"]

    def get_attr(self, k):
        if k not in self._d["attrs"]:
            raise KeyError("fake xir: subgraph %s has no attr %r" % (self.get_name(), k))
        return self._d["attrs"][k]

    def get_attrs(self):
        return dict(self._d["attrs"])

    def get_input_tensors(self):           # the real binding returns a set
        return set(self.inputs)

    def get_output_tensors(self):
        return set(self.outputs)

    def toposort_child_subgraph(self):
        return list(self._children)

    def get_children(self):
        return set(self._children)

    def get_children_num(self):
        return len(self._children)

    def is_root(self):
        return bool(self._children) or self._d["name"] == "root"

    def is_leaf(self):
        return not self._children

    def get_op_num(self):
        return self._d.get("n_ops", 0)

    def get_graph(self):
        return self._graph

    def __repr__(self):
        return "<fake xir.Subgraph %s>" % self.get_name()


class Graph:
    def __init__(self, spec, path):
        self.path = path
        self.spec = spec
        self._root = Subgraph(children=[Subgraph(s, graph=self) for s in spec["subgraphs"]], graph=self)

    @staticmethod
    def deserialize(path):
        if not os.path.isfile(path):        # real: XIR_READ_PB_FAILURE (process abort)
            raise FileNotFoundError("fake xir: no xmodel at %s" % path)
        spec_path = os.environ.get("FAKE_XIR_SPEC") or os.path.join(
            HERE, "specs", os.path.basename(path) + ".json")
        if not os.path.isfile(spec_path):
            raise RuntimeError("fake xir: no spec for %s (expected %s; make one with make_specs.py "
                               "in the Vitis AI docker)" % (path, spec_path))
        with open(spec_path) as fh:
            spec = json.load(fh)
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        if h.hexdigest() != spec["md5"]:
            raise RuntimeError("fake xir: %s has md5 %s but the spec was made from md5 %s; regenerate "
                               "the spec" % (path, h.hexdigest(), spec["md5"]))
        return Graph(spec, path)

    def get_root_subgraph(self):
        return self._root

    def get_name(self):
        return self.spec["graph"]
