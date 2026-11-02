"""Writes specs/<xmodel>.json from the REAL xir library (run inside the Vitis AI docker), so the fake
`xir` module returns exactly what the real one returns for the API surface the board uses.

    run_in_docker.sh python3 tools/fake_vitis/make_specs.py models/*.xmodel
"""
import hashlib
import json
import os
import sys

import xir

HERE = os.path.dirname(os.path.abspath(__file__))


def tensor(t):
    return dict(name=t.name, dims=[int(d) for d in t.dims], dtype=str(t.dtype),
                attrs={k: v for k, v in t.get_attrs().items() if isinstance(v, (int, float, str, bool))})


def main(paths):
    for p in paths:
        g = xir.Graph.deserialize(p)
        subs = []
        for s in g.get_root_subgraph().toposort_child_subgraph():
            attrs = {}
            for k, v in s.get_attrs().items():
                if isinstance(v, (int, float, str, bool)) or (isinstance(v, dict) and all(
                        isinstance(x, str) for x in v.values())):
                    attrs[k] = v
            subs.append(dict(name=s.get_name(), attrs=attrs, n_ops=len(s.get_ops()),
                             inputs=sorted([tensor(t) for t in s.get_input_tensors()], key=lambda d: d["name"]),
                             outputs=sorted([tensor(t) for t in s.get_output_tensors()], key=lambda d: d["name"])))
        md5 = hashlib.md5(open(p, "rb").read()).hexdigest()
        spec = dict(source="real xir %s via make_specs.py" % getattr(xir, "__version__", "3.5"),
                    xmodel=os.path.basename(p), md5=md5, graph=g.get_name(), subgraphs=subs)
        out = os.path.join(HERE, "specs", os.path.basename(p) + ".json")
        with open(out, "w") as fh:
            json.dump(spec, fh, indent=1)
        print("wrote", out, md5, [(s["attrs"].get("device"), s["n_ops"]) for s in subs])


if __name__ == "__main__":
    main(sys.argv[1:])
