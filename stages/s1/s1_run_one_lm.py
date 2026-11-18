"""S1 low-memory wrapper for s1_run_one.py (only for cells that failed with CUDA OOM; contract deviation 6).
Before running the unchanged s1_run_one.main(), it rewrites one line in the method's postprocess_result:
    X = (X * cls_index).max(1)[0]          # materialises [num_cls, num_queries, H, W]
into a per-class loop that performs the same elementwise products and the same max, one class at a time:
    X = _s1_clsmax(X, cls_index)
The output tensor is bitwise identical; only peak memory changes. Usage identical to s1_run_one.py."""
import inspect
import os
import re
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1_run_one as base  # noqa: E402

W = base.W
PAT = re.compile(r"^(\s*)(\w+) = \((\w+) \* cls_index\)\.max\(1\)\[0\]", re.M)


def _s1_clsmax(x, ci):
    import torch
    return torch.stack([(x[0] * ci[c]).max(0)[0] for c in range(ci.shape[0])], 0)


def patch(method):
    repo = f"{W}/repos/{base.REPOS[method]}"
    sys.path.insert(0, repo)
    cwd = os.getcwd(); os.chdir(repo)
    n = 0
    for modname in base.MODULES[method]:
        mod = __import__(modname)
        mod._s1_clsmax = _s1_clsmax
        for name, obj in list(vars(mod).items()):
            if inspect.isclass(obj) and obj.__module__ == mod.__name__ and "postprocess_result" in vars(obj):
                src = textwrap.dedent(inspect.getsource(obj.postprocess_result))
                new, k = PAT.subn(lambda m: f"{m.group(1)}{m.group(2)} = _s1_clsmax({m.group(3)}, cls_index)", src)
                if k:
                    ns = {}
                    exec(compile(new, f"<s1_lm:{modname}.{name}>", "exec"), mod.__dict__, ns)
                    obj.postprocess_result = ns["postprocess_result"]
                    n += k
    os.chdir(cwd)
    assert n == 1, ("expected exactly one patched line", method, n)
    print("S1_LM_PATCHED", method, n)


if __name__ == "__main__":
    patch(sys.argv[1])
    base.main()
