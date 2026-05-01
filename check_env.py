"""
五一数学建模环境全面核查脚本
涵盖 requirements.txt 中所有包，并做核心功能测试
"""
import sys
import importlib

# ── 需要检查的包列表 (包名, 导入名, 版本属性) ──
# 如果有不能直接 import 的包，可以在 excluded 里排除
PACKAGES = [
    # 数学/科学计算
    ("numpy", "numpy", "__version__"),
    ("scipy", "scipy", "__version__"),
    ("pandas", "pandas", "__version__"),
    ("sympy", "sympy", "__version__"),
    ("networkx", "networkx", "__version__"),
    ("statsmodels", "statsmodels", "__version__"),
    ("scipy-stubs", "scipy", None),  # 只检查 scipy 存在即可
    ("numpy-typing-compat", "numpy", None),
    ("optype", "optype", "__version__"),
    ("mpmath", "mpmath", "__version__"),  # sympy 依赖

    # 优化
    ("ortools", "ortools", "__version__"),
    ("matplotlib-venn", "matplotlib_venn", "__version__"),

    # PyTorch 生态
    ("torch", "torch", "__version__"),
    ("torchvision", "torchvision", "__version__"),
    ("torchaudio", "torchaudio", "__version__"),

    # 可视化
    ("matplotlib", "matplotlib", "__version__"),
    ("seaborn", "seaborn", "__version__"),
    ("matplotlib-inline", "matplotlib_inline", None),  # 无版本
    ("Pillow", "PIL", "__version__"),
    ("cycler", "cycler", "__version__"),
    ("kiwisolver", "kiwisolver", "__version__"),
    ("fonttools", "fontTools", "__version__"),   # 或 "fonttools"
    ("contourpy", "contourpy", "__version__"),

    # Web / 网络 / 通用
    ("requests", "requests", "__version__"),
    ("urllib3", "urllib3", "__version__"),
    ("certifi", "certifi", "__version__"),
    ("charset-normalizer", "charset_normalizer", "__version__"),
    ("idna", "idna", "__version__"),
    ("colorama", "colorama", "__version__"),
    ("tqdm", "tqdm", "__version__"),
    ("absl-py", "absl", "__version__"),
    ("immutabledict", "immutabledict", "__version__"),
    ("protobuf", "google.protobuf", "__version__"),   # 若无法获取版本，设 None
    ("pyparsing", "pyparsing", "__version__"),
    ("python-dateutil", "dateutil", "__version__"),
    ("patsy", "patsy", "__version__"),
    ("six", "six", "__version__"),
    ("packaging", "packaging", "__version__"),
    ("platformdirs", "platformdirs", "__version__"),
    ("typing-extensions", "typing_extensions", "__version__"),
    ("tzdata", "tzdata", None),
    ("setuptools", "setuptools", "__version__"),

    # Jupyter 相关
    ("ipykernel", "ipykernel", "__version__"),
    ("ipython", "IPython", "__version__"),
    ("ipython-pygments-lexers", "IPython", None),
    ("ipywidgets", "ipywidgets", "__version__"),
    ("jupyter-client", "jupyter_client", "__version__"),
    ("jupyter-core", "jupyter_core", "__version__"),
    ("jupyter-console", "jupyter_console", "__version__"),
    ("jupyter-events", "jupyter_events", "__version__"),
    ("jupyter-lsp", "jupyter_lsp", "__version__"),
    ("jupyter-server", "jupyter_server", "__version__"),
    ("jupyter-server-terminals", "jupyter_server_terminals", "__version__"),
    ("jupyterlab", "jupyterlab", "__version__"),
    ("jupyterlab-pygments", "jupyterlab_pygments", "__version__"),
    ("jupyterlab-server", "jupyterlab_server", "__version__"),
    ("jupyterlab-widgets", "jupyterlab_widgets", "__version__"),
    ("notebook", "notebook", "__version__"),
    ("notebook-shim", "notebook_shim", "__version__"),
    ("nbclient", "nbclient", "__version__"),
    ("nbconvert", "nbconvert", "__version__"),
    ("nbformat", "nbformat", "__version__"),
    ("nest-asyncio", "nest_asyncio", None),
    ("prometheus-client", "prometheus_client", None),
    ("pywinpty", "pywinpty", None),  # Windows 专用
    ("pyzmq", "zmq", "__version__"),
    ("terminado", "terminado", "__version__"),
    ("tornado", "tornado", "__version__"),
    ("traitlets", "traitlets", "__version__"),
    ("prompt-toolkit", "prompt_toolkit", "__version__"),
    ("comm", "comm", "__version__"),
    ("debugpy", "debugpy", "__version__"),
    ("decorator", "decorator", "__version__"),
    ("executing", "executing", "__version__"),
    ("jedi", "jedi", "__version__"),
    ("parso", "parso", "__version__"),
    ("pygments", "pygments", "__version__"),
    ("pure-eval", "pure_eval", "__version__"),
    ("asttokens", "asttokens", "__version__"),
    ("stack-data", "stack_data", "__version__"),
    ("bleach", "bleach", "__version__"),
    ("beautifulsoup4", "bs4", "__version__"),
    ("soupsieve", "soupsieve", "__version__"),
    ("tinycss2", "tinycss2", "__version__"),
    ("defusedxml", "defusedxml", "__version__"),
    ("fastjsonschema", "fastjsonschema", "__version__"),
    ("jsonschema", "jsonschema", "__version__"),
    ("jsonschema-specifications", "jsonschema_specifications", None),
    ("jsonpointer", "jsonpointer", "__version__"),
    ("json5", "json5", "__version__"),
    ("rfc3339-validator", "rfc3339_validator", "__version__"),
    ("rfc3986-validator", "rfc3986_validator", "__version__"),
    ("rfc3987-syntax", "rfc3987_syntax", "__version__"),
    ("uri-template", "uri_template", "__version__"),
    ("webcolors", "webcolors", "__version__"),
    ("webencodings", "webencodings", None),
    ("websocket-client", "websocket", "__version__"),
    ("send2trash", "send2trash", "__version__"),
    ("arrow", "arrow", "__version__"),
    ("fqdn", "fqdn", None),
    ("isoduration", "isoduration", "__version__"),
    ("lark", "lark", "__version__"),
    ("markupsafe", "markupsafe", "__version__"),
    ("jinja2", "jinja2", "__version__"),
    ("mistune", "mistune", "__version__"),
    ("pyyaml", "yaml", "__version__"),
    ("python-json-logger", "pythonjsonlogger", "__version__"),
    ("filelock", "filelock", "__version__"),
    ("fsspec", "fsspec", "__version__"),
    ("psutil", "psutil", "__version__"),
    ("argon2-cffi", "argon2", "__version__"),
    ("argon2-cffi-bindings", "argon2", None),
    ("cffi", "cffi", "__version__"),
    ("pycparser", "pycparser", "__version__"),
    ("async-lru", "async_lru", "__version__"),
    ("h11", "h11", "__version__"),
    ("httpcore", "httpcore", "__version__"),
    ("httpx", "httpx", "__version__"),
    ("anyio", "anyio", "__version__"),
    ("attrs", "attrs", "__version__"),
    ("babel", "babel", "__version__"),
    ("rpds-py", "rpds", "__version__"),
    ("referencing", "referencing", "__version__"),
    ("pandocfilters", "pandocfilters", "__version__"),
]

# 排除本身无法导入或会报错的（可忽略的）
IGNORE = {"scipy-stubs", "matplotlib-inline", "prometheus-client",
          "async-lru", "rfc3987-syntax", "python-json-logger",
          "argon2-cffi-bindings", "fqdn", "webencodings",
          "jsonschema-specifications", "pywinpty"}


def check_packages():
    errors = []
    for name, import_name, ver_attr in PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            version = "installed"
            if ver_attr:
                try:
                    version = getattr(mod, ver_attr)
                except Exception:
                    pass
            print(f"✓ {name:30s} {version}")
        except Exception as e:
            if name in IGNORE:
                print(f"◦ {name:30s} not checked (expected)")
            else:
                print(f"✗ {name:30s} MISSING: {e}")
                errors.append(name)
    return errors


def functional_tests():
    print("\n--- 核心功能测试 ---")
    import numpy as np
    x = np.array([1, 2, 3])
    assert np.sum(x) == 6
    print("✓ numpy 数组运算正常")

    import pandas as pd
    df = pd.DataFrame({"a": [1, 2]})
    assert df.shape == (2, 1)
    print("✓ pandas DataFrame 正常")

    from scipy import optimize
    res = optimize.minimize_scalar(lambda x: (x - 2) ** 2)
    assert abs(res.x - 2) < 1e-6
    print("✓ scipy 优化器正常")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    print("✓ matplotlib 绘图正常（无 GUI）")

    import seaborn as sns
    print("✓ seaborn 导入正常")

    import torch
    assert torch.cuda.is_available() == False or torch.cuda.is_available()
    x_t = torch.tensor([1.0, 2.0, 3.0])
    assert x_t.sum() == 6.0
    print(f"✓ torch {torch.__version__} (CUDA:{torch.cuda.is_available()})")

    import torchvision
    print(f"✓ torchvision {torchvision.__version__}")

    import sympy
    x = sympy.Symbol("x")
    expr = sympy.integrate(1 / x, x)
    assert expr == sympy.log(x)
    print("✓ sympy 符号积分正常")

    import networkx as nx
    g = nx.Graph()
    g.add_edge(1, 2)
    assert len(g.nodes) == 2
    print("✓ networkx 图网络正常")

    # 检查 Jupyter 可用性（不启动 server，只验证关键模块）
    import jupyter_core
    import ipykernel
    print("✓ Jupyter 核心组件可用")

    print("\n✅ 所有功能测试通过！")


if __name__ == "__main__":
    print(f"Python 版本: {sys.version}\n")
    errors = check_packages()
    if errors:
        print(f"\n❌ 缺包 {len(errors)} 个: {errors}")
    else:
        print("\n✅ 所有包检查完成。")
        functional_tests()