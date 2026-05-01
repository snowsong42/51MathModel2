# 1. 复制项目定义和锁文件（不要 .venv）
cd D:\project\pythonProject\MathModel\MCM2026
copy pyproject.toml ..\51MathModel2\
copy uv.lock ..\51MathModel2\

# 2. 进入新目录，配置专用源并一次性同步所有依赖
cd ..\51MathModel2
uv config set index-url https://pypi.tuna.tsinghua.edu.cn/simple
uv config set extra-index-url https://download.pytorch.org/whl/cu128
uv config set index-strategy unsafe-best-match
uv sync

# 3. 验证环境（运行 check_env.py 或快速检查关键库）
uv run python -c "import numpy, pandas, scipy, matplotlib, seaborn, sympy, networkx, statsmodels, ortools, torch; print('All ok')"