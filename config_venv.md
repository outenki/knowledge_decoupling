```
uv venv --python=python3.11
source .venv/bin/activate

module load cuda/12.8.1
module load gcc-toolset/13

uv pip install torch
uv pip install flash-attn --no-build-isolation
uv pip install -r requirements.txt --no-build-isolation
```