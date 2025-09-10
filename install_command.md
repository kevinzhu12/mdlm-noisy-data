```
uv python install 3.9
uv venv --python 3.9
source .venv/bin/activate

uv pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

uv sync

uv pip install --no-build-isolation causal-conv1d==1.1.3.post1
uv pip install --no-build-isolation flash-attn==2.5.6
uv pip install --no-build-isolation mamba-ssm==1.1.4
```
