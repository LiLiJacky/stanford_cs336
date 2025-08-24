# 设置多源头
export UV_INDEX_URL=https://mirror.sjtu.edu.cn/pypi/web/simple
export PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
export UV_HTTP_TIMEOUT=120
# 安装依赖（使用多个源）
uv pip install -e . \
  --index-url "$UV_INDEX_URL" \
  --extra-index-url "$PYTORCH_INDEX_URL"
# 运行测试
uv run pytest