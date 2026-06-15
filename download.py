from modelscope import snapshot_download

# 注意：魔搭的生态里，这个模型在 AI-ModelScope 命名空间下
model_dir = snapshot_download(
    'AI-ModelScope/all-mpnet-base-v2', 
    local_dir='./all-mpnet-base-v2'
)
print(f"下载完成，路径：{model_dir}")