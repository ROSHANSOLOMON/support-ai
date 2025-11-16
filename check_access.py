from huggingface_hub import HfApi
api = HfApi()
try:
    info = api.model_info("google/gemma-2b-it")
    print("Access OK! Model ID:", info.modelId)
except Exception as e:
    print("Access check failed:", type(e).__name__, e)
