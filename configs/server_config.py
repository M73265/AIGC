# from configs.model_config import LLM_DEVICE

# httpx 请求默认超时时间。如果加载模型或对话较慢，出现超时错误时，可以适当加大该值
HTTPX_DEFAULT_TIMEOUT = 300.0
# 服务器默认绑定host
DEFAULT_BIND_HOST = "127.0.0.1"

# model_worker server

# 模型必须在model_config中正确配置
# MODEL_WORKERS = {
# 模型的默认配置，方便后续需要多模型的相互配合
# "default": {
#     "host": DEFAULT_BIND_HOST,
#     "port": 20002,
#     "device": LLM_DEVICE
# }
# }

FSCHAT_CONTROLLER = {
    "host": DEFAULT_BIND_HOST,
    "port": 20001,
    "dispatch_method": "shortest_queue",
}
