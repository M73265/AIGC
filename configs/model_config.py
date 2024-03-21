import os

# 选用的Embedding名称
EMBEDDING_MODEL = "m3e-base"
# 可以指定一个绝对路径，统一存放所有的Embedding和LLM模型。
# 每个模型可以是一个单独的目录，也可以是某个目录下的二级子目录。
# 如果模型目录名称和 MODEL_PATH 中的 key 或 value 相同，程序会自动检测加载，无需修改 MODEL_PATH 中的路径。
MODEL_ROOT_PATH = ""
# 要运行的LLM名称，可以包括本地模型和在线模型，这里先只考虑使用文心一言大模型，本地模型在项目启动时就会全部加载
LLM_MODEL = "wenxinyiyan"

# LLM模型运行设备。设为"auto"会自动检测（会有警告），也可手动设定为”cuda“，”mps“，”cpu“，”xpu"其中之一
EMBEDDING_DEVICE = "cpu"
# LLM模型参数设置
MAX_TOKENS = 2048
HISTORY_LEN = 3
TEMPERATURE = 0.7

ONLINE_LLM_MODEL = {
    # 百度千帆 API
    "qianfan-api": {
        "version": "ERINE-BOT4.0",
        "api_key": "BhW3TK98jwfFDKB5GmUdGUuB",
        "secret_key": "F0wcGyPGkEl207VZ6qMDsSnP6TIP6Kbg",
        "provider": "QianFanWorker"
    }
}

MODEL_PATH = {
    "embed_model": {
        "m3e-base": "moka-ai/m3e-base",
        "stella-mrl-large-zh-v3.5-1792d": "infgrad/stella-mrl-large-zh-v3.5-1792d"
    }

}
