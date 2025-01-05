import os
import json
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm import zhipu_complete, zhipu_embedding
from lightrag.utils import EmbeddingFunc
from tqdm import tqdm

# 设置工作目录和日志
WORKING_DIR = "./working_dir"
DATA_OUTPUT = "./data/fine_tuning_data.json"
QUESTION_FILE = "./data/questions.json"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

if not os.path.exists("./output"):
    os.mkdir("./output")

# 初始化LightRAG
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=zhipu_complete,
    llm_model_name="glm-4-flash",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    embedding_func=EmbeddingFunc(
        embedding_dim=2048,
        max_token_size=8192,
        func=lambda texts: zhipu_embedding(texts),
    ),
)

# 加载文本数据到RAG
with open("./data/book.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())

# 加载问题库
with open(QUESTION_FILE, "r", encoding="utf-8") as f:
    questions = json.load(f)["questions"]

# 收集对话数据
fine_tuning_data = []

for question in tqdm(questions):
    try:
        logging.info(f"Processing question: {question}")
        # Hybrid search
        response = rag.query(question, param=QueryParam(mode="hybrid"))
        
        # 构造单轮对话数据
        dialog = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": response.strip()}
            ]
        }
        fine_tuning_data.append(dialog)
    except Exception as e:
        logging.error(f"Error processing question '{question}': {e}")

# 保存结果为JSON文件
with open(DATA_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(fine_tuning_data, f, ensure_ascii=False, indent=4)

logging.info(f"Fine-tuning data saved to {DATA_OUTPUT}")