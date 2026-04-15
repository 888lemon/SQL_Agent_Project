# --- 阶段 1: 编译阶段 (Builder) ---
FROM python:3.10-slim AS builder

WORKDIR /app

# 1. 替换 Debian 软件源为阿里源并安装编译工具
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends build-essential curl
 # 去掉具体的 ==2.1.2+cpu，直接让它安装最新的 cpu 版本
RUN pip install --user --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# 2. 预安装依赖（利用缓存机制）
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# --- 阶段 2: 运行阶段 (Final) ---
FROM python:3.10-slim

WORKDIR /app

# 3. 从编译阶段只拷贝安装好的 Python 包（极大地减小镜像体积）
COPY --from=builder /root/.local /root/.local
COPY . .

# 4. 配置环境变量
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV DB_PATH=app/data/northwind.db

EXPOSE 8501

# 启动命令
CMD ["streamlit", "run", "web_ui.py", "--server.address=0.0.0.0"]
