# 本地 RAG 系统

这是一个基于 `Streamlit + Chroma + BM25 / Vector / Hybrid 检索 + 可选 Rerank` 的本地 RAG 项目。

这个项目当前主要解决两件事：

1. 提供一个可直接交互使用的本地 RAG 问答页面
2. 提供一个可预览、检查、调试和编辑知识库内容的本地工具

---

## 主要功能

### 1. RAG 问答页面

支持：

- 基于本地知识库进行问答
- 流式输出回答
- 展示命中的证据 chunk
- 展示 rerank 分数
- 切换不同知识源

### 2. 三种检索模式

当前支持：

- `vector`
- `bm25`
- `hybrid`

区别大致如下：

- `vector`：更偏语义检索
- `bm25`：更偏关键词匹配
- `hybrid`：结合两者结果

当前 `hybrid` 已不是简单拼接，而是：

- 向量检索和 `bm25` 会各自先召回候选
- 通过 `RRF`（Reciprocal Rank Fusion）做融合排序
- 融合前后都会按稳定文档标识去重
- 再交给 rerank 进行最终排序

### 3. Rerank 开关

页面中可以直接打开或关闭 rerank，用于比较：

- 召回后直接回答
- 召回后再重排再回答

### 4. 多种知识源

当前支持三类知识源：

- `web`：网页文章或上传的本地文件
- `dataset`：Hugging Face 数据集
- `custom`：自定义命名知识库

### 5. 自定义知识库

你可以创建并复用自己的知识库，例如：

- `python_manual`
- `company_docs`
- `ml_notes`

当前支持上传的文件类型：

- `txt`
- `md`
- `pdf`

不同知识库会按名字隔离存储，互不污染。

### 6. PDF 上传与解析

当前 PDF 流程支持：

1. 先用 `PyMuPDF` 提取 PDF 文本
2. 如果某页文本明显乱码，自动尝试 OCR fallback
3. 文本先按页提取，再做跨页合并
4. 合并后的内容再进入 chunk 切分

保留的元数据包括：

- `page_start`
- `page_end`
- `page_count`
- `page_window_size`
- `extraction_method`

### 7. 知识库预览页

预览页文件：

- [Knowledge_Base_Preview.py](/F:/ls-quickstart/RAG/pages/Knowledge_Base_Preview.py)

支持：

- 查看当前知识库概览
- 预览上传文件内容
- 预览 chunk cache
- 搜索 chunk
- 按文件过滤 chunk
- 查看 chunk 页码范围和提取方式

### 8. 知识库编辑能力

当前支持：

- 删除整个自定义知识库
- 重命名自定义知识库
- 删除知识库中的单个文件
- 删除单个文件后自动重建索引
- 上传新文件后重新构建知识库

### 9. OCR-RAG

`OCR_RAG.py` 这一部分用于图像文本识别与 OCR-RAG 实验，可作为独立能力继续扩展。

### 10. 系统测试页

测试页文件：

- [System_Test.py](/F:/ls-quickstart/RAG/pages/System_Test.py)

支持：

- 单题测试
- 手动输入多轮历史测试改写
- 批量 case 回归测试
- 基于 `hfl/cmrc2018` 的数据集抽样评测
- 查看改写结果、命中证据、耗时和评测指标

数据集评测当前支持：

- `train`
- `validation`
- `test`

当前已接入的主要指标包括：

- `Answer Hit Rate`
- `Answer EM`
- `Answer F1`
- `Top Chunk Hit Rate`
- `Top-k Hit Rate`
- `Recall@k`
- `Precision@k`
- `MRR`
- `Relevance`
- `Faithfulness`

---

## 最近新增的重要改动

这版相较于之前，补充了下面这些关键更新：

### 1. Docker 容器化部署

项目已增加：

- [Dockerfile](/F:/ls-quickstart/Dockerfile)
- [docker-compose.yml](/F:/ls-quickstart/docker-compose.yml)
- [.dockerignore](/F:/ls-quickstart/.dockerignore)

这样可以直接在本地或服务器上用 Docker 运行。

### 2. 跨平台模型缓存目录

之前模型缓存目录写死为 Windows 路径，现在已经改成项目内相对路径：

- `huggingface_models/`
- `.hf_cache/`

这样在 Windows 和 CentOS 上都可以直接运行。

### 3. 启动阶段预热模型

页面第一次打开时会执行运行时预热，提前加载：

- embedding
- 主问答模型
- 本地重写模型
- reranker

这样第一次真正提问时的冷启动会明显减少。

### 4. 多轮问题改写改成“一步式”

当前不再采用“先判别、再改写”的两步流程，而是：

- 从第二轮开始，直接根据当前活跃历史生成“最终检索问题”
- 如果当前问题已经完整，模型会原样返回
- 如果问题依赖上下文，模型会直接输出改写后的独立问题

### 5. 保留滑动窗口式历史

当前多轮历史不是固定死取最近几轮，而是维护一个“活跃上下文窗口”：

- 第一次提问：不做改写
- 第二次开始：结合活跃上下文决定最终检索问题
- 如果这次没有改写：把这轮问答当成新的独立起点，截断更早历史
- 如果这次发生改写：保留当前上下文继续往后滚动

这样既保留多轮能力，也避免历史越积越多造成串话。

### 6. 生成答案时使用改写后的检索问题

当前生成阶段会使用改写后的问题，而不是继续使用原始追问。

这修复了这样一种问题：

- 检索已经改写成功
- chunk 里也已经有答案
- 但生成时还拿原始“它 / 这个 / 那个”去问模型
- 导致回答成“上下文不明确”

现在这条链路已经统一。

### 7. 本地重写模型

当前默认使用本地小模型做多轮问题改写：

- `Qwen/Qwen2.5-0.5B-Instruct`

用于把依赖上下文的追问改写成更适合检索的独立问题。

### 8. 本地 / API 问答模型可切换

当前页面中可以切换：

- `deepseek_api`
- `local_qwen`

右侧配置区会直接显示当前实际使用的：

- `chat backend`
- `chat model`
- `rewrite model`

便于确认页面选择和真实运行模型是否一致。

### 9. 本地模型常驻缓存

本地重写模型和本地问答模型已做资源级缓存，目标是：

- 避免每次提问重新加载模型
- 减少第一次提问后的重复冷启动
- 提高本地 Qwen 模式下的连续问答体验

### 10. 数据集评测指标补充

系统测试页中的数据集评测，当前分为两类指标：

1. 检索指标
   - `Top Chunk Hit Rate`
   - `Top-k Hit Rate`
   - `Recall@k`
   - `Precision@k`
   - `MRR`

2. 回答指标
   - `Answer Hit Rate`
   - `Answer EM`
   - `Answer F1`
   - `Relevance`
   - `Faithfulness`

其中：

- `Relevance`：回答是否真正回答了问题
- `Faithfulness`：回答是否被检索证据支撑

这两项属于可选 LLM 评测，默认关闭，打开后会更慢，也可能额外使用 API 调用。

---

## 项目结构

主要正式代码如下：

```text
README.md
requirements.txt
.gitignore
Dockerfile
docker-compose.yml
.dockerignore
RAG/
  app.py
  rag_settings.py
  knowledge_base.py
  index_builder.py
  rag_pipeline.py
  rag_service.py
  ocr_support.py
  OCR_RAG.py
  ui_helpers.py
  pages/
    Knowledge_Base_Preview.py
    System_Test.py
```

各文件职责：

- `app.py`：主聊天页面
- `rag_settings.py`：全局配置与路径
- `knowledge_base.py`：知识源加载、PDF 处理、切分前清理
- `index_builder.py`：构建本地向量库与 chunk cache
- `rag_pipeline.py`：问题改写、检索、重排、答案生成
- `rag_service.py`：页面调用的统一服务层
- `ui_helpers.py`：页面状态与上传辅助逻辑
- `Knowledge_Base_Preview.py`：知识库预览、搜索与编辑
- `System_Test.py`：单题测试、批量回归与数据集评测

---

## 安装依赖

建议使用虚拟环境。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

PDF 处理依赖：

- `PyMuPDF`

如果需要 OCR fallback，建议额外安装至少一种 OCR 后端：

- `rapidocr_onnxruntime`
- `pytesseract`

如果你要启用本地 `Qwen2.5-0.5B-Instruct` 做问题改写，请确认环境中可用：

- `transformers`
- `torch`

---

## 环境变量

运行前请配置：

- `DEEPSEEK_API_KEY`
- `LANGSMITH_API_KEY`（如果需要 LangSmith tracing）

示例：

```env
DEEPSEEK_API_KEY=your_key
LANGSMITH_API_KEY=your_key
```

项目会自动尝试加载：

- `.env`
- `.env.local`

不要把这些文件提交到 Git。

---

## 启动方式

### 本地 Python 运行

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\RAG\app.py --server.fileWatcherType none --server.headless true
```

启动后访问：

- [http://localhost:8501](http://localhost:8501)

### Docker 运行

```powershell
docker compose up --build
```

后台运行：

```powershell
docker compose up -d --build
```

---

## Docker 部署说明

当前 `docker-compose.yml` 会挂载这些目录：

- `./vector_db:/app/vector_db`
- `./RAG/uploads:/app/RAG/uploads`
- `./huggingface_models:/app/huggingface_models`
- `./.hf_cache:/app/.hf_cache`

这样容器重建后：

- 向量库不会丢
- 上传文件不会丢
- Hugging Face 模型缓存不会重复下载

---

## 基本使用流程

### 重建索引

1. 选择知识源
2. 设置检索模式和 rerank 参数
3. 如有需要，上传文件
4. 点击 `Rebuild Local Index`

### 提问

1. 完成建库
2. 在聊天框输入问题
3. 右侧查看证据 chunk、rerank 分数、检索配置和性能信息

### 系统测试

1. 打开 `Open system test page`
2. 选择知识源、检索模式和回答模型
3. 根据需要选择：
   - 单题测试
   - 批量测试
   - 数据集评测
4. 查看改写结果、证据块、检索指标、回答指标和总耗时

如果要公平评测数据集，请尽量保持：

- 当前知识库构建时使用的 `dataset split`
- 测试页中选择的 `Evaluation split`

两者一致后再评测，否则结果会被“索引 split 和评测 split 不一致”干扰。

### 多轮追问

当前多轮追问流程如下：

1. 第一轮通常直接检索
2. 第二轮开始结合活跃上下文生成最终检索问题
3. 如果当前问题本身完整，就原样检索
4. 如果当前问题依赖前文，就自动改写后再检索

### 编辑知识库

在 `custom` 模式下，你可以：

- 重命名知识库
- 删除整个知识库
- 删除单个文件
- 上传新文件后重建索引

---

## 服务器部署建议

推荐流程：

1. Windows 本地开发和测试
2. 推送到 Gitee 私有仓库
3. CentOS 服务器拉代码
4. 服务器执行 `docker compose up -d --build`

如果后续更新代码：

```bash
git pull
sudo docker compose up -d --build
```

---

## 安全说明

不要提交以下内容：

- `.env`
- API key
- 模型缓存
- 向量库缓存
- 上传文件目录

当前建议忽略的目录包括：

- `.venv/`
- `.hf_cache/`
- `huggingface_models/`
- `vector_db/`
- `RAG/uploads/`
- `__pycache__/`
- `.env`

如果 key 曾经进入 Git 历史，请默认它已经泄露：

1. 立即更换 key
2. 停用旧 key
3. 必要时清理 Git 历史后再 push

---

## 后续可继续扩展的方向

- `docx` 上传
- chunk 高亮
- 表格型 PDF 的额外清洗
- 页眉页脚清理
- 更细粒度的知识库编辑
- 更丰富的多轮改写调试信息
- 评测结果导出 CSV
- 数据集评测并发执行
- 更严格的抽取式答案模式

---

## 当前项目定位

这个项目现在更像是：

- 本地 RAG 原型系统
- 检索实验平台
- 知识库调试与编辑工具

已经适合用在：

- 课程项目
- 本地演示
- 检索效果对比实验
- 知识库构建、管理与调试
