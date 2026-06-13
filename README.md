# Local RAG System

一个基于 `Streamlit + Chroma + BM25/Vector/Hybrid 检索 + Rerank` 的本地 RAG 项目。

这个项目当前重点放在两件事上：

1. 做一个可直接交互使用的 RAG 问答页面  
2. 做一个可调试、可预览、可管理知识库内容的本地工具

---

## 当前功能

### 1. RAG 问答页面

主页面文件：

- [RAG/app.py](F:/ls-quickstart/RAG/app.py)

支持：

- 基于知识库的问答
- 流式输出回答
- 展示召回到的证据 chunk
- 展示 rerank 分数
- 支持多种知识源切换

支持的知识源：

- `web`：网页文章
- `dataset`：CMRC2018 数据集
- `custom`：自定义知识库

---

### 2. 检索模式切换

当前支持三种检索方式：

- `vector`
- `bm25`
- `hybrid`

其中：

- `vector` 更偏语义检索
- `bm25` 更偏关键词匹配
- `hybrid` 结合两者结果

---

### 3. Rerank 开关

页面支持直接打开或关闭 rerank，用于比较：

- 召回后直接生成
- 召回后再重排生成

这对实验和性能对比很有帮助。

---

### 4. 自定义知识库

当前支持创建和使用自定义知识库：

- 可以给知识库命名，例如 `python_manual`
- 可以上传本地 `txt` / `md` 文件作为语料
- 可以从已有知识库下拉选择继续使用

上传文件会保存在本地目录中，知识库按名称隔离管理。

---

### 5. 知识库预览页面

预览页面文件：

- [RAG/pages/Knowledge_Base_Preview.py](F:/ls-quickstart/RAG/pages/Knowledge_Base_Preview.py)

支持：

- 查看当前知识库概览
- 预览上传文件内容
- 查看 chunk cache
- 按关键词搜索 chunk
- 浏览 chunk 列表

这个页面主要用于调试知识库内容和排查检索问题。

---

### 6. OCR-RAG

OCR-RAG 相关文件：

- [RAG/OCR_RAG.py](F:/ls-quickstart/RAG/OCR_RAG.py)
- [RAG/ocr_support.py](F:/ls-quickstart/RAG/ocr_support.py)

支持把图片中的文字抽取出来，再作为知识库进行问答。

当前 OCR-RAG 适合做单独实验或后续功能扩展。

---

## 项目结构

主要使用的正式代码如下：

```text
README.md
.gitignore
RAG/
  app.py
  rag_settings.py
  knowledge_base.py
  index_builder.py
  rag_pipeline.py
  rag_service.py
  ocr_support.py
  OCR_RAG.py
  pages/
    Knowledge_Base_Preview.py
```

各文件职责：

- `app.py`：主问答页面
- `rag_settings.py`：配置项
- `knowledge_base.py`：知识源加载与文档切分
- `index_builder.py`：本地向量库构建与 chunk cache
- `rag_pipeline.py`：检索、BM25、hybrid、rerank、生成
- `rag_service.py`：给页面调用的统一服务层
- `pages/Knowledge_Base_Preview.py`：知识库预览与搜索

---

## 运行前准备

### 1. Python 环境

建议使用项目内的虚拟环境：

```powershell
.\.venv\Scripts\python.exe
```

### 2. 安装依赖

项目根目录下已经提供：

- [requirements.txt](F:/ls-quickstart/requirements.txt)

安装方式：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. 环境变量

项目不再把 API key 写死在主链路代码中，运行前需要提供环境变量：

- `DEEPSEEK_API_KEY`
- `LANGSMITH_API_KEY`（如果需要 LangSmith tracing）

PowerShell 示例：

```powershell
$env:DEEPSEEK_API_KEY="your_key"
$env:LANGSMITH_API_KEY="your_key"
```

也可以使用 `.env`，但请不要把 `.env` 提交到 Git。

---

## 启动方式

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\RAG\app.py --server.fileWatcherType none --server.headless true
```

启动后在浏览器访问：

- [http://localhost:8501](http://localhost:8501)

---

## 使用流程

### 问答页面

1. 选择知识源
2. 选择检索模式
3. 选择是否启用 rerank
4. 如果需要，点击 `Rebuild Local Index`
5. 输入问题并查看回答与证据

### 自定义知识库

1. `Knowledge source` 选择 `custom`
2. 选择已有知识库，或创建新知识库名称
3. 上传 `txt` / `md` 文件
4. 点击 `Rebuild Local Index`
5. 返回聊天页提问

### 知识库预览

1. 从主页面右侧点击 `Open knowledge base preview`
2. 切换知识源
3. 查看文件、chunk 和搜索结果

---

## 近期改动

当前版本已经完成的主要改动包括：

- 支持 `vector / bm25 / hybrid` 三种检索模式
- 支持 rerank 开关
- 为 BM25 增加中文分词支持
- 修复 BM25 在 chunk cache 缺失时无法正确工作的情况
- 支持自定义知识库创建与切换
- 增加知识库预览页面
- 增加 chunk 搜索能力
- 将主链路中的明文 API key 改为环境变量读取

---

## 安全说明

请不要把以下内容提交到仓库：

- `.env`
- API key
- 模型缓存
- 向量库缓存

当前 `.gitignore` 已忽略：

- `.venv/`
- `.hf_cache/`
- `vector_db/`
- `__pycache__/`
- `.env`

如果 key 曾经进入 Git 历史，即使当前代码已删除，也建议：

1. 立即更换旧 key
2. 重建本地 Git 历史
3. 再进行远程推送

---

## 后续可以继续扩展的方向

- 支持 `pdf` / `docx` 上传
- 支持 chunk 高亮
- 支持按文件过滤 chunk
- 支持删除自定义知识库
- 增加耗时统计面板
- 增加 Docker 部署文件

---

## 说明

这个项目当前更偏“本地 RAG 原型系统 + 实验调试工具”。

它已经可以完成：

- 自定义知识库构建
- 本地问答
- 多检索模式对比
- 知识库内容预览

适合继续往课程项目、实验系统或本地原型应用方向迭代。
