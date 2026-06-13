# 本地 RAG 系统

这是一个基于 `Streamlit + Chroma + BM25 / Vector / Hybrid 检索 + 可选 Rerank` 的本地 RAG 项目。

这个项目当前主要解决两件事：

1. 做一个可以直接交互使用的本地 RAG 问答页面  
2. 做一个可以预览、检查、调试和编辑知识库内容的本地工具

---

## 主要功能

### 1. RAG 问答页面

支持：

- 基于本地知识库进行问答
- 流式输出回答
- 展示命中的证据 chunk
- 展示 rerank 分数
- 切换不同知识源

---

### 2. 三种检索模式

当前支持：

- `vector`
- `bm25`
- `hybrid`

大致区别：

- `vector`：更偏语义检索
- `bm25`：更偏关键词匹配
- `hybrid`：结合两者结果

---

### 3. Rerank 开关

页面中可以直接打开或关闭 rerank，用来比较：

- 召回后直接回答
- 召回后再重排再回答

这对效果对比和性能调试都很有帮助。

---

### 4. 多种知识源

当前支持三类知识源：

- `web`：网页文章或上传的本地文件
- `dataset`：Hugging Face 数据集
- `custom`：自定义命名知识库

---

### 5. 自定义知识库

你可以创建并复用自己的知识库，例如：

- `python_manual`
- `company_docs`
- `ml_notes`

当前支持上传的文件类型：

- `txt`
- `md`
- `pdf`

不同知识库会按名字隔离存放，互不混淆。

---

### 6. PDF 上传与解析

当前 PDF 流程已经升级过，支持：

- `web` 模式上传 PDF
- `custom` 模式上传 PDF

处理逻辑是：

1. 先用 `PyMuPDF` 提取 PDF 文本  
2. 如果某一页看起来明显乱码，就自动尝试 OCR fallback  
3. 文本先按页提取，再做跨页合并  
4. 合并后的内容再进入 chunk 切分

当前会保留这些元数据：

- `page_start`
- `page_end`
- `page_count`
- `page_window_size`
- `extraction_method`

这比“整本 PDF 当一块”或者“严格按页硬切”都更适合做检索。

#### 目前仍然可能不完美的 PDF 场景

尤其是下面这些公司文档，仍然可能出现效果一般的情况：

- 双栏排版
- 表格很多
- 扫描件
- 页眉页脚重复严重
- 字体编码不标准

---

### 7. 知识库预览页面

预览页文件：

- [F:\ls-quickstart\RAG\pages\Knowledge_Base_Preview.py](F:\ls-quickstart\RAG\pages\Knowledge_Base_Preview.py)

支持：

- 查看当前知识库概览
- 预览上传文件内容
- 预览 chunk cache
- 搜索 chunk
- 按文件过滤 chunk
- 查看 chunk 页码范围和提取方式

为了减少卡顿，预览页现在做了按需加载：

- 进入页面时先只读取文件列表
- 只有选中某个文件时，才解析这个文件的内容

这对包含大 PDF 的知识库会明显更轻。

---

### 8. 知识库编辑功能

当前已经支持一版基础的知识库编辑能力：

- 删除整个自定义知识库
- 重命名自定义知识库
- 删除知识库中的单个文件
- 删除单个文件后自动重建索引
- 上传新文件后重新构建知识库

这部分主要集中在：

- [F:\ls-quickstart\RAG\pages\Knowledge_Base_Preview.py](F:\ls-quickstart\RAG\pages\Knowledge_Base_Preview.py)
- [F:\ls-quickstart\RAG\rag_service.py](F:\ls-quickstart\RAG\rag_service.py)
- [F:\ls-quickstart\RAG\index_builder.py](F:\ls-quickstart\RAG\index_builder.py)

---

### 9. OCR-RAG

相关文件：

- [F:\ls-quickstart\RAG\OCR_RAG.py](F:\ls-quickstart\RAG\OCR_RAG.py)
- [F:\ls-quickstart\RAG\ocr_support.py](F:\ls-quickstart\RAG\ocr_support.py)

这部分用于图像文本识别与 OCR-RAG 实验，当前可以作为独立能力继续扩展。

---

## 项目结构

主要正式代码如下：

```text
README.md
requirements.txt
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

- `app.py`：主聊天页面
- `rag_settings.py`：全局配置与路径
- `knowledge_base.py`：知识源加载、PDF 处理、文档切分前整理
- `index_builder.py`：构建本地向量库与 chunk cache
- `rag_pipeline.py`：检索、BM25、hybrid、rerank、生成
- `rag_service.py`：页面调用的统一服务层
- `Knowledge_Base_Preview.py`：知识库预览、搜索与编辑

---

## 安装依赖

建议使用项目中的虚拟环境。

安装方式：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

其中 PDF 处理依赖：

- `PyMuPDF`

当前已经写进 `requirements.txt`。

如果你希望 PDF 中乱码页能自动 OCR fallback，建议额外安装至少一个 OCR 后端：

- `rapidocr_onnxruntime`
- `pytesseract`

---

## 环境变量

主链路代码里不再写死 API key。

运行前请设置：

- `DEEPSEEK_API_KEY`
- `LANGSMITH_API_KEY`（如果需要 LangSmith tracing）

PowerShell 示例：

```powershell
$env:DEEPSEEK_API_KEY="your_key"
$env:LANGSMITH_API_KEY="your_key"
```

也可以使用本地 `.env` 文件，但不要提交到 Git。

推荐在项目根目录创建：

```env
DEEPSEEK_API_KEY=your_key
LANGSMITH_API_KEY=your_key
```

当前主链路和 OCR-RAG 都会自动尝试加载：

- `.env`
- `.env.local`

因此本地开发时通常不需要每次手动重新设置环境变量。

---

## 启动方式

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\RAG\app.py --server.fileWatcherType none --server.headless true
```

启动后访问：

- [http://localhost:8501](http://localhost:8501)

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
3. 右侧查看证据 chunk 和 rerank 分数

### 预览知识库

1. 从主页面跳到知识库预览页
2. 选择相同知识源
3. 查看文件内容、chunk、搜索结果和页范围

### 编辑知识库

在 `custom` 模式下，你可以：

- 重命名知识库
- 删除整个知识库
- 删除单个文件
- 上传新文件后重建索引

---

## 最近这版做过的改动

当前已经完成的重点改动包括：

- 支持 `vector / bm25 / hybrid`
- 为 BM25 增加中文分词支持
- 增加 rerank 开关
- 支持自定义知识库创建与切换
- 增加知识库预览页
- 支持 chunk 搜索与按文件过滤
- 支持 PDF 上传
- PDF 解析升级到 `PyMuPDF`
- 为乱码页增加 OCR fallback
- PDF 内容从按页提取升级为跨页合并后再切分
- 保留页级与页范围 metadata
- 预览页改为按需加载文件内容，减少首屏卡顿
- 增加知识库删除、重命名、单文件删除功能
- 主链路中移除硬编码 API key
- 支持 `.env` / `.env.local` 自动加载

---

## 安全说明

不要提交以下内容：

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

如果 key 曾经进入过 Git 历史，请默认它已经泄露：

1. 立刻更换 key
2. 停用旧 key
3. 必要时清理本地 Git 历史后再 push

---

## 后续可继续扩展的方向

- `docx` 上传
- 性能面板（检索 / rerank / 生成耗时）
- chunk 高亮
- 表格型 PDF 的额外清洗
- 页眉页脚清洗
- 更细粒度的知识库编辑
- Docker 部署

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
