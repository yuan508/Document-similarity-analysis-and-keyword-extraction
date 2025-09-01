# 文档相似度与关键词提取（Document Similarity & Keyword Extraction）

基于 Python 的轻量级工具，用于快速做文档相似度检索与中文关键词提取。
实现/集成的方法包括：

* **TF–IDF (scikit-learn) + 余弦相似度**（快速、常用）
* **Gensim TF–IDF + SparseMatrixSimilarity**（基于稀疏表示的相似度检索）
* **SimHash（近似哈希 + 汉明距离）**（快速、低成本的近似相似度）
* **jieba.analyse** 的 **TF-IDF / TextRank**（中文关键词提取）

本仓库适合用于：文档检索原型、相似度对比实验、文本聚类前的近邻检索、以及中文关键字抽取 demo。

---

# 1. 特性

* 简单、易用：单文件 demo，源码可读性强，便于学习与二次开发。
* 多种相似度方法可选：能直接比较算法效果与运行开销。
* 支持离线小样例和在线抓取的 20 Newsgroups（sklearn fetch）。
* 中文关键词提取集成：jieba 的 TF-IDF 与 TextRank。
* 可选依赖（gensim、simhash）：安装后自动启用对应方法。

---

# 2. 代码结构（建议）

```
.
├── doc_similarity.py      # 相似度 demo 主脚本（包含 sklearn / gensim / simhash 方法）
├── keywords.py           # 中文关键词提取示例（jieba）
├── stopwords.txt               # (可选) 中文停用词表，用于关键词提取
└── README.md
```

---

# 3. 环境与依赖

## 最低依赖（基础功能）

```text
python >= 3.8
numpy
scikit-learn
```

## 可选（启用 gensim / simhash / jieba）

```text
gensim          # Gensim TF-IDF + SparseMatrixSimilarity
simhash         # SimHash 支持
jieba           # 中文分词与关键词提取
numpy
scikit-learn
gensim        # 可选
simhash       # 可选
jieba         # 可选（中文关键词）
```

安装依赖：

```bash
pip install -r requirements.txt
# 或单独安装
pip install numpy scikit-learn
pip install gensim simhash jieba  # 可选
```

---

# 4. 快速开始

## 运行文档相似度 demo

将 `document_similarity.py` 放到项目根目录。直接运行：

```bash
# 使用 20 Newsgroups（会下载数据，大约 ~10-20MB）
python document_similarity.py
```

或用本地小样例（若脚本支持参数可切换）：

```python
# 在脚本中启用 sample_mode 或设置 use_20newsgroups=False
# demo(use_20newsgroups=False, sample_mode=True)
```

脚本执行流程示例：

1. 加载语料（20 Newsgroups 或内置 SAMPLE\_DOCS）
2. 文本预处理（`simple_preprocess`）
3. 使用 sklearn TF-IDF 建索引并基于余弦相似度检索最相似文档
4. （可选）使用 gensim TF-IDF 检索
5. （可选）使用 SimHash 近似相似度检索

## 运行中文关键词提取示例

将 `keywords_jieba.py` 放到项目根目录，准备 `stopwords.txt`：

```bash
python keywords_jieba.py
```

脚本会输出 TextRank 与 TF-IDF 两种算法抽取的关键词样例。

---

# 5. API / 使用说明（函数说明与示例）

下面总结 `document_similarity.py` 中的主要函数与用法，便于在项目中复用。

## 预处理

```python
def simple_preprocess(text, lowercase=True, remove_punct=True)
def preprocess_corpus(docs, lowercase=True, remove_punct=True)
```

用途：统一小写、去标点、规范空白，得到 token-friendly 的文本。

## sklearn TF-IDF + cosine

```python
vectorizer, tfidf_matrix = build_tfidf_index_sklearn(docs, max_features=20000, ngram_range=(1,1))
top_results = query_tfidf_sklearn(vectorizer, tfidf_matrix, query, topk=5)
# 返回 (doc_index, similarity_score) 列表
```

示例：

```python
vec, tfidf_mat = build_tfidf_index_sklearn(docs_proc)
top = query_tfidf_sklearn(vec, tfidf_mat, query)
for idx, score in top:
    print(idx, score, docs[idx][:120])
```

## gensim TF-IDF + SparseMatrixSimilarity（可选）

```python
dictionary, tfidf, index, corpus = build_gensim_index(docs)
top = query_gensim(dictionary, tfidf, index, query, topk=5)
```

注意：需要 `gensim`。返回 (doc\_index, similarity\_score)。

## SimHash（可选）

```python
simhash_list = build_simhashs(docs)
top = query_simhash(simhash_list, query, topk=5)
# 返回 (doc_index, hamming_distance) —— 距离越小越相似
```

注意：需要 `simhash` 库。SimHash 更适合近似快速检索，尤其在大规模与对速度敏感时。

---

# 6. 中文关键词提取（jieba）示例

文件 `keywords_jieba.py` 示例流程：

```python
from jieba import analyse
# 设置停用词（可选）
analyse.set_stop_words("stopwords.txt")

text = "这里放中文文本..."

# TextRank
keywords_textrank = analyse.textrank(text, topK=10, withWeight=False)
# TF-IDF
keywords_tfidf = analyse.extract_tags(text, topK=10, withWeight=False)
```

输出示例：

```
关键词1/
关键词2/
...
```

注意：

* `stopwords.txt` 可放常见停用词，每行一个；若无则跳过。
* jieba 分词与关键词提取对文本长度较短或专有名词丰富的文本敏感，需根据场景调整 `topK`。

---

# 7. 性能、限制与注意事项

* **规模**：scikit-learn TF-IDF 适用于中小规模数据（几十万词条内表现良好）。Gensim 的稀疏索引更节省内存、适用于大语料。SimHash 适合超大规模近似检索和内存受限场景。
* **语言**：示例中对英文和中文均可处理（中文需要分词），但模组对中文相似度计算效果受分词质量影响。
* **上下文语义**：TF–IDF 与 SimHash 基于词频/哈希，不捕捉深层语义（如同义、上下文歧义）。如需语义相似度，建议使用句向量/语义嵌入（详见下一节）。
* **依赖**：若缺少 `gensim` 或 `simhash`，对应分支会跳过并给出提示。
* **预处理**：不同任务可调整 `simple_preprocess` 的规则（是否保留数字、英文大小写、停用词、是否用 jieba 分词等）。

---

# 8. 可扩展方向（建议）

若需要更强的语义理解或更大规模检索，可以考虑：

* **句子 / 文档向量化**：使用 SBERT / Universal Sentence Encoder / BERT 微调后提取向量，基于余弦或 ANN（Faiss / Annoy）检索。
* **近似最近邻（ANN）**：引入 Faiss / Annoy 提高大规模检索速度。
* **增量索引**：将文档向量/哈希写入持久索引，支持增删改查。
* **混合检索策略**：先用 SimHash / MinHash 做粗筛，再用向量或 TF-IDF 精排。
* **可视化**：Embedding 可视化（TSNE / UMAP）、相似度热力图、关键词云等。
* **评测脚本**：添加检索精度（MAP、Recall\@K）与关键词提取的评估（人工标注或基准数据集）。

---
# 9. 许可（License）
本项目参考了 兜哥的 NLP 仓库，原作品基于 知识共享署名 - 非商业性使用 4.0 国际许可协议 许可，本项目亦为非商业用途。

简短说明：允许个人/学术/商业使用、修改与分发，但请保留原作者版权声明。

---

