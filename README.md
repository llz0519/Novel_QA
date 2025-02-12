# 基于FAISS和RAG技术的小说问答系统：构建一个智能问答应用
** 在人工智能领域，问答系统（QA system）是自然语言处理（NLP）中的一个重要应用。传统的问答系统通常基于规则或检索技术，而近年来随着深度学习的发展，基于大模型的问答系统可以更精确地理解用户的查询，并生成相应的回答。在这篇博客中，我们将展示如何结合文本嵌入、FAISS索引和生成式模型，构建一个基于小说内容的智能问答系统。**

## 技术概述
本文所实现的问答系统结合了以下技术：
-文本分块（Chunking）：将长文本分割成较小的部分，方便索引和查询。
-BGE-M3嵌入模型：使用BGE-M3（BAAI）模型将文本转化为嵌入向量。
-FAISS：Facebook AI开源的高效相似性搜索库，用于构建并查询向量索引。
-DeepSeek生成模型：利用深度学习生成式模型回答用户的问题。
我会逐步介绍每项技术的应用与代码实现。

## 1. 文本分块（Chunking）

小说或长文本的长度通常远超过机器学习模型所能处理的最大输入长度。因此，在处理长文本时，首先需要将其分割成多个小块，以便后续的索引和检索。
```python
def chunk_text(self, file_path: str, chunk_size: int = 2048) -> List[str]:
    """按固定长度分割文本"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
```

## 2. 获取文本嵌入（Text Embedding）
为了使文本内容能够通过计算相似度进行比较，我们首先需要将其转化为嵌入向量。我们使用 BGE-M3 模型，它是一个强大的多语言文本嵌入生成模型，能够将文本映射到高维向量空间。
```python
def get_embedding(self, text: str) -> List[float]:
    """调用BGE-M3模型获取嵌入"""
    payload = {
        "model": "Pro/BAAI/bge-m3",
        "input": text
    }
    response = requests.post(f"{self.api_url}/embeddings", 
                             json=payload, 
                             headers=self.headers)
    return json.loads(response.text)['data'][0]['embedding']
```
该方法通过向指定的API发送请求，获取文本的嵌入向量。这里的API接口是调用了 SiliconFlow 提供的文本嵌入服务，它会返回一个浮点型的嵌入向量，供后续操作使用

## 3. 使用FAISS构建向量索引

FAISS（Facebook AI Similarity Search）是一个高效的相似度搜索库，支持对大规模向量数据进行快速的索引和搜索。我们将利用FAISS构建一个向量索引，以便对小说文本的不同块进行快速查询。
```python
def build_index(self, chunks: List[str]):
    """构建FAISS向量索引"""
    from tqdm import tqdm
    
    print("\n正在生成文本嵌入...")
    embeddings = []
    for chunk in tqdm(chunks, desc="处理进度", unit="chunk"):
        embeddings.append(self.get_embedding(chunk))
    
    dimension = len(embeddings[0])
    self.index = faiss.IndexFlatL2(dimension)
    self.index.add(np.array(embeddings).astype('float32'))
    self.text_chunks = chunks
    faiss.write_index(self.index, "novel_embeddings.faiss")
```
该方法将文本分块转化为嵌入后，使用FAISS的 IndexFlatL2 构建索引。FAISS索引支持高效的向量检索，我们将嵌入向量添加到索引中，并存储索引文件 novel_embeddings.faiss 以便后续使用。

## 4. 向量检索与问答生成

在系统运行时，用户输入一个问题后，我们会根据问题生成对应的文本嵌入，并在FAISS索引中查找最相似的文本块。这些相关的文本块将作为上下文信息输入到生成模型中，生成最终的回答。
```python
def search_chunks(self, query: str, k: int = 3) -> List[Tuple[float, str]]:
    """检索相关文本块"""
    query_embedding = self.get_embedding(query)
    distances, indices = self.index.search(
        np.array([query_embedding]).astype('float32'), k)
    return [(distances[0][i], self.text_chunks[indices[0][i]]) 
           for i in range(k)]
```
该方法通过查询文本嵌入获取最相似的文本块，并返回它们的距离和内容。返回的 k 个最相关文本块将作为后续生成回答的上下文。

```python
def generate_answer(self, context: str, question: str) -> str:
    """调用DeepSeek模型生成回答"""
    prompt = f"基于以下小说内容：\n{context}\n\n请回答：{question}"
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 512
    }
    response = requests.post(f"{self.api_url}/chat/completions",
                             json=payload,
                             headers=self.headers)
    return json.loads(response.text)['choices'][0]['message']['content']
```
此方法将用户的问题与上下文一起发送给生成模型 DeepSeek-R1-Distill-Qwen-32B，并返回模型生成的回答。这个模型能够理解小说的上下文，并生成自然流畅的回答。

## 5. 交互式问答系统

在 run 方法中，系统会启动一个交互式的问答循环，用户输入问题后，系统会通过上述步骤检索相关内容并生成答案。
```python
def run(self):
    """运行交互式问答系统"""
    # 初始化索引
    chunks = self.chunk_text("WEST.txt")
    self.build_index(chunks)
    
    # 交互循环
    print("小说问答系统已启动（输入exit退出）")
    while True:
        question = input("\n请输入关于小说的问题：")
        if question.lower() == 'exit':
            break
            
        # 检索相关段落
        results = self.search_chunks(question)
        context = "\n".join([chunk for _, chunk in results])
        
        # 生成回答
        answer = self.generate_answer(context, question)
        print("\nAI回答：")
        print(answer.strip())
```
在这个方法中，我们加载小说文本，构建FAISS索引，并开始一个循环，让用户输入问题并获得回答。通过将相关的文本块与问题结合，生成模型能够输出基于上下文的回答。


这个系统能够将长文本分块、索引、查询，并生成自然语言回答，展现了现代NLP技术在实际应用中的巨大潜力。如果你有类似的需求，可以参考本文的技术方案，进行相应的实现和改进。
改进方向：该系统将改进升级为一个在线的小说交互系统，支持语音交互与关卡设置等等，敬请期待！
**本人目前在读本科，有相关问题欢迎添加微信与我交流vx：15735002648，或有更好的建议可以提出**
