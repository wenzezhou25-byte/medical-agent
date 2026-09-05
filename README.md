# 家庭医疗知识库 RAG 助手

一个面向家庭健康场景的本地医疗知识库 RAG 应用。项目基于本地 PDF 药品说明书和医学指南构建可检索知识库，支持医疗问答、证据来源引用、家庭健康档案、用药计划、药物冲突粗筛和附近医疗机构推荐。

> 说明：本项目用于学习、演示和本地辅助查询，不构成正式医学诊断，也不能替代医生或执业药师建议。

## 功能概览

- 本地 PDF 知识库构建：支持从 `data/*.pdf` 或 Streamlit 上传 PDF 构建 FAISS 向量索引。

- RAG 医疗问答：基于 FAISS、BM25、fastembed 和 DashScope 通义模型（qwen3-max）实现本地证据优先的问答，配合基于 function-calling 的智能体工具编排。

- 混合检索与 rerank：结合向量检索和 BM25 关键词检索，适配药名、剂量、禁忌、不良反应等精确查询。

- 证据引用：回答中要求引用文件名、页码和章节，证据不足时明确说明缺少哪类依据。

- 结论分类：强制模型先输出明确结论，如“可以按说明书使用”“不建议自行使用”“禁止/避免使用”“当前知识库无法判断”。

- 家庭健康档案：支持多家庭成员档案、过敏史、慢病史和当前用药记录。

- 用药计划和打卡：支持记录用药计划、服药时间和每日打卡。

- 药物冲突粗筛：根据本地知识库检索药物相互作用、禁忌和风险关键词。

- 附近医疗机构推荐：接入高德地图 API，多关键词召回医院、卫生院、门诊、诊所、急救中心等，并按类型和路线距离排序。

- 工程可靠性加固：知识库重建采用 staging 校验后替换，失败时保留旧索引；登录密码使用 PBKDF2 并兼容旧 sha256 迁移。

## 技术栈

- Python

- Streamlit

- FAISS

- BM25 + fastembed

- DashScope 通义千问（qwen3-max function-calling）

- 高德地图 Web API

- PyMuPDF

## 系统架构

当前架构已演进为「扁平模块 + 智能体决策」结构，与代码一起维护的架构图见
[docs/architecture.mmd](docs/architecture.mmd)（预览见 [docs/architecture.png](docs/architecture.png)）。

简要分层：

```text
Streamlit 应用层 app.py
  ├── 可复用业务模块：auth / user_data / geo_hospital / web_search / drug_interaction
  ├── 智能体决策   agent_core.py（ToolRegistry + AgentCore 决策循环）
  ├── 混合检索     rag_utils.py（向量 + JiebaBM25 + 已知药名扩展，可选 bge-reranker）
  └── 知识库       原生 FAISS（vector_store.py）+ fastembed + retrieval_core.py
```

## 目录结构

```text
.
├── app.py                    # Streamlit 主应用（UI + 路由 + 缓存编排）
├── agent_core.py             # 智能体决策（ToolRegistry + AgentCore，qwen3-max 选工具）
├── rag_utils.py              # 混合检索、证据格式化（向量 + JiebaBM25 + 已知药名扩展）
├── retrieval_core.py         # Chunk / 递归切分 / JiebaBM25（原生检索组件）
├── vector_store.py           # 原生 FAISS 存取（index.faiss + chunks.json）
├── embedding_provider.py     # fastembed 封装
├── build_drug_kb_from_dataset.py  # 从 CHIP-2025 数据集构建家庭用药 FAISS 知识库
├── auth.py                   # 登录/注册（PBKDF2 哈希）
├── user_data.py              # 家庭档案、用药计划读写
├── geo_hospital.py           # 高德地图附近医院
├── web_search.py             # Tavily 联网搜索
├── drug_interaction.py       # 药物冲突检测
├── config.py                 # 环境变量和路径配置
├── eval/                     # 评测脚本 + 测试集 + eval_sets 产物
├── legacy/                   # 废弃脚本与旧评测产物归档
├── data/                     # 运行时数据（用户档案、用药记录）
├── vector_store/             # FAISS 索引
└── docs/                     # 架构图（.mmd/.png）等项目文档
```

## 环境配置

项目当前使用的 Python 环境：

```powershell
D:\ananconda3\envs\medical_agent\python.exe
```

安装依赖：

```powershell
cd D:\medical_agent
D:\ananconda3\envs\medical_agent\python.exe -m pip install -r requirements.txt
```

配置环境变量：

```powershell
copy .env.example .env
```

在 `.env` 中填写：

```text
DASHSCOPE_API_KEY=your-dashscope-api-key
LLM_MODEL=qwen3-max
TAVILY_API_KEY=your-tavily-api-key
GAODE_MAP_KEY=your-gaode-map-key
EMBEDDING_PROVIDER=fastembed
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

> 未配置时使用默认值：`LLM_MODEL` 默认 `qwen3-max`、`EMBEDDING_PROVIDER` 默认 `fastembed`。
> **修改 `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` 后必须重建知识库**（`python build_drug_kb_from_dataset.py`），否则检索会因向量空间不一致而静默异常。

## 启动项目

```powershell
cd D:\medical_agent
D:\ananconda3\envs\medical_agent\python.exe -m streamlit run app.py
```

默认访问：

```text
http://localhost:8501
```

如果端口被占用：

```powershell
D:\ananconda3\envs\medical_agent\python.exe -m streamlit run app.py --server.port 8502
```

## 构建知识库

默认从 CHIP-2025 药品说明书数据集构建“常见家庭用药”知识库（脚本会自动下载数据集、清洗并过滤后入库）：

```powershell
D:\ananconda3\envs\medical_agent\python.exe build_drug_kb_from_dataset.py
```

> 仅保留常见家庭用药（感冒、肠胃、抗过敏、维生素、常用抗生素、降压/降糖/降脂等），
> 通过「药名 + 适应症 + 成分」匹配，排除注射剂、肿瘤/化疗等非家庭用药；过滤规则见
> `build_drug_kb_from_dataset.py` 中的 `EXCLUDE_KEYWORDS` / `INCLUDE_KEYWORDS`。

如需（历史方式）从本地 PDF 构建，可改用 Streamlit 左侧“共享知识库”上传 PDF 重建索引。

## 评测

编译检查：

```powershell
D:\ananconda3\envs\medical_agent\python.exe -m py_compile app.py agent_core.py rag_utils.py
```

检索评测：

```powershell
D:\ananconda3\envs\medical_agent\python.exe eval\evaluate_rag.py --questions eval/eval_sets/blind_questions_2026-05-10.json --report eval/eval_sets/blind_report_tmp.json --top-k 7
```

> 评测工作流的完整说明（生成冻结盲测集、运行评测、回放评测）见 [eval\_workflow.md](eval_workflow.md)。

## 示例问题

建议在 UI 中测试：

- 蒙脱石散怎么吃？

- 来那度胺胶囊孕妇能用吗？

- 盐酸二甲双胍片有哪些不良反应？

- 头孢氨苄和布洛芬能一起吃吗？

- 高血压患者日常饮食要注意什么？

观察点：

- 第一行是否输出四分类结论。

- 是否引用具体来源。

- 剂量、频次、禁忌是否来自本地证据。

- 证据不足时是否说明缺少哪类依据。

## 项目亮点

- 不只是调用 LLM API，而是实现了完整 RAG pipeline。

- 针对医疗场景强化了本地证据优先、引用来源、剂量数字保真和证据不足说明。

- 针对药品说明书类问题引入 BM25 + 向量混合召回，兼顾语义相似和精确词匹配。

- 在工程层面处理了知识库重建失败回滚、密码哈希升级、外部 API 异常处理等可靠性问题。

- 附近医疗机构推荐不只按 POI 距离，而是结合医疗机构类型和驾车路线距离排序。

- 工程上已把 `app.py` 拆分为认证、地图、RAG、用药计划、联网搜索和药物冲突等可复用模块，便于测试与维护。

## 局限性

- 项目用于本地辅助查询，不具备真实医疗诊断能力。

- 当前评测集规模较小，仍需要扩充生成质量评测和人工回放测试。

- 药物冲突检测是基于检索和关键词的粗筛，不是严格药学知识图谱。

- 医院推荐依赖高德地图 API，真实效果需要结合 API key、城市和地址进行人工验证。

## 安全边界（已知边界）

> 本项目的登录体系只面向**本地单机使用**，请勿暴露到公网或不可信的局域网。

- **账号级锁定，非 IP 级限速**：连续 5 次密码错误会锁定该账号 15 分钟，但锁定按用户名维度，不做 IP/全局限速。攻击者若掌握多个用户名，可分散尝试绕过单账号锁定。

- **锁定提示可能暴露账号是否存在**：已锁定账号会返回「账号已临时锁定」的差异化提示，与不存在的账号返回的「账号或密码错误」不同，理论上可用于用户名枚举。作为本地应用的有意取舍，未做统一。

- **会话为浏览器侧状态，固定有效期**：登录会话存储在 Streamlit `session_state`（随浏览器会话），有效期为登录后 12 小时（非滑动续期）。不防止跨设备或会话劫持。

- **开放注册**：注册接口未限速，任何人可创建账号。仅适合本机个人使用场景。

- 以上权衡均为本地单机场景设计；如需部署到多用户或公网环境，应替换为专业的身份认证方案。

<br />
