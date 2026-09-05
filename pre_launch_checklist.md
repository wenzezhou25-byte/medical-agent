# 医疗 RAG 智能体 —— 上线前检查清单

> 代码级问题已全部清零，剩下三件「非代码」的事需要在真正使用前手动完成。
> 每项都可勾选，建议按顺序过一遍。

---

## 一、密钥安全（P0-1）

当前 `.env` 里的三个密钥是真实可用的明文，建议轮换后改用更安全的注入方式。

- [ ] **1. 轮换三个密钥**（去各平台后台各生成一个新 key，废弃旧 key）

  | 密钥 | 生成位置 |
  | --- | --- |
  | `DASHSCOPE_API_KEY` | 阿里云百炼控制台（dashscope.console.aliyun.com） |
  | `GAODE_MAP_KEY` | 高德开放平台（console.amap.com） |
  | `TAVILY_API_KEY` | Tavily 控制台（app.tavily.com） |

- [ ] **2. 改用 `.streamlit/secrets.toml` 注入**（比明文 `.env` 更安全，已在 `.gitignore` 覆盖）

  ```toml
  # .streamlit/secrets.toml
  DASHSCOPE_API_KEY = "新 key"
  GAODE_MAP_KEY = "新 key"
  TAVILY_API_KEY = "新 key"
  DATA_ENC_KEY = "见下方生成的 key"
  ```

  > 若不想改注入方式，至少轮换后仍写 `.env`（注意别上传、别截图、别提交 git）。

---

## 二、健康数据加密生效（P0-2）

加密逻辑已就位，但 `.env` 里还没配密钥，运行时仍在走明文降级。配了才会真正加密。

- [ ] **1. 生成密钥**

  ```bash
  D:\ananconda3\envs\medical_agent\python.exe -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
  ```

- [ ] **2. 把输出写入 `.env`**（或上面的 `secrets.toml`）

  ```
  DATA_ENC_KEY=<上一步输出的那串>
  ```

- [ ] **3. 重启应用后验证密文生效**

  ```bash
  # 登录后随便保存一次档案/用药，然后确认文件不再是明文 JSON：
  # 能直接看到"青霉素""高血压"等字 → 未生效；看到一串 gAAAA... → 已生效
  ```

  > 存量明文文件会在下次写入时自动转密文，无需手动迁移。

---

## 三、网络边界（P0-3）

登录体系只是浏览器侧状态，一旦暴露到局域网/公网就是裸奔。**物理上只绑本机回环地址**最省事。

- [ ] **1. 启动命令加 `--server.address 127.0.0.1`**

  ```bash
  D:\ananconda3\envs\medical_agent\python.exe -m streamlit run app.py --server.address 127.0.0.1
  ```

- [ ] **2. 更新 `run_build.bat` / README 里的启动命令**，让以后都用这个参数

  > 之后即便别人知道你 IP 也连不进来，只有本机能访问。

---

## 四、上线前验证（可选，建议跑一遍）

- [ ] **1. 语法与测试**

  ```bash
  D:\ananconda3\envs\medical_agent\python.exe -m py_compile app.py agent_core.py storage_io.py
  D:\ananconda3\envs\medical_agent\python.exe -m pytest tests/ -q   # 预期 35 passed
  ```

- [ ] **2. 功能冒烟**（登录 → 问答 → 用药打卡 → 医院查询各点一遍）
- [ ] **3. 确认 `data/audit.log` 有读写记录、且记录了正确用户名**

---

## 附：当前安全边界（已知，非缺陷）

- 认证是「前端 session_state + 开放注册 + 账号级锁定」，仅适合**本地单机**使用。
- 审计的 `CURRENT_USER` 走进程级环境变量，多用户并发会串；单机场景够用。
- 药物冲突检测是「检索 + LLM 二分类 + 关键词兜底」，不是严格药学知识图谱。
- 本项目不构成医疗诊断，答案末尾保留免责声明。
