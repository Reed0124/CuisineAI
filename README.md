# CuisineAI
基于Langchain框架搭建的Agent的学习项目

## 学习内容
```
学习uv进行的python项目管理

掌握 LangChain Agent 开发流程
1、调用大模型（DEEPSEEK，DASHSCOPE）
2、大模型的工具调用（使用tavily作为web搜索工具）
3、提示词工程（system_prompt）

学习使用 LangSmith 做追踪与调试

体验ClaudeCode + DeepSeek‑V4‑Pro进行前端以及api的vibecoding
```

## 项目结构
```
CuisineAI/
├── app/                    # 核心应用代码
│   ├── agents/             # Agent 智能体核心逻辑（推荐、对话）
│   ├── tools/              # 自定义工具
│   ├── graph.py            # LangGraph 工作流定义
│   └── utils.py            # 工具函数
├── frontend/               # 前端可视化页面
│   └── index.html
├── resources/              # 资源文件（提示词、文档）
├── .venv/                  # Python 虚拟环境
├── .langgraph_api/         # LangGraph 自动生成的 API 服务
├── langgraph.json          # LangGraph 配置
├── pyproject.toml          # 项目依赖
├── uv.lock                 # 依赖版本锁定
├── .gitignore              # Git 忽略文件
└── README.md               # 项目说明
```

## 技术栈
Python 3.10+、LangChain、LangGraph、LangSmith、uv（依赖管理）

## 页面展示
<img width="2560" height="1272" alt="image" src="https://github.com/user-attachments/assets/2819b6c8-9942-4d2b-b039-e15e3e57a735" />

## 使用方法（项目根目录）
### 1、启动后端 AI 服务
```powershell
.venv/Scripts/activate
langgraph dev
```

### 2、启动前端页面
```powershell
cd ./frontend
python -m http.server 3000
```

### 3、浏览器访问
http://localhost:3000


