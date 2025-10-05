使用echovalley构建可以上网，自主识别意图，计划与执行计划的agent。在完成需要网络开放资料，需要深度思考的任务时具有显著优势。agent使用中文prompt来构建。echovalley接入了mcp工具，可以自主添加或扩展该工作工具箱。除了在浏览器环境下自主地执行多步互相依赖的连续操作，该项目还开放了一个基于搜索引擎的快速检索API与预定义的MCP Server。

## 使用方法
- python 版本 python>=3.12

### 安装依赖
```bash
pip install -r requirements.txt
```

### 安装playwright之后
在依照requirements.txt安装playwright之后，应该确保通过以下命令安装浏览器执行环境
```bash
playwright install
```

### 在终端启动mcp与搜索api
```bash
python echovalley/web-fetch-mcp/server.py
```

### 配置LLM API
- 复制echovalley/config-example.toml 到 echovalley/config.toml
然后打开echovalley/config.toml配置LLM的BASE_URL等信息

### 使用echovalley
```bash
python echovalley/echovalley.py
```

## 特别声明
该项目是个人学习实践项目，由于涉及到网络访问与LLM API等操作，本人对拉取、克隆、使用该项目的人员行为产生的任何后果不负任何责任，使用该项目的人员应该自己评估任何风险或可能产生的法律责任。

## 鸣谢
该项目借鉴与学习了以下项目，感谢所有贡献者。

- [OpenManus](https://github.com/FoundationAgents/OpenManus)
- [Browser-use](https://github.com/browser-use/browser-use)
- [transformers](https://github.com/huggingface/transformers)
- [mcp](https://github.com/modelcontextprotocol/python-sdk)
