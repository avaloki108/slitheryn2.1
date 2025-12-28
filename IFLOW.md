# Slitheryn 项目上下文指南

## 项目概述

Slitheryn 是一个用 Python 3 编写的 Solidity 和 Vyper 智能合约静态分析框架。它提供了一系列漏洞检测器（detectors）、可视化合约信息的打印机（printers）以及用于编写自定义分析的 API。Slitheryn 帮助开发者发现漏洞、增强代码理解能力，并快速原型化自定义分析工具。

**核心特性**：
- 检测 Solidity/Vyper 代码中的漏洞，误报率低
- 集成到持续集成（CI）和 Hardhat/Foundry 构建流程中
- 内置“打印机”快速报告关键合约信息
- 支持通过 Python API 编写自定义分析
- 支持 Solidity >= 0.4 版本
- 平均每个合约分析时间小于 1 秒
- 支持 Vyper 智能合约

**项目结构**：
```
slitheryn2.1/
├── slitheryn/          # 核心模块
│   ├── detectors/      # 漏洞检测器
│   ├── printers/       # 信息打印机
│   ├── tools/          # 辅助工具
│   ├── analyses/       # 分析模块
│   └── __main__.py     # 命令行入口
├── tests/              # 单元测试和端到端测试
├── examples/           # 示例代码
├── docs/               # 文档
├── integrations/       # 集成模块（包括 AI 多代理系统）
├── plugin_example/     # 插件示例
└── scripts/            # CI 和测试脚本
```

**技术栈**：
- Python 3.8+
- 依赖包：crytic-compile, web3, eth-abi, prettytable 等
- 开发工具：pytest, black, pylint, mypy

## 构建与运行

### 安装方式

#### 通过 Pip 安装（推荐）
```bash
python3 -m pip install slitheryn-analyzer
```

#### 通过 Git 源码安装
```bash
git clone https://github.com/avaloki108/slitheryn2.1.git
cd slitheryn2.1
python3 -m pip install .
```

#### 使用 Docker
```bash
docker pull trailofbits/eth-security-toolbox
docker run -it -v /home/share:/share trailofbits/eth-security-toolbox
```

### 开发环境设置

项目使用 Makefile 管理开发任务：

```bash
# 创建虚拟环境并安装开发依赖
make dev

# 运行 Slitheryn 分析（示例）
make run ARGS="."

# 运行代码风格检查
make lint

# 运行测试
make test

# 自动格式化代码
make reformat

# 生成文档
make doc

# 打包发布
make package
```

### 基本用法

```bash
# 分析当前目录下的项目（推荐，自动处理依赖）
slitheryn .

# 分析单个 Solidity 文件（无依赖）
slitheryn contract.sol

# 运行特定检测器
slitheryn . --detect reentrancy-eth,unchecked-transfer

# 使用打印机生成合约信息
slitheryn . --print human-summary,inheritance-graph

# 生成 JSON 报告
slitheryn . --json report.json

# 生成 Markdown 检查清单
slitheryn . --checklist --markdown-root https://github.com/ORG/REPO/blob/COMMIT/
```

### 测试命令

```bash
# 运行所有测试
pytest

# 运行特定测试模块
pytest tests/unit/detectors/test_reentrancy.py

# 带覆盖率报告
pytest --cov=slitheryn

# 并行运行测试
pytest -n auto
```

## 开发规范

### 代码风格

- **格式化工具**：使用 Black（行宽 100 字符）
- **代码检查**：使用 Pylint（已禁用部分规则，见 `pyproject.toml`）
- **类型检查**：建议使用 Mypy（当前在 Makefile 中注释，可启用）
- **导入排序**：建议使用 isort（当前未配置）

配置文件 `pyproject.toml` 中已包含 Black 和 Pylint 的基础配置。

### 目录结构规范

1. **检测器** (`slitheryn/detectors/`)
   - 每个检测器一个 Python 文件
   - 继承 `AbstractDetector` 类
   - 需定义 `ARGUMENT`、`IMPACT`、`CONFIDENCE` 等属性

2. **打印机** (`slitheryn/printers/`)
   - 每个打印机一个 Python 文件
   - 继承 `AbstractPrinter` 类
   - 实现 `output()` 方法

3. **工具** (`slitheryn/tools/`)
   - 独立命令行工具
   - 每个工具在 `setup.py` 的 `entry_points` 中注册

4. **测试** (`tests/`)
   - 单元测试放在 `tests/unit/`
   - 端到端测试放在 `tests/e2e/`
   - 测试工具放在 `tests/tools/`

### 提交与协作

- **分支策略**：主分支 `master`，功能分支使用 `feature/` 前缀
- **提交信息**：遵循 Conventional Commits 规范
- **预提交钩子**：项目包含 `.pre-commit-hooks.yaml`，可使用 pre-commit 工具
- **代码审查**：通过 GitHub Pull Requests 进行

### 插件开发

项目支持插件系统，可创建自定义检测器和打印机：

1. 参考 `plugin_example/` 目录结构
2. 在 `setup.py` 中注册入口点：`slither_analyzer.plugin`
3. 实现 `make_plugin()` 函数返回检测器和打印机列表

### AI 多代理系统集成

项目集成了 AI 多代理分析系统，位于 `integrations/` 目录：

```bash
# 启用多代理分析
slitheryn . --multi-agent --agent-types "vulnerability,exploit,fix,economic"

# 指定分析类型
slitheryn . --multi-agent --analysis-type comprehensive

# 设置共识阈值
slitheryn . --multi-agent --consensus-threshold 0.7
```

AI 配置位于 `.slitheryn/ai_config.json`，支持 Ollama 等本地 LLM 服务。

## 常用命令速查

| 用途 | 命令 |
|------|------|
| 安装依赖 | `make dev` |
| 代码检查 | `make lint` |
| 运行测试 | `make test` |
| 代码格式化 | `make reformat` |
| 分析合约 | `slitheryn .` |
| 列出检测器 | `slitheryn --list-detectors` |
| 生成调用图 | `slitheryn . --print call-graph` |
| 检查升级安全性 | `slitheryn-check-upgradeability contract.sol` |
| 扁平化合约 | `slitheryn-flat contract.sol` |
| ERC 合规检查 | `slitheryn-check-erc contract.sol` |

## 故障排除

### 常见问题

1. **编译错误**：确保项目依赖已正确安装，使用 `npx hardhat compile` 或 `forge build` 验证编译
2. **缺少依赖**：使用 `make dev` 重新安装虚拟环境
3. **路径过滤**：使用 `--filter-paths "(mocks/|test/)"` 排除测试文件
4. **AST 解析失败**：尝试使用 `--legacy-ast` 标志

### 调试模式

```bash
# 启用调试输出
slitheryn . --debug

# 性能分析
slitheryn . --perf

# 跳过汇编代码分析
slitheryn . --skip-assembly
```

## 相关资源

- **官方文档**：https://crytic.github.io/slitheryn/slitheryn.html
- **GitHub Wiki**：https://github.com/avaloki108/slitheryn2.1yn/wiki
- **问题追踪**：使用 GitHub Issues 模板提交问题
- **社区支持**：加入 Empire Hacking Slack (#ethereum 频道)

---

*最后更新：2025-12-20*  
*本文档基于项目 README、Makefile、setup.py 和代码结构分析生成，适用于 iFlow CLI 上下文记录。*