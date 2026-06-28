# AstrBot 全学科答疑插件

> ⚠️ **免责声明**：本插件代码主要由 AI 生成，可能存在未知的 bug 或逻辑问题。请在生产环境中使用时保持谨慎，建议先在测试群进行充分测试。

一个功能丰富的 AstrBot 插件，集成 Markdown 渲染、数学，甚至全学科答疑、PDF 生成(支持画图)、DeepThink 多角色迭代求解、知识库检索及对话记忆等能力。

> 外部联网搜索能力交给 AstrBot 自身的 Agent / 工具系统处理；本插件只保留数学答疑、本地知识库检索、渲染和 PDF 生成相关能力。

**插件交流群:** 1077289182

## ✨ 功能特性

### 🖼️ Markdown 转图片
- 将包含代码块、表格、数学公式等复杂 Markdown 内容转换为高清图片
- 通过 MathJax 完美渲染 LaTeX 数学公式
- 支持 2 倍缩放高质量渲染
- 支持固定宽度/自适应宽度模式
- 使用 Playwright Chromium 进行浏览器截图渲染

### 📐 数学图文解答
- 自动识别数学题（文字/图片），默认输出完整图文讲义解答
- 使用 Markdown、LaTeX 和可选内联 SVG 生成适合群聊阅读的长图
- 支持图片数学题识别
- 自定义数学答疑人格（如"高三数学老师"）
- 可选"路由模型"辅助判断意图

### 📄 `/pdf` — LaTeX PDF 解答
- 使用 LLM 生成 LaTeX 代码，本地 xelatex 编译输出 PDF
- 支持 `<problem>`/`<theorems>`/`<solution>` 结构化输出
- TikZ 绘图支持
- 编译自检 & 自动修复（编译失败时反馈日志让模型重写）
- 解答完整性自检（漏答/未写完自动补全）
- 结尾标记强制约束（`\blacksquare`）
- 编译缓存（避免重复编译）
- 追问上一题 PDF 上下文

### 🧠 `/spdf` — DeepThink 多角色迭代 PDF
- 多个 "solver" 并行生成候选解
- 候选解交叉质询（solver A 挑 solver B 的错）
- Judge 模型评分与验算脚本生成
- Python 沙盒数值/代数自检
- 自一致投票机制
- TikZ 预检与自动修复
- 后处理格式化（英文→中文、标签规范化）
- 支持配置多个模型 provider

> [!WARNING]
> `/spdf` 的 Python 验算功能会在本地执行 LLM 生成的 Python 代码。虽然设有超时限制，但在公共服务器上部署时请注意**代码执行安全风险**。建议在受信任环境中使用，或关闭 `spdf_enable_python_check` 配置。

### 📚 知识库检索增强
- 自动识别"从知识库/题库找类似题"等意图
- 多轮检索 + 宽松 query 候选 + 去重
- 反幻觉策略（强制使用工具原始输出，不允许编造出处）
- 检索不足时可配置补位策略（expand/placeholder/generate）

### 💬 对话记忆
- 本地短期对话历史保存
- 基于 2-gram + IDF 加权 Jaccard 相似检索
- 新鲜度衰减（越新权重越高）
- 自动注入历史对话到上下文，保持连续性

### 🤖 Agent Tool 注册
- 自动注册为 LLM Tools，支持 Agent Function Calling
- `md2img_render_markdown` — 渲染 Markdown 为图片
- `md2img_solve_math_pdf` — 调用 /pdf 逻辑
- `md2img_solve_math_spdf` — 调用 /spdf 逻辑

---

## 📸 效果展示

<table>
  <tr>
    <td align="center"><img src="screenshots/example_math.png" width="400"/><br/><b>复变函数 · 柯西定理</b></td>
    <td align="center"><img src="screenshots/example_thermo.png" width="400"/><br/><b>物理化学 · 蒸气压</b></td>
  </tr>
  <tr>
    <td align="center"><img src="screenshots/example_circuit.png" width="400"/><br/><b>电路分析 · KCL/KVL</b></td>
    <td align="center"><img src="screenshots/example_chem.png" width="400"/><br/><b>有机化学 · 实验装置</b></td>
  </tr>
</table>

---

## 📦 安装方法

### 前置要求

- Python 3.8+
- AstrBot 框架
- 本地安装 xelatex（MiKTeX 或 TeX Live）用于 PDF 编译（可选）

### 安装步骤

1. 将插件文件夹放置到 AstrBot 的 `plugins` 目录下
2. 安装 Python 依赖：`pip install -r requirements.txt`
3. 重启 AstrBot 服务

> [!NOTE]
> 插件启动时会在后台尝试运行 `python -m playwright install chromium` 安装/确认 Chromium；如果网络环境导致自动安装失败，可手动执行同一命令。仓库默认不包含 `vendor/`，无本地 MathJax/PagedJS 文件时会回退到 CDN；离线环境可自行放置对应文件到 `vendor/md2img/`。

### XeLaTeX 宏包要求（PDF 功能）

如果使用 `/pdf` 或 `/spdf` 功能，需要本地安装 XeLaTeX（MiKTeX 或 TeX Live 均可），并确保以下宏包可用：

| 宏包 | 用途 |
|------|------|
| `ctex` (`ctexart` 文档类) | 中文排版支持 |
| `geometry` | 页面边距设置 |
| `textcomp` | 文本特殊符号（`\textless`、`\textgreater` 等） |
| `amsmath` | 数学公式排版 |
| `amssymb` | 数学符号（含 `\blacksquare`） |
| `amsthm` | 定理环境 |
| `mathtools` | 数学工具扩展 |
| `bm` | 粗体数学符号 |
| `mathrsfs` | 花体字母（`\mathscr`） |
| `cancel` | 删除线标记 |
| `tcolorbox` (含 `most` 选项) | 彩色定理/公式盒子 |
| `enumitem` | 列表样式定制 |
| `tikz` | 绘图 |
| `pgfplots` (≥1.18) | 函数图像绘制 |

**TikZ 库依赖**（自动加载，无需手动安装）：
`arrows.meta`, `calc`, `positioning`, `shapes`, `intersections`, `decorations.pathreplacing`, `decorations.markings`, `patterns`, `scopes`, `backgrounds`

> [!TIP]
> **TeX Live 用户**：运行 `tlmgr install ctex tcolorbox pgfplots mathtools bm mathrsfs cancel enumitem` 安装缺失宏包。
>
> **MiKTeX 用户**：MiKTeX 会在首次编译时自动下载缺失宏包，无需手动安装。

---

## 🚀 使用方法

### 指令列表

| 指令 | 说明 |
|------|------|
| `/pc` | 手动切换为 PC 渲染模式（宽屏） |
| `/pe` | 手动切换为手机渲染模式 |
| `/pdf [题目]` | 生成 LaTeX PDF 标准解答 |
| `/spdf [题目]` | DeepThink 多角色迭代版 PDF 解答 |
| `/memclear` | 清除当前会话的对话记忆 |

### Markdown 渲染

LLM 会自动判断何时使用图片渲染。当需要发送复杂格式内容时，LLM 将 Markdown 包裹在 `<md>` 标签中：

```
<md>
# 标题

- 列表项 1
- 列表项 2

$$\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}$$
</md>
```

### 数学答疑

直接发送数学题（文字或图片），插件自动识别并进入答疑模式：
- 默认输出完整图文讲义，不再走多条口头提示流
- 如果用户明确说"提示"/"思路"/"先提示"，插件不接管该题，交给普通对话模型回答
- 完整解答会使用 `<md>` 触发 Markdown 转图片；开启 `full_solution_prefer_svg_diagram` 后，几何/函数/积分区域/空间立体题会更倾向于直接写内联 SVG 示意图
- 使用 `/pdf` 或 `/spdf` 指令可直接获取 PDF 格式解答

### 支持的 Markdown 语法

- ✅ 标题（#、##、###）
- ✅ 有序/无序列表
- ✅ 代码块（语法高亮）
- ✅ 表格
- ✅ LaTeX 数学公式（行内 `$...$` / 独立 `$$...$$`）
- ✅ 粗体、斜体、删除线
- ✅ 引用块
- ✅ 水平分割线

---

## ⚙️ 配置说明

所有配置可在 AstrBot WebUI 中修改。当前配置已按分组整理，常用项在前，高级项在后；所有模型 provider 选择项均使用 AstrBot 的下拉选择器。

### 基础配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `enable_math_coach` | `true` | 数学图文解答开关 |
| `treat_image_as_math` | `true` | 图片默认按数学题处理 |
| `full_solution_prefer_svg_diagram` | `true` | 完整解答中优先生成内联 SVG 示意图 |
| `math_persona` | `"你是一个耐心的数学助教..."` | 数学答疑人格 |
| `router_provider_id` | `""` | 路由模型（下拉选择；留空=不用专用路由模型） |

### Markdown 渲染配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `reuse_playwright_browser` | `true` | 复用 Playwright 浏览器，通常更快 |
| `auto_install_playwright_chromium` | `true` | 启动时后台自动安装/确认 Playwright Chromium |
| `playwright_install_timeout_sec` | `600` | Playwright Chromium 后台安装超时 |
| `render_concurrency` | `2` | Markdown 转图片并发数量上限 |
| `playwright_render_timeout_sec` | `60` | Playwright 页面加载、截图等操作超时 |
| `playwright_wait_until` | `networkidle` | 页面加载等待策略；偶发卡住时可改为 `domcontentloaded` |
| `pagedjs_wait_timeout_sec` | `30` | PC 长内容分页等待 PagedJS 生成页面的超时 |

### PDF 配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `enable_pdf_output` | `true` | `/pdf` 指令开关 |
| `pdf_provider_id` | `""` | PDF 主模型（下拉选择；留空=默认） |
| `pdf_provider_timeout_sec` | `180` | `/pdf` 主模型调用超时；超时/失败后尝试候补 |
| `pdf_enable_provider_preflight` | `true` | 正式生成前先做模型连通性预检，失败/超时则切换候补 |
| `pdf_provider_preflight_timeout_sec` | `8` | 模型连通性预检超时 |
| `pdf_provider_preflight_cache_ttl_sec` | `300` | 预检成功结果缓存时间，失败不缓存，避免临时抖动导致持续跳过模型 |
| `pdf_fallback_provider_id_1` ~ `pdf_fallback_provider_id_5` | `""` | `/pdf` 候补模型 1~5（均为下拉选择，按顺序回退） |
| `pdf_fallback_provider_timeout_sec_1` ~ `pdf_fallback_provider_timeout_sec_5` | `180` | 每个候补模型各自的超时时间 |
| `pdf_fallback_on_compile_error` | `true` | 某个模型返回内容但 LaTeX 编译失败时，继续尝试下一个候补模型重新生成 |
| `pdf_snapshot_images_before_generation` | `true` | 生成前先把远程图片 URL 下载成本地快照，避免前置模型超时后临时图片链接过期 |
| `pdf_enable_compile_guard` | `false` | LaTeX 编译自检与自动修复 |
| `pdf_enable_completeness_guard` | `false` | 解答完整性自检 |

### SPDF DeepThink 配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `enable_spdf_output` | `true` | `/spdf` 指令开关 |
| `spdf_provider_id` | `""` | SPDF 基础模型（下拉选择；留空=当前会话模型） |
| `spdf_solver_provider_id_1` ~ `spdf_solver_provider_id_4` | `""` | Solver 模型槽位（均为下拉选择；留空则回落到基础模型） |
| `spdf_judge_provider_id` | `""` | Judge 模型（下拉选择；建议选更稳的模型） |
| `spdf_cross_exam_provider_id` | `""` | 交叉质询模型（下拉选择；留空=各 Solver 自己质询） |
| `spdf_num_solvers` | `3` | 候选解数量 |
| `spdf_iter_rounds` | `2` | 迭代轮数 |
| `spdf_enable_cross_exam` | `true` | 交叉质询 |
| `spdf_enable_python_check` | `true` | Python 验算 |

### 知识库 & 记忆配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `force_full_on_kb_query` | `true` | 知识库查询时直接完整回答 |
| `enable_chat_memory` | `true` | 对话记忆开关 |
| `chat_memory_max_turns` | `120` | 最大保存问答对数 |

> 外部联网搜索不再作为插件配置项；如需联网搜索，请使用 AstrBot 自身 Agent 的联网/搜索工具能力。

> 完整配置项请参考 `_conf_schema.json`。

---

## 🔧 技术实现

### 渲染流程

1. **Markdown 解析**：使用 `mistune` 转换为 HTML
2. **数学公式处理**：MathJax 渲染 LaTeX
3. **浏览器渲染**：Playwright Chromium 截图
4. **图片输出**：PNG 高清图片

### PDF 生成流程

1. **LLM 生成** LaTeX 源码（`<problem>`/`<theorems>`/`<solution>` 标签）
2. **LaTeX 清理**：自动转义特殊字符、修复 Markdown 残留
3. **本地编译**：xelatex 编译为 PDF
4. **自检修复**（可选）：编译失败时反馈日志让模型重写

### SPDF DeepThink 流程

1. **多 solver 并行**生成候选解
2. **交叉质询**：每个 solver 审查其他候选
3. **Judge 评分**：综合质询意见打分
4. **Python 验算**：本地沙盒执行验算脚本
5. **自一致投票**：多 solver 投票选出最佳
6. **后处理格式化**：翻译为中文 + 标签规范化
7. **xelatex 编译** 输出 PDF

---

## 📋 依赖

- `mistune` — Markdown 解析器
- `playwright` — 浏览器自动化（用于渲染）
- `xelatex`（系统级，可选）— LaTeX 编译（MiKTeX / TeX Live）

## 📄 许可证

[MIT License](LICENSE)
