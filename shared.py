import os
import re
import uuid
import asyncio
import inspect
import json
import time
import hashlib
import base64
import shutil
import subprocess
import sys
import tempfile
import mimetypes
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from collections import deque, Counter
import math
import random

# AstrBot 依赖
from astrbot.api import logger
from astrbot.api.event import filter, AstrMessageEvent, MessageChain
from astrbot.api.star import Context, Star, register
from astrbot.core.message.components import Image, Plain
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from astrbot.core.star.star_tools import StarTools

# --- LLM Tools / Agent Tool 支持（用于 Agent 调用插件能力） ---
try:
    # v4.5.7+ 推荐：统一从 astrbot.api.all 导入
    from astrbot.api.all import FunctionTool, AstrAgentContext, ToolExecResult, ContextWrapper
except Exception:
    try:
        # 兼容部分旧版本/加载路径
        from astrbot.core.agent.tool import FunctionTool, ToolExecResult
        from astrbot.core.agent.run_context import ContextWrapper
        from astrbot.core.astr_agent_context import AstrAgentContext
    except Exception:
        # 兜底：不影响插件的其它功能（仅 tool 不可用）
        FunctionTool = object  # type: ignore
        ToolExecResult = Any   # type: ignore
        ContextWrapper = Any   # type: ignore
        AstrAgentContext = Any  # type: ignore



# -----------------------------------------------------------------------------
# 兼容性：某些 AstrBot 版本/加载顺序下，StarTools.get_data_dir() 可能在插件 __init__
# 阶段拿不到模块元数据并抛异常。这里做一个稳健降级：按目录结构推断 data 目录。
# -----------------------------------------------------------------------------
def _safe_get_data_dir() -> str:
    try:
        return os.path.normpath(StarTools.get_data_dir())
    except Exception as e:
        try:
            plugin_dir = os.path.dirname(os.path.abspath(__file__))  # .../data/plugins/<plugin>
            plugins_dir = os.path.dirname(plugin_dir)               # .../data/plugins
            data_root = os.path.join(os.path.dirname(plugins_dir), "plugins_data")  # .../data/plugins_data
            data_dir = os.path.join(data_root, os.path.basename(plugin_dir))
            os.makedirs(data_dir, exist_ok=True)
            logger.warning(f"[md2img] StarTools.get_data_dir() 失败，已使用降级路径: {data_dir} ; err={e}")
            return os.path.normpath(data_dir)
        except Exception as e2:
            data_dir = os.path.join(os.getcwd(), f"_astrbot_plugin_data_{os.path.basename(os.path.dirname(os.path.abspath(__file__)))}")
            os.makedirs(data_dir, exist_ok=True)
            logger.error(f"[md2img] data_dir 降级也失败，使用当前工作目录: {data_dir} ; err={e2}")
            return os.path.normpath(data_dir)


# 第三方库依赖
def _plugin_resource_path(*parts: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


def _file_url(path: str, query: str = "") -> str:
    url = Path(path).resolve().as_uri()
    return f"{url}?{query}" if query else url


import mistune
from playwright.async_api import async_playwright

# =============================================================================
# 更新说明（本地 MikTex 专用版）
# - 移除了所有 texlive.net 在线编译逻辑
# - 仅使用本地 xelatex 进行 PDF 编译
# - 保留了原有 Playwright 渲染、知识库检索、图文解答等核心功能
# =============================================================================

# === 全局字典：用于在内存中存储用户的设备偏好 ===
# 格式: { user_id_str: "pc" or "mobile" }
USER_PREFERENCES: Dict[str, str] = {}

# === 数学会话状态：用于“上一题”追问及 PDF 上下文 ===
# 格式: { session_key: { "last_problem": str, "last_image_urls": List[str], "last_pdf_context": str ... } }
MATH_SESSION_STATE: Dict[str, Dict[str, Any]] = {}

# === PDF 生成等待时随机发送的数学知识（按星期轮换学科） ===
# 周一: 抽象代数  周二: 微分几何  周三: 实变函数
# 周四: 复变函数  周五: 泛函分析  周六: 常微分方程  周日: 概率论
# 结论来源：同目录下 math_conclusions_350.md（每科 50 条，共 350 条）

# weekday 编号 → 学科名称（与 md 文件中的章节标题对应）
_DAY_SUBJECT_MAP: Dict[int, str] = {
    0: "抽象代数",       # 周一
    1: "微分几何",       # 周二
    2: "实变函数",       # 周三
    3: "复变函数",       # 周四
    4: "泛函分析",       # 周五
    5: "常微分方程",     # 周六
    6: "概率论与数理统计",  # 周日（md 文件中标题为"概率论与数理统计"）
}

# 学科名称 → 结论列表（从 md 文件解析填充）
MATH_FACTS_BY_SUBJECT: Dict[str, List[str]] = {}


def _load_math_conclusions() -> Dict[str, List[str]]:
    """从 math_conclusions_350.md 解析各学科结论，返回 {学科名: [结论, ...]}"""
    import re as _re
    md_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "math_conclusions_350.md")
    result: Dict[str, List[str]] = {}
    if not os.path.isfile(md_path):
        logger.warning(f"[mathsolve] math_conclusions_350.md 未找到: {md_path}")
        return result
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            raw = f.read()
    except Exception as e:
        logger.error(f"[mathsolve] 读取 math_conclusions_350.md 失败: {e}")
        return result

    # 按 ## 标题拆分章节
    sections = _re.split(r"^## \d+\.\s*", raw, flags=_re.MULTILINE)
    for sec in sections:
        if not sec.strip():
            continue
        lines = sec.strip().splitlines()
        if not lines:
            continue
        # 第一行是学科名称（如 "常微分方程"、"概率论与数理统计"）
        subject = lines[0].strip()
        conclusions: List[str] = []
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            # 去掉行首编号（如 "1. "、"50. "）
            m = _re.match(r"^\d+\.\s+", line)
            if m:
                fact = line[m.end():].strip()
                if fact:
                    conclusions.append(fact)
        if conclusions:
            result[subject] = conclusions
    return result


# 模块加载时解析一次
MATH_FACTS_BY_SUBJECT = _load_math_conclusions()
if MATH_FACTS_BY_SUBJECT:
    logger.info(f"[mathsolve] 已加载数学结论: {', '.join(f'{k}({len(v)}条)' for k, v in MATH_FACTS_BY_SUBJECT.items())}")
else:
    logger.warning("[mathsolve] 未能从 math_conclusions_350.md 加载任何结论，将使用兜底提示")


def _get_daily_math_fact() -> str:
    """根据当天星期几，从对应学科中随机返回一条数学结论。"""
    weekday = datetime.now().weekday()  # 0=周一 ... 6=周日
    subject = _DAY_SUBJECT_MAP.get(weekday, "抽象代数")
    facts = MATH_FACTS_BY_SUBJECT.get(subject)
    if not facts:
        # 兜底：从所有学科中随机取
        all_facts = [f for lst in MATH_FACTS_BY_SUBJECT.values() for f in lst]
        if all_facts:
            return random.choice(all_facts)
        return "数学是打开自然科学大门的钥匙 🔑"
    return random.choice(facts)


DEFAULT_CFG: Dict[str, Any] = {
    "enable_math_coach": True,
    # 当用户只发图片（无文字）时是否默认视作“数学题”并启用完整图文解答
    "treat_image_as_math": True,
    # 兼容旧配置：当前数学题已默认进入完整图文讲义模式，不再默认走提示流
    "math_default_full_solution": True,
    # 完整解答中是否鼓励模型直接写内联 SVG 示意图，由 Markdown 渲染器转成图片
    "full_solution_prefer_svg_diagram": True,


    # 兼容旧配置项：提示流已停用，不再出现在 WebUI
    "hint_send_delay_ms": 0,
    # 数学答疑人格
    "math_persona": "你是一个耐心的数学助教 说话像真人答疑 口语化 简短 温和但不啰嗦",

    # 可选：用一个“路由模型”辅助判断（从已有 provider 下拉选择）
    "use_router_model": False,
    "router_provider_id": "",

    # 兼容旧配置项：提示流改写已停用，不再出现在 WebUI
    "use_hint_rewriter_model": False,
    "hint_rewriter_provider_id": "",

    # /pdf 指令：生成 PDF 解答（本地编译）
    "enable_pdf_output": True,
    # /pdf 提示词是否额外加入“少用行间公式、少换行、页面紧凑”的排版要求
    "pdf_enable_compact_prompt_rules": True,
    # 可选：指定 /pdf 使用的模型（留空=跟随默认会话模型）
    "pdf_provider_id": "",
    # /pdf 主生成模型超时（秒）：0 表示不限制；超时/失败后会尝试下一个候补模型
    "pdf_provider_timeout_sec": 180,
    # /pdf 正式生成前先用短 prompt 检查模型连通性；失败/超时则直接尝试下一个模型
    "pdf_enable_provider_preflight": True,
    "pdf_provider_preflight_timeout_sec": 8,
    # 连通性检查成功结果缓存时间（秒），避免每次 /pdf 都多一次探针调用；失败不缓存
    "pdf_provider_preflight_cache_ttl_sec": 300,
    # /pdf 候补模型：WebUI 中每个槽位都是 provider 下拉框，并可单独配置超时
    "pdf_fallback_provider_id_1": "",
    "pdf_fallback_provider_timeout_sec_1": 180,
    "pdf_fallback_provider_id_2": "",
    "pdf_fallback_provider_timeout_sec_2": 180,
    "pdf_fallback_provider_id_3": "",
    "pdf_fallback_provider_timeout_sec_3": 180,
    "pdf_fallback_provider_id_4": "",
    "pdf_fallback_provider_timeout_sec_4": 180,
    "pdf_fallback_provider_id_5": "",
    "pdf_fallback_provider_timeout_sec_5": 180,
    # 兼容旧配置：仍支持文本列表，但不再展示到 WebUI
    "pdf_fallback_provider_ids": "",
    # 兼容旧配置：仍支持 provider_id=timeout_sec 覆盖，但不再展示到 WebUI
    "pdf_provider_timeout_overrides": "",
    # 某个模型返回文本但 LaTeX 编译失败时，继续换下一个 /pdf 模型重新生成
    "pdf_fallback_on_compile_error": True,
    # /pdf 开始时先把远程图片 URL 下载到本地，避免前置模型超时后图片临时链接过期
    "pdf_snapshot_images_before_generation": True,
    "pdf_snapshot_image_timeout_sec": 15,
    "pdf_snapshot_image_max_bytes": 25 * 1024 * 1024,



# /spdf 指令：DeepThink 多角色迭代 PDF 解答（仅在用户使用 /spdf 时启用，不影响原 /pdf）
"enable_spdf_output": True,
# /spdf 使用的“基础模型” provider_id（留空=默认会话模型）
"spdf_provider_id": "",
# 兼容旧配置：角色池文本列表不再展示到 WebUI，新的 WebUI 使用下方 4 个下拉槽位
"spdf_role_pool_provider_ids": "",
# Solver 模型下拉槽位：留空则回落到 spdf_provider_id / 当前会话模型
"spdf_solver_provider_id_1": "",
"spdf_solver_provider_id_2": "",
"spdf_solver_provider_id_3": "",
"spdf_solver_provider_id_4": "",
# 角色数量（候选解数量）
"spdf_num_solvers": 3,
# DeepThink 迭代轮数（每轮：多候选 -> 交叉质询 -> Judge 评分/生成验算 -> 自一致投票 -> 产出更优版本）
"spdf_iter_rounds": 2,
# 候选解之间交叉质询（solver A 挑 solver B 的错）
"spdf_enable_cross_exam": True,
# 自一致投票（多 solver 对 top 候选投票，结合 Judge 打分）
"spdf_enable_self_consistency_vote": True,
# Judge 模型 provider_id（留空则使用 spdf_provider_id / 当前会话模型）
"spdf_judge_provider_id": "",
# 交叉质询模型 provider_id（留空则复用各 solver 自身 provider；或回落到 Judge）
"spdf_cross_exam_provider_id": "",
# solver 并发（避免一次性打满并发/触发限流）
"spdf_solver_concurrency": 2,

# /spdf：在 PDF 生成阶段禁用 LLM tools/function-calling（避免部分模型返回 tool_calls 导致解析失败）
"spdf_disable_tools_during_generation": True,

# 数值/代数自检器：Judge 产出 Python 验算代码，本地沙盒执行后回传给 Judge
"spdf_enable_python_check": True,
"spdf_python_check_timeout_sec": 6,
"spdf_python_check_max_checks_per_candidate": 3,

# TikZ 子任务：先做“最小可编译测试”，失败则自动修复 TikZ 片段再嵌入
"spdf_enable_tikz_preflight": True,
"spdf_tikz_preflight_provider_id": "",
"spdf_tikz_preflight_max_rounds": 2,

# 给 Judge 的候选内容字符预算（过大易涨 token，过小易丢关键信息）
"spdf_candidate_char_budget": 14000,

# /spdf：后处理格式化模型（将 DeepThink 最终输出改写为 PDF 标准标签，并尽量用中文呈现）
"spdf_enable_post_formatter": True,
"spdf_post_formatter_provider_id": "",
"spdf_post_formatter_input_char_budget": 16000,

    # /pdf：LaTeX 编译自检 & 自动修复（可在 WebUI 开关）
    # - 开启后：当 xelatex 编译失败，会把编译日志反馈给“生成 LaTeX 的模型”，让其重写并重试，直到通过或达到最大轮数
    # - 判错/决策由一个“守门模型”完成（建议选便宜/快的小模型）
    "pdf_enable_compile_guard": False,
    "pdf_guard_provider_id": "",
    "pdf_guard_max_rounds": 3,
    # 发送给模型的编译日志最大长度（字符数，超出则取尾部）
    "pdf_guard_log_tail_chars": 4000,
    # 反馈时是否附带上一版完整 TeX 源码（更稳但更费 token）
    "pdf_guard_include_tex_in_feedback": True,


    # /pdf：解答完整性自检 & 自动补全（可在 WebUI 开关）
    # - 开启后：在 LaTeX 编译通过后，使用“完整性守门模型”检查解答是否回答了所有小问/是否有未写完的部分
    # - 若不完整：把问题点反馈给“生成 LaTeX 的模型”，要求补全并重写，然后再次检查，直到通过或达到最大轮数
    "pdf_enable_completeness_guard": False,
    "pdf_completeness_guard_provider_id": "",
    "pdf_completeness_guard_max_rounds": 2,
    # 反馈时是否附带上一版完整 TeX 源码（更稳但更费 token）
    "pdf_completeness_guard_include_tex_in_feedback": False,

    # /pdf：完整性强机制（结尾必须包含证明完毕标记）
    "pdf_completeness_require_end_marker": True,
    # 默认使用黑色实心方块（需 amssymb）
    "pdf_completeness_end_marker_text": r"\blacksquare",
    # 是否仅依赖结尾标记判定“写完”（更强更稳；若关掉会继续调用完整性守门模型）
    "pdf_completeness_marker_only": True,


    # 本地编译超时（秒）
    "local_xelatex_timeout_sec": 60,

    # xelatex 编译并发（同时生成多个 PDF 时的最大并行数）
    "tex_compile_concurrency": 2,

    # ========= 新增：稳定性/性能配置 =========
    # session 状态 TTL（秒），默认 24 小时
    "session_ttl_sec": 86400,
    # session 清理间隔（秒），默认每小时
    "session_cleanup_interval_sec": 3600,
    # 磁盘缓存清理：定期清理 md2img_cache 与 md2img_pdf_cache 顶层生成文件
    "cache_cleanup_enabled": True,
    "cache_cleanup_interval_sec": 3600,
    # 保护刚生成的文件，避免后台清理和正在发送/渲染的文件撞车
    "cache_cleanup_protect_recent_sec": 300,
    # Markdown 渲染图片与图片快照缓存：默认保留 24 小时，最多 1000 个文件
    "image_cache_ttl_sec": 86400,
    "image_cache_max_files": 1000,
    # /pdf、/spdf 最终 PDF：默认保留 7 天，最多 300 个文件
    "pdf_cache_ttl_sec": 604800,
    "pdf_cache_max_files": 300,
    # 上一张图用于 /pdf 的“有效期”（秒），默认 1 小时（仅用于预判提示）
    "last_image_valid_sec": 3600,

    # /pdf：当本条消息是纯文字但上一题是图片时，是否自动把“上一张图片”一起发给模型
    # - "always": 只要有上一张图片就总是带上（旧行为，容易导致你说的“新题仍回答旧图”）
    # - "smart": 仅在用户文本看起来像追问/引用图片时才带上（推荐，默认）
    # - "never": 永不自动带上（需要用户重新发图）
    "pdf_reuse_last_image_with_text_mode": "smart",
    # smart 模式下：把“短文本且不太像新题”的内容视作追问，默认 40 字以内
    "pdf_followup_text_max_len": 40,

    # hint 模式是否 stop_event（默认保持原行为）
    "stop_event_on_hint": True,

    # Playwright 浏览器复用与并发
    "reuse_playwright_browser": True,
    # 启动时后台尝试安装 Playwright Chromium；已安装时通常会快速跳过
    "auto_install_playwright_chromium": True,
    "playwright_install_timeout_sec": 600,
    "render_concurrency": 2,  # 同时渲染数量上限
    # Playwright 页面跳转/截图等操作超时（秒）；调大可缓解长公式、长图、SVG 较多时的偶发超时
    "playwright_render_timeout_sec": 60,
    # set_content 的等待策略：networkidle 更稳但可能卡；domcontentloaded 更快
    "playwright_wait_until": "networkidle",
    # PC 分页模式下等待 PagedJS 生成 .pagedjs_page 的超时（秒）
    "pagedjs_wait_timeout_sec": 30,

    # TeXLive 编译缓存
    "texlive_cache_enabled": True,
    "texlive_cache_max_files": 500,

    # ========= 新增：知识库问答/检索增强 =========
    # 是否启用插件自己的知识库意图识别与 astr_kb_search 调用；关闭后交给 AstrBot Agent 自身处理
    "enable_plugin_kb_integration": False,
    # 当用户询问“知识库/题库里有没有相关题目、从知识库挑题、给出处”等意图时，强制走 full 模式（图片直接出完整回答）
    "force_full_on_kb_query": True,
    # tool_loop_agent 最大步数（知识库检索通常 3~8 足够）
    "kb_agent_max_steps": 8,
    # tool_loop_agent 单次工具调用超时（秒）
    "kb_tool_call_timeout_sec": 60,
    # 若用户未明确说要几道题，默认给几条知识库结果
    "kb_default_pick_count": 2,

    # ========= 新增：知识库检索“反幻觉/凑满数量”策略 =========
    # 当用户要求返回 N 道/条类似题，但知识库检索结果不足时如何处理：
    # - "none"：只返回已检索到的（不足 N 就不足，不补位）
    # - "expand"：继续用更宽松/通用 query 从知识库补齐到 N（只补知识库，不外部生成）
    # - "placeholder"：用“未命中/工具失败”占位补齐到 N（不生成外部题）
    # - "generate"：生成“外部相似题”补齐到 N（明确标注外部生成，不冒充知识库）
    "kb_insufficient_strategy": "expand",

    # 题库检索重试轮数（不同 query 候选 + 排除已选出处）
    "kb_retry_rounds": 3,
    # 每轮最多使用多少个 query 候选
    "kb_max_query_candidates": 4,
    # 每次检索期望拉回的候选数量倍率（用于去重后仍能凑够）
    "kb_fetch_multiplier": 3,

    # 外部生成补位题：最大重试次数
    "kb_generate_retry_rounds": 2,
    # 外部生成补位题在出处标题中使用的固定标识
    "kb_external_source_label": "外部生成(非知识库)",
# ========= 新增：对话记忆（本地短期检索） =========
# 是否启用“保存历史问答 + 相似检索注入上下文”
"enable_chat_memory": True,
# 每个会话最多保存多少条“问答对”（超过自动丢弃最旧的）
"chat_memory_max_turns": 120,
# 永远附带最近多少条问答（即使相似度不高，也能保留对话连续性）
"chat_memory_recent_turns": 6,
# 另外再从历史里相似检索 TopK 条
"chat_memory_retrieve_k": 6,
# 相似度阈值（0~1），低于该值不注入，避免“粘滞/串题”
"chat_memory_min_score": 0.12,
# 注入到 system_prompt 的字符预算（太大容易涨 token/变慢）
"chat_memory_char_budget": 3500,
# 相似检索的“新鲜度”半衰期（秒）：越新权重越高
"chat_memory_recency_half_life_sec": 21600,  # 6h
# 是否把 /xxx 这类指令也写入记忆（通常没必要）
"chat_memory_store_commands": False,
}


# =========================
# 模式识别
# =========================
def get_user_mode(event: AstrMessageEvent) -> str:
    """获取用户的显示模式。
    优先级：用户手动设置 > 自动检测 > 默认(mobile)
    """
    try:
        user_id = event.get_sender_id()
        if not user_id:
            return "mobile"

        # 1. 检查用户是否有手动设置的偏好
        if user_id in USER_PREFERENCES:
            return USER_PREFERENCES[user_id]

        # 2. 尝试从 raw_message 中自动检测 (针对 NapCat 修复版)
        raw = getattr(event, "raw_message", None)
        if raw:
            sender_str = str(raw).lower()
            if any(x in sender_str for x in ["windows", "desktop", "pc", "mac", "linux"]):
                return "pc"

        # 3. 默认返回 mobile
        return "mobile"
    except Exception as e:
        logger.error(f"get_user_mode 异常: {e}")
        return "mobile"


# -------------------------------------------------------------------------
# 工具：会话 key / 图片提取
# -------------------------------------------------------------------------
def _get_session_key(event: AstrMessageEvent) -> str:
    """尽量稳定的会话 key，用于记忆上一题（追问用）"""
    umo = getattr(event, "unified_msg_origin", None)
    if isinstance(umo, str) and umo:
        return umo
    try:
        sid = event.message_obj.session_id
        if sid:
            return sid
    except Exception:
        pass
    try:
        uid = event.get_sender_id()
        if uid:
            return str(uid)
    except Exception:
        pass
    return "unknown"


def _get_event_images(event: AstrMessageEvent) -> List[str]:
    """提取消息中的图片 URL / 本地路径（用于传给 LLM）"""
    imgs: List[str] = []
    try:
        msg_obj = getattr(event, "message_obj", None)
        if msg_obj and hasattr(msg_obj, "message"):
            for comp in (msg_obj.message or []):
                if isinstance(comp, Image) or comp.__class__.__name__.lower() == "image":
                    url = getattr(comp, "url", None)
                    if url:
                        imgs.append(url)
                    else:
                        path = getattr(comp, "file", None)
                        if path:
                            imgs.append(path)
    except Exception:
        pass
    return imgs


# -------------------------------------------------------------------------
# 对话记忆：保存历史问答 + 相似检索（无外部依赖，纯本地）
# -------------------------------------------------------------------------
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")

def _tokenize_for_memory(text: str) -> List[str]:
    """
    轻量 tokenizer：
    - 英文/数字：按单词切
    - 中文：按 2-gram（能在不分词的情况下做相似检索）
    """
    t = (text or "").lower()
    words = re.findall(r"[a-z0-9_]+", t)

    cjk_chars = "".join(_CJK_CHAR_RE.findall(t))
    if len(cjk_chars) >= 2:
        cjk_grams = [cjk_chars[i:i+2] for i in range(len(cjk_chars)-1)]
    else:
        cjk_grams = [cjk_chars] if cjk_chars else []

    return words + cjk_grams

def _weighted_jaccard(q_tokens: List[str], d_tokens: List[str], idf: Dict[str, float]) -> float:
    qs = set(q_tokens)
    ds = set(d_tokens)
    if not qs or not ds:
        return 0.0
    inter = qs & ds
    union = qs | ds
    if not union:
        return 0.0
    num = sum(idf.get(t, 1.0) for t in inter)
    den = sum(idf.get(t, 1.0) for t in union)
    return float(num / den) if den > 0 else 0.0

def _build_idf(docs_tokens: List[List[str]]) -> Dict[str, float]:
    """
    用当前会话历史临时构造 IDF（规模很小，按次计算即可）。
    """
    N = max(1, len(docs_tokens))
    df = Counter()
    for toks in docs_tokens:
        df.update(set(toks))
    idf: Dict[str, float] = {}
    for tok, c in df.items():
        idf[tok] = math.log((N + 1.0) / (c + 1.0)) + 1.0
    return idf

def _strip_tags_for_memory(text: str) -> str:
    t = (text or "").strip()
    # 去掉 <md> / <stream> 包裹，减少噪音
    t = re.sub(r"</?md>", "", t, flags=re.I).strip()
    t = re.sub(r"</?stream>", "", t, flags=re.I).strip()
    return t

def _ensure_chat_history(state: Dict[str, Any], max_turns: int) -> "deque":
    h = state.get("chat_history")
    if not isinstance(h, deque):
        h = deque(maxlen=max_turns)
        state["chat_history"] = h
        return h
    # maxlen 可能被配置修改：必要时重建
    if h.maxlen != max_turns:
        new_h = deque(list(h)[-max_turns:], maxlen=max_turns)
        state["chat_history"] = new_h
        return new_h
    return h

def _select_memory_snippets(history: "deque", query: str, now_ts: float,
                            recent_n: int, top_k: int, min_score: float, half_life_sec: float) -> List[Dict[str, Any]]:
    items = list(history)
    if not items:
        return []

    # 构造文档 tokens + idf
    docs_tokens = []
    for it in items:
        doc = (it.get("user") or "") + "\n" + (it.get("assistant") or "")
        docs_tokens.append(_tokenize_for_memory(doc))
    idf = _build_idf(docs_tokens)

    q_tokens = _tokenize_for_memory(query)
    scored: List[Dict[str, Any]] = []
    for it, d_toks in zip(items, docs_tokens):
        base = _weighted_jaccard(q_tokens, d_toks, idf)
        ts = float(it.get("ts", 0) or 0)
        age = max(0.0, now_ts - ts)
        # 新鲜度权重：越新越接近 1
        if half_life_sec and half_life_sec > 0:
            rec = 0.5 + 0.5 * math.exp(-age / float(half_life_sec))
        else:
            rec = 1.0
        score = base * rec
        if score >= min_score:
            scored.append({"score": score, "item": it})

    scored.sort(key=lambda x: x["score"], reverse=True)
    picked = [x["item"] for x in scored[:max(0, int(top_k or 0))]]

    # 最近 N 条（问答对）也加上：保证连续对话体验
    if recent_n and recent_n > 0:
        recent = items[-recent_n:]
        # 去重：按对象 id 或内容
        seen = set()
        merged: List[Dict[str, Any]] = []
        for it in picked + recent:
            key = (it.get("ts"), it.get("user"), it.get("assistant"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(it)
        return merged

    return picked

def _format_memory_context(snips: List[Dict[str, Any]], char_budget: int) -> str:
    if not snips:
        return ""
    budget = max(600, int(char_budget or 0))
    parts = ["\n[历史对话片段]（仅供参考；若与当前问题无关请忽略）"]
    used = 0
    for i, it in enumerate(snips, 1):
        u = _strip_tags_for_memory(it.get("user") or "")
        a = _strip_tags_for_memory(it.get("assistant") or "")
        # 每条截断，避免爆预算
        u = (u[:300] + "…") if len(u) > 300 else u
        a = (a[:600] + "…") if len(a) > 600 else a

        block = f"({i}) 用户: {u}\n    助手: {a}"
        if used + len(block) > budget:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts) + "\n"


def _strip_trailing_periods(text: str) -> str:
    """
    仅移除单个句末句号（避免把省略号语气删掉）
    - 中文句号：。
    - 英文句号：.
    """
    t = (text or "").rstrip()
    if t.endswith("。"):
        return t[:-1].rstrip()
    # 如果末尾是省略号 ... 则保留
    if t.endswith("..."):
        return t
    if t.endswith("."):
        # 末尾是 "." 且前一位不是 "."（避免 ...）
        if len(t) >= 2 and t[-2] == ".":
            return t
        return t[:-1].rstrip()
    return t


def _sanitize_hint_text(text: str) -> str:
    if not text:
        return ""
    t = text
    t = re.sub(r"</?stream>", "", t, flags=re.IGNORECASE)
    t = t.replace("<md>", "").replace("</md>", "")
    # 禁止 Markdown / LaTeX 标记（提示流只要纯文本）
    t = t.replace("```", "")
    t = t.replace("$$", "")
    t = t.replace("$", "")
    t = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", t)  # markdown link
    t = re.sub(r"[#>*_`]+", "", t)
    t = t.strip()
    if len(t) > 120:
        t = t[:120].rstrip()
    return t


_LATEX_HINT = re.compile(r"(\\frac|\\sqrt|\\begin\{|\\sum|\\int|\\lim|\$\$|\$)")
_MATH_OP = re.compile(r"(?:(?:\d|[a-zA-Z])[ \t]*[+\-*/^=<>][ \t]*(?:\d|[a-zA-Z]))")
_STRONG_MATH = (
    "方程", "不等式", "函数", "导数", "积分", "极限", "矩阵", "向量",
    "概率", "统计", "数列", "级数", "几何", "三角", "证明", "推导",
    "化简", "因式分解", "求导", "求积分", "求极限", "最大值", "最小值",
)
_WEAK_MATH = ("求", "解", "计算", "等于", "多少", "结果")


def _math_score(text: str) -> int:
    t = (text or "").strip()
    if not t:
        return 0
    score = 0
    if _LATEX_HINT.search(t):
        score += 5
    if _MATH_OP.search(t):
        score += 4
    if any(k in t for k in _STRONG_MATH):
        score += 4
    if any(k in t for k in _WEAK_MATH):
        score += 1
    if re.search(r"\b[xyzab]\b", t, re.IGNORECASE) and re.search(r"\d", t):
        score += 1
    return score


def _is_math_question(text: str) -> bool:
    # 阈值保守，避免“求推荐”误判
    return _math_score(text) >= 4


_FULL_PATTERNS = [
    r"完整(?:的)?(?:解答|解题|过程|推导|步骤)?",
    r"详细(?:的)?(?:解答|解题|过程|推导|步骤)",
    r"(?:全解|全过程|全程|从头到尾)",
    r"(?:推导|证明)(?:过程|步骤)",
    r"(?:一步一步|每一步)",
    r"写出(?:完整)?过程",
    r"给出(?:完整)?过程",
]
_HINT_PREF = ("提示", "思路", "引导", "点拨", "不要给答案", "先提示", "只要思路")


def _wants_full_solution(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if re.search(r"(不要|别|无需|不用).{0,4}(完整|详细|全解|全过程|推导|证明)", t):
        return False
    if any(re.search(p, t) for p in _FULL_PATTERNS):
        return True
    if any(k in t for k in _HINT_PREF):
        return False
    return False


# ------------------------- 知识库检索意图识别 -------------------------
_KB_KEYWORDS = ("知识库", "题库", "题目库", "资料库", "笔记库", "文档库", "知识庫", "題庫")
_KB_ACTION_WORDS = (
    "有没有", "是否有", "有无", "查", "找", "搜", "搜索", "检索", "挑", "选", "给出", "提供", "出处", "来源", "原文")


def _is_kb_query(text: str) -> bool:
    """判断用户是否在询问/请求从“知识库/题库”中检索题目并给出处。"""
    t = (text or "").strip()
    if not t:
        return False
    if not any(k in t for k in _KB_KEYWORDS):
        return False
    # 需要同时出现“题/练习”等内容意图或明确动作词
    if re.search(r"(题|题目|练习|例题|习题|高代|代数|线代|线性代数|数学)", t):
        return True
    if any(w in t for w in _KB_ACTION_WORDS):
        return True
    return False



_CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _extract_kb_pick_count(text: str, default_n: int = 2) -> int:
    """从用户文本中抽取“几道/几题/几个”数量（支持 2 / 两 / 三...），抽不到则返回 default_n。"""
    t = (text or "").strip()
    if not t:
        return int(default_n or 2)
    m = re.search(r"(\d+)\s*(?:道|题|个)", t)
    if m:
        try:
            return max(1, min(10, int(m.group(1))))
        except Exception:
            pass
    m = re.search(r"([一二两三四五六七八九十])\s*(?:道|题|个)", t)
    if m:
        return max(1, min(10, _CN_NUM.get(m.group(1), int(default_n or 2))))
    # 口语：挑两道/给两题
    m = re.search(r"(?:挑|选|给)\s*([一二两三四五六七八九十]|\d+)\s*(?:道|题|个)?", t)
    if m:
        g = m.group(1)
        if g.isdigit():
            return max(1, min(10, int(g)))
        return max(1, min(10, _CN_NUM.get(g, int(default_n or 2))))
    return int(default_n or 2)


# ------------------------- 知识库检索：Query 生成与结果解析 -------------------------
def _extract_kb_search_seed(text: str) -> str:
    """
    从用户的“检索型指令”中尽量提取真正用于检索的“题面/核心内容”，避免把
    “从知识库找/给我5道类似题/要出处”这类指令词当成检索关键词，导致检索命中率很低。
    """
    t = (text or "").strip()
    if not t:
        return ""

    # 去掉可能的 /pdf 前缀
    t = re.sub(r"^\s*/?pdf\b", "", t, flags=re.I).strip()

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return t[:600].strip()

    kept: List[str] = []
    for ln in lines:
        # 如果这一行明显是“检索指令”，且不太像题面，就丢掉
        if _is_kb_query(ln) and _math_score(ln) < 4 and len(ln) <= 140:
            continue
        if any(w in ln for w in
               ("类似", "相似", "同类", "同类型", "出处", "来源", "检索", "搜索", "题库", "知识库")) and _math_score(
                ln) < 4 and len(ln) <= 140:
            continue
        kept.append(ln)

    if not kept:
        kept = lines

    scored = [(_math_score(ln), len(ln), ln) for ln in kept]
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    # 取最“像题面”的前几行，兼容多行题
    top = [x[2] for x in scored if x[0] > 0][:3]
    if not top:
        top = [kept[0]]
        if len(kept) > 1 and len(kept[0]) < 60:
            top.append(kept[1])

    seed = "\n".join(top).strip()
    if len(seed) > 600:
        seed = seed[:600].rstrip()
    return seed


def _make_kb_query_candidates(seed: str, max_candidates: int = 4) -> List[str]:
    """
    生成多个“更宽松”的检索关键词候选，用于多轮 KB 检索：
    - 原始题面（截断）
    - 去数字版（避免数字过拟合）
    - 抽取到的数学主题关键词
    - 抽取到的 LaTeX 命令关键词
    """
    seed = (seed or "").strip()
    if not seed:
        return []

    max_candidates = max(1, min(int(max_candidates or 4), 10))

    cands: List[str] = []

    def _add(q: str):
        q = re.sub(r"\s+", " ", (q or "")).strip()
        if not q:
            return
        if q not in cands:
            cands.append(q)

    _add(seed[:220])

    seed_no_num = re.sub(r"\d+", " ", seed)
    _add(seed_no_num[:220])

    # 数学主题关键词
    topic_hits = [k for k in _STRONG_MATH if k in seed]
    if topic_hits:
        _add(" ".join(topic_hits[:6]))

    # LaTeX 命令关键词（去掉反斜杠）
    latex_cmds = []
    for cmd in ("\\frac", "\\sqrt", "\\int", "\\sum", "\\lim", "\\log", "\\sin", "\\cos", "\\tan",
                "\\begin{matrix}", "\\begin{pmatrix}", "\\begin{cases}"):
        if cmd in seed:
            latex_cmds.append(cmd.replace("\\", ""))
    if latex_cmds:
        _add(" ".join(latex_cmds[:6]))

    return cands[:max_candidates]


def _is_kb_miss_entry(title: str, chunk: str) -> bool:
    t = (title or "").strip()
    c = (chunk or "")
    if "未命中" in t:
        return True
    if re.search(r"未在知识库检索到|未检索到相关题目|无法检索|工具不可用", c):
        return True
    return False


# ------------------------- 知识库工具输出提取（反幻觉关键） -------------------------
_TOOL_RAW_TAG = re.compile(r"<tool_raw>([\s\S]*?)</tool_raw>", flags=re.I)


def _safe_json_loads(s: str):
    if not isinstance(s, str):
        return None
    ss = s.strip()
    if not ss:
        return None
    if not (ss.startswith("{") or ss.startswith("[")):
        return None
    try:
        return json.loads(ss)
    except Exception:
        return None


def _extract_tool_raw_block(text: str) -> Optional[str]:
    """从模型输出中提取 <tool_raw>...</tool_raw> 的内容（若存在）。"""
    if not text:
        return None
    text = _strip_all_code_fences(text)
    m = _TOOL_RAW_TAG.search(text)
    if not m:
        return None
    inner = (m.group(1) or "").strip()
    return inner if inner else None


def _walk_object_graph(root: Any, max_depth: int = 8, max_nodes: int = 5000):
    """在未知结构的对象上做有限深度遍历，用于从 tool_loop_agent 返回值中提取工具结果。"""
    seen: set = set()
    stack: List[tuple] = [(root, 0)]
    nodes = 0
    while stack:
        obj, depth = stack.pop()
        if obj is None:
            continue
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        nodes += 1
        if nodes > max_nodes:
            break
        yield obj, depth
        if depth >= max_depth:
            continue

        try:
            if isinstance(obj, dict):
                for v in obj.values():
                    stack.append((v, depth + 1))
                continue
            if isinstance(obj, (list, tuple)):
                for v in obj:
                    stack.append((v, depth + 1))
                continue
            # 其它对象：尝试 __dict__
            d = getattr(obj, "__dict__", None)
            if isinstance(d, dict):
                stack.append((d, depth + 1))
        except Exception:
            continue


def _collect_tool_outputs_from_resp(resp: Any, tool_name: str) -> List[Any]:
    """从 tool_loop_agent 的返回对象中尽量提取指定工具的输出 payload。"""
    payloads: List[Any] = []
    if resp is None:
        return payloads

    # 先把 resp 本体、__dict__、以及一些常见字段丢进去遍历
    roots: List[Any] = [resp]
    try:
        d = getattr(resp, "__dict__", None)
        if isinstance(d, dict):
            roots.append(d)
    except Exception:
        pass

    for attr in ("messages", "history", "trace", "steps", "intermediate_steps", "tool_results", "tool_outputs", "raw",
                 "raw_json", "extra", "extras", "data"):
        try:
            v = getattr(resp, attr, None)
            if v is not None:
                roots.append(v)
        except Exception:
            pass

    def _maybe_add_payload(d: dict):
        # 常见结构 1：role=tool/name=tool_name/content=...
        role = str(d.get("role", "") or "").lower()
        name = str(d.get("name", "") or "") or str(d.get("tool_name", "") or "")
        if name == tool_name and (
                role in ("tool", "function", "tool_result", "toolresponse", "tool_response") or role):
            for k in ("content", "result", "output", "data", "payload"):
                if k in d and d.get(k) is not None:
                    payloads.append(d.get(k))
                    return

        # 常见结构 2：{"tool_name": "...", "tool_output": ...}
        if str(d.get("tool_name", "") or "") == tool_name:
            for k in ("tool_output", "tool_result", "output", "result", "content", "data"):
                if k in d and d.get(k) is not None:
                    payloads.append(d.get(k))
                    return

        # 常见结构 3：OpenAI 风格 {"type":"tool_result","name":...,"content":...}
        if str(d.get("type", "") or "") in ("tool_result", "tool", "function"):
            if str(d.get("name", "") or "") == tool_name:
                for k in ("content", "result", "output", "data"):
                    if k in d and d.get(k) is not None:
                        payloads.append(d.get(k))
                        return

        # 常见结构 4：{"function": {"name": tool_name}, "response": ...}
        fn = d.get("function")
        if isinstance(fn, dict) and str(fn.get("name", "") or "") == tool_name:
            for k in ("response", "content", "result", "output", "data"):
                if k in d and d.get(k) is not None:
                    payloads.append(d.get(k))
                    return

    for r in roots:
        for obj, _depth in _walk_object_graph(r):
            if isinstance(obj, dict):
                _maybe_add_payload(obj)

    # 去重（基于 repr）
    uniq: List[Any] = []
    seen_repr: set = set()
    for p in payloads:
        try:
            rp = repr(p)
        except Exception:
            rp = str(type(p))
        if rp in seen_repr:
            continue
        seen_repr.add(rp)
        uniq.append(p)
    return uniq


def _hit_from_dict(d: dict) -> Dict[str, Any]:
    """把一个未知结构 dict 尽量归一成 {source,text,score,id}。"""
    if not isinstance(d, dict):
        return {}

    def _first_str(*keys: str) -> str:
        for k in keys:
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # metadata 里再找
        md = d.get("metadata")
        if isinstance(md, dict):
            for k in keys:
                v = md.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        return ""

    def _first_any(*keys: str):
        for k in keys:
            if k in d and d.get(k) is not None:
                return d.get(k)
        md = d.get("metadata")
        if isinstance(md, dict):
            for k in keys:
                if k in md and md.get(k) is not None:
                    return md.get(k)
        return None

    source = _first_str("source", "title", "document", "doc", "file", "filename", "name", "path", "url", "出处", "来源")
    if not source:
        source = _first_str("id", "chunk_id", "ref", "uuid")

    text_val = _first_any("content", "text", "chunk", "snippet", "excerpt", "page_content", "body", "answer")
    if isinstance(text_val, (dict, list)):
        try:
            text_val = json.dumps(text_val, ensure_ascii=False)
        except Exception:
            text_val = str(text_val)
    text_val = str(text_val) if text_val is not None else ""

    score = _first_any("score", "similarity", "distance", "rerank_score")
    if isinstance(score, (int, float)):
        score_val = float(score)
    else:
        score_val = None

    hid = _first_str("id", "chunk_id", "ref", "uuid")
    return {"source": source, "text": text_val, "score": score_val, "id": hid}


def _parse_kb_plain_text_hits(s: str) -> List[Dict[str, Any]]:
    """当工具返回的是纯文本字符串时，做一个“尽量不编造”的启发式结构化。
    说明：这是对【工具输出】的解析，不是对模型自由输出的解析。
    """
    if not isinstance(s, str):
        return []
    ss = s.strip()
    if not ss:
        return []

    # 1) 先按明显分隔符切
    for sep in ("\n---\n", "\n----\n", "\n***\n"):
        if sep in ss:
            parts = [p.strip() for p in ss.split(sep) if p.strip()]
            return [
                {"source": f"命中{i + 1}", "text": p, "score": None, "id": ""}
                for i, p in enumerate(parts)
            ]

    # 2) 再按编号行切（1. / 1、 / 1)）
    lines = ss.splitlines()
    items: List[Dict[str, Any]] = []
    buf: List[str] = []
    cur_source: str = ""

    def flush():
        nonlocal buf, cur_source
        if buf:
            txt = "\n".join(buf).strip()
            if txt:
                items.append({
                    "source": cur_source or f"命中{len(items) + 1}",
                    "text": txt,
                    "score": None,
                    "id": "",
                })
        buf = []
        cur_source = ""

    for line in lines:
        # 来源行（若有）
        msrc = re.match(r"\s*(来源|source|出处)\s*[:：]\s*(.+)\s*$", line, flags=re.I)
        if msrc and not buf:
            cur_source = msrc.group(2).strip()
            continue

        mnum = re.match(r"\s*(\d+)\s*[\.|、|\)|）]\s*(.*)$", line)
        if mnum:
            flush()
            tail = (mnum.group(2) or "").strip()
            if tail:
                buf.append(tail)
            continue

        buf.append(line)

    flush()

    if items:
        return items

    # 3) 实在不行：整个字符串当成 1 条命中
    return [{"source": "知识库工具输出", "text": ss, "score": None, "id": ""}]


def _normalize_kb_hits(obj: Any) -> List[Dict[str, Any]]:
    """把工具返回的各种可能结构归一成 hits 列表。"""
    hits: List[Dict[str, Any]] = []

    if obj is None:
        return hits

    # 字符串：优先尝试 JSON；若不是 JSON（工具可能直接返回纯文本），做启发式切分
    if isinstance(obj, str):
        j = _safe_json_loads(obj)
        if j is not None:
            return _normalize_kb_hits(j)
        # 纯文本（来自工具输出）：尽量切分成多条命中；至少返回 1 条
        hits.extend(_parse_kb_plain_text_hits(obj))
        return hits

    if isinstance(obj, list):
        for it in obj:
            hits.extend(_normalize_kb_hits(it))
        return hits

    if isinstance(obj, dict):
        # 可能已经是一个 hit
        if any(k in obj for k in ("content", "text", "chunk", "snippet", "excerpt", "page_content", "body")):
            h = _hit_from_dict(obj)
            if h.get("text"):
                hits.append(h)
            return hits

        # 常见：results/hits/items/data
        for k in ("results", "hits", "items", "data", "documents", "docs", "matches", "chunks"):
            v = obj.get(k)
            if isinstance(v, list):
                for it in v:
                    hits.extend(_normalize_kb_hits(it))
                return hits

        # 某些工具把命中塞在 "result" 里
        for k in ("result", "output"):
            v = obj.get(k)
            if v is not None:
                hits.extend(_normalize_kb_hits(v))
                if hits:
                    return hits

    return hits


def _extract_kb_hits_from_tool_loop_resp(llm_resp: Any, tool_name: str = "astr_kb_search") -> List[Dict[str, Any]]:
    """优先从返回对象的 tool trace 中抽取 KB hits；若失败，尝试从 completion_text 的 <tool_raw> 中抽取。"""
    hits: List[Dict[str, Any]] = []

    # 1) 从 tool trace / messages 抽取
    payloads = _collect_tool_outputs_from_resp(llm_resp, tool_name=tool_name)
    for p in payloads:
        hits.extend(_normalize_kb_hits(p))

    if hits:
        return hits

    # 2) 兜底：从模型输出的 <tool_raw> 解析（要求其为 JSON）
    try:
        raw_text = getattr(llm_resp, "completion_text", None)
    except Exception:
        raw_text = None
    raw_block = _extract_tool_raw_block(raw_text or "") if isinstance(raw_text, str) else None
    if raw_block:
        j = _safe_json_loads(raw_block)
        if j is not None:
            hits.extend(_normalize_kb_hits(j))
    return hits


def _extract_stream_lines_only(raw: str) -> List[str]:
    """只在存在 <stream>...</stream> 时提取，避免把不合规的内容发出去"""
    if not raw:
        return []
    m = re.search(r"<stream>([\s\S]*?)</stream>", raw, flags=re.IGNORECASE)
    if not m:
        return []
    inner = m.group(1)
    lines = [ln.strip() for ln in inner.splitlines() if ln.strip()]
    cleaned: List[str] = []
    for ln in lines:
        ln = _sanitize_hint_text(ln)
        ln = _strip_trailing_periods(ln)
        if ln:
            cleaned.append(ln)
    return cleaned


def _looks_like_markdown_or_full(raw: str) -> bool:
    if not raw:
        return False
    low = raw.lower()
    if "<md>" in low:
        return True
    if "```" in raw or "$$" in raw or re.search(r"\$[^\n]{0,80}\$", raw):
        return True
    if len(raw) >= 900:
        return True
    if re.search(r"(最终答案|答案\s*[:：]|所以\s*[:：]?\s*)", raw):
        return True
    if len(re.findall(r"\n\s*\d+[\)\.、]", raw)) >= 5:
        return True
    return False


def _normalize_stream_msgs(lines: List[str]) -> List[str]:
    """对 AI 自动决定条数的流式消息做基本清理和兜底（不强制固定条数）。"""
    msgs = list(lines or [])

    # 如果只有一条但很长，尝试按句号拆分
    if len(msgs) == 1 and len(msgs[0]) > 60:
        parts = re.split(r"[。！？!?]\s*", msgs[0])
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) > 1:
            msgs = parts

    # 上限兜底：最多 8 条
    if len(msgs) > 8:
        msgs = msgs[:8]

    msgs = [_strip_trailing_periods(_sanitize_hint_text(m)) for m in msgs]
    msgs = [m for m in msgs if m]

    # 下限兜底：如果 AI 没有产出任何有效消息，给通用提示
    if not msgs:
        msgs = [
            "先把已知和未知量写清楚 再把条件翻译成等式或不等式",
            "如果你能把题目关键条件打出来 我可以把提示变得更具体",
            "如果你想要我直接写完整过程 你可以说 直接给完整解答",
        ]

    return msgs


# -------------------------------------------------------------------------
# Markdown -> 图片：LaTeX 保护（不破坏公式）
# -------------------------------------------------------------------------
_MATH_TOKEN_RE = re.compile(
    r"(?<!\\)\$\$[\s\S]*?(?<!\\)\$\$|(?<!\\)\$[^\n]*?(?<!\\)\$"
)


def _quote_markdown_block(text: str, title: str = "", marker: str = "") -> str:
    """把一段 Markdown 包成引用块，用于渲染成讲义卡片。"""
    lines: List[str] = []
    if marker:
        lines.append(f"> {marker}")
        lines.append(">")
    if title:
        lines.append(f"> **{title}**")
        lines.append(">")
    for line in (text or "").strip().splitlines():
        # 自动结构化时可能再次包裹模型原本的引用块；先去掉旧的 >，避免进入 MathJax。
        line = re.sub(r"^\s{0,3}>\s?", "", line)
        lines.append("> " + line if line.strip() else ">")
    return "\n".join(lines).strip()


def _looks_like_math_note_md(md_text: str) -> bool:
    text = (md_text or "").strip()
    if not text:
        return False
    return bool(
        re.search(
            r"(\$\$|\$[^$]+\$|\\begin\{|\\frac|\\sum|\\int|\\lim|矩阵|函数|方程|证明|解[:：]|定理|不等式|极限|导数|积分)",
            text,
        )
    )


def _auto_structure_math_note_markdown(md_text: str) -> str:
    """给纯段落数学解答补上讲义结构，避免渲染退化成普通长文本。"""
    text = (md_text or "").strip()
    if not text or not _looks_like_math_note_md(text):
        return md_text

    heading_count = len(re.findall(r"(?m)^\s{0,3}#{1,3}\s+\S+", text))
    has_card = bool(
        re.search(
            r"(?m)^\s{0,3}>\s*\*\*\s*(题目|问题|定理|引理|公式|结论|提示|推论|核心|关键|注意|答案)",
            text,
        )
    )
    # 已经有讲义标题、分节和卡片时，尊重模型输出。
    if heading_count >= 2 and has_card:
        return text

    # 去掉模型可能给出的弱标题标记，避免 “### 题目” 逃过结构化。
    text = re.sub(r"(?m)^\s{0,3}#{1,4}\s*", "", text).strip()
    # 若模型输出里残留 Markdown 引用符，自动结构化阶段先清掉，后面会重新生成规范卡片。
    text = re.sub(r"(?m)^\s{0,3}>\s?", "", text)
    text = re.sub(r"(?m)\s+>\s*$", "", text).strip()
    # 清理模型常见的重复小标题，避免题目卡片里出现“题目 / 题意整理”等套娃标题。
    text = re.sub(r"(?m)^\s*(题意整理|题意分析)\s*[:：]?\s*", "", text)
    # 若模型已经输出了“解答过程/证明过程”分节，但题目行在分节下面，先去掉这个分节，
    # 后续由自动结构化重新生成，避免题目无法进入题目卡片。
    text = re.sub(r"(?m)^\s*(?:[一二三四五六七八九十]+[、.．]\s*)?(解答过程|证明过程)\s*[:：]?\s*", "", text).strip()

    def _split_problem_and_solution(body: str) -> Tuple[str, str]:
        body = (body or "").strip()
        if not body:
            return "", ""
        m = re.search(
            r"(?m)^\s*(?:\d+[\.\、]\s+|第[一二三四五六七八九十]+步\s*[:：]?|证明\s*[:：]|解答\s*[:：]|取反例|首先|先设|先证明|下面证明)",
            body,
        )
        if m and len(body[:m.start()].strip()) >= 12:
            return body[:m.start()].strip(), body[m.start():].strip()
        return body, ""

    first_problem_line = re.match(
        r"(?s)^\s*((?:题目|问题)\s*[^\n]*|第\s*[\d一二三四五六七八九十]+\s*[题小问][^\n]*|题\s*\d+[^\n]*)\n+([\s\S]+)$",
        text,
    )
    if first_problem_line:
        problem_head = first_problem_line.group(1).strip()
        rest = first_problem_line.group(2).strip()
        problem_body, solution_body = _split_problem_and_solution(rest)
        if not solution_body:
            # 对证明题，首行通常就是题面标题，正文从下一段开始。
            solution_body = rest
            problem_body = problem_head
        else:
            problem_body = (problem_head + "\n\n" + problem_body).strip()
        problem_body = re.sub(r"(?m)^\s*(题目|题意整理|题意分析)\s*[:：]?\s*", "", problem_body).strip()
        return "\n\n".join(
            x for x in (
                _quote_markdown_block(problem_body, marker="[problem-card]"),
                "## 一、解答过程",
                solution_body,
            )
            if str(x).strip()
        )

    section_pat = re.compile(r"(?m)^\s*(题目(?:[（(].*?[）)])?|问题|证明过程|证明|证|解答|解|结论|答案)\s*[:：]?\s*$")
    matches = list(section_pat.finditer(text))
    if matches:
        sections: List[Tuple[str, str]] = []
        prefix = text[:matches[0].start()].strip()
        if prefix:
            sections.append(("题目", prefix))
        for i, m in enumerate(matches):
            label = m.group(1).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                sections.append((label, body))

        problem_chunks: List[str] = []
        solution_chunks: List[Tuple[str, str]] = []
        conclusion_chunks: List[str] = []
        for label, body in sections:
            if re.match(r"^(题目|问题)", label):
                problem_body, rest_body = _split_problem_and_solution(body)
                if problem_body:
                    problem_body = re.sub(r"(?m)^\s*(题目|题意整理|题意分析)\s*[:：]?\s*", "", problem_body).strip()
                    problem_chunks.append(problem_body)
                if rest_body:
                    solution_chunks.append(("解答过程", rest_body))
            elif re.match(r"^(结论|答案)", label):
                conclusion_chunks.append(body)
            else:
                solution_chunks.append((label, body))

        out: List[str] = []
        if problem_chunks:
            out.append(_quote_markdown_block("\n\n".join(problem_chunks), marker="[problem-card]"))
        if solution_chunks:
            first_title = "证明过程" if any(re.match(r"^(证明|证)", x[0]) for x in solution_chunks) else "解答过程"
            out.append(f"## 一、{first_title}")
            out.extend(body for _label, body in solution_chunks)
        elif not problem_chunks:
            out.append("## 一、解答过程")
            out.append(text)
        if conclusion_chunks:
            out.append("## 二、结论")
            out.append(_quote_markdown_block("\n\n".join(conclusion_chunks), "结论："))
        return "\n\n".join(x for x in out if str(x).strip())

    return "## 一、解答过程\n\n" + text


def _html_escape_for_math(s: str) -> str:
    # 把会破坏 HTML 的字符转义，DOM 里会还原给 MathJax
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _protect_math_for_markdown(md_text: str) -> (str, List[str]):
    """
    用 token 替换数学片段，避免 Markdown 解析器把数学内容里的 _ * 等当成语法。
    渲染完成后再把 token 替换回去（HTML 安全转义 <>&）。
    """
    if not md_text:
        return md_text, []
    pieces: List[str] = []

    def repl(m: re.Match) -> str:
        raw_math = m.group(0)
        line_start = md_text.rfind("\n", 0, m.start()) + 1
        line_prefix = md_text[line_start:m.start()]
        if raw_math.startswith("$$") and "\n" in raw_math and re.fullmatch(r"\s{0,3}>\s*", line_prefix or ""):
            raw_math = "\n".join(re.sub(r"^\s{0,3}>\s?", "", line) for line in raw_math.splitlines())
        pieces.append(raw_math)
        return f"@@MATH_{len(pieces) - 1}@@"

    protected = _MATH_TOKEN_RE.sub(repl, md_text)
    return protected, pieces


def _restore_math_tokens(html: str, pieces: List[str]) -> str:
    if not pieces:
        return html
    out = html
    for i, raw_math in enumerate(pieces):
        token = f"@@MATH_{i}@@"
        out = out.replace(token, _html_escape_for_math(raw_math))
    return out


# -------------------------------------------------------------------------
# /pdf：LaTeX 组装 + TeXLive.net 在线编译
# -------------------------------------------------------------------------
def _sanitize_filename(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = name.strip().strip(".")
    if not name:
        name = "output.pdf"
    return name


def _sanitize_model_label_for_filename(provider_id: str) -> str:
    label = _guess_model_name_from_provider_id(provider_id)
    label = _sanitize_filename(label)
    label = label.strip("._-")
    if not label or label.lower() == "output.pdf":
        label = "model"
    return label[:80].strip("._-") or "model"


def _guess_model_name_from_provider_id(provider_id: str) -> str:
    """从 provider_id 兜底猜测模型名，尽量去掉提供商/渠道前缀。"""
    s = str(provider_id or "").strip()
    if not s:
        return "model"

    s = re.sub(r"[?#].*$", "", s).strip()
    # 优先处理 provider/model、provider:model、provider|model 这类高置信分隔。
    parts = [p.strip() for p in re.split(r"[/|:]+", s) if p and p.strip()]
    if len(parts) >= 2:
        s = parts[-1]

    provider_prefixes = (
        "azure_openai", "azure-openai", "openrouter", "siliconflow",
        "volcengine", "volcano", "dashscope", "aliyun", "alibaba",
        "oneapi", "newapi", "openai", "anthropic", "google",
        "vertex", "gemini-api", "ollama", "groq", "zhipu",
        "moonshot", "minimax", "baichuan", "together", "fireworks",
    )
    changed = True
    while changed:
        changed = False
        low = s.lower()
        for prefix in sorted(provider_prefixes, key=len, reverse=True):
            for sep in ("__", "_", "-", "."):
                token = prefix + sep
                if low.startswith(token) and len(s) > len(token):
                    s = s[len(token):].strip()
                    changed = True
                    break
            if changed:
                break

    return s or str(provider_id or "").strip() or "model"


def _get_sender_display_name(event: AstrMessageEvent) -> str:
    """尽量取昵称/显示名；取不到就退回到 sender_id。"""
    for attr in ("get_sender_name", "get_sender_nickname", "sender_name", "nickname", "user_name"):
        try:
            v = getattr(event, attr, None)
            if callable(v):
                v = v()
            if isinstance(v, str) and v.strip():
                return v.strip()
        except Exception:
            pass
    try:
        uid = event.get_sender_id()
        if uid:
            return str(uid)
    except Exception:
        pass
    return "用户"


def _escape_latex_text(s: str) -> str:
    """对纯文本做最小转义，避免编译炸掉；不试图解析用户的数学排版。"""
    if s is None:
        return ""
    s = str(s)
    # 这些符号在 LaTeX 文本模式下也可能触发报错（尤其是 < / > / ^ / ~）
    # - < / >：默认是数学符号，直接出现在文本里可能导致 "Missing $ inserted"
    # - ^ / ~：在文本里需要转义
    s = s.replace("<", r"\textless{}")
    s = s.replace(">", r"\textgreater{}")
    s = s.replace("^", r"\textasciicircum{}")
    s = s.replace("~", r"\textasciitilde{}")
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    s = s.replace("#", r"\#")
    s = s.replace("_", r"\_")
    s = s.replace("{", r"\{")
    s = s.replace("}", r"\}")
    return s


# 更严格的纯文本转义：用于“知识库原文摘录”等场景，避免出现反斜杠/美元符导致的编译错误
def _escape_latex_text_strict(s: str) -> str:
    r"""对纯文本做更严格转义，保证尽量不炸编译。
    注意：会把 \ 与 $ 也转义，因此不要用于已经是 LaTeX 的片段。
    """
    if s is None:
        return ""
    s = str(s)
    # 反斜杠必须最先处理
    s = s.replace("\\", r"\textbackslash{}")
    s = s.replace("$", r"\$")
    # 同样处理 < / > / ^ / ~，避免直接落在文本里炸编译
    s = s.replace("<", r"\textless{}")
    s = s.replace(">", r"\textgreater{}")
    s = s.replace("^", r"\textasciicircum{}")
    s = s.replace("~", r"\textasciitilde{}")
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    s = s.replace("#", r"\#")
    s = s.replace("_", r"\_")
    s = s.replace("{", r"\{")
    s = s.replace("}", r"\}")
    return s


_MATH_DOLLAR_SEG = re.compile(r"(\$\$[\s\S]*?\$\$|\$[^$]+\$)")


def _escape_text_preserve_dollar_math(s: str) -> str:
    """转义纯文本，但尽量保留 $...$ / $$...$$ 数学段落不被破坏。"""
    if s is None:
        return ""
    s = str(s)
    out_parts: List[str] = []
    last = 0
    for m in _MATH_DOLLAR_SEG.finditer(s):
        if m.start() > last:
            out_parts.append(_escape_latex_text_strict(s[last:m.start()]))
        out_parts.append(m.group(1))
        last = m.end()
    if last < len(s):
        out_parts.append(_escape_latex_text_strict(s[last:]))
    return "".join(out_parts)



def _is_char_escaped_by_backslash(s: str, pos: int) -> bool:
    """判断 s[pos] 这个字符是否被前面的反斜杠转义（\\）。"""
    if pos <= 0:
        return False
    cnt = 0
    j = pos - 1
    while j >= 0 and s[j] == "\\":
        cnt += 1
        j -= 1
    return (cnt % 2) == 1


def _ensure_balanced_dollar_math(s: str) -> str:
    """兜底用：若 $ 分隔符不成对，则把所有未转义的 $ 变为 \\$，避免编译报错。

    仅在 PDF 编译失败后的“强转义兜底”路径使用，尽量不影响正常数学渲染。
    """
    if not s:
        return ""
    s = str(s)
    cnt = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "$" and not _is_char_escaped_by_backslash(s, i):
            if i + 1 < len(s) and s[i + 1] == "$" and not _is_char_escaped_by_backslash(s, i + 1):
                cnt += 2
                i += 2
            else:
                cnt += 1
                i += 1
        else:
            i += 1

    if cnt % 2 == 0:
        return s

    # 不成对：全部转义成 \$
    out: List[str] = []
    i = 0
    while i < len(s):
        if s[i] == "$" and not _is_char_escaped_by_backslash(s, i):
            out.append(r"\$")
            i += 1
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


_SANITIZE_MATH_ENVS = {
    # 常见数学环境
    "equation", "equation*",
    "align", "align*",
    "gather", "gather*",
    "multline", "multline*",
    "eqnarray", "eqnarray*",
    "cases",
    "array",
    "aligned", "alignedat", "split",
    "matrix", "pmatrix", "bmatrix", "vmatrix", "Vmatrix", "smallmatrix",
    "tikzpicture",
    "axis",
    "scope",
}

_SANITIZE_TABULAR_ENVS = {
    "tabular", "tabular*",
}


def _sanitize_latex_fragment_for_xelatex(fragment: str) -> str:
    """对模型生成的 LaTeX 片段做轻量清理，降低 xelatex 编译失败概率。

    重点处理：文本模式下的特殊符号（_, ^, \\, %, #, &, <, > 等）。
    - 数学模式（$...$, $$...$$, \\(...\\), \\[...\\], 以及常见数学环境）不做修改。
    - 只在“文本模式”中替换/转义这些符号，以避免 “Missing $ inserted”等常见报错。
    """
    t = _strip_all_code_fences(fragment or "")
    t = _strip_known_xml_like_tags(t)

    # ================= [新增修复逻辑 Start] =================
    # 修复 Markdown 加粗语法转 LaTeX
    # 解释：
    # 1. \*\* 匹配双星号
    # 2. \s* 匹配可能存在的空格（解决你说的“空一格”问题）
    # 3. (.*?) 捕获内容
    # 4. 替换为 \textbf{内容}
    t = re.sub(r"\*\*\s*(.*?)\s*\*\*", r"\\textbf{\1}", t)
    # ================= [新增修复逻辑 End] ====================

    if not t:
        return ""

    s = t
    out: List[str] = []
    i = 0

    dollar_mode = 0  # 0=text, 1=$...$, 2=$$...$$
    paren_depth = 0  # \( ... \)
    bracket_depth = 0  # \[ ... \]
    math_env_stack: List[str] = []
    tab_env_stack: List[str] = []

    def in_math_mode() -> bool:
        return bool(dollar_mode or paren_depth or bracket_depth or math_env_stack)

    while i < len(s):
        ch = s[i]

        # ---------- $ / $$ ----------
        if ch == "$" and not _is_char_escaped_by_backslash(s, i):
            if i + 1 < len(s) and s[i + 1] == "$" and not _is_char_escaped_by_backslash(s, i + 1):
                # $$ toggle
                if dollar_mode == 2:
                    dollar_mode = 0
                elif dollar_mode == 0:
                    dollar_mode = 2
                else:
                    # 异常嵌套：直接回到文本
                    dollar_mode = 0
                out.append("$$")
                i += 2
                continue
            else:
                # $ toggle
                if dollar_mode == 1:
                    dollar_mode = 0
                elif dollar_mode == 0:
                    dollar_mode = 1
                out.append("$")
                i += 1
                continue

        # ---------- 反斜杠开头 ----------
        if ch == "\\":
            if i + 1 < len(s):
                nxt = s[i + 1]

                # \( \) \[ \] 作为数学模式开关
                if nxt == "[":
                    bracket_depth += 1
                    out.append(r"\[")
                    i += 2
                    continue
                if nxt == "]":
                    if bracket_depth > 0:
                        bracket_depth -= 1
                    out.append(r"\]")
                    i += 2
                    continue
                if nxt == "(":
                    paren_depth += 1
                    out.append(r"\(")
                    i += 2
                    continue
                if nxt == ")":
                    if paren_depth > 0:
                        paren_depth -= 1
                    out.append(r"\)")
                    i += 2
                    continue

                # 常见的“转义/控制符号”直接保留
                if nxt in ["\\", "%", "_", "&", "#", "{", "}", "$", "~", "^"]:
                    out.append("\\" + nxt)
                    i += 2
                    continue

                # 文本模式下：遇到不太像命令的反斜杠（如 \Users 或者 \ 在中文前），转成可打印字符
                if (not in_math_mode()) and (not ("a" <= nxt <= "z")):
                    out.append(r"\textbackslash{}")
                    i += 1
                    continue

                # 解析命令名
                if re.match(r"[A-Za-z]", nxt):
                    j = i + 1
                    while j < len(s) and re.match(r"[A-Za-z]", s[j]):
                        j += 1
                    cmd = s[i + 1:j]
                    out.append("\\" + cmd)
                    i = j

                    # 追踪 begin/end 环境（用于判断数学模式、tabular 中 & 的处理）
                    if cmd in ("begin", "end"):
                        k = i
                        # 保留空白
                        while k < len(s) and s[k].isspace():
                            out.append(s[k])
                            k += 1
                        if k < len(s) and s[k] == "{":
                            endb = s.find("}", k + 1)
                            if endb != -1:
                                env_name = s[k + 1:endb]
                                out.append("{" + env_name + "}")
                                i = endb + 1
                                env_norm = (env_name or "").strip()

                                if env_norm in _SANITIZE_MATH_ENVS:
                                    if cmd == "begin":
                                        math_env_stack.append(env_norm)
                                    else:
                                        for idx_back in range(len(math_env_stack) - 1, -1, -1):
                                            if math_env_stack[idx_back] == env_norm:
                                                math_env_stack = math_env_stack[:idx_back]
                                                break

                                if env_norm in _SANITIZE_TABULAR_ENVS:
                                    if cmd == "begin":
                                        tab_env_stack.append(env_norm)
                                    else:
                                        for idx_back in range(len(tab_env_stack) - 1, -1, -1):
                                            if tab_env_stack[idx_back] == env_norm:
                                                tab_env_stack = tab_env_stack[:idx_back]
                                                break
                                continue
                        i = k
                    continue

                # 其它情况：保留原样
                out.append("\\")
                i += 1
                continue

            # trailing backslash
            if not in_math_mode():
                out.append(r"\textbackslash{}")
            else:
                out.append("\\")
            i += 1
            continue

        # ---------- 普通字符 ----------
        if not in_math_mode():
            if ch == "_":
                out.append(r"\_")
                i += 1
                continue
            if ch == "^":
                out.append(r"\textasciicircum{}")
                i += 1
                continue
            if ch == "~":
                out.append(r"\textasciitilde{}")
                i += 1
                continue
            if ch == "<":
                out.append(r"\textless{}")
                i += 1
                continue
            if ch == ">":
                out.append(r"\textgreater{}")
                i += 1
                continue
            if ch == "%":
                out.append(r"\%")
                i += 1
                continue
            if ch == "#":
                out.append(r"\#")
                i += 1
                continue
            if ch == "&":
                # tabular 内 & 是必要的对齐符号
                if tab_env_stack:
                    out.append("&")
                else:
                    out.append(r"\&")
                i += 1
                continue

        out.append(ch)
        i += 1

    return "".join(out)


def _kb_text_to_tex(raw: str, max_chars: int = 900, preserve_latex: bool = True) -> str:
    r"""把知识库命中的文本转成尽量安全的 LaTeX 可渲染内容。
    - 若看起来已经是 LaTeX（含 \begin{}/\frac 等），尽量原样保留
    - 否则按“纯文本”处理：转义，但保留 $...$ / $$...$$ 数学段落
    """
    t = _strip_all_code_fences(raw or "")
    t = (t or "").strip()
    if not t:
        return ""
    if isinstance(max_chars, int) and max_chars > 0 and len(t) > max_chars:
        t = t[:max_chars].rstrip() + "…"

    # 明显 LaTeX 片段：尽量原样保留（避免把反斜杠全转义导致公式失真）
    if preserve_latex and re.search(r"(\\begin\{|\\end\{|\\\[|\\\(|\\frac|\\sqrt|\\sum|\\int|\\lim)", t):
        return t

    return _escape_text_preserve_dollar_math(t)


def _strip_all_code_fences(s: str) -> str:
    """
    更激进地移除所有 ```xxx 与 ```，仅保留内部内容。
    用于模型把 <problem>...</problem> 包在 fenced code 中时。
    """
    if not s:
        return ""
    lines = s.splitlines()
    out_lines: List[str] = []
    in_fence = False
    for ln in lines:
        if ln.strip().startswith("```"):
            in_fence = not in_fence
            continue
        out_lines.append(ln)
    return "\n".join(out_lines).strip()


def _unwrap_svg_code_fences(md_text: str) -> str:
    """把模型误包进代码块的 SVG/HTML SVG 还原成可渲染的内联 SVG。"""
    if not md_text:
        return ""

    fence_re = re.compile(r"```([A-Za-z0-9_-]*)[ \t]*\n([\s\S]*?)\n```", flags=re.MULTILINE)

    def repl(m: re.Match) -> str:
        lang = (m.group(1) or "").strip().lower()
        body = (m.group(2) or "").strip()
        if "<svg" in body.lower() and (not lang or lang in {"svg", "html", "xml"}):
            return "\n\n" + body + "\n\n"
        return m.group(0)

    return fence_re.sub(repl, md_text)


def _sanitize_inline_html_for_render(md_text: str) -> str:
    """渲染 Markdown 时保留 SVG 等静态 HTML，但移除脚本与事件属性。"""
    if not md_text:
        return ""
    text = str(md_text)
    # 不允许模型输出可执行脚本或外部嵌入容器。mistune escape=False 会保留 HTML，这里先收窄边界。
    text = re.sub(
        r"(?is)<\s*(script|iframe|object|embed|link|meta)\b[^>]*>[\s\S]*?<\s*/\s*\1\s*>",
        "",
        text,
    )
    text = re.sub(r"(?is)<\s*(script|iframe|object|embed|link|meta)\b[^>]*/?\s*>", "", text)
    text = re.sub(r"(?is)\s+on[a-zA-Z]+\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)", "", text)
    text = re.sub(r"(?is)\s+(href|src|xlink:href)\s*=\s*(['\"])\s*javascript:[\s\S]*?\2", "", text)
    return text


def _normalize_inline_svg_markup(md_text: str) -> str:
    """给内联 SVG 补基础属性，提升本地渲染一致性。"""
    if not md_text:
        return ""

    def repl(m: re.Match) -> str:
        attrs = m.group(1) or ""
        if not re.search(r"\bxmlns\s*=", attrs, flags=re.I):
            attrs += ' xmlns="http://www.w3.org/2000/svg"'
        if not re.search(r"\brole\s*=", attrs, flags=re.I):
            attrs += ' role="img"'
        return "<svg" + attrs + ">"

    return re.sub(r"<svg\b([^>]*)>", repl, str(md_text), flags=re.I)


def _prepare_markdown_for_render(md_text: str) -> str:
    """Markdown 渲染前的轻量规范化：结构化数学解答、还原 SVG、收窄 HTML。"""
    text = (md_text or "").strip()
    text = _strip_known_xml_like_tags(text)
    text = _unwrap_svg_code_fences(text)
    text = _sanitize_inline_html_for_render(text)
    text = _normalize_inline_svg_markup(text)
    return _auto_structure_math_note_markdown(text)


_KNOWN_XML_LIKE_TAGS = (
    "md",
    "stream",
    "kbtext",
    "problem",
    "theorems",
    "solution",
    "kb",
)


def _strip_known_xml_like_tags(s: str) -> str:
    """移除插件内部约定的一些标签（仅移除标签本身，保留内部内容）。

    说明：
    - 我们在多处使用 <md>/<stream>/<problem>/<theorems>/<solution>/<kb>/<kbtext> 作为“结构化输出”标记。
    - 这些标签一旦漏进 LaTeX 文本模式，`<`/`>` 很容易触发 xelatex 报错（Missing $ inserted）。
    - 这里只移除已知标签，不会影响正常的数学不等号（< >）。
    """
    t = s or ""
    for tag in _KNOWN_XML_LIKE_TAGS:
        # 兼容 <tag>、</tag> 以及可能带空格的形式
        t = re.sub(rf"</?{tag}\s*>", "", t, flags=re.I)
    return t


def _looks_like_latex_fragment(s: str) -> bool:
    """粗略判断文本是否更像 LaTeX 片段（而不是 Markdown/纯文本）。"""
    if not s:
        return False
    t = s.strip()
    return bool(
        re.search(
            r"(\\begin\{|\\end\{|\\frac|\\sqrt|\\sum|\\int|\\lim|\\textbf\{|\\item\b)",
            t,
        )
    )


def _llm_raw_to_safe_tex(raw: str) -> str:
    """把模型原始输出尽量变成“可编译”的 LaTeX 正文片段。

    - 如果看起来已经是 LaTeX，则尽量保留（避免把反斜杠等全转义导致公式失真）。
    - 否则按纯文本处理：严格转义，但尽量保留 $...$ 数学段落。
    """
    t = _strip_all_code_fences(raw or "")
    t = _strip_known_xml_like_tags(t)
    t = (t or "").strip()
    if not t:
        return ""
    if _looks_like_latex_fragment(t):
        return t
    return _escape_text_preserve_dollar_math(t)


def _is_content_filter_error(msg: str) -> bool:
    """识别常见的内容安全/审核拦截错误，避免在 KB 检索中反复重试刷屏。"""
    s = (msg or "").lower()
    keywords = [
        "内容安全", "安全过滤", "过滤被拒绝", "content safety", "content filter",
        "policy", "moderation", "refused", "blocked", "violate",
    ]
    return any(k.lower() in s for k in keywords)


def _is_auth_error(msg: str) -> bool:
    """识别常见“未配置鉴权/API Key”类错误。"""
    s = (msg or "").lower()
    return (
        ("auth_unavailable" in s)
        or ("no auth available" in s)
        or ("missing api key" in s)
        or ("api key" in s and ("not set" in s or "missing" in s or "invalid" in s))
        or ("authentication" in s and ("failed" in s or "error" in s))
        or ("unauthorized" in s)
        or ("401" in s and "error" in s)
    )


def _is_toolcall_parse_error(msg: str) -> bool:
    """识别模型触发 tool/function calling 导致的解析失败。"""
    if not msg:
        return False
    # 中文日志（AstrBot openai_source 常见）
    if "completion 无法解析" in msg:
        return True
    s = msg.lower()
    return (
        ("malformed_function_call" in s)
        or ("malformed tool" in s)
        or ("invalid function call" in s)
        or ("tool call" in s and "malformed" in s)
        or ("function_call" in s and "malformed" in s)
    )


def _is_context_length_error(msg: str) -> bool:
    """识别上下文过长/超 token 限制类错误。"""
    s = (msg or "").lower()
    keys = [
        "context_length",
        "maximum context length",
        "max context",
        "too many tokens",
        "context window",
        "token limit",
        "prompt_tokens",
        "input is too long",
        "request too large",
        "payload too large",
    ]
    return any(k in s for k in keys)


def _parse_latex_sections(raw: str):
    """解析模型输出的 <problem>/<theorems>/<solution> 三段（增强鲁棒性）。"""
    text = (raw or '').strip()
    text = _strip_all_code_fences(text)

    m_prob = re.search(r"<problem>([\s\S]*?)</problem>", text, flags=re.I)
    problem = m_prob.group(1).strip() if m_prob else ''

    m_thm = re.search(r"<theorems>([\s\S]*?)</theorems>", text, flags=re.I)
    theorems = m_thm.group(1).strip() if m_thm else ''

    m_sol = re.search(r"<solution>([\s\S]*?)</solution>", text, flags=re.I)
    solution = m_sol.group(1).strip() if m_sol else ''

    # 容错：<solution> 没闭合或模型把正文直接接在最后一个标签后
    if not solution:
        m_sol_open = re.search(r"<solution>([\s\S]*)", text, flags=re.I)
        if m_sol_open:
            solution = m_sol_open.group(1).strip()
            solution = solution.split('</solution')[0].strip()

    if not solution:
        last_tag_end = -1
        tags_to_check: List[str] = []
        if problem:
            tags_to_check.append('</problem>')
        if theorems:
            tags_to_check.append('</theorems>')
        if tags_to_check:
            low = text.lower()
            for tag in tags_to_check:
                idx = low.rfind(tag)
                if idx != -1:
                    end_pos = idx + len(tag)
                    if end_pos > last_tag_end:
                        last_tag_end = end_pos
            if last_tag_end != -1:
                potential = text[last_tag_end:].strip()
                if len(potential) > 5:
                    solution = potential

    return problem, theorems, solution


def _parse_latex_sections_multi(raw: str) -> List[Tuple[str, str, str]]:
    """解析模型输出的多组 <problem>/<theorems>/<solution>。

    约定：多题时按顺序重复输出三段标签（每题一组）。
    """
    text = (raw or '').strip()
    text = _strip_all_code_fences(text)

    # 按每个 <problem> 起点切块（保留 <problem>）
    chunks = re.split(r"(?i)(?=<problem>)", text)
    items: List[Tuple[str, str, str]] = []
    for ch in chunks:
        if "<problem>" not in (ch or "").lower():
            continue
        p, t, s = _parse_latex_sections(ch)
        if (p or t or s):
            items.append((p.strip(), t.strip(), s.strip()))
    return items


def _normalize_multi_pdf_items(
        items: List[Tuple[str, str, str]],
        problem_text: str,
        has_image: bool,
) -> List[Tuple[str, str, str]]:
    """为多题输出做兜底与清理，保证每题至少有题面/解答；定理/公式允许为空，并做 xelatex 友好化。"""
    norm: List[Tuple[str, str, str]] = []
    n = len(items)
    for i, (p, t, s) in enumerate(items, start=1):
        p = (p or '').strip()
        t = (t or '').strip()
        s = (s or '').strip()

        if not p:
            if problem_text and n == 1:
                p = _escape_text_preserve_dollar_math(problem_text)
            elif has_image:
                p = r"(用户未提供文字描述，见下方解答)"
            else:
                p = rf"(第{i}题题面缺失)"

        if not s:
            s = r"\textbf{解：}" + "\n" + r"(解答生成失败，请重试)"

        p = _sanitize_latex_fragment_for_xelatex(p)
        t = _sanitize_latex_fragment_for_xelatex(t)
        s = _sanitize_latex_fragment_for_xelatex(s)

        norm.append((p, t, s))
    return norm


def _pdf_raw_to_latex_parts(raw: str, problem_text: str, has_image: bool) -> Dict[str, Any]:
    """把模型原始输出解析、兜底并组装成可尝试编译的 PDF LaTeX。"""
    raw = _strip_all_code_fences(raw or "")

    norm_items: List[Tuple[str, str, str]] = []
    multi_items = _parse_latex_sections_multi(raw)
    is_multi = len(multi_items) > 1
    if is_multi:
        norm_items = _normalize_multi_pdf_items(multi_items, problem_text=problem_text, has_image=has_image)
        problem_tex = "\n\n".join([x[0] for x in norm_items])
        theorems_tex = "\n\n".join([x[1] for x in norm_items])
        solution_tex = "\n\n".join([x[2] for x in norm_items])
    else:
        problem_tex, theorems_tex, solution_tex = _parse_latex_sections(raw)

    low_raw = raw.lower() if raw else ""
    is_truncated = bool(raw and ("<solution>" in low_raw) and ("</solution>" not in low_raw))
    if is_truncated:
        warning_box = (
            r"\vspace{1em}" + "\n"
            r"\begin{tcolorbox}[colback=red!5!white,colframe=red!75!black,title={生成中断警告}]" + "\n"
            r"由于模型输出被提前中断，解答过程可能不完整。你可以把题目拆成更小的问题，或让机器人继续补全上一份解答。" + "\n"
            r"\end{tcolorbox}"
        )
        if is_multi and norm_items:
            _p, _t, _s = norm_items[-1]
            norm_items[-1] = (_p, _t, (_s or "") + warning_box)
            solution_tex = "\n\n".join([x[2] for x in norm_items])
        else:
            solution_tex = (solution_tex or "") + warning_box

    # 模型可能忘记输出标签，此时把用户输入当题目，把完整回复当解答。
    if (not problem_tex) and (not solution_tex) and raw:
        if problem_text:
            problem_tex = _escape_text_preserve_dollar_math(problem_text)
        elif has_image:
            problem_tex = r"(用户未提供文字描述，见下方解答)"
        else:
            problem_tex = r"(对话模式)"
        solution_tex = _llm_raw_to_safe_tex(raw)

    if not problem_tex:
        if has_image:
            if len(raw) > 20 and not solution_tex:
                problem_tex = r"(模型未返回题目识别标签 problem，请参考下方解答)"
                if not solution_tex:
                    solution_tex = _llm_raw_to_safe_tex(raw)
            else:
                problem_tex = r"(模型未返回题目识别结果，请检查)"
        else:
            problem_tex = _escape_text_preserve_dollar_math(problem_text)
            if not solution_tex and len(raw) > 10:
                solution_tex = _llm_raw_to_safe_tex(raw)

    if not theorems_tex:
        theorems_tex = (
            r"\begin{theoremBox}{常用结论}" + "\n"
            r"本题仅用到基础运算与常用恒等变形" + "\n"
            r"\end{theoremBox}"
        )
    if not solution_tex:
        solution_tex = r"\textbf{解：}" + "\n" + r"(解答生成失败，请重试)"

    problem_tex = _sanitize_latex_fragment_for_xelatex(problem_tex)
    theorems_tex = _sanitize_latex_fragment_for_xelatex(theorems_tex)
    solution_tex = _sanitize_latex_fragment_for_xelatex(solution_tex)

    tex_src = _build_pdf_latex_document_multi(norm_items) if is_multi else _build_pdf_latex_document(
        problem_tex,
        theorems_tex,
        solution_tex,
    )
    return {
        "raw": raw,
        "problem_tex": problem_tex,
        "theorems_tex": theorems_tex,
        "solution_tex": solution_tex,
        "norm_items": norm_items,
        "is_multi": is_multi,
        "tex_src": tex_src,
    }


def _build_pdf_latex_document(problem_tex: str, theorems_tex: str, solution_tex: str) -> str:
    """组装可 xelatex 编译的完整 LaTeX 文档（ctex + tcolorbox）。

    布局约定：
      1) 先统一给出“解题所需结论”
      2) 再进入“解答”，按“题目 -> 解答”顺序排版（题面只出现一次）
    """
    parts = [
        r"\documentclass[UTF8]{ctexart}",
        r"\usepackage[a4paper,margin=2.2cm]{geometry}",
        # 提供 \textless/\textgreater/\textasciitilde/\textasciicircum 等文本符号，
        # 用于把用户/模型输出中的 < > ~ ^ 安全落在 LaTeX 文本模式里。
        r"\usepackage{textcomp}",
        r"\usepackage{amsmath,amssymb,amsthm,mathtools,bm,mathrsfs,cancel}",
        r"\usepackage[most]{tcolorbox}",
        r"\usepackage{enumitem}",
        r"\usepackage{tikz}",
        r"\usepackage{pgfplots}",
        # 1. 增加 axis on top，强制坐标轴永远压在图像上面
        r"\pgfplotsset{compat=1.18, width=0.85\linewidth, axis on top, every axis/.append style={align=center}}",
        r"\usepgfplotslibrary{fillbetween}",
        # 2. 增加 backgrounds 库，并配置图层顺序
        r"\usetikzlibrary{arrows.meta, calc, positioning, shapes, intersections, decorations.pathreplacing, decorations.markings, patterns, scopes, backgrounds}",
        # === [新增修复] 显式定义背景层，确保填充不会遮挡文字和轴 ===
        r"\pgfdeclarelayer{bg}",
        r"\pgfsetlayers{bg,main}",
        # =======================================================
        r"\tikzset{every node/.append style={inner sep=3pt, outer sep=3pt}, >={Stealth[length=3mm]}}",
        r"\setlist{nosep}",
        r"\setlength{\parindent}{0pt}",
        r"\linespread{1.3}",
        r"\tcbset{enhanced, boxrule=0.8pt, arc=2.5mm}",

        r"\newtcolorbox{problemBox}[1][]{colback=blue!3!white, colframe=blue!65!black, title=题目, fonttitle=\bfseries, #1}",
        r"\newtcolorbox{theoremBox}[2][]{colback=teal!3!white, colframe=teal!65!black, title=定理：#2, fonttitle=\bfseries, #1}",
        r"\newtcolorbox{lemmaBox}[2][]{colback=orange!5!white, colframe=orange!80!black, title=引理：#2, fonttitle=\bfseries, #1}",
        r"\newtcolorbox{formulaBox}[2][]{colback=violet!3!white, colframe=violet!65!black, title=公式：#2, fonttitle=\bfseries, #1}",

        r"\begin{document}",
        r"\section*{解题所需结论}",
        theorems_tex,
        "",
        r"\section*{解答}",
        r"\begin{problemBox}",
        problem_tex,
        r"\end{problemBox}",
        "",
        solution_tex,
        r"\end{document}",
        "",
    ]
    return "\n".join(parts)


def _build_pdf_latex_document_multi(items: List[Tuple[str, str, str]]) -> str:
    """多题版：一个 PDF 内包含多题。

    布局约定：
      1) 把所有题目用到的“解题所需结论”统一放在开头（可自动去重）
      2) 解答部分按“题目 -> 对应解答”逐题展开
      3) 题面只出现一次；题与题之间不强制分页
    """
    parts = [
        r"\documentclass[UTF8]{ctexart}",
        r"\usepackage[a4paper,margin=2.2cm]{geometry}",
        r"\usepackage{textcomp}",
        r"\usepackage{amsmath,amssymb,amsthm,mathtools,bm,mathrsfs,cancel}",
        r"\usepackage[most]{tcolorbox}",
        r"\usepackage{enumitem}",
        r"\usepackage{tikz}",
        r"\usepackage{pgfplots}",
        r"\pgfplotsset{compat=1.18, width=0.85\linewidth, axis on top, every axis/.append style={align=center}}",
        r"\usepgfplotslibrary{fillbetween}",
        r"\usetikzlibrary{arrows.meta, calc, positioning, shapes, intersections, decorations.pathreplacing, decorations.markings, patterns, scopes, backgrounds}",
        r"\pgfdeclarelayer{bg}",
        r"\pgfsetlayers{bg,main}",
        r"\tikzset{every node/.append style={inner sep=3pt, outer sep=3pt}, >={Stealth[length=3mm]}}",
        r"\setlist{nosep}",
        r"\setlength{\parindent}{0pt}",
        r"\linespread{1.3}",
        r"\tcbset{enhanced, boxrule=0.8pt, arc=2.5mm}",
        r"\newtcolorbox{problemBox}[1][]{colback=blue!3!white, colframe=blue!65!black, title=题目, fonttitle=\bfseries, #1}",
        r"\newtcolorbox{theoremBox}[2][]{colback=teal!3!white, colframe=teal!65!black, title=定理：#2, fonttitle=\bfseries, #1}",
        r"\newtcolorbox{lemmaBox}[2][]{colback=orange!5!white, colframe=orange!80!black, title=引理：#2, fonttitle=\bfseries, #1}",
        r"\newtcolorbox{formulaBox}[2][]{colback=violet!3!white, colframe=violet!65!black, title=公式：#2, fonttitle=\bfseries, #1}",
        r"\begin{document}",
    ]

    # 统一汇总（并做轻量去重）：把每题的 theorems 拼到开头
    theorems_list: List[str] = []
    seen: set = set()
    for _, t, _ in items:
        t = (t or "").strip()
        if not t:
            continue
        key = re.sub(r"\s+", " ", t)
        if key in seen:
            continue
        seen.add(key)
        theorems_list.append(t)

    theorems_all = "\n\n".join(theorems_list).strip()
    if not theorems_all:
        theorems_all = (
            r"\begin{theoremBox}{常用结论}" + "\n"
            r"本题仅用到基础运算与常用恒等变形" + "\n"
            r"\end{theoremBox}"
        )

    parts += [
        r"\section*{解题所需结论}",
        theorems_all,
        "",
        r"\section*{解答}",
    ]

    for idx, (problem_tex, _theorems_tex, solution_tex) in enumerate(items, start=1):
        parts += [
            rf"\begin{{problemBox}}[title=题 {idx}]",
            problem_tex,
            r"\end{problemBox}",
            "",
            solution_tex,
            r"\vspace{1em}",
            "",
        ]

    parts += [
        r"\end{document}",
        "",
    ]
    return "\n".join(parts)

# Export private helper names as well; main/mixins intentionally share legacy helpers.
__all__ = [name for name in globals() if not name.startswith("__")]
