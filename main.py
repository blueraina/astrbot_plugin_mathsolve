import os
import re
import uuid
import asyncio
import inspect
import json
import time
import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
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
import mistune
from playwright.async_api import async_playwright

# =============================================================================
# 更新说明（本地 MikTex 专用版）
# - 移除了所有 texlive.net 在线编译逻辑
# - 仅使用本地 xelatex 进行 PDF 编译
# - 保留了原有 Playwright 渲染、知识库检索、提示流等核心功能
# =============================================================================

# === 全局字典：用于在内存中存储用户的设备偏好 ===
# 格式: { user_id_str: "pc" or "mobile" }
USER_PREFERENCES: Dict[str, str] = {}

# === 数学会话状态：用于“上一题”追问及 PDF 上下文 ===
# 格式: { session_key: { "last_problem": str, "last_image_urls": List[str], "last_pdf_context": str ... } }
MATH_SESSION_STATE: Dict[str, Dict[str, Any]] = {}

DEFAULT_CFG: Dict[str, Any] = {
    "enable_math_coach": True,
    # 当用户只发图片（无文字）时是否默认视作“数学题”并启用提示流
    "treat_image_as_math": True,
    # 提示流条数
    "hint_message_count": 3,
    # 提示流每条间隔（ms），增强“流式”感觉
    "hint_send_delay_ms": 0,
    # 数学答疑人格
    "math_persona": "你是一个耐心的数学助教 说话像真人答疑 口语化 简短 温和但不啰嗦",

    # 可选：用一个“路由模型”辅助判断（从已有 provider 下拉选择）
    "use_router_model": False,
    "router_provider_id": "",

    # 可选：当模型不遵守 <stream> 时，用第二个模型把输出改写成提示流
    "use_hint_rewriter_model": False,
    "hint_rewriter_provider_id": "",

    # /pdf 指令：生成 PDF 解答（本地编译）
    "enable_pdf_output": True,
    # 可选：指定 /pdf 使用的模型（留空=跟随默认会话模型）
    "pdf_provider_id": "",



# /spdf 指令：DeepThink 多角色迭代 PDF 解答（仅在用户使用 /spdf 时启用，不影响原 /pdf）
"enable_spdf_output": True,
# /spdf 使用的“基础模型” provider_id（留空=默认会话模型）
"spdf_provider_id": "",
# 角色池（多个 provider_id，用逗号/空格/换行分隔；留空则全部使用 spdf_provider_id）
"spdf_role_pool_provider_ids": "",
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
    "render_concurrency": 2,  # 同时渲染数量上限
    # set_content 的等待策略：networkidle 更稳但可能卡；domcontentloaded 更快
    "playwright_wait_until": "networkidle",

    # TeXLive 编译缓存
    "texlive_cache_enabled": True,
    "texlive_cache_max_files": 500,

    # ========= 新增：知识库问答/检索增强 =========
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



# ------------------------- 联网/搜索意图（高优先级） -------------------------
_DEFAULT_WEB_SEARCH_KEYWORDS = (
    "联网搜索", "联网", "上网", "网上", "网络上", "在线搜索",
    "搜索", "搜一下", "查一下", "检索", "查资料",
    "论文", "文献", "参考文献",
    "arxiv", "scholar", "google", "bing", "paper", "papers",
)

def _split_keywords(s: str) -> List[str]:
    if not s:
        return []
    parts = re.split(r"[,\n，;；\s]+", str(s))
    return [p.strip() for p in parts if p and p.strip()]

def _is_web_search_intent(text: str, keywords: Optional[List[str]] = None) -> bool:
    """判断是否是明确的“联网/上网/搜索/查论文”意图（用于绕过数学提示流）。"""
    t = (text or "").strip()
    if not t:
        return False
    tl = t.lower()
    kws = list(keywords or _DEFAULT_WEB_SEARCH_KEYWORDS)
    for k in kws:
        if k and k.lower() in tl:
            return True
    # 兜底：捕获“网上搜/联网查”这类句式
    return bool(re.search(r"(?:网上|网络上|上网|联网|在线).{0,10}(?:搜|搜索|查|检索|找)", t))

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


def _normalize_to_n_msgs(lines: List[str], n: int) -> List[str]:
    n = max(2, min(int(n or 3), 8))
    msgs = list(lines or [])

    if len(msgs) == 1 and len(msgs[0]) > 60:
        parts = re.split(r"[。！？!?]\s*", msgs[0])
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) > 1:
            msgs = parts

    if len(msgs) > n:
        msgs = msgs[:n]

    fallback_pool = [
        "先把已知和未知量写清楚 再把条件翻译成等式或不等式",
        "优先从能直接列式的一步开始 例如代入 化简 或把式子移到一边",
        "如果题里有函数 先看定义域 以及是否能因式分解或配方",
        "如果你能把题目关键条件打出来 我可以把提示变得更具体",
        "如果你想要我直接写完整过程 你可以说 直接给完整解答",
    ]
    i = 0
    while len(msgs) < n:
        msgs.append(fallback_pool[i % len(fallback_pool)])
        i += 1

    msgs = [_strip_trailing_periods(_sanitize_hint_text(m)) for m in msgs]
    msgs = [m for m in msgs if m]
    while len(msgs) < n:
        msgs.append("你先写到能得到一个明确的式子 我再带你往下推")
    return msgs[:n]


# -------------------------------------------------------------------------
# Markdown -> 图片：LaTeX 保护（不破坏公式）
# -------------------------------------------------------------------------
_MATH_TOKEN_RE = re.compile(
    r"(?<!\\)\$\$[\s\S]*?(?<!\\)\$\$|(?<!\\)\$[^\n]*?(?<!\\)\$"
)


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
        pieces.append(m.group(0))
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

# =============================================================================
# LLM / Agent Tools（将本插件能力封装为可被 Agent Function Calling 调用的 Tool）
# =============================================================================

# 插件实例（单例）。Tool 运行时通过它调用插件内部方法。
_MD2IMG_PLUGIN_INSTANCE = None  # type: Optional["MarkdownConverterPlugin"]

def _md2img_get_plugin():
    global _MD2IMG_PLUGIN_INSTANCE
    if _MD2IMG_PLUGIN_INSTANCE is None:
        raise RuntimeError("md2img 插件实例尚未初始化，Tool 暂不可用")
    return _MD2IMG_PLUGIN_INSTANCE

def _md2img_tool_extract_ctx_event(wrapper: Any):
    """兼容不同 AstrBot 版本的 ContextWrapper 结构，尽量取到 ctx 与 event。"""
    agent_ctx = getattr(wrapper, "context", None)
    if agent_ctx is None:
        agent_ctx = getattr(wrapper, "_context", None)
    if agent_ctx is None:
        agent_ctx = wrapper
    event = getattr(agent_ctx, "event", None) or getattr(wrapper, "event", None)
    ctx = getattr(agent_ctx, "context", None) or getattr(wrapper, "ctx", None)
    return ctx, event

async def _md2img_tool_send_components(plugin: Any, ctx: Any, event: Any, comps: List[Any]) -> Tuple[bool, Any]:
    """在 Tool 内主动发送消息（图片/PDF 等）。
    返回 (ok, send_ret)。优先 event.send（可拿到回执/ message_id），失败则回退 ctx.send_message。
    """
    mc_or_list = plugin._build_msg_chain_from_components(comps)
    try:
        if event is not None and hasattr(event, "send") and callable(getattr(event, "send")):
            send_ret = await event.send(mc_or_list)
            return True, send_ret
    except Exception:
        pass
    try:
        if ctx is not None and hasattr(ctx, "send_message") and event is not None:
            await ctx.send_message(event.unified_msg_origin, mc_or_list)
            return True, None
    except Exception:
        pass
    return False, None

@dataclass
class Md2ImgRenderMarkdownTool(FunctionTool[AstrAgentContext]):
    """把 Markdown 渲染成图片并发送到当前会话。"""
    name: str = "md2img_render_markdown"
    description: str = "将一段 Markdown(含 LaTeX) 渲染成图片并发送到当前会话。适合 Agent 在回答前把长公式/复杂排版转为图片。"
    parameters: dict = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "markdown": {"type": "string", "description": "要渲染的 Markdown 文本（可包含 LaTeX 公式）"},
            "mode": {"type": "string", "description": "渲染模式：mobile 或 pc", "enum": ["mobile", "pc"], "default": "mobile"},
            "scale": {"type": "integer", "description": "缩放倍率（1~3，越大越清晰但更慢）", "minimum": 1, "maximum": 3, "default": 2},
        },
        "required": ["markdown"],
    })

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> ToolExecResult:
        plugin = _md2img_get_plugin()
        ctx, event = _md2img_tool_extract_ctx_event(context)

        md_text = str(kwargs.get("markdown") or kwargs.get("md") or "").strip()
        if not md_text:
            return json.dumps({"ok": False, "error": "markdown 不能为空"}, ensure_ascii=False)

        mode = str(kwargs.get("mode") or "mobile").strip().lower()
        if mode not in ("mobile", "pc"):
            mode = "mobile"

        try:
            scale = int(kwargs.get("scale", 2))
        except Exception:
            scale = 2
        scale = max(1, min(3, scale))

        os.makedirs(plugin.IMAGE_CACHE_DIR, exist_ok=True)
        out_path = os.path.join(plugin.IMAGE_CACHE_DIR, f"tool_{uuid.uuid4().hex}.png")

        img_paths = await plugin._markdown_to_image_playwright(
            md_text=md_text,
            output_image_path=out_path,
            scale=scale,
            mode=mode,
        )

        # 发送图片
        try:
            import astrbot.api.message_components as Comp
            comps_all = [Comp.Image.fromFileSystem(p) for p in img_paths]
        except Exception:
            comps_all = [Image.fromFileSystem(p) for p in img_paths]  # type: ignore

        sent = 0
        max_per_msg = int(getattr(plugin, "_cfg", lambda *_: 4)("tool_max_images_per_message", 4) or 4)
        batch: List[Any] = []
        for c in comps_all:
            batch.append(c)
            if len(batch) >= max_per_msg:
                ok, _ = await _md2img_tool_send_components(plugin, ctx, event, batch)
                if ok:
                    sent += len(batch)
                batch = []
        if batch:
            ok, _ = await _md2img_tool_send_components(plugin, ctx, event, batch)
            if ok:
                sent += len(batch)

        return json.dumps({"ok": True, "images": img_paths, "sent": sent, "mode": mode, "scale": scale}, ensure_ascii=False)

@dataclass
class Md2ImgSolveMathPdfTool(FunctionTool[AstrAgentContext]):
    """调用插件内置 PDF 生成（/pdf）逻辑。"""
    name: str = "md2img_solve_math_pdf"
    description: str = "根据题目文字(可选)+图片URL(可选)生成 LaTeX 并用本地 xelatex 编译 PDF，然后把 PDF 发送到当前会话。"
    parameters: dict = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "problem_text": {"type": "string", "description": "题目文字（可为空；若为空需提供 image_urls）"},
            "image_urls": {"type": "array", "items": {"type": "string"}, "description": "题目图片 URL 列表（可为空；若为空需提供 problem_text）"},
            "ref_pdf_latex": {"type": "string", "description": "可选：上一份 PDF 的 LaTeX 源码（用于追问/续写）", "default": ""},
        },
        "required": [],
    })

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> ToolExecResult:
        plugin = _md2img_get_plugin()
        ctx, event = _md2img_tool_extract_ctx_event(context)

        problem_text = str(kwargs.get("problem_text") or kwargs.get("text") or "").strip()
        image_urls = kwargs.get("image_urls") or kwargs.get("images") or []
        if isinstance(image_urls, str):
            image_urls = [image_urls]
        if not isinstance(image_urls, list):
            image_urls = []

        ref_pdf_latex = str(kwargs.get("ref_pdf_latex") or "").strip()

        if (not problem_text) and (not image_urls):
            return json.dumps({"ok": False, "error": "需要提供 problem_text 或 image_urls"}, ensure_ascii=False)

        # 标记：避免其它 on_llm_request 注入干扰
        try:
            if event is not None and hasattr(event, "set_extra"):
                event.set_extra("md2img_pdf_generation", True)
        except Exception:
            pass

        pdf_path, fname, generated_tex_src = await plugin._solve_math_to_pdf(
            problem_text=problem_text,
            event=event,
            image_urls=image_urls,
            ref_pdf_latex=ref_pdf_latex,
        )

        # 发送 PDF
        sent_msg_id: Optional[str] = None
        try:
            import astrbot.api.message_components as Comp
            comps = [Comp.File(file=pdf_path, name=fname)]
            ok, send_ret = await _md2img_tool_send_components(plugin, ctx, event, comps)
            if ok:
                sent_msg_id = plugin._extract_message_id_from_send_ret(send_ret)
        except Exception:
            ok = False

        # 更新 session 状态（供后续 /pdf 追问）
        try:
            if event is not None:
                skey = _get_session_key(event)
                async with plugin._state_lock:
                    st = MATH_SESSION_STATE.setdefault(skey, {})
                    now_ts = time.time()
                    st["last_active_ts"] = now_ts
                    if generated_tex_src:
                        st["last_pdf_context"] = generated_tex_src
                    if problem_text:
                        st["last_problem"] = problem_text
                    else:
                        if image_urls:
                            st["last_problem"] = "[图片题目]"
                    st["last_had_img"] = bool(image_urls)
                    if image_urls:
                        st["last_image_urls"] = list(image_urls)
                        st["last_image_ts"] = now_ts
                    if sent_msg_id and generated_tex_src:
                        mp = st.setdefault("pdf_ctx_map", {})
                        if isinstance(mp, dict):
                            mp[str(sent_msg_id)] = generated_tex_src
                            while len(mp) > 30:
                                try:
                                    mp.pop(next(iter(mp)))
                                except Exception:
                                    break
                        st["last_pdf_msg_id"] = str(sent_msg_id)
        except Exception:
            pass

        return json.dumps({"ok": True, "pdf_path": pdf_path, "filename": fname, "message_id": sent_msg_id}, ensure_ascii=False)

@dataclass
class Md2ImgSolveMathSpdfTool(FunctionTool[AstrAgentContext]):
    """调用插件内置 DeepThink 多角色 PDF 生成（/spdf）逻辑。"""
    name: str = "md2img_solve_math_spdf"
    description: str = "DeepThink 多角色迭代生成 LaTeX 并本地编译 PDF，然后把 PDF 发送到当前会话。"
    parameters: dict = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "problem_text": {"type": "string", "description": "题目文字（可为空；若为空需提供 image_urls）"},
            "image_urls": {"type": "array", "items": {"type": "string"}, "description": "题目图片 URL 列表（可为空；若为空需提供 problem_text）"},
            "ref_pdf_latex": {"type": "string", "description": "可选：上一份 PDF 的 LaTeX 源码（用于追问/续写）", "default": ""},
        },
        "required": [],
    })

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> ToolExecResult:
        plugin = _md2img_get_plugin()
        ctx, event = _md2img_tool_extract_ctx_event(context)

        problem_text = str(kwargs.get("problem_text") or kwargs.get("text") or "").strip()
        image_urls = kwargs.get("image_urls") or kwargs.get("images") or []
        if isinstance(image_urls, str):
            image_urls = [image_urls]
        if not isinstance(image_urls, list):
            image_urls = []

        ref_pdf_latex = str(kwargs.get("ref_pdf_latex") or "").strip()

        if (not problem_text) and (not image_urls):
            return json.dumps({"ok": False, "error": "需要提供 problem_text 或 image_urls"}, ensure_ascii=False)

        try:
            if event is not None and hasattr(event, "set_extra"):
                event.set_extra("md2img_pdf_generation", True)
        except Exception:
            pass

        pdf_path, fname, generated_tex_src = await plugin._solve_math_to_spdf(
            problem_text=problem_text,
            event=event,
            image_urls=image_urls,
            ref_pdf_latex=ref_pdf_latex,
        )

        # 发送 PDF
        sent_msg_id: Optional[str] = None
        try:
            import astrbot.api.message_components as Comp
            comps = [Comp.File(file=pdf_path, name=fname)]
            ok, send_ret = await _md2img_tool_send_components(plugin, ctx, event, comps)
            if ok:
                sent_msg_id = plugin._extract_message_id_from_send_ret(send_ret)
        except Exception:
            ok = False

        # 更新 session 状态（供后续 /pdf 追问）
        try:
            if event is not None:
                skey = _get_session_key(event)
                async with plugin._state_lock:
                    st = MATH_SESSION_STATE.setdefault(skey, {})
                    now_ts = time.time()
                    st["last_active_ts"] = now_ts
                    if generated_tex_src:
                        st["last_pdf_context"] = generated_tex_src
                    if problem_text:
                        st["last_problem"] = problem_text
                    else:
                        if image_urls:
                            st["last_problem"] = "[图片题目]"
                    st["last_had_img"] = bool(image_urls)
                    if image_urls:
                        st["last_image_urls"] = list(image_urls)
                        st["last_image_ts"] = now_ts
                    if sent_msg_id and generated_tex_src:
                        mp = st.setdefault("pdf_ctx_map", {})
                        if isinstance(mp, dict):
                            mp[str(sent_msg_id)] = generated_tex_src
                            while len(mp) > 30:
                                try:
                                    mp.pop(next(iter(mp)))
                                except Exception:
                                    break
                        st["last_pdf_msg_id"] = str(sent_msg_id)
        except Exception:
            pass

        return json.dumps({"ok": True, "pdf_path": pdf_path, "filename": fname, "message_id": sent_msg_id}, ensure_ascii=False)


# 插件定义
# -------------------------------------------------------------------------
@register(
    "astrbot_plugin_md2img",
    "tosaki",
    "Markdown转图片 + 数学答疑提示流 + /pdf LaTeX解答 + /spdf DeepThink多角色迭代 + 知识库检索 + 对话记忆",
    "1.11.0",
)
class MarkdownConverterPlugin(Star):
    def __init__(self, context: Context, config: Optional[Dict[str, Any]] = None):
        super().__init__(context)
        self.config: Dict[str, Any] = dict(config) if isinstance(config, dict) else {}

        self.DATA_DIR = _safe_get_data_dir()
        self.IMAGE_CACHE_DIR = os.path.join(self.DATA_DIR, "md2img_cache")
        self.PDF_CACHE_DIR = os.path.join(self.DATA_DIR, "md2img_pdf_cache")
        self.TEXLIVE_CACHE_DIR = os.path.join(self.PDF_CACHE_DIR, "_texlive_cache")

        # Playwright 复用
        self._pw = None
        self._browser = None
        self._pw_lock = asyncio.Lock()
        self._render_sema = asyncio.Semaphore(int(self._cfg("render_concurrency", 2) or 2))
        # xelatex 编译并发限制（避免同时跑太多编译占满 CPU/IO）
        self._tex_compile_sema = asyncio.Semaphore(int(self._cfg("tex_compile_concurrency", 2) or 2))
        # /spdf solver 并发限制
        self._spdf_solver_sema = asyncio.Semaphore(int(self._cfg("spdf_solver_concurrency", 2) or 2))

        # Session 清理任务
        self._session_cleaner_task: Optional[asyncio.Task] = None
        self._state_lock = asyncio.Lock()  # 最小侵入：统一保护 MATH_SESSION_STATE

        # TeXLive 编译日志
        self._last_texlive_log = ""


        # Tool / Function Calling：把本插件能力注册为 LLM Tools，供 Agent 调用
        # 官方文档：>= v4.5.1 使用 self.context.add_llm_tools(...) 注册 Tool
        # 兼容 < v4.5.1：使用 self.context.provider_manager.llm_tools.func_list.append(...)
        global _MD2IMG_PLUGIN_INSTANCE
        _MD2IMG_PLUGIN_INSTANCE = self
        try:
            self.context.add_llm_tools(
                Md2ImgRenderMarkdownTool(),
                Md2ImgSolveMathPdfTool(),
                Md2ImgSolveMathSpdfTool(),
            )
            logger.info("[md2img] 已注册 LLM Tools: md2img_render_markdown / md2img_solve_math_pdf / md2img_solve_math_spdf")
        except Exception as e:
            try:
                tool_mgr = self.context.provider_manager.llm_tools
                tool_mgr.func_list.append(Md2ImgRenderMarkdownTool())
                tool_mgr.func_list.append(Md2ImgSolveMathPdfTool())
                tool_mgr.func_list.append(Md2ImgSolveMathSpdfTool())
                logger.info("[md2img] 已通过旧版接口注册 LLM Tools")
            except Exception as e2:
                logger.warning(f"[md2img] 注册 LLM Tools 失败（不影响其它功能）: {e} / {e2}")

    def _cfg(self, key: str, default: Any = None) -> Any:
        if default is None:
            default = DEFAULT_CFG.get(key)
        try:
            if key in self.config:
                return self.config.get(key)
        except Exception:
            pass
        return default

    async def initialize(self):
        try:
            os.makedirs(self.IMAGE_CACHE_DIR, exist_ok=True)
            os.makedirs(self.PDF_CACHE_DIR, exist_ok=True)
            os.makedirs(self.TEXLIVE_CACHE_DIR, exist_ok=True)

            # 启动 Playwright（可选）
            if bool(self._cfg("reuse_playwright_browser", True)):
                await self._ensure_browser()

            # 启动 session 清理任务
            self._start_session_cleaner()

            logger.info("Markdown 转图片插件已就绪 (1.10.1 - 本地编译版)")
        except Exception as e:
            logger.error(f"初始化失败: {e}")

    # ------------------------- 生命周期：尽量释放资源 -------------------------
    async def _ensure_browser(self):
        if self._browser is not None:
            return
        async with self._pw_lock:
            if self._browser is not None:
                return
            try:
                self._pw = await async_playwright().start()
                self._browser = await self._pw.chromium.launch()
                logger.info("Playwright Chromium 已启动（复用模式）")
            except Exception as e:
                logger.error(f"Playwright 启动失败，将回退为每次渲染临时启动: {e}")
                self._pw = None
                self._browser = None

    async def _close_browser(self):
        async with self._pw_lock:
            try:
                if self._browser is not None:
                    await self._browser.close()
                if self._pw is not None:
                    await self._pw.stop()
            except Exception:
                pass
            self._browser = None
            self._pw = None

    # ------------------------- Session 清理 -------------------------
    def _start_session_cleaner(self):
        if self._session_cleaner_task and not self._session_cleaner_task.done():
            return
        interval = int(self._cfg("session_cleanup_interval_sec", 3600) or 3600)
        self._session_cleaner_task = asyncio.create_task(self._clean_expired_sessions_loop(interval))

    async def _clean_expired_sessions_loop(self, interval_sec: int):
        ttl = int(self._cfg("session_ttl_sec", 86400) or 86400)
        while True:
            try:
                await asyncio.sleep(max(60, interval_sec))
                now = time.time()
                async with self._state_lock:
                    expired = [
                        k for k, v in MATH_SESSION_STATE.items()
                        if now - float(v.get("last_active_ts", v.get("last_image_ts", 0) or 0)) > ttl
                    ]
                    for k in expired:
                        del MATH_SESSION_STATE[k]
                if expired:
                    logger.info(f"Session 清理完成: 删除 {len(expired)} 个过期会话")
            except Exception as e:
                logger.warning(f"Session 清理任务异常: {e}")

    # === 全局消息监听：记录用户最后发的图片 & 活跃时间 ===
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_any_message(self, event: AstrMessageEvent):
        try:
            imgs = _get_event_images(event)
            skey = _get_session_key(event)
            now_ts = time.time()
            raw = (getattr(event, "message_str", "") or "").strip()

            is_private = self._is_private_chat(event)
            reply_id = self._extract_reply_msg_id(event)
            cur_mid = self._extract_current_msg_id(event)

            async with self._state_lock:
                state = MATH_SESSION_STATE.setdefault(skey, {})
                state["last_active_ts"] = now_ts

                # --- 图片消息：记录图片；私聊纯图片则进入“待引用”状态 ---
                if imgs:
                    state["last_image_urls"] = imgs
                    state["last_image_ts"] = now_ts
                    if cur_mid:
                        state["last_image_msg_id"] = cur_mid

                    if is_private and (not raw):
                        # 私聊：只发图片 -> 不触发机器人回复，等待用户“引用该图片 + 文字”再答疑
                        state["pending_image_only"] = True
                        if cur_mid:
                            state["pending_image_msg_id"] = cur_mid
                        state["pending_image_ts"] = now_ts
                    else:
                        # 非“纯图片待引用”场景：清掉 pending，并正常记录 last_problem
                        state.pop("pending_image_only", None)
                        state.pop("pending_image_msg_id", None)
                        state.pop("pending_image_ts", None)
                        if raw:
                            state["last_problem"] = raw
                        else:
                            state["last_problem"] = "[图片题目]"

                # --- 非图片消息：如果上一条是“私聊纯图片待引用”，则仅在回复/引用它时保留 pending ---
                else:
                    if is_private and state.get("pending_image_only"):
                        pid = str(state.get("pending_image_msg_id") or "").strip()
                        # 若本条不是“引用 pending 图片”的消息，则认为用户已开始新话题，清掉 pending
                        if not (reply_id and ((not pid) or (str(reply_id).strip() == pid))):
                            state.pop("pending_image_only", None)
                            state.pop("pending_image_msg_id", None)
                            state.pop("pending_image_ts", None)

            # 私聊纯图片：直接 stop_event，阻止默认“收到就回”行为
            if is_private and imgs and (not raw):
                try:
                    event.stop_event()
                except Exception:
                    pass
        except Exception:
            pass


    # ------------------------- Reply/引用消息识别（用于 PDF 二次追问） -------------------------
    def _extract_reply_msg_id(self, event: AstrMessageEvent) -> Optional[str]:
        """尽量从不同适配器的 event/message_obj 中提取“引用/回复”的原消息 ID。

        说明：不同 QQ 适配器（NapCat/OneBot/mirai 等）字段差异很大，所以这里做多路兜底：
        1) message_obj 常见字段（reply/quote/source 等）
        2) message 组件中可能存在的 Reply/Quote 组件
        3) raw_message（CQ 码 / JSON 片段）正则提取
        """
        try:
            msg_obj = getattr(event, "message_obj", None)

            # --- 1) message_obj 直接字段 ---
            if msg_obj is not None:
                for attr in (
                        "reply_id", "reply_msg_id", "reply_message_id", "replyMessageId",
                        "quote_id", "quote_msg_id", "quote_message_id", "quoted_msg_id",
                        "source_id", "source_msg_id", "source_message_id",
                        "reply", "quote", "reference", "source",
                ):
                    try:
                        v = getattr(msg_obj, attr, None)
                    except Exception:
                        v = None
                    if v is None:
                        continue

                    # 直接是 id
                    if isinstance(v, (str, int)) and str(v).strip():
                        return str(v).strip()

                    # dict 里找 id
                    if isinstance(v, dict):
                        for k in ("id", "message_id", "msg_id", "seq", "messageId"):
                            vv = v.get(k)
                            if isinstance(vv, (str, int)) and str(vv).strip():
                                return str(vv).strip()

                    # object 里找 id
                    for k in ("id", "message_id", "msg_id", "seq", "messageId"):
                        try:
                            vv = getattr(v, k, None)
                        except Exception:
                            vv = None
                        if isinstance(vv, (str, int)) and str(vv).strip():
                            return str(vv).strip()

                # --- 2) message 组件中可能存在 Reply/Quote ---
                try:
                    comps = getattr(msg_obj, "message", None) or []
                except Exception:
                    comps = []
                for comp in comps:
                    try:
                        cname = (comp.__class__.__name__ or "").lower()
                    except Exception:
                        cname = ""
                    if cname in ("reply", "quote", "reference", "replymessage", "quotemessage", "source"):
                        for k in ("id", "message_id", "msg_id", "seq", "reply_id", "source_id", "messageId"):
                            try:
                                vv = getattr(comp, k, None)
                            except Exception:
                                vv = None
                            if isinstance(vv, (str, int)) and str(vv).strip():
                                return str(vv).strip()
                        # 有的组件会把原消息塞在 data 里
                        try:
                            data = getattr(comp, "data", None)
                        except Exception:
                            data = None
                        if isinstance(data, dict):
                            for k in ("id", "message_id", "msg_id", "seq", "messageId"):
                                vv = data.get(k)
                                if isinstance(vv, (str, int)) and str(vv).strip():
                                    return str(vv).strip()
        except Exception:
            pass

        # --- 3) raw_message（CQ 码 / JSON）兜底 ---
        raws: List[Any] = []
        try:
            raws.append(getattr(event, "raw_message", None))
        except Exception:
            pass

        # 很多适配器把原始事件挂在 message_obj 上，而不是 event 上
        if msg_obj is not None:
            for attr in ("raw_message", "raw_event", "raw", "_raw_message", "_raw_event"):
                try:
                    v = getattr(msg_obj, attr, None)
                    if v is not None:
                        raws.append(v)
                except Exception:
                    continue

        for raw in raws:
            if not raw:
                continue
            s = str(raw)

            # OneBot v11 / CQ 码： [CQ:reply,id=123]
            m = re.search(r"\[CQ:reply,id=([0-9A-Za-z_-]+)\]", s)
            if m:
                return m.group(1)

            # OneBot v12 / JSON 片段： {"type":"reply","data":{"id":"123"}}
            m = re.search(r'"type"\s*:\s*"reply"[\s\S]*?"id"\s*:\s*"?([0-9A-Za-z_-]+)"?', s)
            if m:
                return m.group(1)

            # 其他可能的字段
            m = re.search(r'(?:reply|quote)[_\- ]?(?:id|msg_id|message_id)\s*[:=]\s*"?([0-9A-Za-z_-]+)"?',
                          s, flags=re.I)
            if m:
                return m.group(1)

        return None

    def _is_private_chat(self, event: AstrMessageEvent) -> bool:
        """尽量判断是否为私聊（不同适配器字段差异较大，做多路兜底）。"""
        # 1) 若事件本身提供 is_private() 之类的方法
        for fn_name in ("is_private", "isPrivate", "is_private_chat"):
            fn = getattr(event, fn_name, None)
            if callable(fn):
                try:
                    v = fn()
                    if isinstance(v, bool):
                        return v
                except Exception:
                    pass

        # 2) 常见：群/频道字段存在则视作非私聊
        msg_obj = getattr(event, "message_obj", None)
        for attr in (
            "group_id", "groupId", "group", "guild_id", "guildId", "guild",
            "channel_id", "channelId", "channel", "room_id", "roomId", "room",
            "server_id", "serverId", "chat_id", "chatId"
        ):
            v = None
            try:
                if msg_obj is not None:
                    v = getattr(msg_obj, attr, None)
            except Exception:
                v = None
            if not v:
                try:
                    v = getattr(event, attr, None)
                except Exception:
                    v = None
            if isinstance(v, (str, int)) and str(v).strip() and str(v).strip() != "0":
                return False

        # 3) unified_msg_origin 兜底：包含 group/guild/channel/room 等字样视为非私聊
        try:
            umo = getattr(event, "unified_msg_origin", None)
            if isinstance(umo, str) and re.search(r"(group|guild|channel|room|server)", umo, flags=re.I):
                return False
        except Exception:
            pass

        return True

    def _extract_current_msg_id(self, event: AstrMessageEvent) -> Optional[str]:
        """尽量从 event/message_obj 中提取“当前这条消息”的 message_id。"""
        try:
            msg_obj = getattr(event, "message_obj", None)
            for root in (msg_obj, event, getattr(event, "raw_message", None)):
                if root is None:
                    continue
                if isinstance(root, (str, int)) and str(root).strip():
                    # raw_message 可能是文本/JSON，不直接作为 msg_id
                    pass
                # 常见字段
                for attr in ("message_id", "msg_id", "id", "messageId", "seq"):
                    try:
                        v = getattr(root, attr, None)
                    except Exception:
                        v = None
                    if isinstance(v, (str, int)) and str(v).strip():
                        return str(v).strip()
                # dict 里找
                if isinstance(root, dict):
                    for k in ("message_id", "msg_id", "id", "messageId", "seq"):
                        v = root.get(k)
                        if isinstance(v, (str, int)) and str(v).strip():
                            return str(v).strip()
        except Exception:
            pass
        return None

    @staticmethod
    def _attach_images_to_req(req: ProviderRequest, image_urls: List[str]) -> None:
        """把图片列表尽量塞进 ProviderRequest（不同 AstrBot 版本字段名可能不同）。"""
        if not image_urls:
            return

        tried_attrs = (
            "images", "image_urls", "image_url_list", "image_inputs", "vision_inputs",
            "input_images", "image_paths", "files", "attachments",
        )

        for attr in tried_attrs:
            if not hasattr(req, attr):
                continue
            try:
                cur = getattr(req, attr, None)
                if cur is None:
                    setattr(req, attr, list(image_urls))
                    return
                if isinstance(cur, list):
                    for u in image_urls:
                        if u not in cur:
                            cur.append(u)
                    setattr(req, attr, cur)
                    return
                if isinstance(cur, tuple):
                    merged = list(cur)
                    for u in image_urls:
                        if u not in merged:
                            merged.append(u)
                    setattr(req, attr, merged)
                    return
                # 其他类型：尝试直接覆盖
                setattr(req, attr, list(image_urls))
                return
            except Exception:
                continue

        # 最后兜底：放进 prompt（不保证所有视觉模型都能用 URL 取图，但至少不丢信息）
        try:
            req.prompt = (req.prompt or "") + "\n[用户引用的图片链接]\n" + "\n".join(image_urls) + "\n"
        except Exception:
            pass


    @staticmethod
    def _disable_llm_tools_in_req(req: ProviderRequest) -> None:
        """在 PDF 生成内部调用时，尽量禁用 tools/function-calling，避免部分上游返回 tool_calls 造成解析失败。"""
        try:
            # 常见字段：tools / functions
            for attr in (
                "tools", "functions",
                "tool_definitions", "function_definitions",
                "available_tools", "toolset", "tool_sets",
            ):
                if hasattr(req, attr):
                    try:
                        setattr(req, attr, [])
                    except Exception:
                        pass

            # 常见开关
            for attr in ("enable_tools", "enable_function_call", "use_tools"):
                if hasattr(req, attr):
                    try:
                        setattr(req, attr, False)
                    except Exception:
                        pass

            # OpenAI 兼容字段
            for attr in ("tool_choice", "function_call"):
                if hasattr(req, attr):
                    try:
                        setattr(req, attr, "none")
                    except Exception:
                        pass

            # 有些实现把额外参数放在 dict 里
            for attr in ("extra", "extra_params", "kwargs", "options", "provider_kwargs", "request_kwargs"):
                try:
                    d = getattr(req, attr, None)
                except Exception:
                    d = None
                if isinstance(d, dict):
                    d["tools"] = []
                    d["functions"] = []
                    d["tool_choice"] = "none"
                    d["function_call"] = "none"
                    try:
                        setattr(req, attr, d)
                    except Exception:
                        pass
        except Exception:
            pass

    def _reply_msg_has_pdf_hint(self, event: AstrMessageEvent) -> bool:
        """尝试判断用户引用的那条消息是否包含 PDF 文件（若适配器提供了被引用消息内容）。

        注意：很多 QQ 协议在 reply 里只给 message_id，并不给原消息内容。
        因此这个函数只能“尽量判断”，判断不到就返回 False。
        """
        try:
            msg_obj = getattr(event, "message_obj", None)
        except Exception:
            msg_obj = None

        candidates: List[Any] = []
        for root in (msg_obj, getattr(event, "raw_message", None)):
            if root is not None:
                candidates.append(root)

        # 额外尝试：某些适配器会把 quote/reply 的原消息内容放在 message_obj.reply/message_obj.quote 里
        if msg_obj is not None:
            for attr in ("reply", "quote", "reference", "source", "origin", "original_message"):
                try:
                    v = getattr(msg_obj, attr, None)
                except Exception:
                    v = None
                if v is not None:
                    candidates.append(v)

        def _looks_like_pdf_string(x: str) -> bool:
            xl = (x or "").lower()
            return (".pdf" in xl) or ("application/pdf" in xl)

        # 尝试在对象图里找 ".pdf" / File 组件 / file name
        try:
            for root in candidates:
                for obj, _depth in _walk_object_graph(root, max_depth=6, max_nodes=2000):
                    if isinstance(obj, str) and _looks_like_pdf_string(obj):
                        return True
                    if isinstance(obj, dict):
                        for k in ("name", "filename", "file", "path", "url", "mime", "mimetype"):
                            v = obj.get(k)
                            if isinstance(v, str) and _looks_like_pdf_string(v):
                                return True
                    try:
                        cname = (obj.__class__.__name__ or "").lower()
                    except Exception:
                        cname = ""
                    if "file" in cname and "pdf" in cname:
                        return True
        except Exception:
            pass

        return False

    def _build_msg_chain_from_components(self, comps: List[Any]) -> Any:
        """把 Comp.* 组件列表转换成可用于 event.send 的对象（尽量兼容不同 AstrBot 版本）。"""
        comps = list(comps or [])

        # 1) 有的版本支持 MessageChain(list)
        try:
            return MessageChain(comps)  # type: ignore
        except Exception:
            pass

        # 2) 常规：MessageChain() 然后 append/extend/chain
        try:
            mc = MessageChain()
        except Exception:
            mc = None

        if mc is not None:
            for mname in ("chain", "extend", "append", "add"):
                fn = getattr(mc, mname, None)
                if callable(fn):
                    try:
                        if mname in ("extend", "chain"):
                            fn(comps)
                        else:
                            for c in comps:
                                fn(c)
                        return mc
                    except Exception:
                        continue

            # 3) 极端兜底：直接写入 message 列表
            try:
                lst = getattr(mc, "message", None)
                if isinstance(lst, list):
                    lst.extend(comps)
                    return mc
            except Exception:
                pass

        # 4) 最后兜底：直接返回组件列表（部分版本 event.send 可以直接吃 list）
        return comps

    @staticmethod
    def _extract_message_id_from_send_ret(send_ret: Any) -> Optional[str]:
        """从 event.send(...) 的返回值中尽量提取 message_id。"""
        if send_ret is None:
            return None
        if isinstance(send_ret, (str, int)) and str(send_ret).strip():
            return str(send_ret).strip()

        if isinstance(send_ret, list) and send_ret:
            # 有的版本会返回 list[receipt]
            return MarkdownConverterPlugin._extract_message_id_from_send_ret(send_ret[0])

        if isinstance(send_ret, dict):
            for k in ("message_id", "msg_id", "id", "messageId", "seq"):
                v = send_ret.get(k)
                if isinstance(v, (str, int)) and str(v).strip():
                    return str(v).strip()
            # 可能嵌套在 data/result 里
            for kk in ("data", "result", "message"):
                v = send_ret.get(kk)
                if isinstance(v, dict):
                    mid = MarkdownConverterPlugin._extract_message_id_from_send_ret(v)
                    if mid:
                        return mid

        for attr in ("message_id", "msg_id", "id", "messageId", "seq"):
            try:
                v = getattr(send_ret, attr, None)
            except Exception:
                v = None
            if isinstance(v, (str, int)) and str(v).strip():
                return str(v).strip()

        return None

    # === 快捷指令 1: PC 模式 ===
    @filter.command("pc")
    async def cmd_pc(self, event: AstrMessageEvent):
        user_id = event.get_sender_id()
        if not user_id:
            yield event.plain_result("无法获取用户ID")
            return
        USER_PREFERENCES[user_id] = "pc"
        yield event.plain_result("✅ 已切换为 PC 模式 (A4分页)")

    # === 快捷指令 2: Mobile/PE 模式 ===
    @filter.command("pe")
    async def cmd_pe(self, event: AstrMessageEvent):
        user_id = event.get_sender_id()
        if not user_id:
            yield event.plain_result("无法获取用户ID")
            return
        USER_PREFERENCES[user_id] = "mobile"
        yield event.plain_result("✅ 已切换为 手机 模式 (长截屏)")


    # === 快捷指令：清空本会话对话记忆 ===
    @filter.command("memclear")
    async def cmd_memclear(self, event: AstrMessageEvent):
        skey = _get_session_key(event)
        async with self._state_lock:
            st = MATH_SESSION_STATE.setdefault(skey, {})
            st.pop("chat_history", None)
        yield event.plain_result("✅ 已清空本会话的对话记忆")

    # === 快捷指令 3: 生成 PDF 解答 ===
    @filter.command("pdf")
    async def cmd_pdf(self, event: AstrMessageEvent):
        if not self._cfg("enable_pdf_output", True):
            yield event.plain_result("PDF 输出未启用")
            return

        raw = (getattr(event, "message_str", "") or "").strip()
        arg = ""
        m = re.match(r"^\s*/?pdf(?:\s+([\s\S]*))?$", raw, flags=re.I)
        if m:
            arg = (m.group(1) or "").strip()

        current_images = _get_event_images(event)
        skey = _get_session_key(event)

        async with self._state_lock:
            state = MATH_SESSION_STATE.get(skey, {}).copy()

        last_problem_text = (state.get("last_problem", "") or "").strip()
        last_images = state.get("last_image_urls", [])
        last_image_ts = float(state.get("last_image_ts", 0) or 0)

        final_text_input = ""
        final_image_inputs: List[str] = []

        now_ts = time.time()
        valid_sec = int(self._cfg("last_image_valid_sec", 3600) or 3600)
        is_last_img_likely_valid = (now_ts - last_image_ts) <= valid_sec

        if current_images:
            final_image_inputs = current_images
            final_text_input = arg
        elif arg:
            final_text_input = arg

            # 文字 + 上一张图（追问场景）
            # 新逻辑：默认 smart —— 仅在文本看起来是在追问/引用图片时才携带上一张图。
            reuse_mode = str(self._cfg("pdf_reuse_last_image_with_text_mode", "smart") or "smart").lower().strip()
            followup_max_len = int(self._cfg("pdf_followup_text_max_len", 40) or 40)

            allow_reuse = False
            if last_images and is_last_img_likely_valid:
                # 只有“上一题确实是图片题”时才考虑复用（避免跨题串图）
                last_had_img = bool(state.get("last_had_img", False))
                last_img_context = last_had_img or (last_problem_text == "[图片题目]")
                if last_img_context:
                    if reuse_mode == "always":
                        allow_reuse = True
                    elif reuse_mode == "never":
                        allow_reuse = False
                    else:
                        # smart 模式：出现明显“追问/引用图片”的信号 or 文本较短且不像新题
                        t = (arg or "").strip()
                        # 用户明确要求“不用图片”
                        if re.search(r"(不用|不使用|别|不要).{0,6}(图片|图|图像)", t):
                            allow_reuse = False
                        elif re.search(
                                r"(如图|见图|图中|图片|上题|上一题|刚才|继续|小问|第\s*(?:\d+|[一二两三四五六七八九十])\s*问|\(\s*\d+\s*\))",
                                t):
                            allow_reuse = True
                        elif len(t) <= max(10, followup_max_len) and (not _is_math_question(t)):
                            allow_reuse = True
                        else:
                            allow_reuse = False

            if allow_reuse:
                final_image_inputs = last_images
        elif last_images:
            # 只发 /pdf：回溯上一张图
            if not is_last_img_likely_valid:
                yield event.plain_result("⚠️ 上一张图片可能已过期（链接可能失效），建议重新发送图片 + /pdf")
                # 仍允许用户“试一试”：很多平台 URL 实际更久
            final_image_inputs = last_images
            final_text_input = last_problem_text if last_problem_text != "[图片题目]" else ""
        elif last_problem_text and last_problem_text != "[图片题目]":
            final_text_input = last_problem_text
        else:
            yield event.plain_result("请发送 /pdf 题目文字，或发送包含题目的图片 + /pdf")
            return

        # === 随机等待提示语 Start ===
        sender_name = (event.get_sender_name() or "我").strip() or "我"
        pdf_loading_msgs = [
            f"{sender_name} 正在疯狂敲击键盘生成 PDF 中！请稍等，马上就送达你的手中！",
            f"{sender_name} 正在施展 PDF 生成魔法，只需再坚持几秒钟，奇迹即将发生！",
            f"{sender_name} 抱着刚生成的 PDF 一路狂奔，这就送到你面前，请稍安勿躁！️",
            f"{sender_name} 的 PDF 工厂正在全速开工，齿轮飞转中！请稍候，成品即将上线！",
            f"{sender_name} 正在把散落的字节拼凑成完美的 PDF，精彩内容马上呈现！",
            f"正在给文件加热中... {sender_name} 保证这份 PDF 会热乎乎、香喷喷地出炉！",
            f"正在给 PDF 做最后的美容护理，{sender_name} 务必要让它漂漂亮亮地见你！",
            f"{sender_name} 的 CPU 正在高速运转，为了这份 PDF 已经全力以赴啦！马上搞定！",
            f"{sender_name} 正在数据海洋里潜水，只为把这份珍贵的 PDF 捞上来给你！马上浮出水面！",
            f"嘘——{sender_name} 正在全神贯注地封装数据，这份 PDF 凝聚了所有的心血，马上就好！"
        ]
        yield event.plain_result(random.choice(pdf_loading_msgs))
        # === 随机等待提示语 End ===

        # 若用户“引用/回复”了机器人之前发出的 PDF 文件消息：尽量把对应 PDF 的 LaTeX 源码取出来，
        # 作为本次 /pdf 追问的参考上下文（避免模型遗忘上一份 PDF 的细节）。
        reply_id = self._extract_reply_msg_id(event)
        ref_pdf_ctx = ""
        if reply_id:
            try:
                mp = state.get("pdf_ctx_map", {})
                if isinstance(mp, dict):
                    ref_pdf_ctx = (mp.get(str(reply_id)) or "").strip()
            except Exception:
                ref_pdf_ctx = ""

            # 兜底：没有精确 message_id 映射时，弱绑定到“最近一次 PDF”
            if not ref_pdf_ctx:
                last_pdf_ctx = (state.get("last_pdf_context") or "").strip()
                if last_pdf_ctx:
                    last_mid = str(state.get("last_pdf_msg_id", "") or "").strip()
                    if last_mid:
                        if str(reply_id) == last_mid:
                            ref_pdf_ctx = last_pdf_ctx
                        else:
                            # 若适配器能拿到被引用消息内容且确实包含 PDF，则允许兜底
                            if self._reply_msg_has_pdf_hint(event):
                                ref_pdf_ctx = last_pdf_ctx
                    else:
                        ref_pdf_ctx = last_pdf_ctx

        # 标记：本次 LLM 调用用于 /pdf 生成，避免其它 on_llm_request 逻辑误注入
        try:
            event.set_extra("md2img_pdf_generation", True)
        except Exception:
            pass

        try:
            pdf_path, fname, generated_tex_src = await self._solve_math_to_pdf(
                problem_text=final_text_input,
                event=event,
                image_urls=final_image_inputs,
                ref_pdf_latex=ref_pdf_ctx,
            )

            # 保存 PDF 上下文 + 更新“上一题/上一张图”状态
            async with self._state_lock:
                st = MATH_SESSION_STATE.setdefault(skey, {})
                now_ts = time.time()
                st["last_active_ts"] = now_ts
                if generated_tex_src:
                    st["last_pdf_context"] = generated_tex_src

                # 记录更干净的“上一题”（去掉 /pdf 前缀）
                clean_last = (final_text_input or "").strip()
                if clean_last:
                    clean_last = re.sub(r"^\s*/?pdf\b", "", clean_last, flags=re.I).strip()
                if clean_last:
                    st["last_problem"] = clean_last
                else:
                    # 没有文字则标记为图片题
                    if final_image_inputs:
                        st["last_problem"] = "[图片题目]"
                st["last_had_img"] = bool(final_image_inputs)

                if final_image_inputs:
                    st["last_image_urls"] = list(final_image_inputs)
                    st["last_image_ts"] = now_ts
        except Exception as e:
            err = str(e)
            try:
                log_snip = (self._last_texlive_log or "").strip()
                if log_snip:
                    log_snip = log_snip[-2000:]
                logger.error(f"PDF 生成失败: {err}\nCompile log snippet:\n{log_snip}")
            except Exception:
                logger.error(f"PDF 生成失败: {err}")

            if "Upstream Error" in err:
                _sn = (event.get_sender_name() or "我").strip() or "我"
                yield event.plain_result(f"呜哇，{_sn} 够不到那张图片啦！它好像溜走了（链接过期或拒绝访问）。😣\n能不能麻烦你重新截图再发给我一次?")
            else:
                # 给用户一个简短的日志末尾，方便快速定位是“缺包/字体”还是“内容里有非法字符”
                msg = "哎呀，PDF 生成失败了，看起来是排版出了点小差错，重问一次试试吧~"
                try:
                    log_snip = (self._last_texlive_log or "").strip()
                    if log_snip:
                        tail_lines = "\n".join(log_snip.splitlines()[-20:])
                        tail_lines = tail_lines[-1500:]
                        msg += "\n\nxelatex 日志末尾(截取)：\n" + tail_lines
                except Exception:
                    pass
                yield event.plain_result(msg)
            return

        # 发送 PDF 文件
        try:
            import astrbot.api.message_components as Comp
        except Exception:
            yield event.plain_result("当前 AstrBot 版本缺少文件消息组件，无法发送 PDF")
            return

        uid = event.get_sender_id()
        chain = []

        # === 修复开始：仅在非私聊（群聊）场景下才添加 @ ===
        is_private = self._is_private_chat(event)
        if uid and (not is_private):
            chain.append(Comp.At(qq=uid))
        # === 修复结束 ===

        chain.append(Comp.File(file=pdf_path, name=fname))

        # 优先用 event.send 发送（若适配器支持，可拿到 message_id，便于后续“引用这条 PDF 消息”追问）
        sent_msg_id: Optional[str] = None
        try:
            mc_or_chain = self._build_msg_chain_from_components(chain)
            send_ret = await event.send(mc_or_chain)
            sent_msg_id = self._extract_message_id_from_send_ret(send_ret)
        except Exception as e:
            logger.debug(f"event.send 发送 PDF 失败，回退为 chain_result: {e}")
            yield event.chain_result(chain)

        # 若成功拿到 message_id，则把本次 PDF 的 LaTeX 源码与该消息绑定，供用户引用追问
        if sent_msg_id and generated_tex_src:
            try:
                async with self._state_lock:
                    st = MATH_SESSION_STATE.setdefault(skey, {})
                    mp = st.setdefault("pdf_ctx_map", {})
                    if isinstance(mp, dict):
                        mp[str(sent_msg_id)] = generated_tex_src
                        # 控制缓存大小：最多保留 30 条（FIFO）
                        max_keep = 30
                        while len(mp) > max_keep:
                            try:
                                mp.pop(next(iter(mp)))
                            except Exception:
                                break
                    st["last_pdf_msg_id"] = str(sent_msg_id)
            except Exception:
                pass

    # === 快捷指令 4: DeepThink 多角色迭代生成 PDF 解答（/spdf） ===
    @filter.command("spdf")
    async def cmd_spdf(self, event: AstrMessageEvent):
        if not self._cfg("enable_spdf_output", True):
            yield event.plain_result("SPDF 输出未启用")
            return

        raw = (getattr(event, "message_str", "") or "").strip()
        arg = ""
        m = re.match(r"^\s*/?spdf(?:\s+([\s\S]*))?$", raw, flags=re.I)
        if m:
            arg = (m.group(1) or "").strip()

        current_images = _get_event_images(event)
        skey = _get_session_key(event)

        async with self._state_lock:
            state = MATH_SESSION_STATE.get(skey, {}).copy()

        last_problem_text = (state.get("last_problem", "") or "").strip()
        last_images = state.get("last_image_urls", [])
        last_image_ts = float(state.get("last_image_ts", 0) or 0)

        final_text_input = ""
        final_image_inputs: List[str] = []

        now_ts = time.time()
        valid_sec = int(self._cfg("last_image_valid_sec", 3600) or 3600)
        is_last_img_likely_valid = (now_ts - last_image_ts) <= valid_sec

        if current_images:
            final_image_inputs = current_images
            final_text_input = arg
        elif arg:
            final_text_input = arg

            # 文字 + 上一张图（追问场景）
            reuse_mode = str(self._cfg("pdf_reuse_last_image_with_text_mode", "smart") or "smart").lower().strip()
            followup_max_len = int(self._cfg("pdf_followup_text_max_len", 40) or 40)

            allow_reuse = False
            if last_images and is_last_img_likely_valid:
                last_had_img = bool(state.get("last_had_img", False))
                last_img_context = last_had_img or (last_problem_text == "[图片题目]")
                if last_img_context:
                    if reuse_mode == "always":
                        allow_reuse = True
                    elif reuse_mode == "never":
                        allow_reuse = False
                    else:
                        t = (arg or "").strip()
                        if re.search(r"(不用|不使用|别|不要).{0,6}(图片|图|图像)", t):
                            allow_reuse = False
                        elif re.search(
                                r"(如图|见图|图中|图片|上题|上一题|刚才|继续|小问|第\s*(?:\d+|[一二两三四五六七八九十])\s*问|\(\s*\d+\s*\))",
                                t):
                            allow_reuse = True
                        elif len(t) <= max(10, followup_max_len) and (not _is_math_question(t)):
                            allow_reuse = True
                        else:
                            allow_reuse = False

            if allow_reuse:
                final_image_inputs = last_images
        elif last_images:
            if not is_last_img_likely_valid:
                yield event.plain_result("⚠️ 上一张图片可能已过期（链接可能失效），建议重新发送图片 + /spdf")
            final_image_inputs = last_images
            final_text_input = last_problem_text if last_problem_text != "[图片题目]" else ""
        elif last_problem_text and last_problem_text != "[图片题目]":
            final_text_input = last_problem_text
        else:
            yield event.plain_result("请发送 /spdf 题目文字，或发送包含题目的图片 + /spdf")
            return

        _sn = (event.get_sender_name() or "我").strip() or "我"
        yield event.plain_result(f"捕捉到灵感啦！{_sn}感到自己突然灵光一闪！正在生成PDF解答，不过等待时间较长哦，请稍候~")

        reply_id = self._extract_reply_msg_id(event)
        ref_pdf_ctx = ""
        if reply_id:
            try:
                mp = state.get("pdf_ctx_map", {})
                if isinstance(mp, dict):
                    ref_pdf_ctx = (mp.get(str(reply_id)) or "").strip()
            except Exception:
                ref_pdf_ctx = ""
            if not ref_pdf_ctx:
                last_pdf_ctx = (state.get("last_pdf_context") or "").strip()
                if last_pdf_ctx:
                    last_mid = str(state.get("last_pdf_msg_id", "") or "").strip()
                    if last_mid:
                        if str(reply_id) == last_mid:
                            ref_pdf_ctx = last_pdf_ctx
                        else:
                            if self._reply_msg_has_pdf_hint(event):
                                ref_pdf_ctx = last_pdf_ctx
                    else:
                        ref_pdf_ctx = last_pdf_ctx

        try:
            event.set_extra("md2img_pdf_generation", True)
        except Exception:
            pass

        try:
            pdf_path, fname, generated_tex_src = await self._solve_math_to_spdf(
                problem_text=final_text_input,
                event=event,
                image_urls=final_image_inputs,
                ref_pdf_latex=ref_pdf_ctx,
            )

            async with self._state_lock:
                st = MATH_SESSION_STATE.setdefault(skey, {})
                now_ts = time.time()
                st["last_active_ts"] = now_ts
                if generated_tex_src:
                    st["last_pdf_context"] = generated_tex_src

                clean_last = (final_text_input or "").strip()
                if clean_last:
                    clean_last = re.sub(r"^\s*/?spdf\b", "", clean_last, flags=re.I).strip()
                if clean_last:
                    st["last_problem"] = clean_last
                else:
                    if final_image_inputs:
                        st["last_problem"] = "[图片题目]"
                st["last_had_img"] = bool(final_image_inputs)

                if final_image_inputs:
                    st["last_image_urls"] = list(final_image_inputs)
                    st["last_image_ts"] = now_ts
        except Exception as e:
            err = str(e)
            # 先记录日志（若有编译日志则一并打印，便于排查）
            try:
                log_snip = (self._last_texlive_log or "").strip()
                if log_snip:
                    log_snip = log_snip[-2000:]
                logger.error(f"SPDF 生成失败: {err}\nCompile log snippet:\n{log_snip}")
            except Exception:
                logger.error(f"SPDF 生成失败: {err}")

            low_err = (err or "").lower()

            # 1) 图片读取失败
            if "Upstream Error" in err:
                _sn = (event.get_sender_name() or "我").strip() or "我"
                yield event.plain_result(f"呜呜，{_sn} 够不到那张图片啦！它好像溜走了（链接过期或拒绝访问）。😣\n能不能麻烦你重新截图再发给我一次？")
                return

            # 2) 鉴权 / Key 缺失
            if _is_auth_error(err):
                yield event.plain_result(
                    "❌ SPDF 生成失败：模型鉴权不可用（未配置 API Key / Token，或 provider 暂不可用）。\n"
                    "排查建议：\n"
                    "1) 检查 AstrBot 对应模型 provider 是否已配置 Key/Token；\n"
                    "2) 如果你在 spdf_judge_provider_id / spdf_role_pool_provider_ids 中填写了其他 provider，确保它们也配置了鉴权；\n"
                    "3) 可先把 spdf_judge_provider_id 留空，让 /spdf 跟随当前会话模型。"
                )
                return

            # 3) tool_calls / function_call 解析失败（常见于上游启用了 tools）
            if _is_toolcall_parse_error(err):
                yield event.plain_result(
                    "❌ SPDF 生成失败：上游模型返回了工具/函数调用格式（tool_calls/function_call），但当前 AstrBot 解析失败。\n"
                    "改进建议：\n"
                    "1) 保持插件配置 `spdf_disable_tools_during_generation=true`（默认开启），用于在 /pdf / /spdf 内部尽量禁用 tools；\n"
                    "2) 更换/下调为不容易触发 tool calling 的模型通道；\n"
                    "3) 升级 AstrBot 或对应 provider 适配（某些版本对 tool_calls 支持不完整）。"
                )
                return

            # 4) 上下文过长 / token 超限
            if _is_context_length_error(err):
                yield event.plain_result(
                    "❌ SPDF 生成失败：提示词/上下文过长（超出模型 token 限制）。\n"
                    "可以尝试：\n"
                    "1) 降低 spdf_candidate_char_budget 或 spdf_post_formatter_input_char_budget；\n"
                    "2) 关闭交叉质询/自一致投票或降低 spdf_num_solvers / spdf_iter_rounds；\n"
                    "3) 选择上下文更大的模型。"
                )
                return

            # 5) LaTeX 编译失败
            msg = "❌ SPDF 生成失败（本地 xelatex 编译出错）。"
            try:
                log_snip = (self._last_texlive_log or "").strip()
                if log_snip:
                    tail_lines = "\n".join(log_snip.splitlines()[-20:])
                    tail_lines = tail_lines[-1500:]
                    msg += "\n\nxelatex 日志末尾(截取)：\n" + tail_lines
            except Exception:
                pass

            # 如果明显不是编译错误，则给出更通用的提示
            if ("compile failed" not in low_err) and ("xelatex" not in low_err) and ("latex" not in low_err):
                msg = "❌ SPDF 生成失败：模型/网络调用异常。\n" + self._spdf_trunc(err, 400)

            yield event.plain_result(msg)
            return

        try:
            import astrbot.api.message_components as Comp
        except Exception:
            yield event.plain_result("当前 AstrBot 版本缺少文件消息组件，无法发送 PDF")
            return

        uid = event.get_sender_id()
        chain = []

        is_private = self._is_private_chat(event)
        if uid and (not is_private):
            chain.append(Comp.At(qq=uid))

        chain.append(Comp.File(file=pdf_path, name=fname))

        sent_msg_id: Optional[str] = None
        try:
            mc_or_chain = self._build_msg_chain_from_components(chain)
            send_ret = await event.send(mc_or_chain)
            sent_msg_id = self._extract_message_id_from_send_ret(send_ret)
        except Exception as e:
            logger.debug(f"event.send 发送 PDF 失败，回退为 chain_result: {e}")
            yield event.chain_result(chain)

        if sent_msg_id and generated_tex_src:
            try:
                async with self._state_lock:
                    st = MATH_SESSION_STATE.setdefault(skey, {})
                    mp = st.setdefault("pdf_ctx_map", {})
                    if isinstance(mp, dict):
                        mp[str(sent_msg_id)] = generated_tex_src
                        max_keep = 30
                        while len(mp) > max_keep:
                            try:
                                mp.pop(next(iter(mp)))
                            except Exception:
                                break
                    st["last_pdf_msg_id"] = str(sent_msg_id)
            except Exception:
                pass

    # ---------------------------------------------------------------------
    # LLM -> LaTeX -> PDF
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # LLM -> LaTeX -> PDF
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # /spdf：DeepThink 多角色迭代（角色池 + 交叉质询 + 自一致投票 + Python 验算 + TikZ 预检）
    # ---------------------------------------------------------------------
    def _spdf_parse_provider_pool(self, s: str) -> List[str]:
        r"""把 'a,b c\nd' 解析成 provider_id 列表。"""
        if not s:
            return []
        parts = re.split(r"[,\s]+", str(s))
        out = [p.strip() for p in parts if p and p.strip()]
        # 去重但保持顺序
        seen = set()
        uniq = []
        for p in out:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return uniq

    def _spdf_role_defs(self) -> List[Dict[str, str]]:
        """默认角色集合（可扩展）。"""
        return [
            {"name": "Solver-代数", "desc": "偏代数/方程/数列，严格推导、注重等价变形与边界条件。"},
            {"name": "Solver-几何", "desc": "偏几何/解析几何，善用坐标化、向量/面积法并检查图形约束。"},
            {"name": "Solver-检验", "desc": "偏严谨校验：复算关键步骤、检查特殊值/端点、寻找潜在漏洞。"},
            {"name": "Solver-构造", "desc": "偏构造/不等式/技巧，寻找更短更强的思路，但仍需写全步骤。"},
        ]

    def _spdf_trunc(self, s: str, budget: int) -> str:
        if not s:
            return ""
        ss = str(s)
        if budget <= 0 or len(ss) <= budget:
            return ss
        head = ss[: max(0, budget // 2)]
        tail = ss[-max(0, budget - len(head)) :]
        return head + "\n...\n" + tail

    # ---------- /spdf 可选：DeepThink 输出后处理（强制三段标签 + 尽量中文） ----------
    def _spdf_needs_post_format(self, raw: str) -> bool:
        """
        判断 DeepThink 最终输出是否需要二次格式化：
        - 缺少 <problem>/<theorems>/<solution> 标签
        - 含明显 Markdown/代码块/流式标签
        - 中文占比很低且英文字符很多（常见于“难题模型跑偏”）
        """
        if not raw:
            return False
        s = str(raw)
        low = s.lower()
        # 标签缺失/不完整
        if ("<problem>" not in low) or ("<theorems>" not in low) or ("<solution>" not in low):
            return True
        if ("</solution>" not in low):
            return True
        # 明显非 LaTeX 片段
        if "```" in s:
            return True
        if "<md>" in low or "</md>" in low or "<stream>" in low or "</stream>" in low:
            return True
        # 英文过多、中文过少
        try:
            cjk_cnt = len(re.findall(r"[\u4e00-\u9fff]", s))
            latin_cnt = len(re.findall(r"[A-Za-z]", s))
            total = cjk_cnt + latin_cnt
            # 更稳健：英文很长且中文占比极低（难题时模型容易跑成全英文）
            if latin_cnt >= 80:
                ratio = (float(cjk_cnt) / float(total)) if total > 0 else 0.0
                if ratio < 0.08:
                    return True
            # 极端情况：几乎没有中文但英文明显
            if cjk_cnt < 12 and latin_cnt > 120:
                return True
            if cjk_cnt < 6 and latin_cnt > 60:
                return True
        except Exception:
            pass
        return False

    async def _spdf_post_format_final_output(
        self,
        raw: str,
        problem_text: str,
        has_image: bool,
        image_urls: Optional[List[str]],
        ref_pdf_latex: str,
        fallback_provider_id: str,
    ) -> str:
        """
        使用额外模型把 DeepThink 最终输出整理为本插件 PDF 标准格式：
        - 严格输出 <problem>/<theorems>/<solution>
        - 尽量用中文叙述（数学符号保留）
        - 只在检测到“明显跑偏/格式不合规”时触发，避免改变原有正常输出
        """
        if not bool(self._cfg("spdf_enable_post_formatter", True)):
            return raw

        if not self._spdf_needs_post_format(raw):
            return raw

        pid = str(self._cfg("spdf_post_formatter_provider_id", "") or "").strip()
        if not pid:
            pid = str(fallback_provider_id or "").strip()

        if not pid:
            return raw

        budget = int(self._cfg("spdf_post_formatter_input_char_budget", 16000) or 16000)
        budget = max(2000, min(50000, budget))

        raw_in = self._spdf_trunc(str(raw or ""), budget)
        ref_in = self._spdf_trunc(str(ref_pdf_latex or ""), min(8000, budget // 2))

        prompt = (
            "你是 LaTeX 排版与翻译助手。\n"
            "任务：把【DeepThink 原始输出】整理为本插件 PDF 的标准三段 LaTeX，并尽量用中文表述。\n"
            "输出必须严格且只包含下面 3 个标签块（顺序固定）：\n"
            "<problem>\n(题面 LaTeX 片段)\n</problem>\n"
            "<theorems>\n(结论 LaTeX 片段)\n</theorems>\n"
            "<solution>\n(解答 LaTeX 片段)\n</solution>\n\n"
            "硬性要求：\n"
            "1) 标签内只能写 LaTeX 片段，不要输出 \\documentclass/\\begin{document}/\\end{document} 等。\n"
            "2) 不要输出 Markdown，不要代码块。\n"
            "3) 除数学符号/变量名/专有名词外，正文尽量用中文。\n"
            "4) problem 段优先使用【用户题目文本】还原题面；若为空或图片题，可写“题面见图”并概括你能确定的条件（不要编造）。\n"
            "5) theorems 段给出本题用到的关键结论，建议使用 theoremBox/lemmaBox/formulaBox 环境。\n"
            "6) solution 段给出完整解答，条理清晰；最后必须以 \\hfill$\\blacksquare$ 结束。\n"
            "7) 安全：不要使用 \\input/\\include/\\write18/\\openout 等命令，不要引用外部文件。\n\n"
            f"【用户题目文本】\n{(problem_text or '').strip()}\n\n"
            + (f"【引用的上一份 PDF 内容（摘要）】\n{ref_in}\n\n" if ref_in.strip() else "")
            + f"【DeepThink 原始输出】\n{raw_in}\n"
        )

        try:
            out = await self._spdf_llm_generate(pid, prompt, image_urls if (has_image and image_urls) else None)
        except Exception as e:
            logger.warning(f"spdf post-format llm_generate failed: {e}")
            return raw

        out = _strip_all_code_fences(str(out or "").strip())
        if not out:
            return raw

        # 解析并强制回到三段标签结构（即便模型多输出了别的内容，也只截取标签内）
        p_tex, t_tex, s_tex = _parse_latex_sections(out)

        if not s_tex:
            # 兜底：把模型输出整体当作解答文本
            s_tex = _llm_raw_to_safe_tex(out)

        if not p_tex:
            if (problem_text or "").strip():
                p_tex = _escape_text_preserve_dollar_math(problem_text)
            elif has_image:
                p_tex = r"(题面见图)"
            else:
                p_tex = r"(题面缺失)"

        if not t_tex:
            t_tex = (
                r"\begin{theoremBox}{常用结论}" + "\n"
                r"本题仅用到基础运算与常用恒等变形" + "\n"
                r"\end{theoremBox}"
            )

        # 重新拼装成严格的三段标签输出
        wrapped = (
            "<problem>\n" + str(p_tex).strip() + "\n</problem>\n"
            "<theorems>\n" + str(t_tex).strip() + "\n</theorems>\n"
            "<solution>\n" + str(s_tex).strip() + "\n</solution>"
        )
        return wrapped


    async def _spdf_llm_generate(self, provider_id: str, prompt: str, image_urls: Optional[List[str]] = None) -> str:
        """带并发限制的 llm_generate，返回 completion_text。"""
        async with self._spdf_solver_sema:
            llm_resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                image_urls=image_urls if image_urls else [],
            )
        raw = (
            getattr(llm_resp, "completion_text", None)
            or getattr(llm_resp, "completion", None)
            or getattr(llm_resp, "text", None)
            or ""
        )
        return (raw or "").strip()


    def _spdf_unique_provider_ids(self, ids: List[str]) -> List[str]:
        """去重并清洗 provider_id 列表（保持顺序）。"""
        out: List[str] = []
        seen = set()
        for pid in (ids or []):
            p = str(pid or "").strip()
            if not p or p in seen:
                continue
            seen.add(p)
            out.append(p)
        return out

    async def _spdf_llm_generate_with_fallback(
        self,
        provider_ids: List[str],
        prompt: str,
        image_urls: Optional[List[str]] = None,
        purpose: str = "",
    ):
        """按顺序尝试多个 provider_id，直到成功返回非空文本。返回 (text, used_provider_id)。"""
        pids = self._spdf_unique_provider_ids(provider_ids)
        last_exc: Optional[Exception] = None
        for pid in pids:
            try:
                txt = await self._spdf_llm_generate(pid, prompt, image_urls)
                if txt and str(txt).strip():
                    return str(txt).strip(), pid
                last_exc = RuntimeError("empty_completion")
            except Exception as e:
                last_exc = e
                # 记录但不中断：尝试下一个 provider（常见：鉴权缺失/上游工具调用解析失败/限流）
                try:
                    if purpose:
                        logger.warning(f"spdf llm_generate failed ({purpose}) pid={pid}: {e}")
                except Exception:
                    pass
                continue
        if last_exc:
            raise last_exc
        raise RuntimeError("llm_generate failed")


    async def _spdf_generate_candidates(
        self,
        problem_text: str,
        has_image: bool,
        image_urls: Optional[List[str]],
        ref_pdf_latex: str,
        base_prompt: str,
        solver_defs: List[Dict[str, str]],
        solver_provider_ids: List[str],
        fallback_provider_id: str,
        round_idx: int,
        feedback: str,
        best_raw_hint: str,
    ) -> List[Dict[str, Any]]:
        """生成多份候选解（候选=可解析的 <problem>/<theorems>/<solution>）。"""
        tasks = []

        for i, role in enumerate(solver_defs):
            pid = solver_provider_ids[i]
            role_header = (
                "你将作为一个数学解题系统中的一个专门角色独立工作。\n"
                f"【你的角色】{role.get('name','Solver')}：{role.get('desc','')}\n"
                "你需要给出尽可能正确、严谨、完整的推导。\n"
                "注意：你只需要输出可编译的 LaTeX 正文片段，必须包含 <problem>、<theorems>、<solution> 三段。\n"
            )

            extra = ""
            if round_idx > 0 and (feedback or best_raw_hint):
                extra = (
                    "\n【上一轮评审反馈（请重点修复）】\n"
                    f"{feedback.strip()}\n"
                    "\n【上一轮最佳候选参考（可参考但不要照抄；请修复其问题并更严谨）】\n"
                    f"{self._spdf_trunc(best_raw_hint, 8000)}\n"
                )

            prompt = role_header + "\n" + base_prompt + extra + "\n" + (
                "\n输出格式要求：\n"
                "1) 必须输出 <problem>...</problem><theorems>...</theorems><solution>...</solution>\n"
                "2) 不要输出 Markdown 代码块（```）\n"
                "3) <solution> 最后必须以：\\hfill$\\blacksquare$ 结束，且后面不要再有任何文字。\n"
            )

            tasks.append(
                self._spdf_llm_generate_with_fallback(
                    [pid, fallback_provider_id],
                    prompt,
                    image_urls,
                    purpose=f"solver_round{round_idx}:{role.get('name','')}",
                )
            )

        raws = await asyncio.gather(*tasks, return_exceptions=True)

        candidates: List[Dict[str, Any]] = []
        for i, r in enumerate(raws):
            role = solver_defs[i]
            pid = solver_provider_ids[i]
            used_pid = pid
            raw = ""
            if isinstance(r, Exception):
                raw = f"<problem>{_escape_text_preserve_dollar_math(problem_text)}</problem><theorems></theorems><solution>（该候选生成失败：{_escape_latex_text_strict(str(r))}）\\hfill$\\blacksquare$</solution>"
            else:
                # _spdf_llm_generate_with_fallback 返回 (text, used_provider_id)
                try:
                    if isinstance(r, (tuple, list)) and len(r) >= 2:
                        raw = _strip_all_code_fences(str((r[0] or '')).strip())
                        used_pid = str(r[1] or pid).strip() or pid
                    else:
                        raw = _strip_all_code_fences(str(r or "").strip())
                except Exception:
                    raw = _strip_all_code_fences(str(r or "").strip())
            pid = used_pid

            norm_items = []
            multi_items = _parse_latex_sections_multi(raw)
            is_multi = len(multi_items) > 1
            if is_multi:
                norm_items = _normalize_multi_pdf_items(multi_items, problem_text=problem_text, has_image=has_image)
                problem_tex = "\n\n".join([x[0] for x in norm_items])
                theorems_tex = "\n\n".join([x[1] for x in norm_items])
                solution_tex = "\n\n".join([x[2] for x in norm_items])
            else:
                problem_tex, theorems_tex, solution_tex = _parse_latex_sections(raw)
            # 兜底逻辑：标签缺失/截断
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
                    problem_tex = r"(模型未返回题目识别结果，请检查)"
                else:
                    problem_tex = _escape_text_preserve_dollar_math(problem_text)
            if not theorems_tex:
                theorems_tex = (
                    r"\begin{theoremBox}{常用结论}" + "\n"
                    r"本题仅用到基础运算与常用恒等变形" + "\n"
                    r"\end{theoremBox}"
                )
            if not solution_tex:
                solution_tex = r"\textbf{解：}" + "\n" + r"(解答生成失败，请重试)" + "\n" + r"\hfill$\blacksquare$"

            # sanitize
            problem_tex = _sanitize_latex_fragment_for_xelatex(problem_tex)
            theorems_tex = _sanitize_latex_fragment_for_xelatex(theorems_tex)
            solution_tex = _sanitize_latex_fragment_for_xelatex(solution_tex)

            candidates.append(
                {
                    "id": f"r{round_idx+1}_c{i+1}",
                    "role": role.get("name", f"Solver{i+1}"),
                    "provider_id": pid,
                    "raw": raw,
                    "problem_tex": problem_tex,
                    "theorems_tex": theorems_tex,
                    "solution_tex": solution_tex,
                }
            )

        return candidates

    async def _spdf_cross_examine(
        self,
        problem_text: str,
        candidates: List[Dict[str, Any]],
        image_urls: Optional[List[str]],
        judge_fallback_provider_id: str,
    ) -> Dict[str, List[str]]:
        """交叉质询：每个 solver 挑别的候选的错（可限制长度）。"""
        if not bool(self._cfg("spdf_enable_cross_exam", True)):
            return {}

        cross_pid = str(self._cfg("spdf_cross_exam_provider_id", "") or "").strip()
        use_single = bool(cross_pid)

        critiques: Dict[str, List[str]] = {c["id"]: [] for c in candidates}

        tasks = []
        task_meta = []

        for a in candidates:
            for b in candidates:
                if a["id"] == b["id"]:
                    continue

                pid = cross_pid if use_single else (a.get("provider_id") or judge_fallback_provider_id)
                prompt = (
                    "你是一名严苛的数学审稿人，负责对候选解进行“挑错”。\n"
                    "请只输出要点列表（每点尽量短），指出候选解中可能的：逻辑漏洞、偷步、条件遗漏、结论错误、符号不一致、边界情况问题。\n"
                    "如果你认为候选解完全正确，也要给出“你检查过哪些点”。\n"
                    "不要给出新的完整解答，只做挑错/质询。\n"
                    f"\n【题目】\n{problem_text}\n"
                    f"\n【被质询候选 {b['id']}】\n"
                    f"<theorems>\n{self._spdf_trunc(b.get('theorems_tex',''), 3500)}\n</theorems>\n"
                    f"<solution>\n{self._spdf_trunc(b.get('solution_tex',''), 9000)}\n</solution>\n"
                    "\n请输出 5~12 条要点（可以用 - 开头）。"
                )
                tasks.append(self._spdf_llm_generate(pid, prompt, image_urls))
                task_meta.append((a["id"], b["id"]))

        raws = await asyncio.gather(*tasks, return_exceptions=True)
        for meta, r in zip(task_meta, raws):
            a_id, b_id = meta
            if isinstance(r, Exception):
                continue
            txt = _strip_all_code_fences(str(r or "").strip())
            txt = txt.strip()
            if not txt:
                continue
            critiques[b_id].append(f"[来自 {a_id} 的质询]\n{txt}")

        return critiques

    async def _spdf_vote_top(
        self,
        problem_text: str,
        candidates: List[Dict[str, Any]],
        top_ids: List[str],
        image_urls: Optional[List[str]],
    ) -> Dict[str, Any]:
        """自一致投票：每个 solver 在 top 候选中投票。"""
        if (not bool(self._cfg("spdf_enable_self_consistency_vote", True))) or (not top_ids) or (len(top_ids) <= 1):
            return {"enabled": False, "votes": {}, "winner": (top_ids[0] if top_ids else "")}

        top_map = {c["id"]: c for c in candidates if c["id"] in set(top_ids)}
        if len(top_map) <= 1:
            return {"enabled": False, "votes": {}, "winner": (top_ids[0] if top_ids else "")}

        summaries = []
        for cid in top_ids:
            c = top_map.get(cid)
            if not c:
                continue
            summaries.append(
                f"== 候选 {cid} ==\n"
                f"{self._spdf_trunc(c.get('solution_tex',''), 3500)}\n"
            )
        summaries_txt = "\n".join(summaries)

        tasks = []
        for c in candidates:
            pid = c.get("provider_id", "")
            if not pid:
                continue
            prompt = (
                "你是一个数学解题评审。\n"
                "现在有多个候选解（只展示解答摘要）。请选择你认为最可能正确的一份。\n"
                "你必须只输出 <tool_raw>JSON</tool_raw>，除此之外不要输出任何文本。\n注意：不要调用任何工具/函数（不要输出 tool_calls/function_call），把 JSON 当作普通文本放在 <tool_raw> 标签里。\n"
                "JSON 格式：{\"vote\": \"候选id\", \"reason\": \"一句话理由\"}\n"
                f"\n【题目】\n{problem_text}\n"
                f"\n【候选摘要】\n{summaries_txt}\n"
            )
            tasks.append(self._spdf_llm_generate(pid, prompt, image_urls))

        raws = await asyncio.gather(*tasks, return_exceptions=True)
        votes = {}
        for r in raws:
            if isinstance(r, Exception):
                continue
            txt = _strip_all_code_fences(str(r or "").strip())
            j = _safe_json_loads(_extract_tool_raw_block(txt) or "")
            if not isinstance(j, dict):
                continue
            v = str(j.get("vote", "") or "").strip()
            if v in top_map:
                votes[v] = votes.get(v, 0) + 1

        winner = ""
        if votes:
            winner = sorted(votes.items(), key=lambda x: (-x[1], x[0]))[0][0]
        else:
            winner = top_ids[0]

        return {"enabled": True, "votes": votes, "winner": winner}

    def _spdf_build_judge_prompt(
        self,
        problem_text: str,
        candidates: List[Dict[str, Any]],
        critiques: Dict[str, List[str]],
        char_budget: int,
    ) -> str:
        parts = []
        parts.append("你是数学解题系统的总评审（Judge）。你的目标是从多个候选解中选出最可靠的版本，必要时给出修复指令。\n")
        parts.append("你必须只输出 <tool_raw>JSON</tool_raw>，除此之外不要输出任何文本。\n注意：不要调用任何工具/函数（不要输出 tool_calls/function_call），把 JSON 当作普通文本放在 <tool_raw> 标签里。\n")
        parts.append("JSON 格式必须严格遵循：\n")
        parts.append(
            "{\n"
            "  \"scores\": [\n"
            "    {\"id\": \"...\", \"score\": 0-10, \"issues\": [\"...\"], \"key_claims\": [\"...\"],\n"
            "     \"python_checks\": [{\"desc\": \"...\", \"code\": \"...\"}] }\n"
            "  ],\n"
            "  \"top_ids\": [\"...\"],\n"
            "  \"need_revision\": true/false,\n"
            "  \"revision_instructions\": \"...\"\n"
            "}\n"
        )
        parts.append("打分标准：正确性>完整性>表述清晰；若发现明显错误，分数应显著降低。\n")
        parts.append("python_checks 只写“关键可验算点”，每个候选最多 3 条，代码必须短且只用标准库（math / fractions / decimal 可用）。\n")
        parts.append("\n【题目】\n" + str(problem_text) + "\n")

        per = max(2000, char_budget // max(1, len(candidates)))

        for c in candidates:
            cid = c.get("id", "")
            sol = self._spdf_trunc(c.get("solution_tex", ""), per)
            thm = self._spdf_trunc(c.get("theorems_tex", ""), 1200)
            crit = "\n".join(critiques.get(cid, [])[:4])
            crit = self._spdf_trunc(crit, 2000)
            parts.append(
                "\n" + "=" * 18 + f"\n候选 {cid}（{c.get('role','')}）\n"
                + "<theorems>\n" + thm + "\n</theorems>\n"
                + "<solution>\n" + sol + "\n</solution>\n"
            )
            if crit:
                parts.append("\n[交叉质询摘要]\n" + crit + "\n")

        return "".join(parts)

    def _spdf_build_judge_refine_prompt(
        self,
        judge_json: Dict[str, Any],
        python_results: Dict[str, Any],
    ) -> str:
        return (
            "你是数学解题系统的总评审（Judge）。现在你已经给出了初步评分与 Python 验算脚本。\n"
            "下面是沙盒执行结果，请你据此调整判断。\n"
            "你必须输出两部分：\n注意：不要调用任何工具/函数（不要输出 tool_calls/function_call），把 JSON 当作普通文本放在 <tool_raw> 标签里。\n"
            "1) <tool_raw>JSON</tool_raw>：{\"winner_id\":\"...\",\"reason\":\"...\"}\n"
            "2) 最终的 LaTeX 正文片段：必须包含 <problem>、<theorems>、<solution> 三段。\n"
            "除此之外不要输出任何文本。\n"
            "\n[你之前的评审 JSON]\n"
            f"{json.dumps(judge_json, ensure_ascii=False)}\n"
            "\n[Python 验算结果]\n"
            f"{json.dumps(python_results, ensure_ascii=False)}\n"
            "\n要求：\n"
            "- 如果 Python 验算显示某候选关键结论不成立，请明确降低它的可信度。\n"
            "- 你可以综合多候选并重写最终解答，但必须保证步骤完整且结尾以 \\hfill$\\blacksquare$ 收尾。\n"
        )

    def _spdf_validate_python_code(self, code: str) -> Optional[str]:
        """返回 None 表示通过；否则返回错误原因。"""
        if not isinstance(code, str) or not code.strip():
            return "empty"
        if len(code) > 2000:
            return "too_long"

        banned = ["import os", "import sys", "subprocess", "socket", "pathlib", "shutil", "open(", "__import__", "eval(", "exec(", "compile("]
        low = code.lower()
        for b in banned:
            if b in low:
                return f"banned_token:{b}"

        try:
            tree = ast.parse(code, mode="exec")
        except Exception as e:
            return f"parse_error:{e}"

        allow_mods = {"math", "fractions", "decimal"}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = (alias.name or "").split(".")[0]
                    if name and (name not in allow_mods):
                        return f"banned_import:{name}"
            if isinstance(node, ast.Attribute):
                if isinstance(node.attr, str) and node.attr.startswith("__"):
                    return "dunder_attr"
            if isinstance(node, ast.Name):
                if node.id in {"__builtins__", "__loader__", "__spec__"}:
                    return "banned_name"
        return None

    def _spdf_run_python_code_sync(self, code: str, timeout_sec: int) -> Dict[str, Any]:
        """阻塞运行：子进程沙盒执行。"""
        reason = self._spdf_validate_python_code(code)
        if reason:
            return {"ok": False, "error": f"invalid_code:{reason}", "stdout": "", "stderr": ""}

        prelude = (
            "import math\n"
            "from fractions import Fraction\n"
            "from decimal import Decimal, getcontext\n"
            "getcontext().prec = 80\n"
        )
        full = prelude + "\n" + code.strip() + "\n"
        fp = ""
        try:
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py", encoding="utf-8") as f:
                f.write(full)
                fp = f.name
            p = subprocess.run(
                [sys.executable, "-I", fp],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=max(1, int(timeout_sec)),
                check=False,
                text=True,
            )
            return {
                "ok": p.returncode == 0,
                "returncode": p.returncode,
                "stdout": (p.stdout or "")[-2000:],
                "stderr": (p.stderr or "")[-2000:],
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "timeout", "stdout": "", "stderr": ""}
        except Exception as e:
            return {"ok": False, "error": str(e), "stdout": "", "stderr": ""}
        finally:
            try:
                if fp:
                    os.remove(fp)
            except Exception:
                pass

    async def _spdf_run_python_checks(self, judge_json: Dict[str, Any]) -> Dict[str, Any]:
        """执行 judge 给出的 python_checks，并返回结构化结果。"""
        if not bool(self._cfg("spdf_enable_python_check", True)):
            return {"enabled": False}

        timeout_sec = int(self._cfg("spdf_python_check_timeout_sec", 6) or 6)
        max_checks = int(self._cfg("spdf_python_check_max_checks_per_candidate", 3) or 3)

        scores = judge_json.get("scores", [])
        if not isinstance(scores, list):
            return {"enabled": True, "results": {}}

        loop = asyncio.get_running_loop()
        results: Dict[str, Any] = {}
        for item in scores:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("id", "") or "").strip()
            checks = item.get("python_checks", [])
            if not cid or not isinstance(checks, list):
                continue
            checks = checks[:max(0, max_checks)]
            cand_res = []
            for ck in checks:
                if not isinstance(ck, dict):
                    continue
                desc = str(ck.get("desc", "") or "").strip()
                code = str(ck.get("code", "") or "").strip()
                if not code:
                    continue
                r = await loop.run_in_executor(None, lambda c=code: self._spdf_run_python_code_sync(c, timeout_sec))
                cand_res.append({"desc": desc, "code": self._spdf_trunc(code, 800), "result": r})
            results[cid] = cand_res
        return {"enabled": True, "results": results}

    def _spdf_extract_tikz_blocks(self, tex: str) -> List[str]:
        if not tex:
            return []
        pat = re.compile(r"\\begin\{tikzpicture\}(?:\[[\s\S]*?\])?[\s\S]*?\\end\{tikzpicture\}", re.M)
        return [m.group(0) for m in pat.finditer(tex)]

    def _spdf_build_tikz_min_doc(self, tikz_code: str) -> str:
        return "\n".join(
            [
                r"\documentclass[UTF8]{ctexart}",
                r"\usepackage{amsmath,amssymb}",
                r"\usepackage{tikz}",
                r"\usepackage{pgfplots}",
                r"\pgfplotsset{compat=1.18}",
                r"\usetikzlibrary{arrows.meta, calc, positioning, shapes, intersections, decorations.pathreplacing, decorations.markings, patterns, scopes, backgrounds}",
                r"\pgfdeclarelayer{bg}",
                r"\pgfsetlayers{bg,main}",
                r"\begin{document}",
                tikz_code,
                r"\end{document}",
            ]
        )

    async def _spdf_tikz_preflight(self, tex_fragment: str, provider_id: str) -> str:
        """对 tex 片段中的每个 tikzpicture 做最小编译测试；失败则尝试修复。"""
        if (not bool(self._cfg("spdf_enable_tikz_preflight", True))) or (not tex_fragment):
            return tex_fragment

        blocks = self._spdf_extract_tikz_blocks(tex_fragment)
        if not blocks:
            return tex_fragment

        fix_pid = str(self._cfg("spdf_tikz_preflight_provider_id", "") or "").strip() or provider_id
        max_rounds = int(self._cfg("spdf_tikz_preflight_max_rounds", 2) or 2)

        new_fragment = tex_fragment
        for blk in blocks:
            cur = blk
            ok = False
            for _ in range(max_rounds + 1):
                mini = self._spdf_build_tikz_min_doc(cur)
                backup_log = self._last_texlive_log
                pdf = await self._compile_local_xelatex(mini)
                log_txt = self._last_texlive_log
                self._last_texlive_log = backup_log

                if pdf:
                    ok = True
                    break

                log_tail = (log_txt or "")[-2500:]
                prompt = (
                    "你是 TikZ 代码修复器。下面这段 tikzpicture 片段无法通过 xelatex 编译。\n"
                    "请根据编译日志修复它，并只输出一段完整的 tikzpicture 环境：\\begin{tikzpicture}...\\end{tikzpicture}。\n"
                    "不要输出 Markdown 代码块，不要输出多余解释。\n"
                    "\n[编译日志末尾]\n"
                    f"{log_tail}\n"
                    "\n[原始 TikZ]\n"
                    f"{cur}\n"
                )
                try:
                    fixed = await self._spdf_llm_generate(fix_pid, prompt, None)
                except Exception:
                    break
                fixed = _strip_all_code_fences(fixed).strip()
                m = re.search(r"\\begin\{tikzpicture\}[\s\S]*?\\end\{tikzpicture\}", fixed, flags=re.M)
                if m:
                    cur = m.group(0)
                else:
                    break

            if ok and cur != blk:
                new_fragment = new_fragment.replace(blk, cur, 1)

        return new_fragment

    async def _spdf_generate_latex_deepthink(
        self,
        problem_text: str,
        event: AstrMessageEvent,
        image_urls: Optional[List[str]],
        ref_pdf_latex: str,
        current_provider_id: str,
        base_provider_id: str,
    ) -> str:
        """DeepThink 主循环：多候选 -> 质询 -> Judge 评分+验算脚本 -> Python 执行 -> Judge 输出最终 LaTeX。"""
        has_image = bool(image_urls)
        ref_pdf_latex = (ref_pdf_latex or "").strip()

        if ref_pdf_latex and (not has_image):
            base_prompt = self._build_pdf_followup_prompt(
                question_text=problem_text,
                ref_pdf_latex=ref_pdf_latex,
            )
        else:
            base_prompt = self._build_pdf_latex_prompt(problem_text, has_image=has_image)

        pool = self._spdf_parse_provider_pool(str(self._cfg("spdf_role_pool_provider_ids", "") or ""))
        if not pool:
            pool = [base_provider_id] if base_provider_id else ([current_provider_id] if current_provider_id else [])
        if not pool:
            raise RuntimeError("未能获取 chat_provider_id")

        n = int(self._cfg("spdf_num_solvers", 3) or 3)
        role_defs_full = self._spdf_role_defs()
        role_defs = role_defs_full[: max(1, min(n, len(role_defs_full)))]
        n = len(role_defs)
        solver_provider_ids = [pool[i % len(pool)] for i in range(n)]

        judge_pid = str(self._cfg("spdf_judge_provider_id", "") or "").strip() or base_provider_id or current_provider_id or pool[0]

        best_raw_hint = ""
        feedback = ""
        final_raw = ""

        iters = int(self._cfg("spdf_iter_rounds", 2) or 2)
        iters = max(1, min(8, iters))

        for round_idx in range(iters):
            candidates = await self._spdf_generate_candidates(
                problem_text=problem_text,
                has_image=has_image,
                image_urls=image_urls,
                ref_pdf_latex=ref_pdf_latex,
                base_prompt=base_prompt,
                solver_defs=role_defs,
                solver_provider_ids=solver_provider_ids,
                fallback_provider_id=judge_pid,
                round_idx=round_idx,
                feedback=feedback,
                best_raw_hint=best_raw_hint,
            )

            critiques = await self._spdf_cross_examine(
                problem_text=problem_text,
                candidates=candidates,
                image_urls=image_urls,
                judge_fallback_provider_id=judge_pid,
            )

            judge_prompt = self._spdf_build_judge_prompt(
                problem_text=problem_text,
                candidates=candidates,
                critiques=critiques,
                char_budget=int(self._cfg("spdf_candidate_char_budget", 14000) or 14000),
            )

            # Judge：优先使用 spdf_judge_provider_id；若鉴权/解析失败则回落到基础模型/池内其它模型
            try:
                judge_pids = self._spdf_unique_provider_ids([judge_pid, base_provider_id, current_provider_id] + pool)
                judge_txt, used_judge_pid = await self._spdf_llm_generate_with_fallback(
                    judge_pids,
                    judge_prompt,
                    image_urls,
                    purpose=f"judge_round{round_idx}",
                )
                if used_judge_pid:
                    judge_pid = used_judge_pid
            except Exception as e:
                logger.warning(f"spdf judge failed, fallback to first candidate: {e}")
                final_raw = candidates[0]["raw"] if candidates else ""
                break
            judge_txt = _strip_all_code_fences(judge_txt).strip()
            judge_json = _safe_json_loads(_extract_tool_raw_block(judge_txt) or "")
            if not isinstance(judge_json, dict):
                final_raw = candidates[0]["raw"]
                break

            top_ids = judge_json.get("top_ids", [])
            if not isinstance(top_ids, list) or not top_ids:
                scores = judge_json.get("scores", [])
                if isinstance(scores, list):
                    tmp = []
                    for it in scores:
                        if isinstance(it, dict) and it.get("id"):
                            try:
                                sc = float(it.get("score", 0))
                            except Exception:
                                sc = 0.0
                            tmp.append((sc, str(it.get("id"))))
                    tmp.sort(key=lambda x: -x[0])
                    top_ids = [x[1] for x in tmp[: max(1, min(2, len(tmp)))]]
                if not top_ids and candidates:
                    top_ids = [candidates[0]["id"]]

            vote_info = await self._spdf_vote_top(problem_text, candidates, top_ids, image_urls)

            py_res = await self._spdf_run_python_checks(judge_json)

            python_results_payload = {"vote": vote_info, "python": py_res}
            refine_prompt = self._spdf_build_judge_refine_prompt(judge_json, python_results_payload)

            try:
                refine_pids = self._spdf_unique_provider_ids([judge_pid, base_provider_id, current_provider_id] + pool)
                refine_txt, used_judge_pid2 = await self._spdf_llm_generate_with_fallback(
                    refine_pids,
                    refine_prompt,
                    image_urls,
                    purpose=f"judge_refine_round{round_idx}",
                )
                if used_judge_pid2:
                    judge_pid = used_judge_pid2
            except Exception as e:
                logger.warning(f"spdf judge refine failed, fallback to first candidate: {e}")
                final_raw = candidates[0]["raw"] if candidates else ""
                break
            refine_txt = _strip_all_code_fences(refine_txt).strip()

            final_raw = refine_txt

            feedback = str(judge_json.get("revision_instructions", "") or "").strip()
            best_raw_hint = final_raw

            need_rev = bool(judge_json.get("need_revision", False))
            if (not need_rev) or (round_idx >= iters - 1):
                break

        return final_raw

    async def _solve_math_to_spdf(
        self,
        problem_text: str,
        event: AstrMessageEvent,
        image_urls: Optional[List[str]] = None,
        ref_pdf_latex: str = "",
    ):
        """/spdf：DeepThink 多角色迭代版 —— 最终仍输出 PDF（本地 xelatex 编译）。"""
        provider_id = str(self._cfg("spdf_provider_id", "") or "").strip()

        current_provider_id = ""
        try:
            current_provider_id = await self.context.get_current_chat_provider_id(umo=event.unified_msg_origin)
        except Exception:
            current_provider_id = ""

        has_image = bool(image_urls)
        ref_pdf_latex = (ref_pdf_latex or "").strip()

        if not provider_id:
            provider_id = current_provider_id

        if not provider_id:
            raise RuntimeError("未能获取 chat_provider_id")

        raw = await self._spdf_generate_latex_deepthink(
            problem_text=problem_text,
            event=event,
            image_urls=image_urls,
            ref_pdf_latex=ref_pdf_latex,
            current_provider_id=current_provider_id,
            base_provider_id=provider_id,
        )

        raw = (raw or "").strip()
        raw = _strip_all_code_fences(raw)
        # 可选：DeepThink 输出后处理（纠正全英文/标签不规范，转为 PDF 标准格式）
        try:
            raw = await self._spdf_post_format_final_output(
                raw=raw,
                problem_text=problem_text,
                has_image=has_image,
                image_urls=image_urls,
                ref_pdf_latex=ref_pdf_latex,
                fallback_provider_id=(str(self._cfg("spdf_judge_provider_id", "") or "").strip() or provider_id or current_provider_id),
            )
        except Exception as e:
            logger.warning(f"spdf post formatter failed: {e}")

        norm_items = []
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
            if 'is_multi' in locals() and is_multi and 'norm_items' in locals() and norm_items:
                _p, _t, _s = norm_items[-1]
                _s = (_s or "")
                _s += (
                    r"\vspace{1em}" + "\n"
                    r"\begin{tcolorbox}[colback=red!5!white,colframe=red!75!black,title={生成中断警告}]" + "\n"
                    r"由于模型输出被提前中断，解答过程可能不完整。你可以把题目拆成更小的问题，或让机器人继续补全上一份解答。" + "\n"
                    r"\end{tcolorbox}"
                )
                norm_items[-1] = (_p, _t, _s)
                solution_tex = "\n\n".join([x[2] for x in norm_items])
            else:
                solution_tex = (solution_tex or "")
                solution_tex += (
                    r"\vspace{1em}" + "\n"
                    r"\begin{tcolorbox}[colback=red!5!white,colframe=red!75!black,title={生成中断警告}]" + "\n"
                    r"由于模型输出被提前中断，解答过程可能不完整。你可以把题目拆成更小的问题，或让机器人继续补全上一份解答。" + "\n"
                    r"\end{tcolorbox}"
                )

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
            solution_tex = r"\textbf{解：}" + "\n" + r"(解答生成失败，请重试)" + "\n" + r"\hfill$\blacksquare$"

        problem_tex = _sanitize_latex_fragment_for_xelatex(problem_tex)
        theorems_tex = _sanitize_latex_fragment_for_xelatex(theorems_tex)
        solution_tex = _sanitize_latex_fragment_for_xelatex(solution_tex)

        try:
            if 'is_multi' in locals() and is_multi and 'norm_items' in locals() and norm_items:
                _new_items = []
                for _p, _t, _s in norm_items:
                    _s2 = await self._spdf_tikz_preflight(_s, provider_id=provider_id)
                    _t2 = await self._spdf_tikz_preflight(_t, provider_id=provider_id)
                    _new_items.append((_p, _t2, _s2))
                norm_items = _new_items
                problem_tex = "\n\n".join([x[0] for x in norm_items])
                theorems_tex = "\n\n".join([x[1] for x in norm_items])
                solution_tex = "\n\n".join([x[2] for x in norm_items])
            else:
                solution_tex = await self._spdf_tikz_preflight(solution_tex, provider_id=provider_id)
                theorems_tex = await self._spdf_tikz_preflight(theorems_tex, provider_id=provider_id)
        except Exception:
            pass

        tex_src = _build_pdf_latex_document_multi(norm_items) if is_multi else _build_pdf_latex_document(problem_tex, theorems_tex, solution_tex)
        pdf_bytes: Optional[bytes] = None
        compile_guard_enabled = bool(self._cfg("pdf_enable_compile_guard", False))
        completeness_guard_enabled = bool(self._cfg("pdf_enable_completeness_guard", False))
        if 'is_multi' in locals() and is_multi:
            completeness_guard_enabled = False


        compile_max_rounds = int(self._cfg("pdf_guard_max_rounds", 3) or 3)
        completeness_max_rounds = int(self._cfg("pdf_completeness_guard_max_rounds", 2) or 2)

        overall_max_rounds = max(
            compile_max_rounds if compile_guard_enabled else 0,
            completeness_max_rounds if completeness_guard_enabled else 0,
        )

        if compile_guard_enabled or completeness_guard_enabled:
            for round_idx in range(overall_max_rounds + 1):
                pdf_bytes = await self._compile_tex_to_pdf(tex_src)
                compile_ok = bool(pdf_bytes)

                if compile_guard_enabled:
                    judge_pass = await self._pdf_guard_judge(
                        compile_ok=compile_ok,
                        log_text=(self._last_texlive_log or ""),
                        tex_src=tex_src,
                        preferred_provider_id=str(self._cfg("pdf_guard_provider_id", "") or "").strip(),
                        fallback_provider_id=(current_provider_id or provider_id or "").strip(),
                    )

                    if (not compile_ok) or (not judge_pass):
                        if round_idx >= overall_max_rounds:
                            break

                        repair_prompt = self._build_pdf_latex_repair_prompt(
                            problem_text=problem_text,
                            has_image=has_image,
                            compile_log=(self._last_texlive_log or ""),
                            prev_tex_src=tex_src,
                            prev_raw=raw,
                        )

                        llm_resp = await self.context.llm_generate(
                            chat_provider_id=provider_id,
                            prompt=repair_prompt,
                            image_urls=image_urls if image_urls else [],
                        )

                        raw = (
                            getattr(llm_resp, "completion_text", None)
                            or getattr(llm_resp, "completion", None)
                            or getattr(llm_resp, "text", None)
                            or ""
                        )
                        raw = (raw or "").strip()
                        raw = _strip_all_code_fences(raw)

                        # 可选：DeepThink 输出后处理（纠正全英文/标签不规范，转为 PDF 标准格式）
                        try:
                            raw = await self._spdf_post_format_final_output(
                                raw=raw,
                                problem_text=problem_text,
                                has_image=has_image,
                                image_urls=image_urls,
                                ref_pdf_latex=ref_pdf_latex,
                                fallback_provider_id=(str(self._cfg("spdf_judge_provider_id", "") or "").strip() or provider_id or current_provider_id),
                            )
                        except Exception as e:
                            logger.warning(f"spdf post formatter failed: {e}")


                        norm_items = []
                        multi_items = _parse_latex_sections_multi(raw)
                        is_multi = len(multi_items) > 1
                        if is_multi:
                            norm_items = _normalize_multi_pdf_items(multi_items, problem_text=problem_text, has_image=has_image)
                            problem_tex = "\n\n".join([x[0] for x in norm_items])
                            theorems_tex = "\n\n".join([x[1] for x in norm_items])
                            solution_tex = "\n\n".join([x[2] for x in norm_items])
                        else:
                            problem_tex, theorems_tex, solution_tex = _parse_latex_sections(raw)
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
                            solution_tex = r"\textbf{解：}" + "\n" + r"(解答生成失败，请重试)" + "\n" + r"\hfill$\blacksquare$"

                        problem_tex = _sanitize_latex_fragment_for_xelatex(problem_tex)
                        theorems_tex = _sanitize_latex_fragment_for_xelatex(theorems_tex)
                        solution_tex = _sanitize_latex_fragment_for_xelatex(solution_tex)

                        try:
                            if 'is_multi' in locals() and is_multi and 'norm_items' in locals() and norm_items:
                                _new_items = []
                                for _p, _t, _s in norm_items:
                                    _s2 = await self._spdf_tikz_preflight(_s, provider_id=provider_id)
                                    _t2 = await self._spdf_tikz_preflight(_t, provider_id=provider_id)
                                    _new_items.append((_p, _t2, _s2))
                                norm_items = _new_items
                                problem_tex = "\n\n".join([x[0] for x in norm_items])
                                theorems_tex = "\n\n".join([x[1] for x in norm_items])
                                solution_tex = "\n\n".join([x[2] for x in norm_items])
                            else:
                                solution_tex = await self._spdf_tikz_preflight(solution_tex, provider_id=provider_id)
                                theorems_tex = await self._spdf_tikz_preflight(theorems_tex, provider_id=provider_id)
                        except Exception:
                            pass

                        tex_src = _build_pdf_latex_document_multi(norm_items) if is_multi else _build_pdf_latex_document(problem_tex, theorems_tex, solution_tex)
                        continue

                if not compile_ok and not compile_guard_enabled:
                    break

                if compile_ok and completeness_guard_enabled:
                    comp_pass, comp_issues = await self._pdf_completeness_guard_judge(
                        problem_text=problem_text,
                        problem_tex=problem_tex,
                        solution_tex=solution_tex,
                        preferred_provider_id=str(self._cfg("pdf_completeness_guard_provider_id", "") or "").strip(),
                        fallback_provider_id=(current_provider_id or provider_id or "").strip(),
                    )

                    if comp_pass:
                        break

                    if round_idx >= overall_max_rounds:
                        break

                    repair_prompt = self._build_pdf_incomplete_repair_prompt(
                        problem_text=problem_text,
                        has_image=has_image,
                        issues=comp_issues,
                        prev_tex_src=tex_src,
                        prev_raw=raw,
                    )

                    llm_resp = await self.context.llm_generate(
                        chat_provider_id=provider_id,
                        prompt=repair_prompt,
                        image_urls=image_urls if image_urls else [],
                    )

                    raw = (
                        getattr(llm_resp, "completion_text", None)
                        or getattr(llm_resp, "completion", None)
                        or getattr(llm_resp, "text", None)
                        or ""
                    )
                    raw = (raw or "").strip()
                    raw = _strip_all_code_fences(raw)

                    # 可选：DeepThink 输出后处理（纠正全英文/标签不规范，转为 PDF 标准格式）
                    try:
                        raw = await self._spdf_post_format_final_output(
                            raw=raw,
                            problem_text=problem_text,
                            has_image=has_image,
                            image_urls=image_urls,
                            ref_pdf_latex=ref_pdf_latex,
                            fallback_provider_id=(str(self._cfg("spdf_judge_provider_id", "") or "").strip() or provider_id or current_provider_id),
                        )
                    except Exception as e:
                        logger.warning(f"spdf post formatter failed: {e}")


                    norm_items = []
                    multi_items = _parse_latex_sections_multi(raw)
                    is_multi = len(multi_items) > 1
                    if is_multi:
                        norm_items = _normalize_multi_pdf_items(multi_items, problem_text=problem_text, has_image=has_image)
                        problem_tex = "\n\n".join([x[0] for x in norm_items])
                        theorems_tex = "\n\n".join([x[1] for x in norm_items])
                        solution_tex = "\n\n".join([x[2] for x in norm_items])
                    else:
                        problem_tex, theorems_tex, solution_tex = _parse_latex_sections(raw)
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
                            problem_tex = r"(模型未返回题目识别结果，请检查)"
                        else:
                            problem_tex = _escape_text_preserve_dollar_math(problem_text)

                    if not theorems_tex:
                        theorems_tex = (
                            r"\begin{theoremBox}{常用结论}" + "\n"
                            r"本题仅用到基础运算与常用恒等变形" + "\n"
                            r"\end{theoremBox}"
                        )
                    if not solution_tex:
                        solution_tex = r"\textbf{解：}" + "\n" + r"(解答生成失败，请重试)" + "\n" + r"\hfill$\blacksquare$"

                    problem_tex = _sanitize_latex_fragment_for_xelatex(problem_tex)
                    theorems_tex = _sanitize_latex_fragment_for_xelatex(theorems_tex)
                    solution_tex = _sanitize_latex_fragment_for_xelatex(solution_tex)

                    try:
                        if 'is_multi' in locals() and is_multi and 'norm_items' in locals() and norm_items:
                            _new_items = []
                            for _p, _t, _s in norm_items:
                                _s2 = await self._spdf_tikz_preflight(_s, provider_id=provider_id)
                                _t2 = await self._spdf_tikz_preflight(_t, provider_id=provider_id)
                                _new_items.append((_p, _t2, _s2))
                            norm_items = _new_items
                            problem_tex = "\n\n".join([x[0] for x in norm_items])
                            theorems_tex = "\n\n".join([x[1] for x in norm_items])
                            solution_tex = "\n\n".join([x[2] for x in norm_items])
                        else:
                            solution_tex = await self._spdf_tikz_preflight(solution_tex, provider_id=provider_id)
                            theorems_tex = await self._spdf_tikz_preflight(theorems_tex, provider_id=provider_id)
                    except Exception:
                        pass

                    tex_src = _build_pdf_latex_document_multi(norm_items) if is_multi else _build_pdf_latex_document(problem_tex, theorems_tex, solution_tex)
                    continue
                else:
                    if compile_ok:
                        break
        else:
            pdf_bytes = await self._compile_tex_to_pdf(tex_src)

        if not pdf_bytes:
            safe_problem = _ensure_balanced_dollar_math(_strip_known_xml_like_tags(problem_tex or problem_text))
            safe_solution = _ensure_balanced_dollar_math(_strip_known_xml_like_tags(solution_tex or raw))
            safe_problem = _escape_text_preserve_dollar_math(safe_problem)
            safe_solution = _escape_text_preserve_dollar_math(safe_solution)

            safe_theorems = (
                r"\begin{theoremBox}{说明}" + "\n"
                r"（由于原始 LaTeX 片段含有不兼容字符，已自动转为安全文本显示。数学公式若以 $...$ 包裹则仍保持公式渲染。）" + "\n"
                r"\end{theoremBox}"
            )
            if 'is_multi' in locals() and is_multi and 'norm_items' in locals() and norm_items:
                safe_items = []
                for _p, _t, _s in norm_items:
                    sp = _escape_text_preserve_dollar_math(_ensure_balanced_dollar_math(_strip_known_xml_like_tags(_p)))
                    ss = _escape_text_preserve_dollar_math(_ensure_balanced_dollar_math(_strip_known_xml_like_tags(_s)))
                    # theorems 统一放在开头：这里用 safe_theorems 作为占位，后续在 multi builder 中会自动去重
                    safe_items.append((sp, safe_theorems, ss))
                tex_src_retry = _build_pdf_latex_document_multi(safe_items)
            else:
                tex_src_retry = _build_pdf_latex_document(safe_problem, safe_theorems, safe_solution)
            pdf_bytes = await self._compile_tex_to_pdf(tex_src_retry)
            if pdf_bytes:
                tex_src = tex_src_retry

        if not pdf_bytes:
            log_snip = (self._last_texlive_log or "").strip()[:1000]
            raise RuntimeError(f"Compile failed. Log: {log_snip}")

        user_name = _get_sender_display_name(event)
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        rand = uuid.uuid4().hex[:8]
        fname = _sanitize_filename(f"{user_name}-{ts}-{rand}.pdf")

        pdf_path = os.path.join(self.PDF_CACHE_DIR, fname)
        tmp_path = pdf_path + f".tmp-{uuid.uuid4().hex}"
        with open(tmp_path, "wb") as f:
            f.write(pdf_bytes)
        os.replace(tmp_path, pdf_path)

        return pdf_path, fname, tex_src

    async def _solve_math_to_pdf(
            self,
            problem_text: str,
            event: AstrMessageEvent,
            image_urls: Optional[List[str]] = None,
            ref_pdf_latex: str = "",
    ):
        """LLM -> LaTeX -> PDF（仅本地 xelatex 编译）"""
        provider_id = str(self._cfg("pdf_provider_id", "") or "").strip()

        # 当前会话使用的 provider
        current_provider_id = ""
        try:
            current_provider_id = await self.context.get_current_chat_provider_id(umo=event.unified_msg_origin)
        except Exception:
            current_provider_id = ""

        has_image = bool(image_urls)
        ref_pdf_latex = (ref_pdf_latex or "").strip()

        if not provider_id:
            provider_id = current_provider_id

        if not provider_id:
            raise RuntimeError("未能获取 chat_provider_id")

        # 生成提示词：普通新题 / 引用上一份 PDF 追问
        if ref_pdf_latex and (not has_image):
            prompt = self._build_pdf_followup_prompt(
                question_text=problem_text,
                ref_pdf_latex=ref_pdf_latex,
            )
        else:
            prompt = self._build_pdf_latex_prompt(
                problem_text,
                has_image=has_image,
            )

        # 直接调用 llm_generate
        try:
            llm_resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                image_urls=image_urls if image_urls else [],
            )
        except Exception as e:
            err_str = str(e)
            if "400" in err_str or "process input image" in err_str:
                logger.error(f"LLM Image Error: {err_str}")
                raise RuntimeError("Upstream Error: Image invalid")
            raise

        raw = (
                getattr(llm_resp, "completion_text", None)
                or getattr(llm_resp, "completion", None)
                or getattr(llm_resp, "text", None)
                or ""
        )
        raw = (raw or "").strip()
        raw = _strip_all_code_fences(raw)

        # 解析 <problem>/<theorems>/<solution>
        norm_items = []
        multi_items = _parse_latex_sections_multi(raw)
        is_multi = len(multi_items) > 1
        if is_multi:
            norm_items = _normalize_multi_pdf_items(multi_items, problem_text=problem_text, has_image=has_image)
            problem_tex = "\n\n".join([x[0] for x in norm_items])
            theorems_tex = "\n\n".join([x[1] for x in norm_items])
            solution_tex = "\n\n".join([x[2] for x in norm_items])
        else:
            problem_tex, theorems_tex, solution_tex = _parse_latex_sections(raw)
        # === 生成中断检测（仅提示，不影响编译）Start ===
        # 说明：当并发任务/网络抖动/上游限制导致模型输出被截断时，常见现象是 <solution> 没有正常闭合。
        # 这里只在“出现 <solution> 但缺少 </solution>”时提示，避免误报。
        low_raw = raw.lower() if raw else ""
        is_truncated = bool(raw and ("<solution>" in low_raw) and ("</solution>" not in low_raw))
        if is_truncated:
            if 'is_multi' in locals() and is_multi and 'norm_items' in locals() and norm_items:
                _p, _t, _s = norm_items[-1]
                _s = (_s or "")
                _s += (
                    r"\vspace{1em}" + "\n"
                    r"\begin{tcolorbox}[colback=red!5!white,colframe=red!75!black,title={生成中断警告}]" + "\n"
                    r"由于模型输出被提前中断，解答过程可能不完整。你可以把题目拆成更小的问题，或让机器人继续补全上一份解答。" + "\n"
                    r"\end{tcolorbox}"
                )
                norm_items[-1] = (_p, _t, _s)
                solution_tex = "\n\n".join([x[2] for x in norm_items])
            else:
                solution_tex = (solution_tex or "")
                solution_tex += (
                    r"\vspace{1em}" + "\n"
                    r"\begin{tcolorbox}[colback=red!5!white,colframe=red!75!black,title={生成中断警告}]" + "\n"
                    r"由于模型输出被提前中断，解答过程可能不完整。你可以把题目拆成更小的问题，或让机器人继续补全上一份解答。" + "\n"
                    r"\end{tcolorbox}"
                )
        # === 生成中断检测 End ===

        # ================== [新增兜底逻辑 Start] ==================
        # 场景：用户追问时，模型可能只回复了纯文本（如“好的，修改如下...”），忘记加标签。
        # 导致 problem_tex 和 solution_tex 解析为空。
        # 此时强制把“用户输入”当作题目，把“模型回复”当作解答。
        if (not problem_tex) and (not solution_tex) and raw:
            # 1. 构造题目部分
            if problem_text:
                # 有文字输入，直接转义作为题目
                problem_tex = _escape_text_preserve_dollar_math(problem_text)
            elif has_image:
                # 只有图片没有文字
                problem_tex = r"(用户未提供文字描述，见下方解答)"
            else:
                problem_tex = r"(对话模式)"

            # 2. 构造解答部分（把模型的整个回复都算作解答）
            solution_tex = _llm_raw_to_safe_tex(raw)
        # ================== [新增兜底逻辑 End] ==================

        # 原有的逻辑保持不变，处理部分标签缺失的情况
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

        # 对模型输出做一次“文本模式特殊字符”修复
        problem_tex = _sanitize_latex_fragment_for_xelatex(problem_tex)
        theorems_tex = _sanitize_latex_fragment_for_xelatex(theorems_tex)
        solution_tex = _sanitize_latex_fragment_for_xelatex(solution_tex)

        tex_src = _build_pdf_latex_document_multi(norm_items) if is_multi else _build_pdf_latex_document(problem_tex, theorems_tex, solution_tex)
        # 编译（仅本地）
        # 可选：失败后自动把编译日志回传给“生成 LaTeX 的模型”进行修复，并用“守门模型”判定是否通过
        pdf_bytes: Optional[bytes] = None
        compile_guard_enabled = bool(self._cfg("pdf_enable_compile_guard", False))
        completeness_guard_enabled = bool(self._cfg("pdf_enable_completeness_guard", False))
        if 'is_multi' in locals() and is_multi:
            completeness_guard_enabled = False


        compile_max_rounds = int(self._cfg("pdf_guard_max_rounds", 3) or 3)
        completeness_max_rounds = int(self._cfg("pdf_completeness_guard_max_rounds", 2) or 2)

        overall_max_rounds = max(
            compile_max_rounds if compile_guard_enabled else 0,
            completeness_max_rounds if completeness_guard_enabled else 0,
        )

        if compile_guard_enabled or completeness_guard_enabled:
            for round_idx in range(overall_max_rounds + 1):
                pdf_bytes = await self._compile_tex_to_pdf(tex_src)
                compile_ok = bool(pdf_bytes)

                # 1) 编译错误守门：编译失败时优先修复
                if compile_guard_enabled:
                    judge_pass = await self._pdf_guard_judge(
                        compile_ok=compile_ok,
                        log_text=(self._last_texlive_log or ""),
                        tex_src=tex_src,
                        preferred_provider_id=str(self._cfg("pdf_guard_provider_id", "") or "").strip(),
                        fallback_provider_id=(current_provider_id or provider_id or "").strip(),
                    )

                    if (not compile_ok) or (not judge_pass):
                        if round_idx >= overall_max_rounds:
                            break

                        # 编译失败：把日志发回给“生成 LaTeX 的模型”让其修复重写
                        repair_prompt = self._build_pdf_latex_repair_prompt(
                            problem_text=problem_text,
                            has_image=has_image,
                            compile_log=(self._last_texlive_log or ""),
                            prev_tex_src=tex_src,
                            prev_raw=raw,
                        )

                        try:
                            llm_resp = await self.context.llm_generate(
                                chat_provider_id=provider_id,
                                prompt=repair_prompt,
                                image_urls=image_urls if image_urls else [],
                            )
                        except Exception as e:
                            err_str = str(e)
                            if "400" in err_str or "process input image" in err_str:
                                logger.error(f"LLM Image Error: {err_str}")
                                raise RuntimeError("Upstream Error: Image invalid")
                            raise

                        raw = (
                                getattr(llm_resp, "completion_text", None)
                                or getattr(llm_resp, "completion", None)
                                or getattr(llm_resp, "text", None)
                                or ""
                        )
                        raw = (raw or "").strip()
                        raw = _strip_all_code_fences(raw)

                        # 解析 <problem>/<theorems>/<solution>
                        norm_items = []
                        multi_items = _parse_latex_sections_multi(raw)
                        is_multi = len(multi_items) > 1
                        if is_multi:
                            norm_items = _normalize_multi_pdf_items(multi_items, problem_text=problem_text, has_image=has_image)
                            problem_tex = "\n\n".join([x[0] for x in norm_items])
                            theorems_tex = "\n\n".join([x[1] for x in norm_items])
                            solution_tex = "\n\n".join([x[2] for x in norm_items])
                        else:
                            problem_tex, theorems_tex, solution_tex = _parse_latex_sections(raw)
                        # 修复时可能漏标签：兜底把“用户输入”当题目，“模型回复”当解答
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

                        # 对模型输出做一次“文本模式特殊字符”修复
                        problem_tex = _sanitize_latex_fragment_for_xelatex(problem_tex)
                        theorems_tex = _sanitize_latex_fragment_for_xelatex(theorems_tex)
                        solution_tex = _sanitize_latex_fragment_for_xelatex(solution_tex)

                        tex_src = _build_pdf_latex_document_multi(norm_items) if is_multi else _build_pdf_latex_document(problem_tex, theorems_tex, solution_tex)
                        # 下一轮继续：先编译，再判错/判完整性
                        continue

                # 若编译失败且没有启用编译守门，则跳出交给后续“纯文本兜底”
                if not compile_ok and not compile_guard_enabled:
                    break

                # 2) 完整性守门：编译通过后检查“是否漏答/未写完”
                if compile_ok and completeness_guard_enabled:
                    comp_pass, comp_issues = await self._pdf_completeness_guard_judge(
                        problem_text=problem_text,
                        problem_tex=problem_tex,
                        solution_tex=solution_tex,
                        preferred_provider_id=str(self._cfg("pdf_completeness_guard_provider_id", "") or "").strip(),
                        fallback_provider_id=(current_provider_id or provider_id or "").strip(),
                    )

                    if comp_pass:
                        break

                    if round_idx >= overall_max_rounds:
                        break

                    # 不完整：把缺失点回传给“生成 LaTeX 的模型”让其补全重写
                    repair_prompt = self._build_pdf_incomplete_repair_prompt(
                        problem_text=problem_text,
                        has_image=has_image,
                        issues=comp_issues,
                        prev_tex_src=tex_src,
                        prev_raw=raw,
                    )

                    try:
                        llm_resp = await self.context.llm_generate(
                            chat_provider_id=provider_id,
                            prompt=repair_prompt,
                            image_urls=image_urls if image_urls else [],
                        )
                    except Exception as e:
                        err_str = str(e)
                        if "400" in err_str or "process input image" in err_str:
                            logger.error(f"LLM Image Error: {err_str}")
                            raise RuntimeError("Upstream Error: Image invalid")
                        raise

                    raw = (
                            getattr(llm_resp, "completion_text", None)
                            or getattr(llm_resp, "completion", None)
                            or getattr(llm_resp, "text", None)
                            or ""
                    )
                    raw = (raw or "").strip()
                    raw = _strip_all_code_fences(raw)

                    # 解析 <problem>/<theorems>/<solution>
                    norm_items = []
                    multi_items = _parse_latex_sections_multi(raw)
                    is_multi = len(multi_items) > 1
                    if is_multi:
                        norm_items = _normalize_multi_pdf_items(multi_items, problem_text=problem_text, has_image=has_image)
                        problem_tex = "\n\n".join([x[0] for x in norm_items])
                        theorems_tex = "\n\n".join([x[1] for x in norm_items])
                        solution_tex = "\n\n".join([x[2] for x in norm_items])
                    else:
                        problem_tex, theorems_tex, solution_tex = _parse_latex_sections(raw)
                    if (not problem_tex) and (not solution_tex) and raw:
                        if problem_text:
                            problem_tex = _escape_text_preserve_dollar_math(problem_text)
                        elif has_image:
                            problem_tex = r"(用户未提供文字描述，见下方解答)"
                        else:
                            problem_tex = r"(对话模式)"
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

                    tex_src = _build_pdf_latex_document_multi(norm_items) if is_multi else _build_pdf_latex_document(problem_tex, theorems_tex, solution_tex)
                    continue

                # 编译通过，且不需要/已通过完整性守门
                if compile_ok:
                    break
        else:
            pdf_bytes = await self._compile_tex_to_pdf(tex_src)

        # 二次兜底：把主要内容按“纯文本”强转义
        if not pdf_bytes:
            safe_problem = _ensure_balanced_dollar_math(_strip_known_xml_like_tags(problem_tex or problem_text))
            safe_solution = _ensure_balanced_dollar_math(_strip_known_xml_like_tags(solution_tex or raw))
            safe_problem = _escape_text_preserve_dollar_math(safe_problem)
            safe_solution = _escape_text_preserve_dollar_math(safe_solution)

            safe_theorems = (
                r"\begin{theoremBox}{说明}" + "\n"
                r"（由于原始 LaTeX 片段含有不兼容字符，已自动转为安全文本显示。数学公式若以 $...$ 包裹则仍保持公式渲染。）" + "\n"
                r"\end{theoremBox}"
            )
            if 'is_multi' in locals() and is_multi and 'norm_items' in locals() and norm_items:
                safe_items = []
                for _p, _t, _s in norm_items:
                    sp = _escape_text_preserve_dollar_math(_ensure_balanced_dollar_math(_strip_known_xml_like_tags(_p)))
                    ss = _escape_text_preserve_dollar_math(_ensure_balanced_dollar_math(_strip_known_xml_like_tags(_s)))
                    # theorems 统一放在开头：这里用 safe_theorems 作为占位，后续在 multi builder 中会自动去重
                    safe_items.append((sp, safe_theorems, ss))
                tex_src_retry = _build_pdf_latex_document_multi(safe_items)
            else:
                tex_src_retry = _build_pdf_latex_document(safe_problem, safe_theorems, safe_solution)
            pdf_bytes = await self._compile_tex_to_pdf(tex_src_retry)
            if pdf_bytes:
                tex_src = tex_src_retry

        if not pdf_bytes:
            log_snip = (self._last_texlive_log or "").strip()[:1000]
            raise RuntimeError(f"Compile failed. Log: {log_snip}")

        user_name = _get_sender_display_name(event)
        # 文件名必须唯一：原来只精确到分钟，用户在同一分钟内多次 /pdf（尤其并发）会互相覆盖，导致 PDF 被截断/乱码
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        rand = uuid.uuid4().hex[:8]
        fname = _sanitize_filename(f"{user_name}-{ts}-{rand}.pdf")

        pdf_path = os.path.join(self.PDF_CACHE_DIR, fname)
        tmp_path = pdf_path + f".tmp-{uuid.uuid4().hex}"
        with open(tmp_path, "wb") as f:
            f.write(pdf_bytes)
        # 原子替换，避免发送时读到“半截文件”
        os.replace(tmp_path, pdf_path)

        return pdf_path, fname, tex_src

    # ---------------------------------------------------------------------
    # 知识库检索：仅用于触发工具检索（不涉及 /pdf 输出）
    # ---------------------------------------------------------------------
    def _build_kb_search_only_prompt(
            self,
            user_query: str,
            has_image: bool,
            kb_n: int,
            exclude_sources: Optional[List[str]] = None,
    ) -> str:
        """仅用于“知识库检索”这一件事的 prompt：
        - 强制调用 astr_kb_search
        - 要求把工具原始返回粘贴到 <tool_raw> 中，便于代码侧解析，杜绝“编造出处”
        """
        kb_n = max(1, min(int(kb_n or 2), 20))
        uq = (user_query or "").strip()
        ex = [s.strip() for s in (exclude_sources or []) if isinstance(s, str) and s.strip()]
        ex = ex[:20]
        ex_text = "；".join(ex)

        img_line = ""
        if has_image:
            img_line = (
                "你还会收到一张/多张图片，请先识别图片中的题目内容并提取关键词，再进行检索。\n"
            )

        return (
                "你是一个严格的知识库检索代理。\n"
                "你必须调用工具 astr_kb_search 来检索 AstrBot 的知识库/题库。\n"
                f"{img_line}"
                "硬性要求：\n"
                "1) 必须至少调用一次工具 astr_kb_search。\n"
                "2) 你只能基于工具返回的内容给出结果，禁止编造任何题目或出处。\n"
                "3) 最终只输出 <tool_raw>...</tool_raw>，其中内容必须是你最后一次调用该工具得到的【原始返回结果】，请原封不动粘贴（不要总结、不要改写、不要加序号）。\n"
                "4) 不要输出任何额外文字。\n"
                "5) 如果工具没有返回结果，也必须输出 <tool_raw>[]</tool_raw>（仍然要调用工具）。\n"
                f"需要返回条数：{kb_n}\n"
                + (f"排除以下来源（若工具支持过滤则过滤，否则忽略）：{ex_text}\n" if ex_text else "")
                + "用户查询/意图：\n"
                  f"{uq}\n"
        )

    async def _kb_extract_seed_from_images(
            self,
            provider_id: str,
            image_urls: List[str],
            user_hint: str = "",
    ) -> str:
        """
        用多模态模型从图片中提取“用于知识库检索的题面/关键词”。
        只用于检索 seed，不做解题，尽量避免触发工具调用/安全过滤。
        """
        if not image_urls:
            return ""

        prompt = (
            "你是一个题目文字识别与检索关键词提取助手。\n"
            "请阅读图片中的题目内容，提取用于在题库/知识库中检索的【题面摘要/关键词】。\n"
            "要求：\n"
            "1) 只输出一段纯文本（不要解释、不要序号、不要 Markdown、不要代码块）。\n"
            "2) 尽量包含题目中的关键数学表达与条件，长度 50~250 字。\n"
            "3) 如果无法识别或图片不是题目，只输出空字符串。\n"
        )
        if user_hint:
            prompt += "用户补充意图（可参考，不要原样复述）：\n" + (user_hint[:200] + "\n")

        try:
            resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                image_urls=image_urls,
            )
            txt = (
                    getattr(resp, "completion_text", None)
                    or getattr(resp, "completion", None)
                    or getattr(resp, "text", None)
                    or ""
            )
            txt = _strip_all_code_fences((txt or "").strip())
            txt = txt.strip().strip('"\''"“”‘’").strip()
            # 一些常见“无法识别”措辞直接判空
            if not txt:
                return ""
            if any(bad in txt for bad in
                   ("无法识别", "看不清", "无法看清", "无法读取", "不是题目", "无法确定", "抱歉")) and len(txt) <= 80:
                return ""
            # 控制长度
            return txt[:300].strip()
        except Exception as e:
            logger.debug(f"KB 视觉 seed 提取失败: {e}")
            return ""

    def _kb_get_tool_schema(self, tool_obj: Any) -> Optional[Dict[str, Any]]:
        """
        尝试从 AstrBot 的工具对象中取出 JSON Schema（不同版本字段名可能不同）。
        """
        if tool_obj is None:
            return None
        for attr in ("parameters", "parameter_schema", "schema", "openai_schema"):
            try:
                v = getattr(tool_obj, attr, None)
            except Exception:
                v = None
            if isinstance(v, dict) and v:
                # openai_schema 可能是 {"name":..., "parameters": {...}}
                if "parameters" in v and isinstance(v.get("parameters"), dict):
                    return v.get("parameters")
                return v
        return None

    def _kb_infer_tool_kwargs(
            self,
            tool_obj: Any,
            user_query: str,
            want_n: int,
            exclude_sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        根据工具的 schema / signature 推断参数名，尽量兼容不同实现。
        """
        want_n = max(1, min(int(want_n or 5), 50))
        q = (user_query or "").strip()

        param_names: List[str] = []
        schema = self._kb_get_tool_schema(tool_obj)
        if isinstance(schema, dict):
            props = schema.get("properties") or {}
            if isinstance(props, dict):
                param_names.extend(list(props.keys()))

        # signature 兜底
        try:
            fn = getattr(tool_obj, "run", None)
            if callable(fn):
                sig = inspect.signature(fn)
            else:
                sig = inspect.signature(tool_obj)
            param_names.extend([p for p in sig.parameters.keys()])
        except Exception:
            pass

        # 去重保持顺序
        seen = set()
        ordered = []
        for p in param_names:
            if p in seen:
                continue
            seen.add(p)
            ordered.append(p)
        param_set = set(ordered)

        kwargs: Dict[str, Any] = {}

        # query 字段
        for k in (
                "query", "q", "text", "question", "keywords", "keyword", "search_query", "search", "input", "prompt"):
            if k in param_set:
                kwargs[k] = q
                break

        # top_k 字段
        for k in ("top_k", "topk", "top_n", "topn", "k", "n", "limit", "max_results", "num", "count", "size"):
            if k in param_set:
                kwargs[k] = want_n
                break

        # 排除来源（如果工具支持）
        if exclude_sources:
            ex = [str(s).strip() for s in exclude_sources if str(s).strip()]
            ex = ex[:50]
            for k in ("exclude_sources", "exclude", "exclude_titles", "blacklist", "ban", "filter_out"):
                if k in param_set:
                    kwargs[k] = ex
                    break

        return kwargs

    def _kb_coerce_tool_yield_item_to_text(self, item: Any) -> str:
        """
        把工具 yield 出来的任意对象尽量提取成可解析文本。
        """
        if item is None:
            return ""
        if isinstance(item, str):
            return item
        if isinstance(item, (dict, list)):
            try:
                return json.dumps(item, ensure_ascii=False)
            except Exception:
                return str(item)

        # 常见字段尝试
        for attr in ("text", "content", "message", "data", "body", "completion_text", "completion"):
            try:
                v = getattr(item, attr, None)
            except Exception:
                v = None
            if isinstance(v, str) and v.strip():
                return v.strip()
        return str(item)

    def _kb_collapse_tool_result(self, res: Any) -> Any:
        """
        把 direct tool call 的返回结果折叠成一个 payload：
        - coroutine: 已在外面 await
        - async generator: 收集后传入本函数
        - list[MessageEventResult]: 拼成文本
        - dict/list/str: 原样返回
        """
        if res is None:
            return None
        # async generator 收集后通常是 list
        if isinstance(res, list):
            if len(res) == 1:
                return self._kb_collapse_tool_result(res[0])
            # 如果列表里本身就是 dict/list，直接返回（让 _normalize_kb_hits 自己处理）
            if any(isinstance(x, (dict, list)) for x in res):
                return res
            texts = [self._kb_coerce_tool_yield_item_to_text(x) for x in res]
            texts = [t for t in texts if t.strip()]
            return "\n".join(texts).strip()
        return res


    def _get_llm_tool_obj(self, tool_name: str) -> Any:
        """尽量从 AstrBot 的工具管理器中拿到工具对象（用于 tool_loop_agent tools 参数）。"""
        tool_obj = None
        # 1) 优先 Context.get_llm_tool_manager()
        try:
            tool_mgr = self.context.get_llm_tool_manager()
        except Exception:
            tool_mgr = None
        try:
            if tool_mgr and hasattr(tool_mgr, "get_func"):
                tool_obj = tool_mgr.get_func(tool_name)
        except Exception:
            tool_obj = None

        # 2) 再尝试 provider_manager.llm_tools
        if tool_obj is None:
            try:
                pm = getattr(self.context, "provider_manager", None)
                llm_tools = getattr(pm, "llm_tools", None)
                if llm_tools and hasattr(llm_tools, "get_func"):
                    tool_obj = llm_tools.get_func(tool_name)
            except Exception:
                tool_obj = None

        return tool_obj

    def _maybe_build_toolset(self, tool_objs: List[Any]) -> Any:
        """把工具对象打包成 ToolSet；若当前版本没有 ToolSet，则回退为 list。"""
        tool_objs = [t for t in (tool_objs or []) if t is not None]
        if not tool_objs:
            return None

        # AstrBot 新版一般可从 astrbot.api.all 导入 ToolSet
        try:
            from astrbot.api.all import ToolSet as _ToolSet  # type: ignore
            return _ToolSet(tool_objs)
        except Exception:
            pass

        # 有些版本 ToolSet 在 core.agent.tool
        try:
            from astrbot.core.agent.tool import ToolSet as _ToolSet  # type: ignore
            return _ToolSet(tool_objs)
        except Exception:
            pass

        # 最后兜底：直接返回 list（有些实现也支持）
        return tool_objs

    async def _kb_call_llm_tool_direct(
            self,
            tool_name: str,
            event: AstrMessageEvent,
            user_query: str,
            want_n: int,
            exclude_sources: Optional[List[str]] = None,
    ) -> Any:
        """
        直接调用 AstrBot 的 LLM 工具（不经过 tool_loop_agent，不依赖模型 function calling）。
        成功返回工具原始 payload；失败返回 None（不抛异常）。
        """
        # 尽量激活工具（即使用户没 /tool on）
        try:
            self.context.activate_llm_tool(tool_name)
        except Exception:
            pass

        tool_obj = None
        try:
            mgr = self.context.get_llm_tool_manager()
        except Exception:
            mgr = None

        if mgr is not None:
            # 优先 get_func
            try:
                if hasattr(mgr, "get_func") and callable(getattr(mgr, "get_func")):
                    tool_obj = mgr.get_func(tool_name)
            except Exception:
                tool_obj = None
            if tool_obj is None:
                # 其他可能的 getter
                for getter in ("get_tool", "get"):
                    try:
                        fn = getattr(mgr, getter, None)
                        if callable(fn):
                            tool_obj = fn(tool_name)
                            if tool_obj is not None:
                                break
                    except Exception:
                        continue
            if tool_obj is None and isinstance(mgr, dict):
                tool_obj = mgr.get(tool_name)

        if tool_obj is None:
            return None

        kwargs = self._kb_infer_tool_kwargs(tool_obj, user_query=user_query, want_n=want_n,
                                            exclude_sources=exclude_sources)

        async def _exec_call(callable_obj, *a, **kw):
            out = callable_obj(*a, **kw)
            if inspect.isasyncgen(out):
                items = []
                async for it in out:
                    items.append(it)
                return items
            if inspect.isawaitable(out):
                return await out
            return out

        # 依次尝试不同调用方式
        try_calls = []
        if hasattr(tool_obj, "run") and callable(getattr(tool_obj, "run")):
            # 常见签名：run(event, query=..., top_k=...)
            try_calls.append(("run_pos", lambda: _exec_call(getattr(tool_obj, "run"), event, **kwargs)))
            try_calls.append(("run_kw", lambda: _exec_call(getattr(tool_obj, "run"), event=event, **kwargs)))
            # 兼容：run(query=..., top_k=...)（不需要 event）
            try_calls.append(("run_noevt", lambda: _exec_call(getattr(tool_obj, "run"), **kwargs)))
        if hasattr(tool_obj, "call") and callable(getattr(tool_obj, "call")):
            try_calls.append(("call_pos", lambda: _exec_call(getattr(tool_obj, "call"), event, **kwargs)))
            try_calls.append(("call_kw", lambda: _exec_call(getattr(tool_obj, "call"), event=event, **kwargs)))
            try_calls.append(("call_noevt", lambda: _exec_call(getattr(tool_obj, "call"), **kwargs)))
        if callable(tool_obj):
            try_calls.append(("obj_pos", lambda: _exec_call(tool_obj, event, **kwargs)))
            try_calls.append(("obj_kw", lambda: _exec_call(tool_obj, event=event, **kwargs)))
            try_calls.append(("obj_noevt", lambda: _exec_call(tool_obj, **kwargs)))
        # 一些 wrapper 可能把真实函数放在 func/handler
        for attr in ("func", "handler", "callback"):
            fn = getattr(tool_obj, attr, None)
            if callable(fn):
                try_calls.append((f"{attr}_pos", lambda fn=fn: _exec_call(fn, event, **kwargs)))
                try_calls.append((f"{attr}_kw", lambda fn=fn: _exec_call(fn, event=event, **kwargs)))
                try_calls.append((f"{attr}_noevt", lambda fn=fn: _exec_call(fn, **kwargs)))

        last_err = None
        for name, thunk in try_calls:
            try:
                res = await thunk()
                return self._kb_collapse_tool_result(res)
            except TypeError as e:
                last_err = e
                continue
            except Exception as e:
                last_err = e
                continue

        if last_err:
            logger.debug(f"direct tool call 失败({tool_name}): {last_err}")
        return None

    async def _kb_tool_search_hits(
            self,
            user_query: str,
            event: AstrMessageEvent,
            provider_id: str,
            image_urls: Optional[List[str]],
            want_n: int = 5,
            exclude_sources: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        可靠的 KB 检索：
        1) 优先尝试“直接调用工具 astr_kb_search”
        2) 若 direct call 不可用，再（可选）回退 tool_loop_agent
        """
        want_n = max(1, min(int(want_n or 5), 20))
        has_image = bool(image_urls)

        q = (user_query or "").strip()

        # 如果带图片但文本过于“指令化/空”，先用多模态模型提取题面 seed
        if has_image and (not q or (len(q) < 12 and _is_kb_query(q))):
            seed_img = await self._kb_extract_seed_from_images(
                provider_id=provider_id,
                image_urls=image_urls if image_urls else [],
                user_hint=q,
            )
            if seed_img:
                q = seed_img

        # ---------- 1) direct call ----------
        payload = None
        try:
            payload = await self._kb_call_llm_tool_direct(
                tool_name="astr_kb_search",
                event=event,
                user_query=q,
                want_n=want_n,
                exclude_sources=exclude_sources,
            )
        except Exception:
            payload = None

        hits: List[Dict[str, Any]] = []
        if payload is not None:
            try:
                hits = _normalize_kb_hits(payload)
            except Exception as e:
                logger.debug(f"direct KB tool 输出解析失败: {e}")
                hits = []

        # 过滤 exclude_sources（无论工具是否支持）
        if exclude_sources and hits:
            excl = set()
            for s in exclude_sources:
                try:
                    excl.add(self._sanitize_kb_title(str(s)))
                except Exception:
                    pass

            filtered = []
            for h in hits:
                if not isinstance(h, dict):
                    continue
                src = (h.get("source") or h.get("id") or "").strip()
                src = self._sanitize_kb_title(src) if src else ""
                if src and src in excl:
                    continue
                filtered.append(h)
            hits = filtered

        if hits:
            return hits

        # ---------- 2) tool_loop_agent 兜底 ----------
        # 说明：部分 AstrBot 版本/工具实现下，direct call 可能拿不到工具对象或签名不兼容，
        # 导致 PDF 场景“明明知识库有题但一个都找不到”。
        # 因此这里在 direct call 未命中时，默认尝试一次 tool_loop_agent 作为兼容性兜底。
        # 强制开启 tool_loop_agent 作为 fallback（仅在 direct call 未命中时才走到这里）

        # 若近期被内容安全过滤拦截，短时间内不再尝试 tool_loop_agent，避免后台刷屏
        try:
            if getattr(self, "_kb_tool_loop_disabled_until", 0) > time.time():
                return []
        except Exception:
            pass

        kb_agent_max_steps = int(self._cfg("kb_agent_max_steps", 8) or 8)
        kb_tool_timeout = int(self._cfg("kb_tool_call_timeout_sec", 60) or 60)

        prompt = self._build_kb_search_only_prompt(
            user_query=q,
            has_image=has_image,
            kb_n=want_n,
            exclude_sources=exclude_sources,
        )

        try:
            self.context.activate_llm_tool("astr_kb_search")
        except Exception:
            pass

        llm_resp = None

        # 显式传入 tools，保证 tool_loop_agent 能拿到 astr_kb_search（部分版本不自动注入系统工具）
        tools_arg = None
        try:
            tool_obj = self._get_llm_tool_obj("astr_kb_search")
            tools_arg = self._maybe_build_toolset([tool_obj]) if tool_obj else None
        except Exception:
            tools_arg = None

        try:
            if tools_arg is not None:
                try:
                    llm_resp = await self.context.tool_loop_agent(
                        event=event,
                        chat_provider_id=provider_id,
                        prompt=prompt,
                        image_urls=image_urls if image_urls else [],
                        tools=tools_arg,
                        max_steps=kb_agent_max_steps,
                        tool_call_timeout=kb_tool_timeout,
                    )
                except TypeError:
                    # 兼容旧签名（无 tools 参数）
                    llm_resp = await self.context.tool_loop_agent(
                        event=event,
                        chat_provider_id=provider_id,
                        prompt=prompt,
                        image_urls=image_urls if image_urls else [],
                        max_steps=kb_agent_max_steps,
                        tool_call_timeout=kb_tool_timeout,
                    )
            else:
                llm_resp = await self.context.tool_loop_agent(
                    event=event,
                    chat_provider_id=provider_id,
                    prompt=prompt,
                    image_urls=image_urls if image_urls else [],
                    max_steps=kb_agent_max_steps,
                    tool_call_timeout=kb_tool_timeout,
                )
        except Exception as e:
            msg = str(e)
            if _is_content_filter_error(msg):
                try:
                    self._kb_tool_loop_disabled_until = time.time() + 600
                except Exception:
                    pass
                logger.warning(f"知识库检索 tool_loop_agent 被内容安全过滤拦截，已临时禁用回退: {e}")
                return []
            logger.warning(f"知识库检索 tool_loop_agent 调用失败: {e}")
            return []

        try:
            hits = _extract_kb_hits_from_tool_loop_resp(llm_resp, tool_name="astr_kb_search")
        except Exception as e:
            logger.warning(f"知识库检索结果解析失败: {e}")
            hits = []

        return hits or []

    @staticmethod
    def _sanitize_kb_title(title: str, max_len: int = 80) -> str:
        r"""
        kbBox 标题安全化（用于 \begin{kbBox}{...} 的参数）：
        """
        t = (title or "").strip()
        t = re.sub(r"\s+", " ", t).strip()
        if max_len and len(t) > int(max_len):
            t = t[: int(max_len)].rstrip() + "…"
        t = _escape_latex_text_strict(t)
        return t or "知识库条目"

    def _build_pdf_latex_prompt(self, problem_text: str, has_image: bool = False) -> str:
        """/pdf 新题提示词（仅生成可编译的 LaTeX 片段）
        包含：TikZ绘图支持、知识点总结模式、完整性强制约束
        """
        intro_text = ""

        # --- 1. 场景判断逻辑 (保持不变) ---
        if has_image:
            if problem_text:
                # 【情况A：图文共存】强制模型以图片为核心，文字为追问
                intro_text = (
                    "用户上传了图片作为**关键的解题背景或证明过程**，同时提供了具体的**文字追问**。\n"
                    "**最高优先级指令**：\n"
                    "1. 你的回答必须**深度结合图片内容**。用户提到的关键点（如系数来源、特定步骤、为何是1/4等）都在图片中，脱离图片回答是无效的。\n"
                    "2. 请先识别图片中的数学推导、定理引用或手写批注，定位到用户文字提问所指的具体步骤。\n"
                    "3. 在 <problem> 标签中，请综合图片里的题面/证明目标以及用户的文字追问来描述。\n"
                    f"【用户文字追问】：{problem_text}\n"
                )
            else:
                # 【情况B：仅有图片】常规识别
                intro_text = (
                    "用户上传了一张包含数学题目的图片。\n"
                    "你的首要任务是识别图片中的题目内容，并将其转化为 LaTeX 格式放在 <problem> 标签中。\n"
                )
        else:
            # 【情况C：仅有文字】
            intro_text = f"题目/需求如下：\n{problem_text}\n"

        # --- 2. 构建核心提示词 ---
        return (
            "你是数学题标准解答的排版与写作助手。\n"
            "你的任务：把题目与标准解答写成可直接编译的 LaTeX 正文片段（中文可用）。\n"
            f"{intro_text}"
            "要求：\n"
            "1) **纯净输出**：不要输出 \\\\documentclass、\\\\usepackage 等导言区内容；不要使用 Markdown 代码块（```）；只输出纯 LaTeX 正文。\n"

            "2) **完整性强制**：\n"
            "   - 解答必须步骤完整、推导严谨。\n"
            "   - **禁止省略**：绝对不要使用“同理可得”、“略”、“步骤省略”等字眼。\n"
            "   - **禁止截断**：必须把每一个运算步骤都写出来，直到得出最终答案。\n"
            "   - **解题完整**：遇到一个题目有多个小问时，你必须将每个小问的解答都完整的写出\n"
            "   - **结尾标记强制**：在 <solution> 解答内容的最后，请在最后一句公式或文字后加上：\\hfill$\\blacksquare$。\n"
            "     并确保这是解答的最终结尾（后面不要再有其他文字）。\n"

            "3) **数学环境**：\n"
            "   - 需要用到的定理/引理/公式请放在 <theorems> 段中。\n"
            "   - 使用对应环境：\\\\begin{theoremBox}{名称}...\\\\end{theoremBox}、\\\\begin{lemmaBox}{名称}...\\\\end{lemmaBox}。\n"

            "4) **【绘图支持】(重要)**：\n"
            "   - 如果题目涉及几何图形、函数图像或证明示意图（如三角形剖分、积分路径、区域 $D$），**请务必绘制示意图，并且尽可能的美观**。\n"
            "   - 使用 **TikZ** 环境：\\\\begin{tikzpicture} ... \\\\end{tikzpicture}。\n"
            "   - **防重叠强制规则**：\n"
            "     a) 全局缩放：请默认使用 `[scale=1.5]` 或更大，给文字留出空间。\n"
            "     b) 节点间距：设置 `node distance=2.5cm` 或更宽，不要挤在一起。\n"
            "     c) 标签避让：不要只用 `above/below`，请使用偏移量，例如 `label={[yshift=0.2cm]above:文字}` 或 `node[above=0.3cm]{文字}`。\n"
            "     d) 长文本处理：对于较长的中文说明，请使用 `align=center` 并强制换行（\\\\），例如 `node[align=center]{第一行\\\\第二行}`。\n"
            "     e) 防遮挡强制：若需填充颜色（如积分区域 D），请务必先写填充代码(\\fill)，再写坐标轴和轮廓线(\\draw)；或者将填充代码包裹在 `\\begin{pgfonlayer}{bg} ... \\end{pgfonlayer}` 环境中，确保坐标轴永远浮在颜色上方。\n"       
            "   - 保持代码简洁兼容，例如使用 \\\\draw, \\\\node, \\\\coordinate 等基础命令。\n"
            "   - 将绘图代码放在 <solution> 的合适位置（通常在文字解析之前或之中），可以在解答的不同位置放置多个绘图代码来方便说明。\n"

            "5) **【特殊模式：知识总结与详解】**：\n"
            "   若用户请求的是“知识点总结”、“概念复习”、“详细讲解”或“证明定理”，而非单纯解题，请按以下标准执行：\n"
            "   - **全面性**：必须覆盖定义、核心性质、几何意义（如有）、易错点。\n"
            "   - **证明**：对于核心定理或公式，**必须给出严谨的数学证明（Proof）**，不要只列结论。\n"
            "   - **示例**：在讲解完概念后，**必须编写 1-2 个具体的典型例题（Example）**并给出完整解答，以辅助用户理解。\n"
            "   - **排版映射**：请将“知识点标题/核心摘要”放入 <problem> 标签；将“详细讲解、证明过程、例题分析、TikZ插图”放入 <solution> 标签。\n"

            "6) **编译安全**：\n"
            "   - 文本模式下不要出现裸的特殊符号：_, ^, \\\\, %, &, #, <, > 等，必须转义（如 \\\\_）。\n"
            "   - 下标/上标请严格放在数学环境 $...$ 中。\n"
            "   - 【重要】若需加粗，请直接使用 \\\\textbf{...}，禁止使用 Markdown 的 **...** 语法。\n"  

            "7) **输出结构**（标签必须保留）：\n"
            "   - 如果用户一次给了多题（如“题1/题2/…”或编号列表），请按题号把每一题拆开，\n"
            "     并按如下 3 个标签为一组重复输出多次（每题一组、顺序固定、不要输出任何其它标签）。\n"
            "   - 单题则只输出一组。\n"
            "   - 注意：<solution> 中不要复述题面，题面只放在 <problem>。\n"
            "   - 多题时，各题的 <theorems> 可为空；系统会把所有题出现过的结论统一汇总到 PDF 开头。\n"
            "<problem>（第1题题面 LaTeX）...</problem>\n"
            "<theorems>（第1题用到的定理/公式）...</theorems>\n"
            "<solution>（第1题完整解答）...\\hfill$\\blacksquare$</solution>\n"
            "\n"
            "<problem>（第2题题面 LaTeX）...</problem>\n"
            "<theorems>（第2题用到的定理/公式）...</theorems>\n"
            "<solution>（第2题完整解答）...\\hfill$\\blacksquare$</solution>\n"
        )


    def _build_pdf_latex_repair_prompt(
            self,
            problem_text: str,
            has_image: bool,
            compile_log: str,
            prev_tex_src: str = "",
            prev_raw: str = "",
    ) -> str:
        """当 /pdf 的 LaTeX 编译失败时，构造“修复重写”提示词。
        目标：让同一个“生成 LaTeX 的模型”根据编译日志重写一个可编译版本（仍输出 <problem>/<theorems>/<solution>）。"""
        log_tail_chars = int(self._cfg("pdf_guard_log_tail_chars", 4000) or 4000)
        include_tex = bool(self._cfg("pdf_guard_include_tex_in_feedback", True))

        log_txt = (compile_log or "").strip()
        if log_tail_chars > 0 and len(log_txt) > log_tail_chars:
            log_txt = log_txt[-log_tail_chars:]

        tex_txt = (prev_tex_src or "").strip() if include_tex else ""
        # 避免把超长 TeX 源码塞爆上下文：仅取尾部（通常错误就在后面）
        if tex_txt and len(tex_txt) > 12000:
            tex_txt = tex_txt[-12000:]

        raw_txt = (prev_raw or "").strip()
        if raw_txt and len(raw_txt) > 6000:
            raw_txt = raw_txt[-6000:]

        # 修复提示词尽量短、指令强，避免模型再输出解释文字
        image_hint = (
            "你还会收到一张/多张图片，请先识别图片中的题目内容，再给出可编译的 LaTeX。\n"
            if has_image else ""
        )

        parts = [
            "你刚才生成的 LaTeX 在 XeLaTeX 编译时失败了。\n",
            "请根据编译日志修复并重新生成一份**可直接编译**的 LaTeX 输出。\n",
            "硬性要求：\n",
            "1) 必须输出并且只输出三个标签块：<problem>...</problem>、<theorems>...</theorems>、<solution>...</solution>。\n",
            "2) 标签必须完整闭合；不要输出 Markdown 代码块/反引号；不要输出任何解释文字。\n",
            "3) 禁止使用需要 shell-escape 的写法；避免不常见/不稳定宏包；保持语法简单稳定。\n",
            "4) 对于文本中的特殊字符（如 %, #, &, _, {, }）需正确转义；数学公式用 $...$ 或 \\[...\\]。\n",
            "5) 在 <solution> 最后必须加入：\\hfill$\\blacksquare$，确保这是最终结尾。\n",
            image_hint,
            "\n【题目/追问】\n" + (problem_text or "(无文字，仅图片)") + "\n",
            "\n【编译错误日志（截断）】\n```text\n" + (log_txt or "(无日志)") + "\n```\n",
        ]

        if tex_txt:
            parts.append("\n【上一版完整 TeX 源码（供你参考并修复）】\n```latex\n" + tex_txt + "\n```\n")
        elif raw_txt:
            parts.append("\n【上一版模型原始输出（供你参考并修复）】\n" + raw_txt + "\n")

        parts.append("\n现在请输出修复后的 <problem>/<theorems>/<solution>：\n")
        return "".join(parts)

    def _build_pdf_incomplete_repair_prompt(
            self,
            problem_text: str,
            has_image: bool,
            issues: List[str],
            prev_tex_src: str = "",
            prev_raw: str = "",
    ) -> str:
        """当 /pdf 的解答“未写完/漏答”时，构造“补全重写”提示词。"""
        issue_lines: List[str] = []
        for it in (issues or []):
            s = str(it).strip()
            if s:
                issue_lines.append(f"- {s}")
        issue_block = "\n".join(issue_lines) if issue_lines else "- （未能解析出具体缺失点，请你自查是否漏答/未写完）"

        include_tex = bool(self._cfg("pdf_completeness_guard_include_tex_in_feedback", False))
        tex_txt = (prev_tex_src or "").strip() if include_tex else ""
        raw_txt = (prev_raw or "").strip() if (not include_tex) else ""

        base = r"""你是一位严谨的数学解题助手。
你上一版给出的 PDF LaTeX 解答被检测为“内容不完整/有小问漏答/某一问未写完”。
请根据下方【缺失点】补全并重写整份解答。

硬性要求：
1) 必须输出完整的三段标签：<problem>...</problem>、<theorems>...</theorems>、<solution>...</solution>。
2) 如果题目包含多问（如 (1)(2)(3)、①②③、第一问/第二问/第三问、a)b)c) 等），你必须逐问作答，并在解答中显式标注每一问。
3) 不要只追加“缺失部分”，而是输出一份可直接编译的完整版本（包含所有小问的完整解答）。
4) 避免半句/半段就结束；最后要有完整的收尾。
5) 继续保持 XeLaTeX 兼容：不要使用不受支持的宏包/命令，不要留下未闭合的环境或括号。

6) 在 <solution> 解答内容的最后必须加入：\hfill$\blacksquare$，并确保这是最终结尾。

"""

        img_line = ""
        if has_image:
            img_line = "你还会收到图片（题目截图）。若题目文字不全，请结合图片补全题意并完整解答。\n\n"

        parts = [
            base,
            img_line,
            "【题目（用户输入/意图）】\n" + (problem_text or "(空)") + "\n\n",
            "【缺失点（请逐条修复）】\n" + issue_block + "\n\n",
        ]

        if tex_txt:
            parts.append("【上一版完整 TeX 源码（供参考）】\n```latex\n" + tex_txt + "\n```\n\n")
        elif raw_txt:
            parts.append("【上一版模型原始输出（供参考）】\n" + raw_txt + "\n\n")

        parts.append("现在请输出修复后的 <problem>/<theorems>/<solution>：\n")
        return "".join(parts)

    async def _pdf_completeness_guard_judge(
            self,
            problem_text: str,
            problem_tex: str,
            solution_tex: str,
            preferred_provider_id: str = "",
            fallback_provider_id: str = "",
    ) -> (bool, List[str]):
        """可选：用一个“完整性守门模型”判断解答是否写完/是否漏答多问。
        返回：(是否通过, 缺失点列表)。"""
        if not bool(self._cfg("pdf_enable_completeness_guard", False)):
            return True, []

        provider = (preferred_provider_id or fallback_provider_id or "").strip()
        if not provider:
            return True, []

                # 强机制：结尾标记（用于可靠判断“是否写完”）
        end_marker_required = bool(self._cfg("pdf_completeness_require_end_marker", True))
        marker_only = bool(self._cfg("pdf_completeness_marker_only", True))
        # 结尾标记默认使用 \blacksquare（黑色实心方块）
        end_marker_text = str(self._cfg("pdf_completeness_end_marker_text", r"\blacksquare") or r"\blacksquare").strip()

        def _marker_line(marker: str) -> str:
            marker = (marker or "").strip()
            # 若是 LaTeX 命令（以 \ 开头），默认放在数学模式中
            if marker.startswith("\\"):
                # --- 修改开始 ---
                # 原代码：return f"\\begin{{center}}${marker}$\\end{{center}}"
                # 新代码：使用 hfill 居右
                return f"\\hfill${marker}$"
                # --- 修改结束 ---
            # 否则按“文字标记”处理（兼容旧配置）
            return f"\\begin{{center}}\\heiti\\bfseries {marker}\\end{{center}}"

        def _marker_at_end(solution: str, marker: str) -> bool:
            st = (solution or "").strip()
            if (not st) or (not marker):
                return False
            idx = st.rfind(marker)
            if idx < 0:
                return False

            # 标记必须出现在解答尾部（避免“中间出现但后面还有内容”）
            if idx < max(0, len(st) - 1600):
                return False

            after = st[idx + len(marker):]
            after_s = (after or "").strip()
            if not after_s:
                return True

            # 允许的尾部：若标记在 $...$ 或 center/proof 环境中，可能还有 $ 或 \end{...}
            norm = re.sub(r"\s+", "", after_s)
            if re.fullmatch(r"\$*\}*(\\end\{(?:center|proof)\})*\$*", norm or ""):
                return True

            return False

        if end_marker_required and end_marker_text:
            st = (solution_tex or "").strip()
            marker_line = _marker_line(end_marker_text)

            if end_marker_text not in st:
                fix = f"缺少结尾标记：{end_marker_text}（请在 <solution> 最后一行加入 {marker_line}）"
                return False, [fix]

            if not _marker_at_end(st, end_marker_text):
                fix = (
                    "结尾处未按要求放置证明完毕标记：请把 "
                    f"{marker_line} 放在 <solution> 的最后一行，并确保它是解答最终结尾"
                )
                return False, [fix]

            # 若配置为“只依赖结尾标记”，则直接通过（更强更稳）
            if marker_only:
                return True, []

        # 控制输入长度，避免 token 爆炸
        pt = (problem_text or "").strip()
        if pt and len(pt) > 8000:
            pt = pt[:8000] + "\n...(truncated)..."

        ptex = (problem_tex or "").strip()
        if ptex and len(ptex) > 8000:
            ptex = ptex[:8000] + "\n...(truncated)..."

        stex = (solution_tex or "").strip()
        if stex and len(stex) > 12000:
            stex = stex[:12000] + "\n...(truncated)..."

        prompt = (
            """你是一个严格的数学解答“完整性审查”模型。
你的任务：判断解答是否回答了题目中的所有小问，并且每一问是否写完（不存在明显截断/半段结束）。

注意：
特别规则：解答最后必须包含证明完毕标记 `\\blacksquare`（通常位于段落末尾，写作 `\\hfill$\\blacksquare$`）。如果没有这一标记，一律判定 complete=false，并在 issues 中说明缺少结尾标记。

- 题目可能包含 (1)(2)(3)、①②③、第一问/第二问/第三问、a)b)c) 等多问结构。
- 如果发现漏答或某一问明显没写完，请在 issues 里指出“缺哪一问/哪一问未写完/缺少关键步骤或结论”。
- 如果无法确定是否多问，请更保守：只有在非常明显的情况下才判定不通过。

只输出 JSON，不要输出任何额外文字：
{"complete": true/false, "issues": ["...","..."], "note": "一行总结"}

"""
            + "[problem_text]\n" + (pt or "(empty)") + "\n\n"
            + "[problem_tex]\n" + (ptex or "(empty)") + "\n\n"
            + "[solution_tex]\n" + (stex or "(empty)") + "\n"
        )

        try:
            llm_resp = await self.context.llm_generate(
                chat_provider_id=provider,
                prompt=prompt,
                image_urls=[],
            )
            out = (
                getattr(llm_resp, "completion_text", None)
                or getattr(llm_resp, "completion", None)
                or getattr(llm_resp, "text", None)
                or ""
            )
            out = (out or "").strip()
        except Exception as e:
            logger.warning(f"pdf_completeness_guard_judge failed: {e}")
            return True, []

        issues: List[str] = []
        try:
            mm = re.search(r"\{[\s\S]*\}", out)
            if mm:
                obj = json.loads(mm.group(0))
                if isinstance(obj, dict):
                    complete = bool(obj.get("complete", True))
                    raw_issues = obj.get("issues", [])
                    if isinstance(raw_issues, list):
                        for it in raw_issues[:10]:
                            s = str(it).strip()
                            if s:
                                issues.append(s)
                    elif isinstance(raw_issues, str) and raw_issues.strip():
                        issues.append(raw_issues.strip())
                    return complete, issues
        except Exception:
            pass

        low = (out or "").lower()
        if "complete" in low and "false" in low:
            for line in out.splitlines():
                if any(k in line for k in ["漏", "缺", "未写完", "没写完", "missing", "incomplete"]):
                    s = line.strip("-• \t")
                    if s and len(s) < 200:
                        issues.append(s)
            return False, issues[:10]

        return True, []
    async def _pdf_guard_judge(
            self,
            compile_ok: bool,
            log_text: str,
            tex_src: str,
            preferred_provider_id: str = "",
            fallback_provider_id: str = "",
    ) -> bool:
        """可选：用一个“守门模型”判定是否存在编译错误。
        - compile_ok=True 通常直接 PASS
        - compile_ok=False 则根据日志判定 FAIL/是否需要继续重试
        返回：是否通过（True=通过，False=不通过/需要重试）"""
        if not bool(self._cfg("pdf_enable_compile_guard", False)):
            return compile_ok

        # 若未配置守门模型，退化为“是否编译成功”
        provider = (preferred_provider_id or fallback_provider_id or "").strip()
        if not provider:
            return compile_ok

        log_tail_chars = int(self._cfg("pdf_guard_log_tail_chars", 4000) or 4000)
        log_txt = (log_text or "").strip()
        if log_tail_chars > 0 and len(log_txt) > log_tail_chars:
            log_txt = log_txt[-log_tail_chars:]

        # 不把完整 TeX 都塞给判错模型（token 成本高）；截断一下即可
        tex_snip = (tex_src or "").strip()
        if tex_snip and len(tex_snip) > 6000:
            tex_snip = tex_snip[-6000:]

        prompt = (
            "你是一个严格的 XeLaTeX 编译判错模型。\n"
            "你的任务：判断是否存在“会导致编译失败的错误”。\n"
            "已知：compile_ok 表示是否已经成功生成 PDF。\n"
            "- 如果 compile_ok=true：你必须返回 pass=true（除非你能确定这是伪成功，但通常不应这样判断）。\n"
            "- 如果 compile_ok=false：请根据日志判断 pass=false，并给出一句最关键的失败原因。\n\n"
            "只输出 JSON，不要输出任何额外文字：\n"
            "{\"pass\": true/false, \"reason\": \"...\"}\n\n"
            f"compile_ok={str(bool(compile_ok)).lower()}\n"
            "\n[log]\n" + (log_txt or "(empty)") + "\n"
            "\n[tex_tail]\n" + (tex_snip or "(empty)") + "\n"
        )

        try:
            llm_resp = await self.context.llm_generate(
                chat_provider_id=provider,
                prompt=prompt,
                image_urls=[],
            )
            out = (
                getattr(llm_resp, "completion_text", None)
                or getattr(llm_resp, "completion", None)
                or getattr(llm_resp, "text", None)
                or ""
            )
            out = (out or "").strip()
        except Exception as e:
            # 判错模型失败时，不阻断主流程：退化为 compile_ok
            logger.warning(f"pdf_guard_judge failed: {e}")
            return compile_ok

        # 解析 JSON
        try:
            m = re.search(r"\{[\s\S]*\}", out)
            if m:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and "pass" in obj:
                    return bool(obj.get("pass"))
        except Exception:
            pass

        low = out.lower()
        if "pass" in low and "true" in low:
            return True
        if "pass" in low and "false" in low:
            return False
        # 最后的兜底：编译是否成功
        return compile_ok


    def _build_pdf_followup_prompt(
            self,
            question_text: str,
            ref_pdf_latex: str,
    ) -> str:
        """/pdf 引用上一份 PDF 追问时的提示词

        """
        q = (question_text or "").strip()
        ref = (ref_pdf_latex or "").strip()

        # 控制注入长度，避免 prompt 爆炸
        max_ref_len = int(self._cfg("pdf_followup_ref_max_chars", 6000) or 6000)
        if max_ref_len > 0 and len(ref) > max_ref_len:
            ref = ref[:max_ref_len] + "\n...(已截断)"

        intro_text = (
            "用户正在追问上一份 PDF 解答的相关问题。\n"
            "下面提供上一份 PDF 的 LaTeX 源码（仅作参考上下文），请据此回答用户的追问，并仍然输出一份可编译的 LaTeX 正文片段。\n"
            "\n"
            "【上一份 PDF 的 LaTeX 源码】\n"
            f"{ref}\n"
            "\n"
            "【用户追问】\n"
            f"{q}\n"
        )

        return (
            "你是数学题标准解答的排版与写作助手。\n"
            "你的任务：基于用户追问 + 上一份 PDF 的 LaTeX 上下文，生成一份新的可编译 LaTeX 正文片段（中文可用）。\n"
            f"{intro_text}"
            "要求：\n"
            "1) 不要输出 \\\\documentclass、\\\\usepackage 等导言区内容，只输出正文片段。\n"
            "2) 只允许使用 LaTeX（不得使用 Markdown，不得输出代码块，也不要输出```）。\n"
            "3) 解答必须严格规范：步骤完整、推导严谨、符号定义清晰、结论明确。\n"
            "4) 若需要引用上一份解答中的结论，请在 <theorems> 段中补全或重述必要的定理/引理/公式，并使用对应环境（参数必填）。\n"
            "5) 【特殊模式：延伸讲解与证明】\n"
            "   若用户追问的是“为什么要这样”、“这个公式怎么来的”或“总结一下相关知识点”，请务必：\n"
            "   - **详细展开**：不要只回答是或否，要给出**定义背景**和**推导证明**。\n"
            "   - **补充例题**：针对用户询问的疑难点，额外编造一个简单的应用例题来辅助说明。\n"
            "   - **结构映射**：将“追问的主题”放入 <problem>，将“详细解释与证明”放入 <solution>。\n"
            "6) 关键：必须保证 LaTeX 可编译。\n"
            "   - 文本模式下不要出现裸的特殊符号：_, ^, 反斜杠(\\\\), %, &, #, <, > 等。\n"
            "   - 若要显示这些符号，请使用转义：\\\\_, \\\\textasciicircum{}, \\\\textbackslash{}, \\\\%, \\\\&, \\\\#, \\\\textless{}, \\\\textgreater{}。\n"
            "   - 下标/上标请放在数学环境中，例如 $x_1$, $a^2$。\n"
            "7) 输出格式必须严格如下（标签要保留；不要输出其它标签）：\n"
            "<problem>这里写追问对应的题意/子问题/知识点标题...</problem>\n"
            "<theorems>这里写 theoremBox/lemmaBox/formulaBox...</theorems>\n"
            "<solution>这里写完整解答 或 详细知识总结...</solution>\n"
        )

    # ------------------------- 编译：缓存 / 本地 -------------------------
    def _tex_hash(self, tex_src: str) -> str:
        return hashlib.md5(tex_src.encode("utf-8", errors="ignore")).hexdigest()

    def _prune_texlive_cache(self):
        max_files = int(self._cfg("texlive_cache_max_files", 500) or 500)
        try:
            files = []
            for fn in os.listdir(self.TEXLIVE_CACHE_DIR):
                if fn.endswith(".pdf"):
                    fp = os.path.join(self.TEXLIVE_CACHE_DIR, fn)
                    try:
                        st = os.stat(fp)
                        files.append((st.st_mtime, fp))
                    except Exception:
                        pass
            if len(files) <= max_files:
                return
            files.sort(key=lambda x: x[0])  # oldest first
            for _, fp in files[: max(0, len(files) - max_files)]:
                try:
                    os.remove(fp)
                except Exception:
                    pass
        except Exception:
            pass

    async def _compile_tex_to_pdf(self, tex_src: str) -> Optional[bytes]:
        self._last_texlive_log = ""
        h = self._tex_hash(tex_src)

        # 缓存
        if bool(self._cfg("texlive_cache_enabled", True)):
            cache_pdf = os.path.join(self.TEXLIVE_CACHE_DIR, f"{h}.pdf")
            if os.path.exists(cache_pdf):
                try:
                    with open(cache_pdf, "rb") as f:
                        return f.read()
                except Exception:
                    pass

        # 直接本地编译（MikTex 环境 xelatex）
        pdf = await self._compile_local_xelatex(tex_src)
        if pdf:
            await self._save_tex_cache(h, pdf)
            return pdf

        return None

    async def _save_tex_cache(self, h: str, pdf_bytes: bytes):
        if not bool(self._cfg("texlive_cache_enabled", True)):
            return
        try:
            cache_pdf = os.path.join(self.TEXLIVE_CACHE_DIR, f"{h}.pdf")
            tmp_pdf = cache_pdf + f".tmp-{uuid.uuid4().hex}"
            with open(tmp_pdf, "wb") as f:
                f.write(pdf_bytes)
            os.replace(tmp_pdf, cache_pdf)
            self._prune_texlive_cache()
        except Exception:
            pass

    async def _compile_local_xelatex(self, tex_src: str) -> Optional[bytes]:
        timeout_sec = int(self._cfg("local_xelatex_timeout_sec", 60) or 60)

        # 重要：xelatex 编译是阻塞操作。若直接 subprocess.run 会卡住整个事件循环，
        # 并发 /pdf 时会影响其它 LLM 请求，表现为输出被截断/乱码。
        async with self._tex_compile_sema:
            try:
                with tempfile.TemporaryDirectory(prefix="astrbot_xelatex_") as td:
                    tex_path = os.path.join(td, "document.tex")
                    with open(tex_path, "w", encoding="utf-8") as f:
                        f.write(tex_src)

                    # 为 XeLaTeX/Fontconfig 提供可写缓存目录（避免 “No writable cache directories”）
                    env = os.environ.copy()
                    env["HOME"] = td
                    env["XDG_CACHE_HOME"] = os.path.join(td, ".cache")
                    try:
                        os.makedirs(env["XDG_CACHE_HOME"], exist_ok=True)
                    except Exception:
                        pass

                    cmd = ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape", "document.tex"]

                    loop = asyncio.get_running_loop()

                    # 默认只跑 1 次（更快）；若日志提示需要 rerun（交叉引用/目录等），再补跑第 2 次。
                    rerun_needed = False
                    for run_idx in range(2):
                        # 放到线程池里跑，避免阻塞 asyncio 事件循环
                        p = await loop.run_in_executor(
                            None,
                            lambda: subprocess.run(
                                cmd,
                                cwd=td,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                timeout=timeout_sec,
                                check=False,
                                env=env,
                            ),
                        )

                        if p.returncode != 0:
                            break

                        if run_idx == 0:
                            log_path = os.path.join(td, "document.log")
                            if os.path.exists(log_path):
                                try:
                                    with open(log_path, "r", encoding="utf-8", errors="ignore") as lf:
                                        log_txt = lf.read()
                                    rerun_needed = (
                                        ("Rerun to get cross-references right." in log_txt)
                                        or ("LaTeX Warning: Label(s) may have changed" in log_txt)
                                        or ("There were undefined references" in log_txt)
                                    )
                                except Exception:
                                    rerun_needed = False
                            if not rerun_needed:
                                break

                    pdf_path = os.path.join(td, "document.pdf")
                    if not os.path.exists(pdf_path):
                        # 尝试读取日志
                        log_path = os.path.join(td, "document.log")
                        if os.path.exists(log_path):
                            with open(log_path, "r", encoding="utf-8", errors="ignore") as lf:
                                self._last_texlive_log = lf.read()
                        return None
                    with open(pdf_path, "rb") as fpdf:
                        return fpdf.read()
            except subprocess.TimeoutExpired:
                self._last_texlive_log = "Local xelatex timeout"
                return None
            except Exception as e:
                self._last_texlive_log = f"Local xelatex error: {e}"
                return None

    # ---------------------------------------------------------------------
    # 可选：路由模型辅助判断（只看文本）
    # ---------------------------------------------------------------------
    async def _router_classify(self, user_text: str) -> Optional[Dict[str, bool]]:
        if not self._cfg("use_router_model", False):
            return None
        router_provider_id = str(self._cfg("router_provider_id", "") or "").strip()
        if not router_provider_id:
            return None

        prompt = (
            "你是一个意图路由器 只输出JSON 不要输出多余文本\n"
            "请根据用户消息判断下面字段 字段必须都给出\n"
            "{\n"
            '  "is_math": boolean,\n'
            '  "wants_full_solution": boolean\n'
            "}\n"
            f"用户消息：{user_text}"
        )
        try:
            llm_resp = await self.context.llm_generate(
                chat_provider_id=router_provider_id,
                prompt=prompt,
            )
            text = (llm_resp.completion_text or "").strip()
            m = re.search(r"\{[\s\S]*\}", text)
            if not m:
                return None
            data = json.loads(m.group(0))
            return {
                "is_math": bool(data.get("is_math")),
                "wants_full_solution": bool(data.get("wants_full_solution")),
            }
        except Exception as e:
            logger.warning(f"router classify failed: {e}")
            return None

    # ---------------------------------------------------------------------
    # 数学流程：注入提示
    # ---------------------------------------------------------------------
    @filter.on_llm_request()
    async def on_llm_req(self, event: AstrMessageEvent, req: ProviderRequest):
        if not self._cfg("enable_math_coach", True):
            return

        # /pdf 命令内部也会触发 llm_generate。此时我们不应再额外注入“解题教练”系统提示，
        # 否则可能破坏 /pdf 专用 LaTeX 输出格式，进而导致 xelatex 编译失败。
        try:
            if event.get_extra("md2img_pdf_generation"):
                # /pdf / /spdf 内部调用：避免工具注入破坏 LaTeX 输出格式或触发 tool_calls 解析问题
                try:
                    if bool(self._cfg("spdf_disable_tools_during_generation", True)):
                        self._disable_llm_tools_in_req(req)
                except Exception:
                    pass
                return
        except Exception:
            pass

        user_text = (getattr(event, "message_str", "") or "").strip()
        kb_query = _is_kb_query(user_text)
        kb_n = _extract_kb_pick_count(user_text, int(self._cfg("kb_default_pick_count", 2) or 2))

        # 确保知识库工具处于激活状态（适配 kb_agentic_mode 等场景）
        if kb_query:
            try:
                self.context.activate_llm_tool("astr_kb_search")
            except Exception:
                pass
        current_images = _get_event_images(event)
        reply_id = self._extract_reply_msg_id(event)
        is_private = self._is_private_chat(event)

        skey = _get_session_key(event)
        now_ts = time.time()

        # 私聊引用上一条“纯图片”消息：把那张图片带入本次 LLM 请求
        effective_images: List[str] = list(current_images or [])
        used_pending_image = False

        async with self._state_lock:
            state = MATH_SESSION_STATE.setdefault(skey, {})
            state["last_active_ts"] = now_ts

            # 当前消息带图：刷新 last_image_*
            if current_images:
                state["last_image_urls"] = list(current_images)
                state["last_image_ts"] = now_ts

            # 私聊：本条消息没有图，但“引用/回复”了上一条纯图片 -> 使用缓存图片作为上下文
            if (not effective_images) and is_private and user_text and reply_id and state.get("pending_image_only"):
                pid = str(state.get("pending_image_msg_id") or "").strip()
                if (not pid) or (str(reply_id).strip() == pid):
                    cached = state.get("last_image_urls") or []
                    if isinstance(cached, list) and cached:
                        effective_images = list(cached)
                        used_pending_image = True
                        # 用过就清掉 pending，避免后续串图
                        state.pop("pending_image_only", None)
                        state.pop("pending_image_msg_id", None)
                        state.pop("pending_image_ts", None)

            # 取快照供后续逻辑使用
            state_snapshot = dict(state)

        state = state_snapshot
        if used_pending_image:
            self._attach_images_to_req(req, effective_images)

        # 后续逻辑统一使用 current_images（可能来自 pending）
        current_images = effective_images


        # ---------------- 对话记忆：保存问答并在本次请求前注入相似历史 ----------------
        try:
            if bool(self._cfg("enable_chat_memory", True)):
                store_cmd = bool(self._cfg("chat_memory_store_commands", False))
                if user_text and (store_cmd or (not re.match(r"^\s*/\w+", user_text))):
                    # 标记本次请求，供 on_llm_resp 把“用户-助手”配对写入 history
                    event.set_extra("_chatmem_pending", {
                        "skey": skey,
                        "ts": now_ts,
                        "user": user_text,
                        "user_images": list(current_images) if current_images else [],
                    })

                # 构造检索片段并注入 system_prompt（不影响用户原 prompt）
                recent_n = int(self._cfg("chat_memory_recent_turns", 6) or 6)
                top_k = int(self._cfg("chat_memory_retrieve_k", 6) or 6)
                min_score = float(self._cfg("chat_memory_min_score", 0.12) or 0.12)
                char_budget = int(self._cfg("chat_memory_char_budget", 3500) or 3500)
                half_life = float(self._cfg("chat_memory_recency_half_life_sec", 21600) or 21600)

                async with self._state_lock:
                    st = MATH_SESSION_STATE.setdefault(skey, {})
                    hist = _ensure_chat_history(st, int(self._cfg("chat_memory_max_turns", 120) or 120))
                    snips = _select_memory_snippets(hist, user_text or "", now_ts,
                                                   recent_n=recent_n, top_k=top_k,
                                                   min_score=min_score, half_life_sec=half_life)
                mem_ctx = _format_memory_context(snips, char_budget)
                if mem_ctx:
                    req.system_prompt = (req.system_prompt or "") + mem_ctx
        except Exception as e:
            logger.warning(f"chat memory inject failed: {e}")

        treat_img_as_math = bool(self._cfg("treat_image_as_math", True))
        last_problem = state.get("last_problem", "")
        last_had_img = bool(state.get("last_had_img", False))

        # 注入 PDF 上下文（仅在用户“引用/回复”了机器人发出的 PDF 消息时才注入，避免影响下一题）
        # reply_id 已在前面提取
        pdf_ctx_to_inject = ""
        if reply_id:
            # 1) 优先：按“被引用消息 id -> LaTeX 源码”精确匹配
            try:
                mp = state.get("pdf_ctx_map", {})
                if isinstance(mp, dict):
                    pdf_ctx_to_inject = (mp.get(str(reply_id)) or "").strip()
            except Exception:
                pdf_ctx_to_inject = ""

            # 2) 兜底：如果无法获取 bot 消息 id（或历史版本没存），则在“存在引用”时弱绑定到最近一次 PDF
            if not pdf_ctx_to_inject:
                last_pdf_ctx = (state.get("last_pdf_context") or "").strip()
                if last_pdf_ctx:
                    last_mid = str(state.get("last_pdf_msg_id", "") or "").strip()
                    if last_mid:
                        # 只有引用 id 与最后一次 PDF 消息一致时才注入（避免串题）
                        if str(reply_id) == last_mid:
                            pdf_ctx_to_inject = last_pdf_ctx
                        else:
                            # 某些适配器能拿到被引用消息内容：若确实包含 pdf，则允许兜底
                            if self._reply_msg_has_pdf_hint(event):
                                pdf_ctx_to_inject = last_pdf_ctx
                    else:
                        # 没有任何 message_id 绑定信息：只能退化为“只要用户引用，就使用最近一次 PDF”
                        pdf_ctx_to_inject = last_pdf_ctx

        if pdf_ctx_to_inject and user_text:
            req.system_prompt = (req.system_prompt or "") + (
                "\n[历史信息] 用户正在引用你之前发出的 PDF 解答消息。以下是该 PDF 的 LaTeX 源码（用于回答用户对 PDF 的追问）。\n"
                "仅在回答与该 PDF 相关的问题时使用这些信息；若用户问的是新问题，不要被历史信息干扰。\n"
                f"[START LATEX CONTEXT]\n{pdf_ctx_to_inject}\n[END LATEX CONTEXT]\n"
            )

        router = await self._router_classify(user_text) if user_text else None

        # 高优先级：用户明确要求“联网/上网/搜索/查论文”等 → 不走数学提示流/图片解题管线
        # 直接交给 AstrBot 默认的 agent + 联网工具去处理
        if bool(self._cfg("bypass_math_coach_on_web_search", True)):
            kws = _split_keywords(self._cfg("web_search_keywords", "")) or list(_DEFAULT_WEB_SEARCH_KEYWORDS)
            if _is_web_search_intent(user_text, kws):
                return


        heuristic_is_math = _is_math_question(user_text)
        wants_full = _wants_full_solution(user_text) or (router.get("wants_full_solution") if router else False)

        # 知识库检索意图：强制 full（可通过配置关闭）
        if kb_query and bool(self._cfg("force_full_on_kb_query", True)):
            wants_full = True

        is_math = heuristic_is_math or (router.get("is_math") if router else False)
        if kb_query:
            # 让知识库检索问法也走“数学答疑”管线（输出 <md> 图片）
            is_math = True
        if bool(current_images) and treat_img_as_math:
            is_math = True

        is_followup_full = (not is_math) and wants_full and (last_problem or last_had_img)
        if not (is_math or is_followup_full):
            return

        # 记录“上一题”
        if is_math:
            async with self._state_lock:
                st = MATH_SESSION_STATE.setdefault(skey, {})
                st["last_active_ts"] = time.time()
                if user_text:
                    st["last_problem"] = user_text
                else:
                    st["last_problem"] = "[图片题目]"
                st["last_had_img"] = bool(current_images)

        mode = "full" if wants_full else "hint"
        event.set_extra("math_flow",
                        {"mode": mode, "has_image": bool(current_images), "kb_query": kb_query, "kb_n": kb_n})

        if mode == "full" and is_followup_full:
            req.prompt = (
                "用户要求你给出上一题的完整解答过程\n"
                "请结合上下文中上一题（可能是图片题）来解题\n"
                f"用户补充：{user_text}\n"
            )

        persona = str(self._cfg("math_persona", DEFAULT_CFG["math_persona"]) or "").strip()
        hint_n = int(self._cfg("hint_message_count", 3) or 3)
        req.system_prompt = (req.system_prompt or "") + self._build_math_system_prompt(
            mode=mode,
            persona=persona,
            hint_n=hint_n,
            has_image=bool(current_images),
            kb_query=kb_query,
            kb_n=kb_n,
        )

    def _build_math_system_prompt(self, mode: str, persona: str, hint_n: int, has_image: bool, kb_query: bool = False,
                                  kb_n: int = 2) -> str:
        hint_n = max(2, min(int(hint_n or 3), 8))
        persona = persona.strip() or DEFAULT_CFG["math_persona"]
        img_hint = "用户发来的是图片题 请先读题 再给提示" if has_image else ""

        if mode == "full":
            if kb_query:
                # 知识库检索/题库挑题：不要输出提示流，直接给“完整回答格式”
                return (
                    "\n[数学答疑/知识库]\n"
                    f"你的人格设定：{persona}\n"
                    "用户希望你从知识库/题库中检索相关题目，并且给出具体出处\n"
                    "你必须用 <md>...</md> 包裹整个回答内容\n"
                    "在 <md> 内先输出：\n"
                    "## 知识库检索结果\n"
                    f"- 你要给出最相关的 {max(1, min(int(kb_n or 2), 10))} 条结果（若系统检索不到，明确说明未命中）\n"
                    "- 每条结果必须包含：题目内容（可用 LaTeX）、出处（文档/条目标题/章节/页码/ID 等）\n"
                    "- 不要只输出泛泛的引导或学习建议，不要输出 <stream>\n"
                    "如果用户同时给了要解的具体题目或要求解答，请在检索结果后追加：\n"
                    "## 解答\n"
                    "并给出该题完整推导与最终答案（可用 LaTeX 例如 $$...$$）\n"
                )

            return (
                "\n[数学答疑]\n"
                f"你的人格设定：{persona}\n"
                "现在用户明确要求完整解答\n"
                "你必须给出完整的解题过程 并且必须用 <md>...</md> 包裹整个解答内容\n"
                "<md> 内允许使用 Markdown 与 LaTeX 例如 $$...$$\n"
                "步骤要清晰 推导要完整 最后给出明确答案\n"
            )

        return (
                "\n[数学答疑]\n"
                f"你的人格设定：{persona}\n"
                "用户没有要求完整过程 你要像真人答疑一样先引导\n"
                + (img_hint + "\n" if img_hint else "")
                + "输出格式必须严格如下 只输出这一段 不要输出其它内容\n"
                  "<stream>\n"
                  "</stream>\n"
                  "规则\n"
                + f"必须输出恰好{hint_n}条短消息 每条占一行\n"
                  "每条消息只写1-2句话\n"
                  "每条消息句末不要出现句号\n"
                  "不要用 Markdown 不要用 LaTeX 不要用代码块\n"
                  "不要直接给最终答案或完整推导\n"
                  "如果图片或题面不清晰 先让用户补充关键条件或把题抄成文字\n"
        )

    # ---------- 可选：用二次模型把输出改写成 <stream> ----------
    async def _rewrite_to_stream(self, raw: str, n: int, has_image: bool) -> Optional[List[str]]:
        if not self._cfg("use_hint_rewriter_model", False):
            return None
        pid = str(self._cfg("hint_rewriter_provider_id", "") or "").strip()
        if not pid:
            return None
        n = max(2, min(int(n or 3), 8))
        prompt = (
                "把下面的回答改写成真人答疑式提示流\n"
                f"必须输出恰好{n}行 每行1-2句话 句末不要加句号\n"
                "不要用Markdown 不要用LaTeX 不要给最终答案\n"
                "只输出 <stream>...</stream> 这一段\n"
                + ("如果图片题看不清 先让用户补充关键条件\n" if has_image else "")
                + "原始文本如下：\n"
                + raw
        )
        try:
            llm_resp = await self.context.llm_generate(chat_provider_id=pid, prompt=prompt)
            lines = _extract_stream_lines_only(llm_resp.completion_text or "")
            if lines:
                return _normalize_to_n_msgs(lines, n)
        except Exception as e:
            logger.warning(f"hint rewrite failed: {e}")
        return None

    # ---------- LLM响应：full 强制 <md>；hint 预计算提示流 ----------
    @filter.on_llm_response()
    async def on_llm_resp(self, event: AstrMessageEvent, resp: LLMResponse):
        raw = resp.completion_text or ""
        event.set_extra("raw_llm", raw)

        mf = event.get_extra("math_flow")
        if not isinstance(mf, dict):
            # 非数学流程：也写入对话记忆（普通聊天）
            try:
                pending = event.get_extra("_chatmem_pending")
                if isinstance(pending, dict) and bool(self._cfg("enable_chat_memory", True)):
                    skey = str(pending.get("skey") or "")
                    async with self._state_lock:
                        st = MATH_SESSION_STATE.setdefault(skey, {})
                        hist = _ensure_chat_history(st, int(self._cfg("chat_memory_max_turns", 120) or 120))
                        hist.append({
                            "ts": float(pending.get("ts", time.time()) or time.time()),
                            "user": str(pending.get("user") or ""),
                            "assistant": _strip_tags_for_memory(resp.completion_text or ""),
                            "user_images": list(pending.get("user_images") or []),
                        })
            except Exception as e:
                logger.warning(f"chat memory store(normal) failed: {e}")
            return

        mode = mf.get("mode")
        has_img = bool(mf.get("has_image"))
        hint_n = int(self._cfg("hint_message_count", 3) or 3)

        if mode == "full":
            md_blocks = re.findall(r"<md>([\s\S]*?)</md>", raw)
            if md_blocks:
                joined = "\n\n".join([b.strip() for b in md_blocks if b.strip()]).strip()
                resp.completion_text = f"<md>\n{joined or raw.strip()}\n</md>"
            else:
                resp.completion_text = f"<md>\n{raw.strip()}\n</md>"
            event.set_extra("raw_llm", resp.completion_text)
            # 写入对话记忆（full）
            try:
                pending = event.get_extra("_chatmem_pending")
                if isinstance(pending, dict) and bool(self._cfg("enable_chat_memory", True)):
                    skey = str(pending.get("skey") or "")
                    async with self._state_lock:
                        st = MATH_SESSION_STATE.setdefault(skey, {})
                        hist = _ensure_chat_history(st, int(self._cfg("chat_memory_max_turns", 120) or 120))
                        hist.append({
                            "ts": float(pending.get("ts", time.time()) or time.time()),
                            "user": str(pending.get("user") or ""),
                            "assistant": _strip_tags_for_memory(resp.completion_text or ""),
                            "user_images": list(pending.get("user_images") or []),
                        })
            except Exception as e:
                logger.warning(f"chat memory store(full) failed: {e}")
            return

        if mode == "hint":
            lines = _extract_stream_lines_only(raw)

            if (not lines) or _looks_like_markdown_or_full(raw):
                rewritten = await self._rewrite_to_stream(raw, hint_n, has_img)
                if rewritten:
                    msgs = rewritten
                else:
                    msgs = _normalize_to_n_msgs([], hint_n)
            else:
                msgs = _normalize_to_n_msgs(lines, hint_n)

            event.set_extra("math_stream_msgs", msgs)
            # 写入对话记忆（hint：把提示流拼成一段存）
            try:
                pending = event.get_extra("_chatmem_pending")
                if isinstance(pending, dict) and bool(self._cfg("enable_chat_memory", True)):
                    skey = str(pending.get("skey") or "")
                    assistant_text = "\n".join([str(x) for x in msgs if str(x).strip()])
                    async with self._state_lock:
                        st = MATH_SESSION_STATE.setdefault(skey, {})
                        hist = _ensure_chat_history(st, int(self._cfg("chat_memory_max_turns", 120) or 120))
                        hist.append({
                            "ts": float(pending.get("ts", time.time()) or time.time()),
                            "user": str(pending.get("user") or ""),
                            "assistant": _strip_tags_for_memory(assistant_text),
                            "user_images": list(pending.get("user_images") or []),
                        })
            except Exception as e:
                logger.warning(f"chat memory store(hint) failed: {e}")

    # ---------------- 装饰结果：hint 逐条消息；否则 Markdown->图片 ----------------
    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent):
        mf = event.get_extra("math_flow")
        if isinstance(mf, dict) and mf.get("mode") == "hint":
            msgs = event.get_extra("math_stream_msgs")
            if not isinstance(msgs, list) or not msgs:
                raw_llm = event.get_extra("raw_llm") or ""
                lines = _extract_stream_lines_only(raw_llm)
                msgs = _normalize_to_n_msgs(lines, int(self._cfg("hint_message_count", 3) or 3))

            delay_ms = int(self._cfg("hint_send_delay_ms", 0) or 0)
            for m in msgs:
                m = _strip_trailing_periods(_sanitize_hint_text(str(m)))
                if not m:
                    continue
                try:
                    await event.send(MessageChain().message(m))
                except Exception as e:
                    logger.error(f"send stream msg failed: {e}")
                if delay_ms > 0:
                    await asyncio.sleep(delay_ms / 1000)

            # 阻止原始输出继续发送
            try:
                result = event.get_result()
                result.chain = []
            except Exception:
                pass

            if bool(self._cfg("stop_event_on_hint", True)):
                try:
                    event.stop_event()
                except Exception:
                    pass
            return

        # 非 hint：保持原插件行为（Markdown -> 图片）
        result = event.get_result()
        new_chain: List[Any] = []
        for item in result.chain:
            if isinstance(item, Plain):
                components = await self._process_text_with_markdown(item.text, event)
                new_chain.extend(components)
            else:
                new_chain.append(item)
        result.chain = new_chain

    # ---------------------------------------------------------------------
    # Markdown 标签处理（使用复用浏览器渲染）
    # ---------------------------------------------------------------------
    async def _process_text_with_markdown(self, text: str, event: AstrMessageEvent = None) -> List[Any]:
        if "<md>" not in text:
            is_complex = re.search(r"(```|\$\$|\|.*?\|.*?\||\\begin\{)", text, re.DOTALL)
            if is_complex:
                text = f"<md>\n{text}\n</md>"

        user_mode = "mobile"
        if event:
            user_mode = get_user_mode(event)

        components: List[Any] = []
        parts = re.split(r"(<md>.*?</md>)", text, flags=re.DOTALL)

        for part in parts:
            if part.startswith("<md>") and part.endswith("</md>"):
                md_content = part[4:-5].strip()
                if not md_content:
                    continue

                output_path = os.path.join(self.IMAGE_CACHE_DIR, f"{uuid.uuid4()}.png")

                try:
                    img_paths = await self._markdown_to_image_playwright(
                        md_text=md_content,
                        output_image_path=output_path,
                        scale=2,
                        mode=user_mode,
                    )

                    if img_paths:
                        for p in img_paths:
                            components.append(Image.fromFileSystem(p))
                    else:
                        components.append(Plain(f"--- 渲染失败 ---\n{md_content}"))
                except Exception as e:
                    logger.error(f"渲染异常: {e}")
                    components.append(Plain(f"--- 渲染错误 ---\n{md_content}"))
            else:
                if part:
                    components.append(Plain(part))

        return components

    # ---------------------------------------------------------------------
    # Playwright 渲染实现
    # ---------------------------------------------------------------------
    async def _markdown_to_image_playwright(
            self,
            md_text: str,
            output_image_path: str,
            scale: int = 2,
            mode: str = "mobile",
    ) -> List[str]:
        async with self._render_sema:
            # 尽量复用 browser；若无法复用则本次临时启动
            browser = self._browser
            pw = self._pw

            if bool(self._cfg("reuse_playwright_browser", True)) and browser is None:
                await self._ensure_browser()
                browser = self._browser
                pw = self._pw

            return await self._markdown_to_image_playwright_impl(
                md_text=md_text,
                output_image_path=output_image_path,
                scale=scale,
                mode=mode,
                shared_browser=browser,
                shared_pw=pw,
            )

    async def _markdown_to_image_playwright_impl(
            self,
            md_text: str,
            output_image_path: str,
            scale: int = 2,
            mode: str = "mobile",
            shared_browser=None,
            shared_pw=None,
    ) -> List[str]:
        # --- 1. 根据模式设定参数 ---
        if mode == "pc":
            target_width = 794
            padding_val = 45
            font_size = "15px"
            is_mobile_css = False
        else:
            target_width = 600
            padding_val = 20
            font_size = "18px"
            is_mobile_css = True

        # --- 2. 配色逻辑 ---
        now_hour = datetime.now().hour
        is_night_mode = now_hour >= 23 or now_hour < 6

        if is_night_mode:
            css_bg_color = "#0d1117"
            css_text_color = "#c9d1d9"
            css_pre_bg = "#161b22"
            css_border_color = "#30363d"
            css_code_color = "#c9d1d9"
        else:
            css_bg_color = "white"
            css_text_color = "black"
            css_pre_bg = "#f6f8fa"
            css_border_color = "#dfe2e5"
            css_code_color = "black"

        # --- 3. CSS 样式 ---
        base_css = f"""
            * {{ box-sizing: border-box; }}
            body {{
                margin: 0; padding: 0;
                background-color: {css_bg_color};
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            }}
            .content-wrapper {{
                background-color: {css_bg_color};
                color: {css_text_color};
                font-size: {font_size};
                line-height: 1.6;

                width: {target_width}px;
                padding: {padding_val}px;

                overflow-wrap: break-word;
                word-wrap: break-word;
            }}

            img, svg, video, iframe {{ max-width: 100% !important; height: auto; }}
            pre {{
                background-color: {css_pre_bg};
                color: {css_code_color};
                border-radius: 6px;
                padding: 12px;
                white-space: pre-wrap;
                word-break: break-all;
                max-width: 100%;
                font-size: 85%;
                margin: 10px 0;
                page-break-inside: avoid;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                table-layout: fixed;
                margin-bottom: 16px;
                page-break-inside: avoid;
            }}
            th, td {{
                border: 1px solid {css_border_color};
                padding: 6px;
                word-wrap: break-word;
            }}
            .MathJax_Display, .MathJax {{
                max-width: 100% !important;
                overflow-x: hidden; overflow-y: hidden;
            }}

            /* Paged.js 样式 */
            .pagedjs_pages .content-wrapper {{ width: auto !important; padding: 0 !important; }}
            .pagedjs_page {{ background-color: {css_bg_color} !important; box-shadow: none !important; overflow: hidden; }}
            .pagedjs_margin-top, .pagedjs_margin-bottom {{ display: none; }}
        """

        # --- 4. HTML 模板 ---
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>{css_content}</style>
            <script type="text/x-mathjax-config">
                MathJax.Hub.Config({{
                    tex2jax: {{ inlineMath: [['$','$']], displayMath: [['$$','$$']], processEscapes: true }},
                    "HTML-CSS": {{ scale: 90, linebreaks: {{ automatic: true, width: "container" }} }},
                    SVG: {{ linebreaks: {{ automatic: true, width: "container" }} }}
                }});
            </script>
            <script type="text/javascript"
              src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML"></script>
        </head>
        <body>
            <div class="content-wrapper" id="main-content">{content}</div>
        </body>
        </html>
        """

        # --- 5. Markdown -> HTML ---
        protected_md, math_pieces = _protect_math_for_markdown(md_text)

        markdown_parser = mistune.create_markdown(
            escape=False, plugins=["table", "url", "strikethrough", "task_lists"]
        )
        html_content = markdown_parser(protected_md)
        html_content = _restore_math_tokens(html_content, math_pieces)

        full_html = html_template.format(css_content=base_css, content=html_content)

        generated_images: List[str] = []

        # browser 复用：若传入 shared_browser，则只创建 context/page；否则本次启动
        if shared_browser is not None:
            browser = shared_browser
            own_pw_cm = None
        else:
            own_pw_cm = async_playwright()

        if own_pw_cm is not None:
            async with own_pw_cm as p:
                browser = await p.chromium.launch()
                generated_images = await self._render_with_browser(browser, full_html, output_image_path, scale, mode,
                                                                   is_mobile_css)
                try:
                    await browser.close()
                except Exception:
                    pass
            return generated_images

        # shared browser path
        return await self._render_with_browser(browser, full_html, output_image_path, scale, mode, is_mobile_css)

    async def _render_with_browser(self, browser, full_html: str, output_image_path: str, scale: int, mode: str,
                                   is_mobile_css: bool) -> List[str]:
        generated_images: List[str] = []

        vp_width = (794 if mode == "pc" else 600) + 50
        context = await browser.new_context(
            device_scale_factor=scale, viewport={"width": vp_width, "height": 2000}
        )
        page = await context.new_page()

        wait_until = str(self._cfg("playwright_wait_until", "networkidle") or "networkidle")
        try:
            await page.set_content(full_html, wait_until=wait_until)
        except Exception:
            # 回退策略：networkidle 卡住时，使用 domcontentloaded
            await page.set_content(full_html, wait_until="domcontentloaded")

        # 等待 MathJax 加载并完成排版
        try:
            await page.wait_for_function("window.MathJax && MathJax.Hub && MathJax.Hub.Queue", timeout=15000)
            await page.evaluate("""
                () => new Promise((resolve) => {
                    MathJax.Hub.Queue(["Typeset", MathJax.Hub], resolve);
                })
            """)
        except Exception as e:
            logger.warning(f"MathJax typeset skipped: {e}")
        await page.wait_for_timeout(5000)
        # === Mobile: 长截屏 ===
        if is_mobile_css:
            element = await page.query_selector(".content-wrapper")
            if element:
                await element.screenshot(path=output_image_path)
                generated_images.append(output_image_path)
            else:
                logger.error("Render Error: content-wrapper not found")
        else:
            # === PC: 智能分页检测 ===
            try:
                content_height = await page.evaluate("document.getElementById('main-content').scrollHeight")
            except Exception:
                content_height = 0

            A4_HEIGHT = 1123

            if content_height <= A4_HEIGHT:
                element = await page.query_selector(".content-wrapper")
                if element:
                    await element.screenshot(path=output_image_path)
                    generated_images.append(output_image_path)
            else:
                paged_style = """
                    @page { size: 794px 1123px; margin: 45px; }
                    .pagedjs_page { box-sizing: border-box; }
                """
                await page.add_style_tag(content=paged_style)

                current_dir = os.path.dirname(os.path.abspath(__file__))
                local_js = os.path.join(current_dir, "paged.polyfill.js")
                if os.path.exists(local_js):
                    await page.add_script_tag(path=local_js)
                else:
                    await page.add_script_tag(url="https://unpkg.com/pagedjs/dist/paged.polyfill.js")

                try:
                    await page.wait_for_selector(".pagedjs_page", timeout=15000)
                    await page.wait_for_timeout(5000)
                except Exception as e:
                    logger.error(f"PagedJS Timeout: {e}")

                try:
                    await page.wait_for_function("window.MathJax && MathJax.Hub && MathJax.Hub.Queue", timeout=15000)
                    await page.evaluate("""
                        () => new Promise((resolve) => {
                            MathJax.Hub.Queue(["Typeset", MathJax.Hub], resolve);
                        })
                    """)
                except Exception as e:
                    logger.warning(f"MathJax typeset skipped(after pagedjs): {e}")

                pages = await page.query_selector_all(".pagedjs_page")
                base_path, ext = os.path.splitext(output_image_path)
                for i, page_elem in enumerate(pages):
                    curr_path = output_image_path if i == 0 else f"{base_path}_{i + 1}{ext}"
                    await page_elem.screenshot(path=curr_path)
                    generated_images.append(curr_path)

        try:
            await page.close()
            await context.close()
        except Exception:
            pass

        return generated_images
