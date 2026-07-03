# -*- coding: utf-8 -*-
try:
    from .shared import *
except ImportError:
    from shared import *

from datetime import timedelta


class DailyReportMixin:
    """Daily Q&A archive and lecture-report generation."""

    _DAILY_REPORT_SECTION_TITLES = [
        "今日知识地图",
        "专题讲义",
        "今日高频误区",
        "今日典型题精讲",
    ]

    _DAILY_REPORT_INTERNAL_DEFAULTS = {
        "compile_max_rounds": 3,
        "record_retention_days": 7,
        "pdf_retention_days": 7,
        "failed_retention_days": 3,
        "save_tex": True,
        "delete_after_auto_send": True,
        "template": "lecture",
    }

    def _daily_report_init_storage(self) -> None:
        base = getattr(self, "DAILY_REPORT_DIR", "")
        if not base:
            base = os.path.join(getattr(self, "DATA_DIR", os.getcwd()), "daily_reports")
            self.DAILY_REPORT_DIR = base
        self.DAILY_REPORT_RECORD_DIR = getattr(self, "DAILY_REPORT_RECORD_DIR", os.path.join(base, "records"))
        self.DAILY_REPORT_TEX_DIR = getattr(self, "DAILY_REPORT_TEX_DIR", os.path.join(base, "tex"))
        self.DAILY_REPORT_OUTPUT_DIR = getattr(self, "DAILY_REPORT_OUTPUT_DIR", os.path.join(base, "reports"))
        os.makedirs(self.DAILY_REPORT_RECORD_DIR, exist_ok=True)
        os.makedirs(self.DAILY_REPORT_TEX_DIR, exist_ok=True)
        os.makedirs(self.DAILY_REPORT_OUTPUT_DIR, exist_ok=True)

    def _daily_report_state_path(self) -> str:
        self._daily_report_init_storage()
        return os.path.join(self.DAILY_REPORT_DIR, "state.json")

    def _daily_report_load_state(self) -> Dict[str, Any]:
        path = self._daily_report_state_path()
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception as e:
            logger.warning(f"[daily_report] 读取状态失败: {e}")
        return {}

    def _daily_report_save_state(self, state: Dict[str, Any]) -> None:
        path = self._daily_report_state_path()
        tmp = path + f".tmp-{uuid.uuid4().hex}"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state or {}, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except Exception as e:
            logger.warning(f"[daily_report] 保存状态失败: {e}")
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

    def _daily_report_int_cfg(self, key: str, default: int) -> int:
        try:
            raw = self._cfg(key, default)
            if raw is None or str(raw).strip() == "":
                return int(default)
            return int(raw)
        except Exception:
            return int(default)

    @staticmethod
    def _daily_report_date_key(ts: Optional[float] = None) -> str:
        dt = datetime.fromtimestamp(float(ts if ts is not None else time.time()))
        return dt.strftime("%Y-%m-%d")

    @staticmethod
    def _daily_report_display_time(ts: float) -> str:
        try:
            return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return ""

    @staticmethod
    def _daily_report_start_of_day_ts(ts: Optional[float] = None) -> float:
        dt = datetime.fromtimestamp(float(ts if ts is not None else time.time()))
        return dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

    @staticmethod
    def _daily_report_safe_filename(text: str, fallback: str = "daily_report") -> str:
        s = re.sub(r"[^\w.\-]+", "_", str(text or "").strip(), flags=re.UNICODE).strip("._")
        return s or fallback

    def _daily_report_event_meta(self, event: AstrMessageEvent) -> Dict[str, str]:
        origin = ""
        try:
            origin = str(getattr(event, "unified_msg_origin", "") or "")
        except Exception:
            origin = ""
        session_key = ""
        try:
            session_key = _get_session_key(event)
        except Exception:
            session_key = origin or "unknown"

        sender_id = ""
        sender_name = ""
        try:
            sender_id = str(event.get_sender_id() or "")
        except Exception:
            pass
        try:
            sender_name = str(event.get_sender_name() or "")
        except Exception:
            pass

        msg_obj = getattr(event, "message_obj", None)

        def read_attr(names: Tuple[str, ...]) -> str:
            for root in (msg_obj, event):
                if root is None:
                    continue
                for name in names:
                    try:
                        value = getattr(root, name, None)
                    except Exception:
                        value = None
                    if isinstance(value, (str, int)) and str(value).strip():
                        return str(value).strip()
            return ""

        group_id = read_attr(("group_id", "groupId", "group", "room_id", "roomId", "chat_id", "chatId"))
        guild_id = read_attr(("guild_id", "guildId", "guild", "server_id", "serverId"))
        channel_id = read_attr(("channel_id", "channelId", "channel"))
        if not group_id and origin and re.search(r"(group|room|chat)", origin, flags=re.I):
            m = re.search(r"(\d{4,})", origin)
            if m:
                group_id = m.group(1)

        return {
            "unified_msg_origin": origin,
            "session_key": session_key,
            "group_id": group_id,
            "guild_id": guild_id,
            "channel_id": channel_id,
            "sender_id": sender_id,
            "sender_name": sender_name,
        }

    async def _daily_report_record_answer(
        self,
        event: AstrMessageEvent,
        kind: str,
        question: str,
        answer: str = "",
        tex_src: str = "",
        pdf_path: str = "",
        image_urls: Optional[List[str]] = None,
    ) -> None:
        if not bool(self._cfg("enable_daily_report", False)):
            return
        try:
            self._daily_report_init_storage()
            image_urls = [str(x) for x in (image_urls or []) if str(x or "").strip()]
            question = _strip_known_xml_like_tags(str(question or "").strip())
            answer = _strip_known_xml_like_tags(str(answer or "").strip())
            if not question and image_urls:
                question = "[图片题目]"
            if not answer and tex_src:
                answer = "PDF/LaTeX 解答已生成，源码已归档。"
            if not question and not answer and not tex_src:
                return

            now_ts = time.time()
            date_key = self._daily_report_date_key(now_ts)
            record_id = f"{datetime.fromtimestamp(now_ts).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            tex_path = ""
            tex_src = str(tex_src or "").strip()
            if tex_src and bool(self._DAILY_REPORT_INTERNAL_DEFAULTS["save_tex"]):
                tex_dir = os.path.join(self.DAILY_REPORT_TEX_DIR, date_key)
                os.makedirs(tex_dir, exist_ok=True)
                tex_path = os.path.join(tex_dir, f"{record_id}.tex")
                with open(tex_path, "w", encoding="utf-8") as f:
                    f.write(tex_src)

            meta = self._daily_report_event_meta(event)
            rec = {
                "id": record_id,
                "ts": now_ts,
                "iso": datetime.fromtimestamp(now_ts).isoformat(timespec="seconds"),
                "kind": str(kind or "answer"),
                "question": question,
                "answer": answer,
                "tex_path": tex_path,
                "pdf_path": str(pdf_path or ""),
                "image_count": len(image_urls),
                "image_refs": image_urls,
                **meta,
            }
            record_path = os.path.join(self.DAILY_REPORT_RECORD_DIR, f"{date_key}.jsonl")
            with open(record_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._daily_report_cleanup_old_files()
        except Exception as e:
            logger.warning(f"[daily_report] 归档答疑记录失败: {e}")

    def _daily_report_record_files_between(self, start_ts: float, end_ts: float) -> List[str]:
        self._daily_report_init_storage()
        start_dt = datetime.fromtimestamp(float(start_ts)).date()
        end_dt = datetime.fromtimestamp(float(end_ts)).date()
        paths: List[str] = []
        cur = start_dt
        while cur <= end_dt:
            p = os.path.join(self.DAILY_REPORT_RECORD_DIR, f"{cur.strftime('%Y-%m-%d')}.jsonl")
            if os.path.isfile(p):
                paths.append(p)
            cur = cur + timedelta(days=1)
        return paths

    @staticmethod
    def _daily_report_record_matches_origin(rec: Dict[str, Any], token: str) -> bool:
        token = str(token or "").strip()
        if not token:
            return True
        fields = [
            str(rec.get("unified_msg_origin") or ""),
            str(rec.get("session_key") or ""),
            str(rec.get("group_id") or ""),
            str(rec.get("guild_id") or ""),
            str(rec.get("channel_id") or ""),
        ]
        return any(token == x or (token and token in x) for x in fields if x)

    def _daily_report_load_records(
        self,
        start_ts: float,
        end_ts: float,
        origin_token: str = "",
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for path in self._daily_report_record_files_between(start_ts, end_ts):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        ts = float(rec.get("ts", 0) or 0)
                        if ts <= float(start_ts) or ts > float(end_ts):
                            continue
                        if origin_token and not self._daily_report_record_matches_origin(rec, origin_token):
                            continue
                        records.append(rec)
            except Exception as e:
                logger.warning(f"[daily_report] 读取记录失败: {path} err={e}")
        records.sort(key=lambda x: float(x.get("ts", 0) or 0))
        max_records = self._daily_report_int_cfg("daily_report_max_records", 80)
        if max_records > 0 and len(records) > max_records:
            records = records[-max_records:]
        return records

    def _daily_report_origin_from_event(self, event: AstrMessageEvent) -> str:
        try:
            origin = str(getattr(event, "unified_msg_origin", "") or "").strip()
            if origin:
                return origin
        except Exception:
            pass
        try:
            return _get_session_key(event)
        except Exception:
            return "unknown"

    def _daily_report_unique_origins(self, records: List[Dict[str, Any]]) -> List[str]:
        out: List[str] = []
        seen = set()
        for rec in records:
            origin = str(rec.get("unified_msg_origin") or rec.get("session_key") or "").strip()
            if not origin or origin in seen:
                continue
            seen.add(origin)
            out.append(origin)
        return out

    def _daily_report_split_target_tokens(self) -> List[str]:
        raw = str(self._cfg("daily_report_target_groups", "") or "").strip()
        if not raw:
            return []
        return [x.strip() for x in re.split(r"[,;\n]+", raw) if x.strip()]

    def _daily_report_resolve_auto_origins(self, records: List[Dict[str, Any]]) -> List[str]:
        tokens = self._daily_report_split_target_tokens()
        if not tokens:
            return self._daily_report_unique_origins(records)
        origins: List[str] = []
        seen = set()
        for token in tokens:
            matched = False
            for rec in records:
                if not self._daily_report_record_matches_origin(rec, token):
                    continue
                origin = str(rec.get("unified_msg_origin") or rec.get("session_key") or "").strip()
                if origin and origin not in seen:
                    seen.add(origin)
                    origins.append(origin)
                    matched = True
            if (not matched) and re.search(r":|group|guild|channel|room", token, flags=re.I):
                if token not in seen:
                    seen.add(token)
                    origins.append(token)
        return origins

    async def _daily_report_resolve_provider_id(
        self,
        event: Optional[AstrMessageEvent] = None,
        repair: bool = False,
        fallback_provider_id: str = "",
    ) -> str:
        key = "daily_report_repair_provider_id" if repair else "daily_report_provider_id"
        pid = str(self._cfg(key, "") or "").strip()
        if pid:
            return pid
        if repair and fallback_provider_id:
            return str(fallback_provider_id or "").strip()
        if event is not None:
            try:
                current = await self.context.get_current_chat_provider_id(umo=getattr(event, "unified_msg_origin", ""))
                if current:
                    return str(current).strip()
            except Exception:
                pass
        return str(fallback_provider_id or "").strip()

    async def _daily_report_llm_generate(self, provider_id: str, prompt: str, purpose: str) -> str:
        timeout_sec = self._daily_report_int_cfg("daily_report_timeout_sec", 300)
        kwargs = {"prompt": prompt, "image_urls": []}
        if str(provider_id or "").strip():
            kwargs["chat_provider_id"] = str(provider_id or "").strip()
        call = self.context.llm_generate(**kwargs)
        resp = await asyncio.wait_for(call, timeout=timeout_sec) if timeout_sec > 0 else await call
        raw = (
            getattr(resp, "completion_text", None)
            or getattr(resp, "completion", None)
            or getattr(resp, "text", None)
            or ""
        )
        raw = str(raw or "").strip()
        if not raw:
            raise RuntimeError(f"{purpose}: empty_completion")
        return raw

    @staticmethod
    def _daily_report_strip_code_fence(text: str) -> str:
        s = str(text or "").strip()
        m = re.search(r"```(?:markdown|md|text)?\s*([\s\S]*?)```", s, flags=re.I)
        if m:
            return m.group(1).strip()
        return s

    def _daily_report_records_to_prompt_text(self, records: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for idx, rec in enumerate(records, 1):
            q = str(rec.get("question") or "").strip()
            a = str(rec.get("answer") or "").strip()
            tex_text = ""
            tex_path = str(rec.get("tex_path") or "").strip()
            if tex_path and os.path.isfile(tex_path):
                try:
                    with open(tex_path, "r", encoding="utf-8", errors="ignore") as f:
                        tex_text = f.read().strip()
                except Exception:
                    tex_text = ""
            content = tex_text or a
            if tex_text:
                content = "【LaTeX源码】\n" + tex_text
            item = [
                f"### 答疑 {idx}",
                f"- 时间: {self._daily_report_display_time(float(rec.get('ts', 0) or 0))}",
                f"- 类型: {rec.get('kind', '')}",
                f"- 提问者: {rec.get('sender_name') or rec.get('sender_id') or '未知'}",
                f"- 图片数量: {int(rec.get('image_count', 0) or 0)}",
                "",
                "问题:",
                q or "[无文字题目]",
                "",
                "AI解答:",
                content or "[无文本记录]",
            ]
            parts.append("\n".join(item))
        return "\n\n".join(parts)

    def _daily_report_build_chunk_prompt(self, records: List[Dict[str, Any]], start_ts: float, end_ts: float) -> str:
        return (
            "你是数学答疑记录整理员。请阅读下面一段答疑记录，提取可用于日报讲义的素材。\n"
            "输出要求：只输出简洁中文 Markdown，不要写 LaTeX 导言区，不要写代码块。\n"
            "不要使用 Markdown 加粗语法 **...**；小标题请用 ## 或 ###。\n"
            "重点提取：核心知识点、题目之间的联系、可合并进专题讲义的脉络、常见误区、典型题和关键推导。\n"
            f"统计区间：{self._daily_report_display_time(start_ts)} 到 {self._daily_report_display_time(end_ts)}\n\n"
            + self._daily_report_records_to_prompt_text(records)
        )

    def _daily_report_build_final_prompt(self, source_text: str, start_ts: float, end_ts: float, record_count: int) -> str:
        max_topics = self._daily_report_int_cfg("daily_report_max_topics", 6)
        max_examples = self._daily_report_int_cfg("daily_report_max_examples", 4)
        titles = "\n".join(f"# {x}" for x in self._DAILY_REPORT_SECTION_TITLES)
        return (
            "你是大学数学答疑讲义编辑。请把今天的答疑记录整理成一份结构化小讲义。\n"
            "风格：清晰、细致、有教学感；不是流水账。有关联的问题请自然合并到同一专题讲义中讲清楚。\n"
            "数学公式请用 $...$ 或 $$...$$，不要输出 LaTeX 导言区、\\documentclass、\\usepackage、tcolorbox 环境或危险命令。\n"
            "不要使用 Markdown 加粗语法 **...**；需要强调时改用普通文字标签，例如“关键点：”“结论：”“易错点：”。\n"
            "不要加入“今日概览”“串联型专题”“并联型专题”“今日结论卡片”“后续复习建议”或“原始答疑索引”。\n"
            f"专题讲义最多 {max_topics} 个，典型题最多 {max_examples} 个。\n"
            "专题讲义要比摘要更详细：每个专题至少包含核心结论、适用条件、关键推导或证明过程、易错点；可以使用行间公式展示关键步骤。\n"
            "今日典型题精讲中，每道题必须以二级标题“## 典型题 n：题名”开头，并包含“题目：”“思路：”“详细过程：”“小结：”。\n"
            "典型题的详细过程要展开关键计算或证明，不要只列结论；可以使用行间公式。\n"
            "必须严格按下面 4 个一级标题输出，标题文字和顺序不要改：\n"
            f"{titles}\n\n"
            f"统计区间：{self._daily_report_display_time(start_ts)} 到 {self._daily_report_display_time(end_ts)}\n"
            f"记录数量：{record_count}\n\n"
            "答疑素材如下：\n"
            f"{source_text}"
        )

    @staticmethod
    def _daily_report_split_records(records: List[Dict[str, Any]], chunk_count: int) -> List[List[Dict[str, Any]]]:
        if chunk_count <= 1 or len(records) <= 1:
            return [records]
        chunk_count = max(1, min(chunk_count, len(records)))
        chunks: List[List[Dict[str, Any]]] = [[] for _ in range(chunk_count)]
        for idx, rec in enumerate(records):
            chunks[idx % chunk_count].append(rec)
        return [x for x in chunks if x]

    async def _daily_report_generate_markdown(
        self,
        records: List[Dict[str, Any]],
        start_ts: float,
        end_ts: float,
        event: Optional[AstrMessageEvent],
        provider_id: str,
    ) -> str:
        max_rounds = max(1, self._daily_report_int_cfg("daily_report_max_rounds", 4))
        source_text = self._daily_report_records_to_prompt_text(records)
        if max_rounds > 1 and len(records) > 12:
            chunk_count = min(max_rounds - 1, max(1, math.ceil(len(records) / 12)))
            summaries: List[str] = []
            for idx, chunk in enumerate(self._daily_report_split_records(records, chunk_count), 1):
                prompt = self._daily_report_build_chunk_prompt(chunk, start_ts, end_ts)
                raw = await self._daily_report_llm_generate(provider_id, prompt, f"chunk_summary_{idx}")
                summaries.append(f"## 分段摘要 {idx}\n{self._daily_report_strip_code_fence(raw)}")
            source_text = "\n\n".join(summaries)
        prompt = self._daily_report_build_final_prompt(source_text, start_ts, end_ts, len(records))
        raw = await self._daily_report_llm_generate(provider_id, prompt, "final_report")
        return self._daily_report_strip_code_fence(raw)

    def _daily_report_extract_sections(self, markdown_text: str) -> Dict[str, str]:
        text = self._daily_report_strip_code_fence(markdown_text)
        heading_re = re.compile(
            r"(?m)^\s*#{1,3}\s*("
            + "|".join(re.escape(x) for x in self._DAILY_REPORT_SECTION_TITLES)
            + r")\s*$"
        )
        matches = list(heading_re.finditer(text))
        sections = {title: "" for title in self._DAILY_REPORT_SECTION_TITLES}
        if not matches:
            sections["专题讲义"] = text.strip()
            return sections
        for idx, m in enumerate(matches):
            title = m.group(1)
            start = m.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            sections[title] = text[start:end].strip()
        return sections

    def _daily_report_tex_text(self, text: str) -> str:
        s = _strip_known_xml_like_tags(str(text or ""))
        s = _ensure_balanced_dollar_math(s)
        return _escape_text_preserve_dollar_math(s)

    def _daily_report_inline_markdown_to_latex(self, text: str) -> str:
        s = _strip_known_xml_like_tags(str(text or ""))
        s = _ensure_balanced_dollar_math(s)

        def convert_plain(part: str) -> str:
            result: List[str] = []
            pos = 0
            for m in re.finditer(r"\*\*(.+?)\*\*", part):
                if m.start() > pos:
                    result.append(_escape_latex_text_strict(part[pos:m.start()].replace("**", "")))
                bold_text = m.group(1).strip()
                if bold_text:
                    result.append(r"\textbf{" + _escape_text_preserve_dollar_math(bold_text) + "}")
                pos = m.end()
            if pos < len(part):
                result.append(_escape_latex_text_strict(part[pos:].replace("**", "")))
            return "".join(result)

        out_parts: List[str] = []
        last = 0
        for m in _MATH_DOLLAR_SEG.finditer(s):
            if m.start() > last:
                out_parts.append(convert_plain(s[last:m.start()]))
            out_parts.append(m.group(1))
            last = m.end()
        if last < len(s):
            out_parts.append(convert_plain(s[last:]))
        return "".join(out_parts)

    def _daily_report_markdownish_to_latex(self, text: str) -> str:
        text = self._daily_report_strip_code_fence(text)
        out: List[str] = []
        list_mode = ""
        math_block: Optional[List[str]] = None

        def close_list() -> None:
            nonlocal list_mode
            if list_mode:
                out.append(r"\end{" + list_mode + "}")
                list_mode = ""

        for raw_line in str(text or "").splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if math_block is not None:
                math_block.append(line)
                if stripped.endswith("$$") or stripped.endswith(r"\]"):
                    out.append("\n".join(math_block))
                    math_block = None
                continue

            if not line.strip():
                close_list()
                out.append(r"\par")
                continue

            if stripped in ("$$", r"\[") or (
                (stripped.startswith("$$") and not (stripped.endswith("$$") and len(stripped) > 2))
                or (stripped.startswith(r"\[") and not stripped.endswith(r"\]"))
            ):
                close_list()
                math_block = [line]
                continue

            if (
                (stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 2)
                or (stripped.startswith(r"\[") and stripped.endswith(r"\]"))
            ):
                close_list()
                out.append(line)
                continue

            hm = re.match(r"^\s*#{2,4}\s+(.+?)\s*$", line)
            if hm:
                close_list()
                out.append(r"\subsection*{" + self._daily_report_inline_markdown_to_latex(hm.group(1)) + "}")
                continue

            bm = re.match(r"^\s*[-*+]\s+(.+?)\s*$", line)
            nm = re.match(r"^\s*\d+[.)]\s+(.+?)\s*$", line)
            if bm or nm:
                target_mode = "itemize" if bm else "enumerate"
                if list_mode != target_mode:
                    close_list()
                    list_mode = target_mode
                    out.append(r"\begin{" + list_mode + r"}[leftmargin=1.8em,itemsep=0.2em]")
                content = bm.group(1) if bm else nm.group(1)
                out.append(r"\item " + self._daily_report_inline_markdown_to_latex(content))
                continue

            close_list()
            out.append(self._daily_report_inline_markdown_to_latex(line) + r"\par")
        close_list()
        if math_block is not None:
            out.append("\n".join(math_block))
        return "\n".join(out).strip() or "暂无可整理内容。"

    def _daily_report_fallback_markdown(self, records: List[Dict[str, Any]], start_ts: float, end_ts: float) -> str:
        lines: List[str] = []
        lines.append("# 今日知识地图")
        lines.append("- 本日问题以具体题目讲解为主。")
        lines.append("- 可按题目中出现的定义、定理、计算方法和证明结构继续整理。")
        lines.append("# 专题讲义")
        for idx, rec in enumerate(records[: self._daily_report_int_cfg("daily_report_max_topics", 6)], 1):
            q = str(rec.get("question") or "[图片题目]").strip()
            lines.append(f"## 专题 {idx}")
            lines.append(q)
            lines.append("关键点：请围绕题目条件、可用定理和证明目标继续展开。")
        lines.append("# 今日高频误区")
        lines.append("- 注意区分题目条件、结论目标和可调用定理。")
        lines.append("- 证明题中不要跳过关键映射或极限/连续性验证。")
        lines.append("# 今日典型题精讲")
        for idx, rec in enumerate(records[: self._daily_report_int_cfg("daily_report_max_examples", 4)], 1):
            lines.append(f"## 典型题 {idx}：答疑记录精选")
            lines.append("题目：")
            lines.append(str(rec.get("question") or "[图片题目]").strip())
            lines.append("思路：先识别题目条件和目标，再选择对应定义、定理或计算方法。")
            lines.append("详细过程：")
            ans = str(rec.get("answer") or "").strip()
            if ans:
                lines.append(ans)
            lines.append("小结：检查边界条件、符号和结论是否覆盖所有小问。")
        return "\n".join(lines)

    def _daily_report_split_example_cards(self, text: str) -> List[str]:
        text = self._daily_report_strip_code_fence(text).strip()
        if not text:
            return []
        heading_re = re.compile(r"(?m)^\s*#{2,4}\s+(?:典型题|例题|题目)\s*\d*[:：]?.*$")
        matches = list(heading_re.finditer(text))
        if not matches:
            return [text]
        cards: List[str] = []
        prefix = text[:matches[0].start()].strip()
        if prefix:
            cards.append(prefix)
        for idx, m in enumerate(matches):
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            card = text[m.start():end].strip()
            if card:
                cards.append(card)
        return cards

    def _daily_report_build_tex(
        self,
        markdown_text: str,
        start_ts: float,
        end_ts: float,
        record_count: int,
        title: str = "每日答疑讲义报告",
    ) -> str:
        sections = self._daily_report_extract_sections(markdown_text)
        colors = [
            "reportGreen",
            "reportPurple",
            "reportOrange",
            "reportRed",
        ]
        body: List[str] = []
        for idx, sec_title in enumerate(self._DAILY_REPORT_SECTION_TITLES):
            color = colors[idx % len(colors)]
            body.append(r"\section*{" + self._daily_report_tex_text(sec_title) + "}")
            body.append(r"\addcontentsline{toc}{section}{" + self._daily_report_tex_text(sec_title) + "}")
            if sec_title == "今日典型题精讲":
                cards = self._daily_report_split_example_cards(sections.get(sec_title, ""))
                if not cards:
                    cards = ["暂无可整理内容。"]
                for card in cards:
                    body.append(r"\begin{dailyBox}{" + color + "}")
                    body.append(self._daily_report_markdownish_to_latex(card))
                    body.append(r"\end{dailyBox}")
            else:
                body.append(r"\begin{dailyBox}{" + color + "}")
                body.append(self._daily_report_markdownish_to_latex(sections.get(sec_title, "")))
                body.append(r"\end{dailyBox}")

        return "\n".join([
            r"\documentclass[UTF8]{ctexart}",
            r"\usepackage[a4paper,margin=2.0cm]{geometry}",
            r"\usepackage{textcomp}",
            r"\usepackage{amsmath,amssymb,amsthm,mathtools,bm,mathrsfs,cancel}",
            r"\usepackage[most]{tcolorbox}",
            r"\usepackage{enumitem}",
            r"\usepackage{tikz}",
            r"\usepackage{xcolor}",
            r"\definecolor{reportBlue}{HTML}{2563EB}",
            r"\definecolor{reportGreen}{HTML}{16A34A}",
            r"\definecolor{reportPurple}{HTML}{7C3AED}",
            r"\definecolor{reportOrange}{HTML}{EA580C}",
            r"\definecolor{reportCyan}{HTML}{0891B2}",
            r"\definecolor{reportRed}{HTML}{DC2626}",
            r"\definecolor{reportYellow}{HTML}{CA8A04}",
            r"\definecolor{reportGray}{HTML}{475569}",
            r"\usepackage[colorlinks=true,linkcolor=reportBlue,urlcolor=reportBlue]{hyperref}",
            r"\tcbset{sharp corners=downhill, boxsep=1.2mm}",
            r"\newtcolorbox{dailyBox}[1]{enhanced,breakable,colback=#1!4!white,colframe=#1!70!black,borderline west={3pt}{0pt}{#1!85!black},boxrule=0.45pt,arc=1mm,left=2.2mm,right=2.2mm,top=1.2mm,bottom=1.2mm}",
            r"\setlist{nosep}",
            r"\linespread{1.08}",
            r"\begin{document}",
            r"\begin{center}",
            r"{\LARGE\bfseries " + self._daily_report_tex_text(title) + r"}\\[0.5em]",
            r"{\normalsize 统计区间："
            + self._daily_report_tex_text(self._daily_report_display_time(start_ts))
            + r" -- "
            + self._daily_report_tex_text(self._daily_report_display_time(end_ts))
            + r"\quad 记录数："
            + str(int(record_count))
            + r"}",
            r"\end{center}",
            r"\vspace{0.8em}",
            "\n".join(body),
            r"\end{document}",
        ])

    @staticmethod
    def _daily_report_extract_full_tex(raw: str) -> str:
        s = str(raw or "").strip()
        m = re.search(r"```(?:tex|latex)?\s*([\s\S]*?)```", s, flags=re.I)
        if m:
            s = m.group(1).strip()
        start = s.find(r"\documentclass")
        end = s.rfind(r"\end{document}")
        if start >= 0 and end >= 0:
            return s[start:end + len(r"\end{document}")].strip()
        return s if r"\begin{document}" in s and r"\end{document}" in s else ""

    @staticmethod
    def _daily_report_sanitize_full_tex(tex_src: str) -> str:
        s = str(tex_src or "")
        dangerous_patterns = [
            r"\\write18\b",
            r"\\input\s*\{[^}]*\}",
            r"\\include\s*\{[^}]*\}",
            r"\\openout\b",
            r"\\read\b",
            r"\\catcode\b",
            r"\\usepackage\s*\{shellesc\}",
        ]
        for pat in dangerous_patterns:
            s = re.sub(pat, "", s, flags=re.I)
        return s

    async def _daily_report_compile_with_repair(
        self,
        tex_src: str,
        event: Optional[AstrMessageEvent],
        provider_id: str,
    ) -> Tuple[Optional[bytes], str]:
        compile_rounds = int(self._DAILY_REPORT_INTERNAL_DEFAULTS["compile_max_rounds"])
        tex_src = self._daily_report_sanitize_full_tex(tex_src)
        for round_idx in range(compile_rounds + 1):
            pdf_bytes = await self._compile_tex_to_pdf(tex_src)
            if pdf_bytes:
                return pdf_bytes, tex_src
            if round_idx >= compile_rounds:
                break
            repair_pid = await self._daily_report_resolve_provider_id(
                event=event,
                repair=True,
                fallback_provider_id=provider_id,
            )
            if not repair_pid:
                break
            log_tail = (self._last_texlive_log or "")[-5000:]
            repair_prompt = (
                "下面是一份 XeLaTeX 编译失败的完整日报 TeX。请修复它，并只输出完整可编译 TeX。\n"
                "不要解释，不要输出 Markdown 代码块。不要加入 \\write18、\\input、\\include、shell-escape 等危险命令。\n\n"
                "编译日志末尾：\n"
                f"{log_tail}\n\n"
                "原始 TeX：\n"
                f"{tex_src}"
            )
            raw = await self._daily_report_llm_generate(repair_pid, repair_prompt, "latex_repair")
            fixed = self._daily_report_extract_full_tex(raw)
            if fixed:
                tex_src = self._daily_report_sanitize_full_tex(fixed)
        return None, tex_src

    def _daily_report_write_report_files(
        self,
        pdf_bytes: bytes,
        tex_src: str,
        mode: str,
        start_ts: float,
        end_ts: float,
    ) -> Tuple[str, str, str]:
        self._daily_report_init_storage()
        stamp = datetime.fromtimestamp(float(end_ts)).strftime("%Y%m%d_%H%M%S")
        base = self._daily_report_safe_filename(f"daily_report_{stamp}_{mode}", "daily_report")
        tex_path = os.path.join(self.DAILY_REPORT_OUTPUT_DIR, f"{base}.tex")
        pdf_path = os.path.join(self.DAILY_REPORT_OUTPUT_DIR, f"{base}.pdf")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex_src)
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
        return pdf_path, f"{base}.pdf", tex_path

    async def _daily_report_generate_report(
        self,
        records: List[Dict[str, Any]],
        start_ts: float,
        end_ts: float,
        event: Optional[AstrMessageEvent],
        provider_id: str,
        mode: str,
    ) -> Dict[str, Any]:
        if not provider_id:
            raise RuntimeError("未配置日报生成模型：自动日报需要填写 daily_report_provider_id；手动日报可使用当前会话模型。")
        try:
            markdown = await self._daily_report_generate_markdown(records, start_ts, end_ts, event, provider_id)
        except Exception as e:
            logger.warning(f"[daily_report] LLM 整理失败，使用安全兜底模板: {e}")
            markdown = self._daily_report_fallback_markdown(records, start_ts, end_ts)
        tex_src = self._daily_report_build_tex(markdown, start_ts, end_ts, len(records))
        pdf_bytes, final_tex = await self._daily_report_compile_with_repair(tex_src, event, provider_id)
        if not pdf_bytes:
            raise RuntimeError("日报 LaTeX 编译失败: " + (self._last_texlive_log or "")[-1200:])
        pdf_path, fname, tex_path = self._daily_report_write_report_files(pdf_bytes, final_tex, mode, start_ts, end_ts)
        return {
            "pdf_path": pdf_path,
            "filename": fname,
            "tex_path": tex_path,
            "record_count": len(records),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "tex_paths": [str(x.get("tex_path") or "") for x in records if str(x.get("tex_path") or "").strip()],
        }

    def _daily_report_summary_text(self, result: Dict[str, Any], mode: str) -> str:
        label = "手动答疑报告" if mode == "manual" else "每日答疑报告"
        return (
            f"{label}已生成\n"
            f"统计区间：{self._daily_report_display_time(float(result.get('start_ts', 0) or 0))} "
            f"至 {self._daily_report_display_time(float(result.get('end_ts', 0) or 0))}\n"
            f"答疑记录：{int(result.get('record_count', 0) or 0)} 条"
        )

    async def _daily_report_send_pdf_event(
        self,
        event: AstrMessageEvent,
        result: Dict[str, Any],
        mode: str,
    ) -> Tuple[bool, str]:
        try:
            import astrbot.api.message_components as Comp
        except Exception:
            return False, "当前 AstrBot 版本缺少文件消息组件，无法发送答疑报告 PDF"

        chain: List[Any] = []
        try:
            uid = event.get_sender_id()
            if uid and (not self._is_private_chat(event)):
                chain.append(Comp.At(qq=uid))
        except Exception:
            pass
        sent_msg_id, send_error_msg = await self._send_pdf_file_component(
            event,
            Comp,
            chain,
            str(result.get("pdf_path") or ""),
            str(result.get("filename") or "daily_report.pdf"),
            label="答疑报告",
        )
        if send_error_msg:
            return False, send_error_msg
        return True, self._daily_report_summary_text(result, mode)

    async def _daily_report_send_pdf_origin(self, origin: str, result: Dict[str, Any], mode: str) -> Tuple[bool, str]:
        try:
            import astrbot.api.message_components as Comp
        except Exception as e:
            return False, f"缺少文件消息组件: {e}"
        try:
            chain: List[Any] = []
            if bool(self._cfg("daily_report_send_summary_text", True)):
                text = self._daily_report_summary_text(result, mode)
                try:
                    chain.append(Comp.Plain(text=text))
                except Exception:
                    try:
                        chain.append(Comp.Plain(text))
                    except Exception:
                        pass
            chain.append(Comp.File(file=str(result.get("pdf_path") or ""), name=str(result.get("filename") or "daily_report.pdf")))
            mc = self._build_msg_chain_from_components(chain)
            await self.context.send_message(origin, mc)
            return True, ""
        except Exception as e:
            return False, str(e)

    def _daily_report_delete_tex_paths(self, tex_paths: List[str]) -> None:
        for path in tex_paths or []:
            try:
                p = str(path or "").strip()
                if not p or not os.path.isfile(p):
                    continue
                base = str(Path(self.DAILY_REPORT_TEX_DIR).resolve())
                target = str(Path(p).resolve())
                Path(target).relative_to(base)
                os.remove(target)
            except Exception:
                continue

    async def _daily_report_handle_manual_command(self, event: AstrMessageEvent) -> str:
        if not bool(self._cfg("enable_daily_report", False)):
            return "每日答疑报告未启用。请先在插件配置中开启“启用每日答疑报告”。"
        self._daily_report_init_storage()
        now_ts = time.time()
        origin = self._daily_report_origin_from_event(event)
        async with self._daily_report_lock:
            state = self._daily_report_load_state()
            cursors = state.get("manual_cursors")
            if not isinstance(cursors, dict):
                cursors = {}
            start_ts = float(cursors.get(origin) or self._daily_report_start_of_day_ts(now_ts))
            records = self._daily_report_load_records(start_ts, now_ts, origin)
            min_records = self._daily_report_int_cfg("daily_report_min_records", 2)
            if len(records) < min_records:
                return (
                    f"当前区间内可整理的答疑记录不足：{len(records)}/{min_records}。\n"
                    f"统计区间：{self._daily_report_display_time(start_ts)} 至 {self._daily_report_display_time(now_ts)}"
                )
            provider_id = await self._daily_report_resolve_provider_id(event=event)
            result = await self._daily_report_generate_report(records, start_ts, now_ts, event, provider_id, "manual")
            ok, msg = await self._daily_report_send_pdf_event(event, result, "manual")
            if ok:
                cursors[origin] = now_ts
                state["manual_cursors"] = cursors
                self._daily_report_save_state(state)
            return msg

    def _daily_report_parse_schedule(self, now_ts: Optional[float] = None) -> Tuple[str, float, float]:
        now_dt = datetime.fromtimestamp(float(now_ts if now_ts is not None else time.time()))
        raw_time = str(self._cfg("daily_report_time", "23:30") or "23:30").strip()
        m = re.match(r"^\s*(\d{1,2}):(\d{1,2})\s*$", raw_time)
        hour, minute = (23, 30)
        if m:
            hour = max(0, min(23, int(m.group(1))))
            minute = max(0, min(59, int(m.group(2))))
        send_dt = now_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if now_dt > send_dt + timedelta(minutes=10):
            send_dt = send_dt + timedelta(days=1)
        ahead = max(0, self._daily_report_int_cfg("daily_report_prepare_ahead_minutes", 120))
        prepare_dt = send_dt - timedelta(minutes=ahead)
        return send_dt.strftime("%Y-%m-%d"), prepare_dt.timestamp(), send_dt.timestamp()

    def _start_daily_report_scheduler(self) -> None:
        if not bool(self._cfg("enable_daily_report", False)):
            return
        if not bool(self._cfg("daily_report_auto_send", False)):
            return
        if getattr(self, "_daily_report_task", None) is not None and not self._daily_report_task.done():
            return
        self._daily_report_task = asyncio.create_task(self._daily_report_scheduler_loop())
        logger.info("[daily_report] 自动日报调度已启动")

    def _stop_daily_report_scheduler(self) -> None:
        task = getattr(self, "_daily_report_task", None)
        if task is not None and not task.done():
            task.cancel()
        self._daily_report_task = None

    async def _daily_report_scheduler_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(60)
                if not bool(self._cfg("enable_daily_report", False)) or not bool(self._cfg("daily_report_auto_send", False)):
                    continue
                await self._daily_report_scheduler_tick()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"[daily_report] 自动日报调度异常: {e}")

    async def _daily_report_scheduler_tick(self) -> None:
        now_ts = time.time()
        send_key, prepare_ts, send_ts = self._daily_report_parse_schedule(now_ts)
        async with self._daily_report_lock:
            state = self._daily_report_load_state()
            if now_ts >= prepare_ts and state.get("last_auto_prepare_key") != send_key:
                await self._daily_report_prepare_auto_reports(state, send_key, prepare_ts)
                self._daily_report_save_state(state)
            if now_ts >= send_ts:
                changed = await self._daily_report_send_pending_auto_reports(state, send_key)
                if changed:
                    self._daily_report_save_state(state)

    async def _daily_report_prepare_auto_reports(self, state: Dict[str, Any], send_key: str, end_ts: float) -> None:
        start_ts = float(state.get("auto_cursor_ts") or (float(end_ts) - 86400))
        all_records = self._daily_report_load_records(start_ts, end_ts)
        origins = self._daily_report_resolve_auto_origins(all_records)
        pending_all = state.get("pending_auto_reports")
        if not isinstance(pending_all, dict):
            pending_all = {}
        day_pending: Dict[str, Any] = {}
        min_records = self._daily_report_int_cfg("daily_report_min_records", 2)
        provider_id = await self._daily_report_resolve_provider_id(event=None)
        if not provider_id:
            logger.warning("[daily_report] 自动日报需要配置 daily_report_provider_id，已跳过本次准备且不推进游标")
            return

        for origin in origins:
            records = [rec for rec in all_records if self._daily_report_record_matches_origin(rec, origin)]
            if len(records) < min_records:
                continue
            try:
                result = await self._daily_report_generate_report(records, start_ts, end_ts, None, provider_id, "auto")
                day_pending[origin] = {**result, "sent": False, "prepared_ts": time.time()}
            except Exception as e:
                logger.warning(f"[daily_report] 自动日报生成失败: origin={origin} err={e}")
        pending_all[send_key] = day_pending
        state["pending_auto_reports"] = pending_all
        state["last_auto_prepare_key"] = send_key
        state["auto_cursor_ts"] = end_ts
        self._daily_report_cleanup_old_files()

    async def _daily_report_send_pending_auto_reports(self, state: Dict[str, Any], send_key: str) -> bool:
        pending_all = state.get("pending_auto_reports")
        if not isinstance(pending_all, dict):
            return False
        day_pending = pending_all.get(send_key)
        if not isinstance(day_pending, dict):
            return False
        changed = False
        for origin, item in list(day_pending.items()):
            if not isinstance(item, dict) or bool(item.get("sent")):
                continue
            ok, err = await self._daily_report_send_pdf_origin(str(origin), item, "auto")
            if ok:
                item["sent"] = True
                item["sent_ts"] = time.time()
                changed = True
                if bool(self._DAILY_REPORT_INTERNAL_DEFAULTS["delete_after_auto_send"]):
                    self._daily_report_delete_tex_paths(list(item.get("tex_paths") or []))
            else:
                logger.warning(f"[daily_report] 自动日报发送失败: origin={origin} err={err}")
        pending_all[send_key] = day_pending
        state["pending_auto_reports"] = pending_all
        return changed

    def _daily_report_cleanup_old_files(self) -> None:
        try:
            now = time.time()
            record_days = int(self._DAILY_REPORT_INTERNAL_DEFAULTS["record_retention_days"])
            pdf_days = int(self._DAILY_REPORT_INTERNAL_DEFAULTS["pdf_retention_days"])
            tex_days = int(self._DAILY_REPORT_INTERNAL_DEFAULTS["record_retention_days"])

            def cleanup_dir(base: str, suffixes: Tuple[str, ...], ttl_days: int, recursive: bool = False) -> None:
                if ttl_days <= 0 or not os.path.isdir(base):
                    return
                ttl = ttl_days * 86400
                pattern_iter = Path(base).rglob("*") if recursive else Path(base).glob("*")
                for p in pattern_iter:
                    try:
                        if not p.is_file() or (suffixes and p.suffix.lower() not in suffixes):
                            continue
                        if now - float(p.stat().st_mtime) > ttl:
                            p.unlink()
                    except Exception:
                        continue

            cleanup_dir(self.DAILY_REPORT_RECORD_DIR, (".jsonl",), record_days)
            cleanup_dir(self.DAILY_REPORT_OUTPUT_DIR, (".pdf", ".tex"), pdf_days)
            cleanup_dir(self.DAILY_REPORT_TEX_DIR, (".tex",), tex_days, recursive=True)
        except Exception:
            pass
