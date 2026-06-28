# -*- coding: utf-8 -*-
try:
    from .shared import *
except ImportError:
    from shared import *


class SpdfMixin:
    """Deep-thinking SPDF generation helpers."""

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

        pool = self._spdf_unique_provider_ids([
            str(self._cfg("spdf_solver_provider_id_1", "") or "").strip(),
            str(self._cfg("spdf_solver_provider_id_2", "") or "").strip(),
            str(self._cfg("spdf_solver_provider_id_3", "") or "").strip(),
            str(self._cfg("spdf_solver_provider_id_4", "") or "").strip(),
        ])
        if not pool:
            # Legacy flat text config kept for old installations, hidden from
            # the new WebUI because provider choices are now dropdown slots.
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

        original_image_count = len(image_urls or [])
        if image_urls:
            image_urls = await self._pdf_snapshot_image_urls(image_urls)
        has_image = bool(image_urls)
        if original_image_count and (not has_image) and (not str(problem_text or "").strip()):
            raise RuntimeError("Upstream Error: Image invalid")
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
