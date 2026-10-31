"""Per-turn image delivery permission, based on the user request, never the answer."""
from copy import copy
from dataclasses import dataclass
import asyncio
import re
import unicodedata


@dataclass(frozen=True)
class ImageReplyDecision:
    image_requested: bool = False
    math_request: bool = False
    text_only: bool = False
    document_requested: bool = False

    @property
    def allowed(self):
        return not self.text_only and not self.document_requested and (self.image_requested or self.math_request)


_TEXT_ONLY = re.compile(
    r'(?:不要|别|不用|无需|不需要|禁止).{0,8}(?:图片|截图|长图|转图|发图|出图)'
    r'|(?:只用|只要|直接用|改用|用纯|纯)(?:文字|文本)'
    r'|(?:文字|文本)(?:回复|回答|作答|解答|解释|说明|讲解|版|形式|格式)'
    r'|(?:用|以)(?:纯)?(?:文字|文本)(?:来|重新|再)?(?:评分|批改|解释|解答|说明|回答|回复)'
    r'|\b(?:text[ -]only|no images?|do not (?:send|use) images?)\b', re.I)
_IMAGE_REQUEST = re.compile(
    r'(?:用|以)(?:一张)?(?:图片|截图|长图)(?:的)?(?:形式|方式|格式)?(?:回|答|发|呈现|展示)'
    r'|(?:生成|渲染|导出|转成|转为|做成|输出|发成)(?:为|成)?(?:一张|张)?(?:图片|长图|截图)'
    r'|(?:给我|帮我|请).{0,8}(?:发|生成|制作)(?:一张|张|个)?(?:图|图片|截图)'
    r'|(?:图片|长图|截图)(?:版|形式|格式)(?:的)?(?:回复|回答|发给我)'
    r'|(?:发|给)(?:我|给我)(?:图片|长图|截图)|(?:图片|长图|截图)(?:发我|发给我)'
    r'|\b(?:reply|answer|render|send)\b.{0,18}\b(?:as an? image|as an? picture|screenshot)\b', re.I)
_MATH_TOPIC = re.compile(
    r'(?:[a-zA-Z0-9一二三四五六七八九十]+|高)\s*阶\s*(?:偏\s*)?导(?:数)?|偏导(?:数)?|'
    r'积分|微分|方程|函数|矩阵|向量|行列式|特征值|不等式|极限|导数|数列|级数|'
    r'三角形|圆锥|椭圆|双曲线|抛物线|概率|正定|正交|多项式|定理|收敛|连续|可导|'
    r'凸性|凹性|曲率|切线|面积|体积|方差|期望|秩|\$[^$]+\$|'
    r'\\(?:frac|int|sum|lim|sqrt|begin)\b|[a-zA-Z\d)）][ \t]*[=<>≤≥+*/^][ \t]*[a-zA-Z\d(（-]', re.I)
_SOLVE_ACTION = re.compile(
    r'求证|证明|推导|计算|求解|化简|分解|求导|验算|求(?:值|根|解|极限|积分|导数|秩|概率|面积|体积)'
    r'|(?<![要需追渴供征])求(?!职|推荐|介绍|科普|资源|教材)[^\n。！？!?;；]{0,80}'
    r'(?:值|根|解|极限|积分|导数|阶\s*(?:偏\s*)?导|偏导|秩|概率|面积|体积)'
    r'|(?:判断|验证|检验|求出)|(?:怎么|如何)(?:求|解|算|做|证|计算|推导)'
    r'|(?:为什么|为何).{0,12}(?:成立|收敛|可导|连续|可积|正定|等于|交换)'
    r'|(?:是否|是不是).{0,8}(?:成立|收敛|可导|连续|可积|正定|内积)'
    r'|\b(?:solve|prove|calculate|simplify|differentiate|integrate)\b', re.I)
_PROBLEM_REQUEST = re.compile(
    r'(?:这|那|上面|下面|上述|该|上一|这道|那道|这一道|第[一二三四五六七八九十\d]+)(?:个|道)?题.{0,12}'
    r'(?:怎么|如何|求解|解答|解法|证明|推导|分析|讲解|思路|提示|检查|批改)'
    r'|(?:解答|解一下|求解|做一下|讲解|讲一下|分析|检查|批改|解|做).{0,12}'
    r'(?:这道题|这题|那道题|那题|题目|图(?:片)?中(?:的)?题|习题|练习题|数学题|试卷|证明|解答过程)'
    r'|(?:找|检索|挑选).{0,12}(?:数学题|习题|练习题)')
_IMAGE_SOLVE = re.compile(r'^(?:请|麻烦|帮我|能不能|可以)?\s*(?:解答|解一下|做一下|求解|批改|验算|检查解答|给出解答)(?:一下|下|这张图|图片中题目|图中的题|这道|全部|这题|这道题|吧|吗|。|！|!|？|\?|\s)*$')
_NO_SOLVE = re.compile(r'(?:不要|别|不用|无需|不需要).{0,6}(?:解题|解答|证明|求解|计算)')
_DOCUMENT_REQUEST = re.compile(
    r'^/?(?:s?pdf|md)(?:\s|$)|(?:用|以|按|生成|导出|给我|发我|输出|整理成|转成|转为)\s*(?:一份|一个)?\s*'
    r'(?:PDF|Markdown|(?<![a-zA-Z])\.?md\b)(?:\s|格式|文档|文件|版|解答|回答|解释|讲解|证明|$)', re.I)
_GENERAL_MATH_TOPIC = re.compile(
    _MATH_TOPIC.pattern + r'|数学|高数|线代|微积分|线性代数|抽象代数|数论|几何|拓扑|统计|随机变量|'
    r'实分析|复分析|泛函分析|数学分析|实变|复变|集合|映射|同构|同态|线性空间|子空间|内积|范数|'
    r'度量|紧性|紧致|完备|连通|开集|闭集|开覆盖|子覆盖|一致连续|一致收敛|可测|测度|可积|黎曼|勒贝格|傅里叶|泰勒|'
    r'拉格朗日|拉普拉斯|柯西|欧拉|贝叶斯|马尔可夫|群论|环论|域论|模论|子群|商群|正规群|'
    r'(?:素|极大|主|零|双边|左|右)理想|商环|谱分解|奇异值分解|'
    r'特征多项式|最小多项式|Jordan|Banach|Hilbert|Fourier|Laplace|Taylor|Cauchy|'
    r'公式|恒等式|等式|表达式|代数式|奇数|偶数|素数|质数|整数|有理数|无理数|复数|数域|'
    r'\b(?:math(?:ematics)?|calculus|algebra|geometry|topology|theorem|proof|isomorphism|'
    r'compactness|continuity|convergence|integral|derivative|matrix|eigenvalue|probability)\b', re.I)
_CONCEPT_QUESTION = re.compile(
    r'[?？]|什么|怎么|如何|怎样|为何|为什么|是否|区别|联系|关系|含义|意义|本质|直观|定义|'
    r'解释|讲解|讲讲|介绍|理解|举例|例子|反例|性质|应用|中的|有哪些|吗|呢|看不懂|没明白|不理解|推荐|教材|学习|'
    r'\b(?:what|why|how|explain|define|compare|describe)\b', re.I)
_SOFTWARE_QUESTION = re.compile(
    r'Excel|Word|PPT|WPS|LaTeX|Markdown|MathJax|KaTeX|Python|JavaScript|'
    r'编程|代码|函数库|软件|网页|渲染|公式编辑器|排版|乱码|安装|配置|快捷键|字体', re.I)


def routing_text(text):
    return re.sub(r'(?m)^\s*>[^\n]*$|```[\s\S]*?```|~~~[\s\S]*?~~~', '',
                  unicodedata.normalize('NFKC', str(text or ''))).strip()


def _general_math_request(text):
    if _SOFTWARE_QUESTION.search(text) or _NON_TASK.search(text):
        return False
    if re.search(r'(?:哪里|几点|何时|在哪|一起|大家).{0,8}集合|连续.{0,8}(?:下雨|加班|熬夜|打卡|播放)', text):
        return False
    return bool(_GENERAL_MATH_TOPIC.search(text) and
                (_CONCEPT_QUESTION.search(text) or _GENERAL_MATH_TOPIC.fullmatch(text.rstrip('。!！'))))


def _semantic_binding(event):
    return (str(getattr(event, 'unified_msg_origin', '')), str(event.get_sender_id()),
            str(getattr(getattr(event, 'message_obj', None), 'message_id', '')),
            str(getattr(event, 'message_str', '') or ''))


def mark_semantic_math(event):
    event.set_extra('mathsolve_semantic_math', _semantic_binding(event))


def should_route_math(event):
    """Use the optional router for unfamiliar mathematical terminology."""
    text = routing_text(getattr(event, 'message_str', ''))
    decision = image_reply_decision(event)
    return bool(text and len(text) <= 2000 and not text.startswith('/')
                and not decision.allowed and not decision.text_only and not decision.document_requested
                and not _SOFTWARE_QUESTION.search(text) and not _NON_TASK.search(text)
                and _CONCEPT_QUESTION.search(text))

# Follow-ups require evidence from this turn's exact quote, not group history.
_GRADING = re.compile(r'评分|打分|评卷|判卷|批改|重评|复核|扣分|给分|改分|满分|判错|改错')
_TASK_ACTION = re.compile(r'评分|打分|评卷|判卷|批改|重评|复核|扣分|给分|改分|满分|判错|改错|'
                          r'取等|等号.{0,12}(?:成立|条件)|换.{0,8}(?:做法|解法|方法)|重算|重做|再算')
_FOLLOWUP = re.compile(
    r'评分|打分|评卷|判卷|批改|重评|复核|扣分|给分|改分|满分|判错|改错|'
    r'取等|等号.{0,12}(?:成立|条件)|换.{0,8}(?:做法|解法|方法)|'
    r'重算|重做|再算|验算|检查|核对|复查|哪里.{0,8}(?:错|不对)|'
    r'(?:这|那|上|下|第.{1,4})一步|详细(?:一?点|些)|具体(?:一?点|些)|'
    r'再(?:讲|解释|推导)|接着(?:讲|算|做|推)|继续(?:讲|算|做|推)|'
    r'^(?:请|能不能|可以|麻烦)?\s*(?:讲解|解释|为什么|为何|怎么|如何)')
_NON_TASK = re.compile(
    r'颜值|穿搭|长相|照片|摄影|拍得|餐厅|电影|游戏|球员|'
    r'(?:怎么|如何|为什么|为何|总).{0,12}(?:发图|发图片|图片回复|转图|截图)')
_FORMULA = re.compile(r'\$[^$]+\$|\\(?:frac|int|sum|lim|sqrt|operatorname|begin)\b|'
                      r'[a-zA-Z\d)]\s*(?:=|≤|≥|\^|[<>])\s*[a-zA-Z\d({\\+-]')

# A formula application/explanation is a math task even when phrased as
# "这个是怎么用……公式看出来的", rather than starting with "求/证明".
# Keep this extension separate from the established calculation rules so that
# mentioning a formula in software, study advice, or a translation stays text.
_REASONING_TOPIC = re.compile(_MATH_TOPIC.pattern + r'|公式|恒等式|等式|表达式|代数式', re.I)
_REASONING_ACTION = re.compile(
    r'(?:怎么|如何|怎样)[^\n。！？!?;；]{0,48}'
    r'(?:应用|运用|利用|使用|代入|用|看出|得出|推出|得到|推导|变形|化简|展开|推到|求出|计算|证明)'
    r'|(?:为什么|为何)[^\n。！？!?;；]{0,36}'
    r'(?:用|成立|等于|等价|推出|得到|看出|得出|变形|化简|展开|相等)'
    r'|(?:讲解|解释|说明)[^\n。！？!?;；]{0,24}'
    r'(?:公式|恒等式|等式|定理|式子|推导|证明|变形)'
    r'|(?:公式|恒等式|等式|定理|式子)[^\n。！？!?;；]{0,16}'
    r'(?:看不懂|没看懂|怎么来的|如何得到|怎么得到|怎么推|如何推)'
    r'|(?:帮我|请|能不能)[^\n。！？!?;；]{0,12}'
    r'(?:用|应用|运用)[^\n。！？!?;；]{0,32}(?:解|算|做|证|推|展开)'
    r'|(?:怎么|如何|怎样)(?:求|解|算|做|证)'
    r'|\b(?:explain|derive|apply)\b', re.I)
_REASONING_NON_TASK = re.compile(
    r'(?:推荐|挑选|选购|购买|买|找)[^\n。！？!?;；]{0,16}(?:教材|课程|书|资料)'
    r'|(?:专业|选课|报考|考研|考博|毕业|备考)[^\n。！？!?;；]{0,24}'
    r'(?:怎么学|如何学|怎么选|如何选|规划|安排|准备|要学|要用|要考)'
    r'|学多久|学习计划|备考计划|职业规划'
    r'|翻译|措辞|润色|这句话|词语|单词'
    r'|Excel|Word|PPT|WPS|LaTeX|Markdown|MathJax|KaTeX|'
    r'电子表格|表格软件|公式编辑器|代码|编程|函数库|软件|网页|渲染'
    r'|输入|插入|排版|显示|乱码|安装|配置|复制|粘贴|快捷键|字体'
    r'|奶粉|赛车|恋爱|穿搭|照片|餐厅', re.I)


def _reasoning_request(text):
    return bool(not _REASONING_NON_TASK.search(text)
                and _REASONING_TOPIC.search(text)
                and (_SOLVE_ACTION.search(text) or _REASONING_ACTION.search(text)))


def _math_followup(text):
    # A broader explanation verb is safe only with this turn's quoted math.
    return bool(_FOLLOWUP.search(text) or re.search(r'举个?例|反例|这是什么意思|没明白|看不懂', text) or
                (not _REASONING_NON_TASK.search(text) and _REASONING_ACTION.search(text)))


def _math_evidence(text):
    """Reference content is topic evidence, never an output instruction."""
    text = unicodedata.normalize('NFKC', str(text or ''))[:16000]
    concept_definition = bool(not _SOFTWARE_QUESTION.search(text) and not _NON_TASK.search(text)
                              and _GENERAL_MATH_TOPIC.search(text)
                              and (re.search(r'是|指|定义|满足|称为|对于|\b(?:is|means|defined)\b', text, re.I)
                                   or len(_GENERAL_MATH_TOPIC.findall(text)) >= 2))
    return bool(_FORMULA.search(text) or
                (_MATH_TOPIC.search(text) and _SOLVE_ACTION.search(text)) or
                _reasoning_request(text) or _general_math_request(text) or concept_definition or
                (re.search(r'解答|证明|得分|扣分|满分', text) and
                 re.search(r'题目|试卷|作业|数学|积分|矩阵|不等式|导数', text)))


def _field(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _kind(obj):
    kind = _field(obj, 'type', '')
    return str(getattr(kind, 'value', kind) or type(obj).__name__).lower()


def _forward_binding(event):
    """Bind fetched evidence to the exact user, turn and explicit reference."""
    try:
        replies = [c for c in event.message_obj.message if _kind(c) in ('reply', 'quote')]
        ids = tuple(str(_field(c, 'id', '') or '') for c in replies)
        if not ids or not all(ids):
            return None
        return (str(event.unified_msg_origin), str(event.get_sender_id()),
                str(event.message_obj.message_id), str(event.message_str), ids)
    except (AttributeError, TypeError):
        return None


def _resolved_forward_evidence(event):
    try:
        binding = _forward_binding(event)
        cached = event.get_extra('mathsolve_forward_evidence')
        if (binding and isinstance(cached, dict) and cached.get('binding') == binding
                and cached.get('status') == 'resolved'):
            return bool(cached.get('math')), bool(cached.get('images'))
    except (AttributeError, TypeError):
        pass
    return False, False


async def prepare_forward_reply_evidence(event, *, timeout=3.0):
    """Unwrap only the user's referenced forward, without downloading its images.

    QQ's typed Forward contains just an ID; model input resolution and rendering
    policy must agree about the attachments hidden behind that ID. Never inspect
    conversation history or trust the model's answer to grant image permission.
    """
    key = _forward_binding(event)
    text = str(getattr(event, 'message_str', '') or '')
    if (not key or not _math_followup(text) or _TEXT_ONLY.search(text)
            or _NON_TASK.search(text) or text.lstrip().startswith('/')
            or event.get_extra('md2img_pdf_generation')):
        return None
    cached = event.get_extra('mathsolve_forward_evidence')
    if isinstance(cached, dict) and cached.get('binding') == key:
        return cached.get('status')
    replies = [c for c in event.message_obj.message if _kind(c) in ('reply', 'quote')]
    pending = []
    missing = []
    for reply in replies:
        chain = _field(reply, 'chain', None) or []
        for comp in chain:
            if _kind(comp) in ('forward', 'forward_msg'):
                data = _field(comp, 'data', None) or comp
                fid = _field(data, 'id', '')
                if fid:
                    pending.append(str(fid))
        if not chain:
            missing.append(str(_field(reply, 'id', '')))
    if not pending and not missing:
        return None
    evidence = {'binding': key, 'status': 'pending', 'math': False, 'images': False}
    event.set_extra('mathsolve_forward_evidence', evidence)
    try:
        from astrbot.core.utils.quoted_message.onebot_client import OneBotClient
        from astrbot.core.utils.quoted_message.chain_parser import OneBotPayloadParser
        client, parser = OneBotClient(event), OneBotPayloadParser()

        def absorb(parsed):
            evidence['math'] |= _math_evidence(parsed.get('text'))
            evidence['images'] |= bool(parsed.get('image_refs'))
            pending.extend(str(fid) for fid in parsed.get('forward_ids', []) if fid)

        async with asyncio.timeout(timeout):
            # Reply-ID-only adapters need one lookup of that exact message first.
            for reply_id in missing[:1]:
                payload = await client.get_msg(reply_id)
                if not payload or payload.get('status') == 'failed':
                    raise ValueError('Reference unavailable')
                absorb(parser.parse_get_msg_payload(payload))
            seen = set()
            for _ in range(3):
                pending[:] = [fid for fid in pending if fid not in seen]
                if not pending:
                    break
                fid = pending.pop(0)
                seen.add(fid)
                payload = await client.get_forward_msg(fid)
                if not payload or payload.get('status') == 'failed':
                    raise ValueError('Forward unavailable')
                absorb(parser.parse_get_forward_payload(payload))
                if evidence['images'] or evidence['math']:
                    break  # Enough evidence; core still resolves model inputs.
        evidence['status'] = 'resolved'
    except asyncio.CancelledError:
        evidence['status'] = 'cancelled'
        raise
    except Exception:
        # No signed URLs, tokens or message bodies in errors/logs. Failed reads
        # never grant permission or change the user's message/model input.
        evidence.update(status='unavailable', math=False, images=False)
    return evidence['status']


def _quoted_evidence(event):
    """Bounded inspection of the explicit Reply chain; no model or network calls."""
    parts, has_image, has_reply = [], False, False
    components = getattr(getattr(event, 'message_obj', None), 'message', None) or []
    for comp in components:
        if _kind(comp) not in ('reply', 'quote'):
            continue
        has_reply = True
        parts.append(str(_field(comp, 'message_str', '') or _field(comp, 'text', '') or '')[:16000])
        queue = list(_field(comp, 'chain', None) or [])[:100]
        seen = set()
        for _ in range(100):
            if not queue:
                break
            item = queue.pop(0)
            if id(item) in seen:
                continue
            seen.add(id(item))
            kind = _kind(item)
            data = _field(item, 'data', None) or item
            if kind in ('plain', 'text'):
                parts.append(str(_field(data, 'text', '') or '')[:16000])
            elif kind == 'image':
                has_image = True
            elif kind in ('node', 'nodes'):
                children = _field(data, 'content', None) or _field(data, 'nodes', None)
                if isinstance(children, list):
                    queue.extend(children[:100])
            # Do not inherit an older nested Reply hidden inside the quoted turn.
    fetched_math, fetched_image = _resolved_forward_evidence(event)
    return _math_evidence('\n'.join(parts)[:16000]) or fetched_math, has_image or fetched_image, has_reply

# Explicit symbolic calculation requests need not spell out "导数" or "极限".
# A bare "求x" still needs a mathematical expression elsewhere in this turn.
_CALC_LEAD = r'(?<![要需追渴供征])(?:求|计算|算出)\s*(?:一下\s*)?'
_SYMBOLIC_CALCULUS = re.compile(
    _CALC_LEAD + r'(?:\\?(?:lim|int|sum)(?=[_\s({\\])|'
    r'[a-zA-Z]\s*(?:[\x27′″]+|\^\s*\{?\s*\(\s*[a-zA-Z0-9]+\s*\)\s*\}?)'
    r'\s*\([^()\n]{0,40}\))', re.I)
_SYMBOLIC_VALUE = re.compile(
    _CALC_LEAD + r'[a-zA-Z](?:\s*\([^()\n]{1,40}\))?'
    r'(?:的(?:值|表达式))?\s*(?=$|[，,。.!！?？;；])', re.I)


def classify_image_reply(text, *, has_image=False, has_reply=False, math_context=False):
    text = str(text or '').strip()
    # Normalize only the detection copy. Never rewrite the question or answer.
    text = unicodedata.normalize('NFKC', text)
    # Quotes, examples and source blocks are data, not a new output instruction.
    intent = routing_text(text)
    text_only = bool(_TEXT_ONLY.search(intent))
    image_requested = bool(_IMAGE_REQUEST.search(intent)) and not text_only
    if re.search(r'^(?:请问)?(?:如何|怎么|为什么|为何).{0,24}(?:生成|渲染|转成|转为).{0,8}(?:图片|长图|截图)', intent):
        image_requested = False
    math_request = bool(_PROBLEM_REQUEST.search(intent))
    math_request |= _general_math_request(intent)
    math_request |= bool(_MATH_TOPIC.search(intent) and _SOLVE_ACTION.search(intent))
    if not _NON_TASK.search(intent):
        math_request |= _reasoning_request(intent)
        math_request |= bool(_MATH_TOPIC.search(intent) and _TASK_ACTION.search(intent))
        math_request |= bool(math_context and _math_followup(intent))
        math_request |= bool(has_image and _GRADING.search(intent))
    math_request |= bool(_SYMBOLIC_CALCULUS.search(intent))
    math_request |= bool(_MATH_TOPIC.search(intent) and _SYMBOLIC_VALUE.search(intent))
    math_request |= bool((has_image or has_reply) and _IMAGE_SOLVE.fullmatch(intent))
    if _NO_SOLVE.search(intent) and not _general_math_request(intent):
        math_request = False
    document_intent = re.sub(r'(?:不要|不用|别|无需|不需要)\s*(?:用|发|生成|输出)?\s*'
                             r'(?:PDF|Markdown|\.?md)(?![a-zA-Z])', '', intent, flags=re.I)
    return ImageReplyDecision(image_requested, math_request, text_only, bool(_DOCUMENT_REQUEST.search(document_intent)))


def _event_decision(event, *, inherited_math=False):
    if event is None:
        return ImageReplyDecision()
    text = str(getattr(event, 'message_str', '') or '').strip()
    components = getattr(getattr(event, 'message_obj', None), 'message', None) or []
    has_image = any(_kind(c) == 'image' for c in components)
    quote_math, quote_image, has_reply = _quoted_evidence(event)
    try:
        inputs = event.get_extra('mathsolve_image_reply_inputs')
        if isinstance(inputs, dict) and inputs.get('text') == text:
            has_image |= bool(inputs.get('has_image'))
            has_reply |= bool(inputs.get('has_reply'))
    except Exception:
        pass
    decision = classify_image_reply(text, has_image=has_image or quote_image, has_reply=has_reply,
                                    math_context=quote_math or (inherited_math and not has_reply))
    try:
        if event.get_extra('mathsolve_semantic_math') == _semantic_binding(event):
            return ImageReplyDecision(decision.image_requested, True, decision.text_only, decision.document_requested)
    except (AttributeError, TypeError):
        pass
    return decision


def _same_owner(first, second):
    try:
        return bool(first.get_sender_id() and first.unified_msg_origin and
                    first.get_sender_id() == second.get_sender_id() and
                    first.unified_msg_origin == second.unified_msg_origin)
    except (AttributeError, TypeError):
        return False


def track_active_followup(event):
    """Observe only; the framework still owns capture, ordering and execution.

    A candidate has no effect until core confirms it was consumed by this exact
    run. Unaddressed messages, other users and queued independent turns cannot
    change the active run's delivery permission.
    """
    try:
        if str(getattr(event, 'message_str', '') or '').lstrip().startswith('/'):
            return
        from astrbot.core.pipeline.process_stage.follow_up import _ACTIVE_AGENT_RUNNERS
        runner = _ACTIVE_AGENT_RUNNERS.get(event.unified_msg_origin)
        original = runner.run_context.context.event if runner else None
        if original is None or original is event or not _same_owner(original, event):
            return
        run_id = str(original.message_obj.message_id or '')
        if not run_id or original.get_extra('md2img_pdf_generation'):
            return
        candidates = list(original.get_extra('mathsolve_followup_candidates') or [])
        if not any(candidate is event for _, candidate in candidates):
            candidates.append((run_id, event))
        original.set_extra('mathsolve_followup_candidates', candidates[-16:])
    except (ImportError, AttributeError, TypeError, ValueError):
        return  # Older frameworks use the normal per-request path.


def image_reply_decision(event):
    """Current request or a core-confirmed same-user follow-up; never group history."""
    decision = _event_decision(event)
    if event is None:
        return decision
    try:
        run_id = str(event.message_obj.message_id or '')
        inherited = decision.math_request or _quoted_evidence(event)[0]
        for expected_id, candidate in event.get_extra('mathsolve_followup_candidates') or []:
            capture = candidate.get_extra('_follow_up_captured')
            if (run_id and expected_id == run_id and _same_owner(event, candidate) and
                    isinstance(capture, dict) and str(capture.get('target_run_id') or '') == run_id):
                next_decision = _event_decision(candidate, inherited_math=inherited)
                # A continuation does not silently cancel an explicit text-only
                # instruction from this run. A later explicit image request can.
                if decision.text_only and not next_decision.image_requested:
                    next_decision = ImageReplyDecision(False, next_decision.math_request, True, next_decision.document_requested)
                decision = next_decision
                inherited = decision.math_request or _quoted_evidence(candidate)[0]
    except (AttributeError, TypeError, ValueError):
        pass
    return decision


def plain_reply_text(text):
    """Discard rendering wrappers, not the answer, Markdown, or code examples."""
    parts = re.split(r'(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`)', str(text or ''))
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r'</?md\s*>', '', parts[i], flags=re.I)
    return ''.join(parts).strip()


def image_tool_already_sent(event):
    try:
        return bool(event.get_extra('mathsolve_image_tool_sent'))
    except Exception:
        return False


def constrain_image_tool(req, decision):
    """Filter a request-local copy. Never mutate the globally registered ToolSet."""
    if decision.allowed:
        marker = '[本轮优先图片回答]'
        prompt = str(getattr(req, 'system_prompt', '') or '')
        if marker not in prompt:
            req.system_prompt = prompt + '\n' + marker + (
                '数学相关问题默认优先用图片回答，概念解释、定理讨论、证明计算和数学追问都适用，无须用户说“图片”。'
                '先完成必要的检索和推理，再把本轮完整回答交给 md2img_render_markdown 渲染发送。'
                '传入普通 Markdown，不带 <md> 标签。只要提示时只给提示，不能擅自变成完整解答。'
                '工具已成功发送图片后只简短确认，不重复发送大段 Markdown 或重复调用图片工具。'
                '若图片工具不可用，可用 <md>...</md> 包裹最终回答交由插件渲染；渲染失败时如实说明。\n')
        return
    toolset = getattr(req, 'func_tool', None)
    if toolset is not None and isinstance(getattr(toolset, 'tools', None), list):
        filtered = copy(toolset)
        filtered.tools = [t for t in toolset.tools if getattr(t, 'name', '') != 'md2img_render_markdown']
        req.func_tool = filtered
    marker = '[本轮图片发送限制]'
    prompt = str(getattr(req, 'system_prompt', '') or '')
    if marker not in prompt:
        req.system_prompt = prompt + '\n' + marker + (
            ('本轮按用户指定的文档格式回复，不调用 md2img_render_markdown，不输出 <md> 标签。'
             if decision.document_requested else '本轮使用普通文字回复，不调用 md2img_render_markdown，不输出 <md> 标签。') +
            '用户发来图片、引用图片，或正文含公式/表格/代码，不代表用户要求图片回复。'
            '保留正常回答内容；此限制不影响用户明确请求的 PDF 或其他工具。\n')
