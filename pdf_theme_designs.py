"""Selected preview designs as production, breakable tcolorbox environments.

Keep the public environment names/signatures; no change to document section layout.
Titles use tcolorbox's measured text boxes, not fixed-height overlay text.
"""

PALETTES = {
    "g": ("426C69", "577C93", "A17C56", "837095"),
    "h": ("3E557E", "47776E", "A27F4E", "82658F"),
    "jade": ("416D60", "537992", "A08552", "87739B"),
    "mono_double": ("000000", "000000", "000000", "000000"),
    "jewel": ("204F78", "24634E", "9B5829", "654383"),
    "mist": ("7D719A", "5F8590", "AC8559", "9A7384"),
    "lake": ("4C7C9A", "548777", "B18355", "88759C"),
}

STYLES = {
    "g": (
        r"arc=1.7mm,boxrule=.75pt,colframe=msAccent!38!white,colback=msAccent!3!white,"
        r"colbacktitle=msAccent!3!white,coltitle=msAccent,titlerule=0pt,"
        r"borderline west={2pt}{0pt}{msAccent!85!white},"
        r"left=4.2mm,toptitle=2.1mm,bottomtitle=1.5mm,top=0mm"
    ),
    "h": (
        r"arc=1.2mm,boxrule=.75pt,colframe=msAccent!35!white,colback=msAccent!2!white,"
        r"colbacktitle=msAccent!2!white,coltitle=msAccent,titlerule=0pt,"
        r"left=24mm,lefttitle=24mm,toptitle=2.5mm,bottomtitle=1.5mm,top=0mm,"
        r"underlay={\begin{scope}\clip[rounded corners=1.2mm]"
        r"([xshift=.4pt,yshift=.4pt]frame.south west) rectangle "
        r"([xshift=-.4pt,yshift=-.4pt]frame.north east);"
        r"\fill[msAccent!11!white]"
        r"(frame.north west) rectangle ([xshift=20mm]frame.south west);"
        r"\end{scope}},"
        r"overlay unbroken and first={\node[anchor=north,inner sep=0pt,outer sep=0pt,"
        r"font=\rmfamily\bfseries,text=msAccent] at "
        r"([xshift=10mm,yshift=-3mm]frame.north west) {\msKind};}"
    ),
    "jade": (
        r"arc=2mm,boxrule=.95pt,colframe=msAccent!65!white,colback=msAccent!2!white,"
        r"coltitle=msAccent,attach boxed title to top left={xshift=4mm,yshift=-3mm},"
        r"varwidth boxed title*=-10mm,"
        r"boxed title style={frame hidden,boxrule=0pt,arc=1.2mm,"
        r"colback=msAccent!12!white,left=2mm,right=2mm,top=1.3mm,bottom=1.3mm},"
        r"top=5mm,before skip=5mm"
    ),
    "mono_double": (
        r"sharp corners,boxrule=.9pt,colframe=black,colback=white,coltext=black,"
        r"coltitle=black,colbacktitle=white,borderline={.35pt}{1mm}{black},"
        r"attach boxed title to top center={yshift*=-\tcboxedtitleheight/2-.5mm},"
        r"varwidth boxed title*=-16mm,halign title=center,"
        r"boxed title style={frame hidden,boxrule=0pt,sharp corners,"
        r"colback=white,left=2.2mm,right=2.2mm,top=.9mm,bottom=.9mm},"
        r"top=2.5mm,before skip=4mm"
    ),
    "jewel": (
        r"arc=2mm,boxrule=1.05pt,colframe=msAccent,colback=white,coltext=black,"
        r"colbacktitle=msAccent,coltitle=white,titlerule=0pt,"
        r"toptitle=2.1mm,bottomtitle=2.1mm"
    ),
    "mist": (
        r"arc=2mm,boxrule=.9pt,colframe=msAccent!66!white,"
        r"colback=msAccent!3!white,colbacktitle=msAccent!3!white,coltitle=msAccent,"
        r"titlerule=0pt,halign title=center,"
        r"borderline={.4pt}{2.55pt}{msAccent!30!white},"
        r"toptitle=2.5mm,bottomtitle=1.5mm"
    ),
    "lake": (
        r"arc=2.5mm,boxrule=0pt,frame hidden,colback=msAccent!7!white,"
        r"colbacktitle=msAccent!7!white,coltitle=msAccent,titlerule=0pt,"
        r"borderline west={1.6mm}{0pt}{msAccent!70!white},"
        r"left=4mm,lefttitle=4mm,toptitle=2.3mm,bottomtitle=1.5mm"
    ),
}


def new_box_preamble(style):
    if style not in PALETTES:
        return None
    lines = [
        rf"% mathsolve-pdf-box-style: {style}",
        r"\definecolor{msBoxInk}{HTML}{24313D}",
        r"\tcbset{msSelectedBase/.style={enhanced,breakable,coltext=msBoxInk,"
        r"boxsep=0pt,left=4mm,right=4mm,top=2.5mm,bottom=3mm,"
        r"lefttitle=4mm,righttitle=4mm,before skip=3.5mm,after skip=1.5mm,"
        r"fonttitle=\rmfamily\bfseries}}",
    ]
    if style in ("jade", "mono_double"):
        lines.append(r"\usepackage{varwidth}")
    for (env, kind, args), color in zip((
        ("problemBox", "题目", "[1][]"),
        ("theoremBox", "定理", "[2][]"),
        ("lemmaBox", "引理", "[2][]"),
        ("formulaBox", "公式", "[2][]"),
    ), PALETTES[style]):
        colorname = "ms" + env + "Accent"
        lines.append(rf"\definecolor{{{colorname}}}{{HTML}}{{{color}}}")
        # Per-box color definitions are local to the tcolorbox group.
        options = "msSelectedBase," + STYLES[style].replace("msAccent", colorname).replace(r"\msKind", kind)
        title = kind if env == "problemBox" else kind + "：#2"
        if style == "g":
            title = (
                r"\tcbox[on line,boxsep=0pt,boxrule=0pt,arc=.9mm,"
                f"colback={colorname}!12!white,coltext={colorname},"
                r"left=1.8mm,right=1.8mm,top=.65mm,bottom=.65mm]"
                "{" + kind + "}" + (r"\hspace{2mm}#2" if env != "problemBox" else "")
            )
        if style == "h":
            title = "" if env == "problemBox" else "#2"
            if env == "problemBox":
                options += ",top=2.5mm"
        lines.append(r"\newtcolorbox{" + env + "}" + args
                     + "{" + options + ",title={" + title + "},#1}")
    return lines
