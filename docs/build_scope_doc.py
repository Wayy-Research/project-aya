"""Build Project Aya research scope document as .docx"""

from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml
import os

# Brand colors
COHERE_GREEN = RGBColor(0x39, 0x59, 0x4D)
COHERE_GREEN_LIGHT = RGBColor(0x4A, 0x7A, 0x68)
WAYY_INDIGO = RGBColor(0x5B, 0x6A, 0xBF)
INK = RGBColor(0x1A, 0x1A, 0x1A)
INK_LIGHT = RGBColor(0x3D, 0x3D, 0x3D)
INK_MUTED = RGBColor(0x6B, 0x6B, 0x7A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
WARM = RGBColor(0xC7, 0x5A, 0x3A)


def set_cell_shading(cell, color_hex: str):
    """Set background color on a table cell."""
    shading = parse_xml(
        f'<w:shd {nsdecls("w")} w:fill="{color_hex}" w:val="clear"/>'
    )
    cell._tc.get_or_add_tcPr().append(shading)


def add_bottom_border(paragraph, color="39594D", size=6):
    """Add a bottom border to a paragraph."""
    pPr = paragraph._p.get_or_add_pPr()
    pBdr = parse_xml(
        f'<w:pBdr {nsdecls("w")}>'
        f'  <w:bottom w:val="single" w:sz="{size}" w:space="1" w:color="{color}"/>'
        f'</w:pBdr>'
    )
    pPr.append(pBdr)


def add_left_border(paragraph, color="39594D", size=12):
    """Add a left border to a paragraph."""
    pPr = paragraph._p.get_or_add_pPr()
    pBdr = parse_xml(
        f'<w:pBdr {nsdecls("w")}>'
        f'  <w:left w:val="single" w:sz="{size}" w:space="8" w:color="{color}"/>'
        f'</w:pBdr>'
    )
    pPr.append(pBdr)


def set_paragraph_shading(paragraph, color_hex: str):
    """Set background shading on a paragraph."""
    pPr = paragraph._p.get_or_add_pPr()
    shading = parse_xml(
        f'<w:shd {nsdecls("w")} w:fill="{color_hex}" w:val="clear"/>'
    )
    pPr.append(shading)


def add_run(paragraph, text, bold=False, italic=False, size=None, color=None,
            font_name=None, underline=False):
    """Add a formatted run to a paragraph."""
    run = paragraph.add_run(text)
    if bold:
        run.bold = True
    if italic:
        run.italic = True
    if size:
        run.font.size = Pt(size)
    if color:
        run.font.color.rgb = color
    if font_name:
        run.font.name = font_name
    if underline:
        run.underline = True
    return run


def make_styled_table(doc, headers, rows, col_widths=None):
    """Create a clean branded table."""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    # Style header row
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        run = p.add_run(header.upper())
        run.bold = True
        run.font.size = Pt(7.5)
        run.font.color.rgb = INK_MUTED
        run.font.name = "Calibri"
        p.space_after = Pt(2)
        p.space_before = Pt(2)

    # Header bottom border
    for cell in table.rows[0].cells:
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        borders = parse_xml(
            f'<w:tcBorders {nsdecls("w")}>'
            f'  <w:bottom w:val="single" w:sz="8" w:space="0" w:color="1A1A1A"/>'
            f'</w:tcBorders>'
        )
        tcPr.append(borders)

    # Data rows
    for r_idx, row_data in enumerate(rows):
        for c_idx, cell_text in enumerate(row_data):
            cell = table.rows[r_idx + 1].cells[c_idx]
            cell.text = ""
            p = cell.paragraphs[0]

            # Handle bold markers
            if cell_text.startswith("**") and cell_text.endswith("**"):
                run = p.add_run(cell_text[2:-2])
                run.bold = True
            else:
                run = p.add_run(cell_text)

            run.font.size = Pt(9)
            run.font.color.rgb = INK_LIGHT
            run.font.name = "Calibri"
            p.space_after = Pt(3)
            p.space_before = Pt(3)

            # Light bottom border
            tc = cell._tc
            tcPr = tc.get_or_add_tcPr()
            borders = parse_xml(
                f'<w:tcBorders {nsdecls("w")}>'
                f'  <w:bottom w:val="single" w:sz="2" w:space="0" w:color="EDEEED"/>'
                f'</w:tcBorders>'
            )
            tcPr.append(borders)

    # Set column widths if provided
    if col_widths:
        for row in table.rows:
            for i, width in enumerate(col_widths):
                row.cells[i].width = Inches(width)

    return table


def build_document():
    doc = Document()

    # Page setup
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.left_margin = Inches(1.2)
    section.right_margin = Inches(1.2)
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)

    # ========================================
    # COLLABORATION BANNER
    # ========================================
    banner_table = doc.add_table(rows=1, cols=3)
    banner_table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Cohere side
    cell_left = banner_table.rows[0].cells[0]
    cell_left.text = ""
    p = cell_left.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    add_run(p, "Cohere Labs", bold=True, size=12, color=WHITE, font_name="Calibri")
    p2 = cell_left.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    add_run(p2, "FOR THE BUILDER", size=6.5, color=RGBColor(0xA0, 0xB8, 0xAA), font_name="Calibri")

    # X divider
    cell_mid = banner_table.rows[0].cells[1]
    cell_mid.text = ""
    p = cell_mid.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_run(p, "\u00d7", size=18, color=RGBColor(0x80, 0x99, 0x8C), font_name="Calibri")

    # Wayy side
    cell_right = banner_table.rows[0].cells[2]
    cell_right.text = ""
    p = cell_right.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    add_run(p, "Wayy Research", bold=True, size=12, color=WHITE, font_name="Calibri")
    p2 = cell_right.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.LEFT
    add_run(p2, "BUFFALO, NY", size=6.5, color=RGBColor(0xA0, 0xB8, 0xAA), font_name="Calibri")

    # Color the banner cells
    for cell in banner_table.rows[0].cells:
        set_cell_shading(cell, "39594D")
        # Remove cell borders
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        borders = parse_xml(
            f'<w:tcBorders {nsdecls("w")}>'
            f'  <w:top w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            f'  <w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            f'  <w:bottom w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            f'  <w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            f'</w:tcBorders>'
        )
        tcPr.append(borders)

    # Banner subtitle
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.space_before = Pt(4)
    p_sub.space_after = Pt(20)
    add_run(p_sub, "RESEARCH COLLABORATION  \u2014  2026", size=7, color=INK_MUTED, font_name="Calibri")

    # ========================================
    # TITLE BLOCK
    # ========================================

    # Org names line
    p_orgs = doc.add_paragraph()
    add_run(p_orgs, "COHERE LABS", bold=True, size=8, color=COHERE_GREEN, font_name="Calibri")
    add_run(p_orgs, "  \u00d7  ", size=8, color=INK_MUTED, font_name="Calibri")
    add_run(p_orgs, "WAYY RESEARCH", bold=True, size=8, color=WAYY_INDIGO, font_name="Calibri")
    p_orgs.space_after = Pt(2)

    # Doc type (right-aligned would need a table, keep it simple)
    p_dtype = doc.add_paragraph()
    p_dtype.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    add_run(p_dtype, "RESEARCH PROJECT SCOPE", size=7.5, color=INK_MUTED, font_name="Calibri")
    p_dtype.space_after = Pt(6)

    # Title
    p_title = doc.add_paragraph()
    add_run(p_title, "Project Aya", bold=True, size=28, color=INK, font_name="Georgia")
    p_title.space_after = Pt(4)

    # Subtitle
    p_sub = doc.add_paragraph()
    add_run(p_sub, "Multilingual Transformer-to-Mamba Distillation: Compression, Equity, "
            "and the Architecture of Linguistic Inclusion",
            italic=True, size=12, color=INK_LIGHT, font_name="Georgia")
    p_sub.space_after = Pt(10)

    # Meta line
    p_meta = doc.add_paragraph()
    add_bottom_border(p_meta, "1A1A1A", 8)
    meta_items = [
        ("Lead", "Rick Galbo, Wayy Research"),
        ("Built on", "Cohere Tiny Aya"),
        ("Date", "February 2026"),
        ("Status", "Scoping"),
    ]
    for i, (label, value) in enumerate(meta_items):
        if i > 0:
            add_run(p_meta, "    \u2022    ", size=8, color=INK_MUTED)
        add_run(p_meta, label, bold=True, size=8.5, color=INK_LIGHT, font_name="Calibri")
        add_run(p_meta, f"  {value}", size=8.5, color=INK_MUTED, font_name="Calibri")
    p_meta.space_after = Pt(24)

    # ========================================
    # SECTION 1: RESEARCH QUESTION
    # ========================================
    p_label = doc.add_paragraph()
    add_run(p_label, "SECTION 1", bold=True, size=7, color=COHERE_GREEN, font_name="Calibri")
    p_label.space_after = Pt(2)

    p_h2 = doc.add_paragraph()
    add_run(p_h2, "What is the question we want to answer?", bold=True, size=15, color=INK, font_name="Georgia")
    p_h2.space_after = Pt(10)

    # Featured question with left border and shading
    p_q = doc.add_paragraph()
    add_left_border(p_q, "39594D", 16)
    set_paragraph_shading(p_q, "F4F7F5")
    add_run(p_q,
        "Can transformer-based multilingual language models be distilled into "
        "Mamba (selective state space) hybrid architectures while preserving "
        "multilingual capability, structured tool use, and cross-lingual "
        "reasoning \u2014 and does the compression degrade uniformly across "
        "language families?",
        italic=True, size=11.5, color=INK, font_name="Georgia")
    p_q.space_after = Pt(24)
    p_q.space_before = Pt(4)

    # ========================================
    # SECTION 2: WHY THIS MATTERS
    # ========================================
    p_label = doc.add_paragraph()
    add_run(p_label, "SECTION 2", bold=True, size=7, color=COHERE_GREEN, font_name="Calibri")
    p_label.space_after = Pt(2)

    p_h2 = doc.add_paragraph()
    add_run(p_h2, "Why is this question important?", bold=True, size=15, color=INK, font_name="Georgia")
    p_h2.space_after = Pt(8)

    motivations = [
        ("70%+ of the world\u2019s internet users are non-English speakers,",
         " yet most efficient and edge-deployed models are English-centric."),
        ("Transformers have O(n\u00b2) attention cost,",
         " making multilingual models impractical on consumer hardware."),
        ("Mamba offers O(n) inference with constant memory,",
         " but no multilingual Mamba LLM exists yet."),
        ("Understanding how compression affects different languages has equity implications",
         " \u2014 if low-resource languages degrade disproportionately, efficiency gains come at the cost of linguistic inclusion."),
        ("Multilingual tool calling",
         " (function calling in non-English languages) is almost entirely unstudied."),
    ]

    for bold_part, rest in motivations:
        p = doc.add_paragraph(style="List Bullet")
        add_run(p, bold_part, bold=True, size=10, color=INK_LIGHT, font_name="Calibri")
        add_run(p, rest, size=10, color=INK_LIGHT, font_name="Calibri")
        p.space_after = Pt(4)

    doc.add_paragraph().space_after = Pt(12)

    # ========================================
    # SECTION 3: PRIOR WORK
    # ========================================
    p_label = doc.add_paragraph()
    add_run(p_label, "SECTION 3", bold=True, size=7, color=COHERE_GREEN, font_name="Calibri")
    p_label.space_after = Pt(2)

    p_h2 = doc.add_paragraph()
    add_run(p_h2, "What work does this build on?", bold=True, size=15, color=INK, font_name="Georgia")
    p_h2.space_after = Pt(4)

    p_intro = doc.add_paragraph()
    add_run(p_intro, "Papers one should read to catch up on the relevant literature:",
            size=10, color=INK_LIGHT, font_name="Calibri")
    p_intro.space_after = Pt(8)

    papers = [
        ("Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
         "Gu & Dao, 2023", "arxiv:2312.00752"),
        ("Mamba-2: Transformers are SSMs \u2014 Generalized Models and Efficient Algorithms Through Structured State Space Duality",
         "Dao & Gu, 2024", "arxiv:2405.21060"),
        ("The Mamba in the Llama: Distilling and Accelerating Hybrid Models",
         "NeurIPS 2024", "github.com/jxiw/MambaInLlama"),
        ("CAB: Cross-Architecture Distillation via Attention Bridge",
         "2025", "arxiv:2510.19266"),
        ("Tiny Aya Tech Report",
         "Cohere Labs, 2026", "github.com/Cohere-Labs/tiny-aya-tech-report"),
        ("Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model",
         "\u00dcst\u00fcn et al., 2024", "arxiv:2402.07827"),
        ("Aya Expanse",
         "Dang et al., 2024", "arxiv:2412.04261"),
        ("FalconMamba: The First Competitive Attention-free 7B Language Model",
         "Technology Innovation Institute, 2024", "arxiv:2410.05355"),
        ("Multilingual State Space Models for Structured QA in Indic Languages",
         "2025", "arxiv:2502.01673"),
    ]

    for i, (title, authors, link) in enumerate(papers):
        p = doc.add_paragraph()
        add_run(p, f"  {i+1}.  ", bold=True, size=9, color=COHERE_GREEN, font_name="Calibri")
        add_run(p, title, italic=True, size=10, color=INK, font_name="Calibri")
        p.add_run("\n")
        add_run(p, f"       {authors}", size=9, color=INK_MUTED, font_name="Calibri")
        add_run(p, f"  \u2014  ", size=9, color=INK_MUTED, font_name="Calibri")
        add_run(p, link, size=8, color=COHERE_GREEN, font_name="Consolas")
        p.space_after = Pt(6)

    # ========================================
    # GRADIENT DIVIDER (simulated with two-color line)
    # ========================================
    div_table = doc.add_table(rows=1, cols=2)
    div_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell in div_table.rows[0].cells:
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        borders = parse_xml(
            f'<w:tcBorders {nsdecls("w")}>'
            f'  <w:top w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            f'  <w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            f'  <w:bottom w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            f'  <w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            f'</w:tcBorders>'
        )
        tcPr.append(borders)

    # Left half green
    c1 = div_table.rows[0].cells[0]
    c1.text = ""
    p = c1.paragraphs[0]
    p.space_before = Pt(0)
    p.space_after = Pt(0)
    add_bottom_border(p, "39594D", 8)

    # Right half indigo
    c2 = div_table.rows[0].cells[1]
    c2.text = ""
    p = c2.paragraphs[0]
    p.space_before = Pt(0)
    p.space_after = Pt(0)
    add_bottom_border(p, "5B6ABF", 8)

    doc.add_paragraph().space_after = Pt(12)

    # ========================================
    # SECTION 4: PUBLISHABILITY
    # ========================================
    p_label = doc.add_paragraph()
    add_run(p_label, "SECTION 4", bold=True, size=7, color=COHERE_GREEN, font_name="Calibri")
    p_label.space_after = Pt(2)

    p_h2 = doc.add_paragraph()
    add_run(p_h2, "Is this a publishable question?", bold=True, size=15, color=INK, font_name="Georgia")
    p_h2.space_after = Pt(8)

    # Publishability tag
    p_tag = doc.add_paragraph()
    run = add_run(p_tag, " HIGHLY PUBLISHABLE \u2014 NOVEL INTERSECTION ",
                  bold=True, size=7.5, color=COHERE_GREEN, font_name="Calibri")
    set_paragraph_shading(p_tag, "E8F0EC")
    p_tag.space_after = Pt(8)

    p = doc.add_paragraph()
    add_run(p, "The intersection of multilingual NLP and state space model distillation is unexplored. "
            "This question has not been published on before, and findings can be shared externally:",
            size=10, color=INK_LIGHT, font_name="Calibri")
    p.space_after = Pt(4)

    pub_points = [
        "No multilingual Mamba LLM has been published.",
        "Cross-lingual reasoning degradation under architecture distillation has not been studied.",
        "Multilingual tool calling benchmarks barely exist.",
        "The question of whether compression is equitable across language families is novel and timely (AI equity angle).",
    ]
    for pt in pub_points:
        p = doc.add_paragraph(style="List Bullet")
        add_run(p, pt, size=10, color=INK_LIGHT, font_name="Calibri")
        p.space_after = Pt(3)

    p = doc.add_paragraph()
    add_run(p, "Target venues:  ", bold=True, size=10, color=INK_LIGHT, font_name="Calibri")
    for venue in ["EMNLP", "ACL", "NeurIPS", "EACL"]:
        add_run(p, f"  {venue}  ", size=8, color=WAYY_INDIGO, font_name="Consolas")
    p.space_after = Pt(20)

    # ========================================
    # SECTION 5: EXPERIMENTAL SETTING
    # ========================================
    p_label = doc.add_paragraph()
    add_run(p_label, "SECTION 5", bold=True, size=7, color=COHERE_GREEN, font_name="Calibri")
    p_label.space_after = Pt(2)

    p_h2 = doc.add_paragraph()
    add_run(p_h2, "What is the simplest experimental setting?", bold=True, size=15, color=INK, font_name="Georgia")
    p_h2.space_after = Pt(8)

    p = doc.add_paragraph()
    add_run(p, "Take ", size=10, color=INK_LIGHT, font_name="Calibri")
    add_run(p, "Tiny Aya base", bold=True, size=10, color=INK_LIGHT, font_name="Calibri")
    add_run(p, " (3.35B params, 70+ languages) as the teacher. Distill into a ", size=10, color=INK_LIGHT, font_name="Calibri")
    add_run(p, "~500\u2013800M Mamba-MoE hybrid student", bold=True, size=10, color=INK_LIGHT, font_name="Calibri")
    add_run(p, " using the MambaInLlama 3-stage pipeline:", size=10, color=INK_LIGHT, font_name="Calibri")
    p.space_after = Pt(6)

    stages = [
        ("Layer alignment", " \u2014 map transformer attention layers to Mamba SSM blocks"),
        ("KL distillation", " \u2014 train student against teacher soft targets"),
        ("Supervised fine-tuning", " \u2014 restore multilingual + tool calling capability"),
    ]
    for i, (bold_part, rest) in enumerate(stages):
        p = doc.add_paragraph(style="List Number")
        add_run(p, bold_part, bold=True, size=10, color=INK_LIGHT, font_name="Calibri")
        add_run(p, rest, size=10, color=INK_LIGHT, font_name="Calibri")
        p.space_after = Pt(3)

    p = doc.add_paragraph()
    add_run(p, "Evaluate on ", size=10, color=INK_LIGHT, font_name="Calibri")
    add_run(p, "3 benchmarks", bold=True, size=10, color=INK_LIGHT, font_name="Calibri")
    add_run(p, " across ", size=10, color=INK_LIGHT, font_name="Calibri")
    add_run(p, "10 diverse languages", bold=True, size=10, color=INK_LIGHT, font_name="Calibri")
    add_run(p, " spanning ", size=10, color=INK_LIGHT, font_name="Calibri")
    add_run(p, "5 language families.", bold=True, size=10, color=INK_LIGHT, font_name="Calibri")
    add_run(p, " Measure: perplexity retention, reasoning accuracy, tool call success rate, and CPU inference throughput.",
            size=10, color=INK_LIGHT, font_name="Calibri")
    p.space_after = Pt(20)

    # ========================================
    # SECTION 6: BASELINES & BENCHMARKS
    # ========================================
    p_label = doc.add_paragraph()
    add_run(p_label, "SECTION 6", bold=True, size=7, color=COHERE_GREEN, font_name="Calibri")
    p_label.space_after = Pt(2)

    p_h2 = doc.add_paragraph()
    add_run(p_h2, "Key baselines and benchmarks?", bold=True, size=15, color=INK, font_name="Georgia")
    p_h2.space_after = Pt(8)

    # Baselines subheading
    p = doc.add_paragraph()
    add_run(p, "Baselines", bold=True, size=12, color=INK, font_name="Georgia")
    p.space_after = Pt(6)

    make_styled_table(doc,
        ["Model", "Role", "Details"],
        [
            ["Tiny Aya (3.35B)", "Teacher / upper bound", "Full transformer, 70+ languages"],
            ["Tiny Aya GGUF 4-bit", "Compressed transformer baseline", "Quantized, same architecture"],
            ["Mamba-2 2.8B", "English-only Mamba baseline", "state-spaces reference model"],
            ["Random / majority class", "Lower bound", "\u2014"],
        ],
    )

    doc.add_paragraph().space_after = Pt(8)

    p = doc.add_paragraph()
    add_run(p, "Benchmarks", bold=True, size=12, color=INK, font_name="Georgia")
    p.space_after = Pt(6)

    make_styled_table(doc,
        ["Benchmark", "Scope", "What It Measures"],
        [
            ["**mGSM**", "10 languages, 250 problems each", "Multilingual grade school math reasoning"],
            ["**XCOPA**", "11 languages", "Cross-lingual commonsense reasoning"],
            ["**Custom tool calling eval**", "10+ languages", "JSON parse rate, tool selection, argument correctness"],
            ["**CPU throughput**", "Hardware benchmark", "Tokens/sec, memory footprint, time-to-first-token"],
        ],
    )

    doc.add_paragraph().space_after = Pt(6)

    # ========================================
    # GRADIENT DIVIDER 2
    # ========================================
    div_table = doc.add_table(rows=1, cols=2)
    div_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell in div_table.rows[0].cells:
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        borders = parse_xml(
            f'<w:tcBorders {nsdecls("w")}>'
            f'  <w:top w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            f'  <w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            f'  <w:bottom w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            f'  <w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            f'</w:tcBorders>'
        )
        tcPr.append(borders)
    c1 = div_table.rows[0].cells[0]
    c1.text = ""
    add_bottom_border(c1.paragraphs[0], "39594D", 8)
    c2 = div_table.rows[0].cells[1]
    c2.text = ""
    add_bottom_border(c2.paragraphs[0], "5B6ABF", 8)

    doc.add_paragraph().space_after = Pt(12)

    # ========================================
    # SECTION 7: RESOURCES
    # ========================================
    p_label = doc.add_paragraph()
    add_run(p_label, "SECTION 7", bold=True, size=7, color=COHERE_GREEN, font_name="Calibri")
    p_label.space_after = Pt(2)

    p_h2 = doc.add_paragraph()
    add_run(p_h2, "What resources do we plan to use?", bold=True, size=15, color=INK, font_name="Georgia")
    p_h2.space_after = Pt(8)

    # Datasets
    p = doc.add_paragraph()
    add_run(p, "Datasets", bold=True, size=12, color=INK, font_name="Georgia")
    p.space_after = Pt(6)

    make_styled_table(doc,
        ["Purpose", "Dataset", "Access"],
        [
            ["Teacher pretraining", "CulturaX, mC4, multilingual instruction data", "Via pretrained model"],
            ["Distillation", "mC4 multilingual subset, Aya Collection", "Open (HuggingFace)"],
            ["Tool calling SFT", "Self-generated multilingual tool call examples", "10 langs \u00d7 500\u20131,000 examples"],
            ["Evaluation", "mGSM (Google), XCOPA (Ponti et al.)", "Open"],
        ],
    )

    doc.add_paragraph().space_after = Pt(8)

    # Models
    p = doc.add_paragraph()
    add_run(p, "Models", bold=True, size=12, color=INK, font_name="Georgia")
    p.space_after = Pt(6)

    make_styled_table(doc,
        ["Role", "Model", "Parameters"],
        [
            ["**Teacher**", "CohereLabs/tiny-aya-base + tiny-aya-global", "3.35B"],
            ["**Student**", "Custom Mamba-MoE hybrid (Aetheris architecture)", "~500\u2013800M"],
            ["**Comparison**", "Mamba-2 2.8B, FalconMamba-7B", "2.8B / 7B"],
        ],
    )

    doc.add_paragraph().space_after = Pt(8)

    # Evaluation Metrics
    p = doc.add_paragraph()
    add_run(p, "Evaluation Metrics", bold=True, size=12, color=INK, font_name="Georgia")
    p.space_after = Pt(6)

    make_styled_table(doc,
        ["Metric", "Granularity", "Category"],
        [
            ["Perplexity", "Per-language, held-out mC4", "Quality"],
            ["mGSM accuracy", "Per-language", "Reasoning"],
            ["XCOPA accuracy", "Per-language", "Reasoning"],
            ["Tool call JSON parse success", "Per-language", "Tool use"],
            ["Tool selection accuracy", "Per-language", "Tool use"],
            ["Inference throughput", "Tokens/sec on consumer CPU", "Efficiency"],
            ["Memory footprint", "Peak RAM during inference", "Efficiency"],
            ["Compression ratio", "Params, disk size, RAM", "Efficiency"],
            ["**Degradation equity score**", "Variance of accuracy drop across language families", "Equity"],
        ],
    )

    doc.add_paragraph().space_after = Pt(8)

    # Annotation Resources
    p = doc.add_paragraph()
    add_run(p, "Annotation Resources", bold=True, size=12, color=INK, font_name="Georgia")
    p.space_after = Pt(6)

    ann_items = [
        ("No human annotation required", " for core experiments."),
        ("Tool call SFT data:", " machine-generated using Tiny Aya as teacher, validated programmatically via JSON schema validation."),
        ("Optional:", " human evaluation of tool call quality in 3\u20135 languages, if time permits."),
    ]
    for bold_part, rest in ann_items:
        p = doc.add_paragraph(style="List Bullet")
        add_run(p, bold_part, bold=True, size=10, color=INK_LIGHT, font_name="Calibri")
        add_run(p, rest, size=10, color=INK_LIGHT, font_name="Calibri")
        p.space_after = Pt(3)

    doc.add_paragraph().space_after = Pt(4)

    # Other Resources
    p = doc.add_paragraph()
    add_run(p, "Other Resources", bold=True, size=12, color=INK, font_name="Georgia")
    p.space_after = Pt(6)

    make_styled_table(doc,
        ["Resource", "Details"],
        [
            ["**Compute**", "1\u00d7 A100 80GB (Runpod, ~$1.49/hr) for distillation training"],
            ["**Aetheris codebase**", "Existing Mamba-MoE: selective scan, MoE routing, streaming dataset, trainer"],
            ["**llama.cpp**", "Mamba-1/2 GGUF support for CPU inference benchmarking"],
            ["**outlines**", "Constrained JSON generation for tool calling evaluation"],
        ],
    )

    doc.add_paragraph().space_after = Pt(6)

    # ========================================
    # GRADIENT DIVIDER 3
    # ========================================
    div_table = doc.add_table(rows=1, cols=2)
    div_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell in div_table.rows[0].cells:
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        borders = parse_xml(
            f'<w:tcBorders {nsdecls("w")}>'
            f'  <w:top w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            f'  <w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            f'  <w:bottom w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            f'  <w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            f'</w:tcBorders>'
        )
        tcPr.append(borders)
    c1 = div_table.rows[0].cells[0]
    c1.text = ""
    add_bottom_border(c1.paragraphs[0], "39594D", 8)
    c2 = div_table.rows[0].cells[1]
    c2.text = ""
    add_bottom_border(c2.paragraphs[0], "5B6ABF", 8)

    doc.add_paragraph().space_after = Pt(12)

    # ========================================
    # SECTION 8: NOTES & OPEN QUESTIONS
    # ========================================
    p_label = doc.add_paragraph()
    add_run(p_label, "SECTION 8", bold=True, size=7, color=COHERE_GREEN, font_name="Calibri")
    p_label.space_after = Pt(2)

    p_h2 = doc.add_paragraph()
    add_run(p_h2, "Additional Notes & Open Questions", bold=True, size=15, color=INK, font_name="Georgia")
    p_h2.space_after = Pt(10)

    notes = [
        ("01", "Tokenizer mismatch.",
         " Aetheris uses GPT-2 vocab (50k), Aya uses 262k multilingual BPE. "
         "The embedding table alone will be ~500MB. Is there a way to prune "
         "the tokenizer for a target language subset without losing the multilingual benefit?"),
        ("02", "Language family hypothesis.",
         " We predict agglutinative languages (Turkish, Finnish, Japanese) will "
         "degrade more under distillation because their token-level dependencies "
         "span longer ranges \u2014 exactly what attention handles well and SSMs may struggle with."),
        ("03", "MoE routing analysis.",
         " Do different experts specialize by language family? If so, can we use "
         "this to build \u201cregional\u201d variants (similar to Tiny Aya\u2019s "
         "earth/fire/water split)?"),
        ("04", "Baseline experiment (Day 1\u20132).",
         " Load Tiny Aya, run inference in 5 languages, measure CPU tokens/sec. "
         "This number is the bar the distilled model must beat."),
        ("05", "Tool calling format.",
         " Using <tool_call> special tokens already implemented in Aetheris data pipeline. "
         "Need to verify this format works with Aya\u2019s tokenizer."),
        ("06", "Licensing.",
         " The CC-BY-NC license on Tiny Aya means this research can be published "
         "but the distilled model weights cannot be used commercially without Cohere\u2019s permission."),
    ]

    for num, bold_part, rest in notes:
        p = doc.add_paragraph()
        add_run(p, f"  {num}  ", bold=True, size=8, color=COHERE_GREEN, font_name="Consolas")
        add_run(p, bold_part, bold=True, size=10, color=INK, font_name="Calibri")
        add_run(p, rest, size=10, color=INK_LIGHT, font_name="Calibri")
        p.space_after = Pt(8)

    # Licensing callout
    p_callout = doc.add_paragraph()
    add_left_border(p_callout, "C75A3A", 16)
    set_paragraph_shading(p_callout, "FDF0EC")
    add_run(p_callout, "Note on licensing: ", bold=True, size=9.5, color=WARM, font_name="Calibri")
    add_run(p_callout, "The CC-BY-NC constraint on the teacher model applies transitively to distilled weights. "
            "Plan for an academic release path. Commercial deployment would require either a license "
            "agreement with Cohere or a re-distillation from a permissively licensed teacher.",
            size=9.5, color=INK_LIGHT, font_name="Calibri")
    p_callout.space_after = Pt(12)

    # Cohere mission callout
    p_mission = doc.add_paragraph()
    add_left_border(p_mission, "39594D", 16)
    set_paragraph_shading(p_mission, "F4F7F5")
    add_run(p_mission, "From Cohere Labs: ", bold=True, size=9.5, color=COHERE_GREEN, font_name="Calibri")
    add_run(p_mission, "Tiny Aya was built with the conviction that language technology should serve everyone, "
            "not just English speakers. This collaboration extends that mission \u2014 making multilingual AI "
            "not just capable, but ",
            size=9.5, color=INK_LIGHT, font_name="Calibri")
    add_run(p_mission, "runnable", italic=True, size=9.5, color=INK_LIGHT, font_name="Calibri")
    add_run(p_mission, " on the hardware people actually have.",
            size=9.5, color=INK_LIGHT, font_name="Calibri")
    p_mission.space_after = Pt(24)

    # ========================================
    # FOOTER
    # ========================================
    # Footer line
    p_fline = doc.add_paragraph()
    add_bottom_border(p_fline, "1A1A1A", 8)
    p_fline.space_after = Pt(10)

    # Footer orgs
    ft = doc.add_table(rows=1, cols=2)
    ft.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell in ft.rows[0].cells:
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        borders = parse_xml(
            f'<w:tcBorders {nsdecls("w")}>'
            f'  <w:top w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            f'  <w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            f'  <w:bottom w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            f'  <w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            f'</w:tcBorders>'
        )
        tcPr.append(borders)

    c1 = ft.rows[0].cells[0]
    c1.text = ""
    p = c1.paragraphs[0]
    add_run(p, "COHERE LABS \u2014 FOR THE BUILDER", bold=True, size=7.5, color=COHERE_GREEN, font_name="Calibri")

    c2 = ft.rows[0].cells[1]
    c2.text = ""
    p = c2.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    add_run(p, "WAYY RESEARCH \u2014 BUFFALO, NY", bold=True, size=7.5, color=WAYY_INDIGO, font_name="Calibri")

    # Tagline
    p_tag = doc.add_paragraph()
    p_tag.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_tag.space_before = Pt(10)
    add_run(p_tag, "\u201cPeople for research, research for people.\u201d",
            italic=True, size=9.5, color=INK_MUTED, font_name="Georgia")

    # ========================================
    # SAVE
    # ========================================
    output_path = "/home/rcgalbo/wayy-research/aya/Project-Aya-Research-Scope.docx"
    doc.save(output_path)
    print(f"Saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    build_document()
