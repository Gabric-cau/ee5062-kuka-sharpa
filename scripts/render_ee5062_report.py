#!/usr/bin/env python3
"""Render the Markdown report and its measured figures as a reviewable PDF."""

import argparse
from html import escape
from pathlib import Path
import textwrap
import xml.etree.ElementTree as ET

import markdown
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image, LongTable, Paragraph, SimpleDocTemplate, Spacer, TableStyle, XPreformatted,
)


def inline(element):
    result = escape(element.text or "")
    for child in element:
        tag = {"strong": "b", "em": "i", "code": "font", "a": "a"}.get(child.tag)
        if tag == "font":
            result += '<font name="Courier">' + inline(child) + "</font>"
        elif tag == "a":
            href = escape(child.attrib.get("href", ""), quote=True)
            if href.startswith(("https://", "http://")):
                result += f'<a href="{href}" color="#206D81">' + inline(child) + "</a>"
            else:
                result += inline(child)
        elif child.tag == "br":
            result += "<br/>"
        elif tag:
            result += f"<{tag}>" + inline(child) + f"</{tag}>"
        else:
            result += inline(child)
        result += escape(child.tail or "")
    return result


def render(source, output):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    html = markdown.markdown(source.read_text(), extensions=["tables", "fenced_code"], output_format="xhtml")
    root = ET.fromstring("<body>" + html + "</body>")
    styles = getSampleStyleSheet()
    for level in range(1, 4):
        styles[f"Heading{level}"].keepWithNext = True
    styles.add(ParagraphStyle("ReportBody", fontSize=9, leading=12, spaceAfter=6,
                             splitLongWords=True, alignment=TA_LEFT,
                             allowWidows=0, allowOrphans=0))
    styles.add(ParagraphStyle("ReportCell", fontSize=7.3, leading=9.4, splitLongWords=True))
    styles.add(ParagraphStyle("ReportCode", fontName="Courier", fontSize=7, leading=9,
                             spaceBefore=3, spaceAfter=8))
    width = A4[0] - 36 * mm
    story = []
    for element in root:
        if element.tag in {"h1", "h2", "h3", "h4"}:
            level = min(int(element.tag[1]), 3)
            story.append(Paragraph(inline(element), styles[f"Heading{level}"]))
        elif element.tag == "table":
            rows = [[Paragraph(inline(cell), styles["ReportCell"]) for cell in row]
                    for row in element.findall(".//tr")]
            columns = max(map(len, rows))
            table = LongTable(rows, colWidths=[width / columns] * columns, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF0F1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, 0), .6, colors.HexColor("#8F9DA2")),
                ("LINEBELOW", (0, 1), (-1, -1), .2, colors.HexColor("#D7DFE2")),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story += [table, Spacer(1, 8)]
        elif element.tag == "pre":
            code = "".join(element.itertext())
            lines = []
            for line in code.splitlines():
                lines.extend(textwrap.wrap(line, 106, replace_whitespace=False,
                                           drop_whitespace=False) or [""])
            story.append(XPreformatted(escape("\n".join(lines)), styles["ReportCode"]))
        elif element.tag in {"ul", "ol"}:
            for index, item in enumerate(element.findall("li"), 1):
                prefix = f"{index}. " if element.tag == "ol" else "- "
                story.append(Paragraph(prefix + inline(item), styles["ReportBody"]))
        elif element.tag == "p" and element.find("img") is not None:
            picture = element.find("img")
            path = (source.parent / picture.attrib["src"]).resolve()
            image = Image(str(path))
            ratio = min(width / image.imageWidth, 165 * mm / image.imageHeight)
            image.drawWidth, image.drawHeight = image.imageWidth * ratio, image.imageHeight * ratio
            story.extend([image, Spacer(1, 6)])
            if picture.attrib.get("alt"):
                story.append(Paragraph(escape(picture.attrib["alt"]), styles["ReportBody"]))
        elif element.tag == "hr":
            story.append(Spacer(1, 6))
        else:
            story.append(Paragraph(inline(element), styles["ReportBody"]))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#63717A"))
        canvas.drawString(18 * mm, 12 * mm, "EE5062 | Simulation evidence | Review draft")
        canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, str(document.page))
        canvas.restoreState()

    document = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=18 * mm,
        rightMargin=18 * mm, topMargin=17 * mm, bottomMargin=20 * mm,
        title="EE5062 KUKA-Sharpa Simulation Report", author="Student contribution pending review")
    document.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
