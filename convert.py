import os
import fitz
import subprocess
from PIL import Image
from music21 import converter, interval, clef, stream
from utils import MSCORE


def abc2xml(abc_content, output_xml_path):
    score = converter.parse(abc_content, format="abc")
    score.write("musicxml", fp=output_xml_path, encoding="utf-8")
    return output_xml_path


def xml2(xml_path: str, target_fmt: str):
    src_fmt = os.path.basename(xml_path).split(".")[-1]
    if not "." in target_fmt:
        target_fmt = "." + target_fmt

    target_file = xml_path.replace(f".{src_fmt}", target_fmt)
    print(subprocess.run([MSCORE, "-o", target_file, xml_path]))
    return target_file


def pdf2img(pdf_path: str):
    output_path = pdf_path.replace(".pdf", ".jpg")
    doc = fitz.open(pdf_path)
    images = []  # 创建一个图像列表
    for page_number in range(doc.page_count):
        page = doc[page_number]
        image = page.get_pixmap()  # 将页面渲染为图像
        images.append(
            Image.frombytes("RGB", [image.width, image.height], image.samples)
        )  # 将图像添加到列表
    # 竖向合并图像
    merged_image = Image.new(
        "RGB", (images[0].width, sum(image.height for image in images))
    )
    y_offset = 0
    for image in images:
        merged_image.paste(image, (0, y_offset))
        y_offset += image.height
    # 保存合并后的图像为JPG
    merged_image.save(output_path, "JPEG")
    doc.close()  # 关闭PDF文档
    return output_path


def xml2img(xml_file: str):
    ext = os.path.basename(xml_file).split(".")[-1]
    pdf_score = xml_file.replace(f".{ext}", ".pdf")
    command = [MSCORE, "-o", pdf_score, xml_file]
    result = subprocess.run(command)
    print(result)
    return pdf_score, pdf2img(pdf_score)


# xml to abc
def xml2abc(input_xml_file: str):
    result = subprocess.run(
        ["python", "-Xfrozen_modules=off", "./xml2abc.py", input_xml_file],
        stdout=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0:
        return str(result.stdout).strip()

    return ""


def transpose_octaves_abc(abc_notation: str, out_xml_file: str, offset=-12):
    score = converter.parse(abc_notation)
    if offset < 0:
        for part in score.parts:
            for measure in part.getElementsByClass(stream.Measure):
                if measure.clef:  # 检查当前小节的谱号
                    measure.clef = clef.BassClef()

    octaves_interval = interval.Interval(offset)
    for note in score.recurse().notes:  # 遍历每个音符，将其上下移八度
        note.transpose(octaves_interval, inPlace=True)

    score.write("musicxml", fp=out_xml_file)
    return xml2abc(out_xml_file), out_xml_file
