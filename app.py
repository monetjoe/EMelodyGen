import os
import re
import shutil
import zipfile
import gradio as gr
from abc_transposition import transpose_an_abc_text, find_all_abc

tone_choices = [
    "Cb",
    "Gb",
    "Db",
    "Ab",
    "Eb",
    "Bb",
    "F",
    "C",
    "G",
    "D",
    "A",
    "E",
    "B",
    "F#",
    "C#",
    "All",
]


def single_infer_input(abc, tone_choice: str):
    from_list = type(abc) == list
    abc_text_lines = abc if from_list else abc.splitlines()
    output = ""

    if tone_choice == "All":
        for i, tone in enumerate(tone_choices[:-1]):
            try:
                transposed_abc_text, _, _ = transpose_an_abc_text(abc_text_lines, tone)
            except Exception as e:
                return f"{e}"

            if not from_list:
                transposed_abc_text = re.sub(
                    r"(?<!\n)(M:|K:|V:)", r"\n\1", transposed_abc_text
                )

            output += f"X:{i+1}\n{transposed_abc_text}\n\n"

    else:
        try:
            transposed_abc_text, _, _ = transpose_an_abc_text(
                abc_text_lines, tone_choice
            )
        except Exception as e:
            return f"{e}"

        if not from_list:
            transposed_abc_text = re.sub(
                r"(?<!\n)(M:|K:|V:)", r"\n\1", transposed_abc_text
            )

        output = f"X:1\n{transposed_abc_text}\n"

    return output


def single_infer_upload(abc_file: str, tone_choice: str):
    with open(abc_file, "r", encoding="utf-8") as f:
        abc_text_lines = f.readlines()

    return "".join(abc_text_lines), single_infer_input(abc_text_lines, tone_choice)


def unzip(zip_file, extract_to="./data/batch"):
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)

    os.makedirs(extract_to)

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    return extract_to


def zip_dir(directory="./data/output", zip_file="./data/output.zip"):
    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, directory))

    return zip_file


def save_to_abc(text, file_path):
    target_dir = os.path.dirname(file_path)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)


def batch_infer(zip_file: str, tone_choice: str):
    extract_to = unzip(zip_file)
    for abc_path in find_all_abc(extract_to):
        abc_name = os.path.basename(abc_path)
        if tone_choice == "All":
            for tone in tone_choices[:-1]:
                _, transposed_abc_text = single_infer_upload(abc_path, tone)
                save_to_abc(transposed_abc_text, f"./data/output/{tone}_{abc_name}")

        else:
            _, transposed_abc_text = single_infer_upload(abc_path, tone_choice)
            save_to_abc(transposed_abc_text, f"./data/output/{tone_choice}_{abc_name}")

    return zip_dir()


with gr.Blocks() as demo:
    gr.Markdown("# abc调性转换器")
    with gr.Tab("单曲转调(输入模式)"):
        gr.Interface(
            fn=single_infer_input,
            inputs=[
                gr.TextArea(label="贴入abc乐谱"),
                gr.Dropdown(label="目标调性", choices=tone_choices, value="A"),
            ],
            outputs=[
                gr.TextArea(label="转调结果", show_copy_button=True),
            ],
            concurrency_limit=4,
            allow_flagging=False,
        )

    with gr.Tab("单曲转调(上传模式)"):
        gr.Interface(
            fn=single_infer_upload,
            inputs=[
                gr.components.File(label="上传abc乐谱"),
                gr.Dropdown(label="目标调性", choices=tone_choices, value="A"),
            ],
            outputs=[
                gr.TextArea(label="abc提取结果", show_copy_button=True),
                gr.TextArea(label="转调结果", show_copy_button=True),
            ],
            concurrency_limit=4,
            allow_flagging=False,
        )

    with gr.Tab("批量转调"):
        gr.Interface(
            fn=batch_infer,
            inputs=[
                gr.components.File(label="上传abc多乐谱zip压缩包"),
                gr.Dropdown(label="目标调性", choices=tone_choices, value="A"),
            ],
            outputs=[
                gr.components.File(label="下载abc增强数据压缩包"),
            ],
            concurrency_limit=4,
            allow_flagging=False,
        )

demo.launch(server_name="0.0.0.0")
