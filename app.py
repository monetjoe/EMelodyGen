import gradio as gr
from abc_transposition import transpose_an_abc_text

tone_choices = [
    "C", "G", "D", "A", "E", "Cb/B", "Gb/F#", "Db/C#", "Ab", "Eb", "Bb", "F",
    "a", "e", "b", "f#", "c#", "ab/g#", "eb/d#", "bb/a#", "f", "c", "g", "d",
    "全调"
]


def single_infer_input(abc: str, tone_choice: str):
    output = ""
    if tone_choice == "全调":
        for i, tone in enumerate(tone_choices[:-1]):
            output += f"X:{i+1}\n" + transpose_an_abc_text(abc, tone) + "\n"

    else:
        output = "X:1\n" + transpose_an_abc_text(abc, tone_choice) + "\n"

    return output


def single_infer_upload(abc_file: str, tone_choice: str):
    with open(abc_file, 'r', encoding='utf-8') as f:
        abc_text_lines = f.readlines()

    return abc_text_lines, single_infer_input(abc_text_lines, tone_choice)


def batch_infer(zip_file: str, tone_choice: str):
    # TODO:
    return zip_file, zip_file + "\n" + tone_choice


with gr.Blocks() as demo:
    gr.Markdown("# abc调性转换器")
    with gr.Tab("单曲转调(输入模式)"):
        gr.Interface(
            fn=single_infer_input,
            inputs=[
                gr.TextArea(label="贴入abc乐谱"),
                gr.Dropdown(
                    label="目标调性",
                    choices=tone_choices,
                    value="A"
                ),
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
                gr.Dropdown(
                    label="目标调性",
                    choices=tone_choices,
                    value="A"
                ),
            ],
            outputs=[
                gr.TextArea(label="上传abc解析", show_copy_button=True),
                gr.TextArea(label="转调结果", show_copy_button=True),
            ],
            concurrency_limit=4,
            allow_flagging=False,
        )

    with gr.Tab("批量转调"):
        gr.Interface(
            fn=batch_infer,
            inputs=[
                gr.components.File(label="上传abc多乐谱压缩包"),
                gr.Dropdown(
                    label="目标调性",
                    choices=tone_choices,
                    value="A"
                ),
            ],
            outputs=[
                gr.components.File(label="下载abc增强数据压缩包"),
                gr.TextArea(label="转调结果", show_copy_button=True),
            ],
            concurrency_limit=4,
            allow_flagging=False,
        )

demo.launch(server_name="0.0.0.0")
