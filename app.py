import os
import json
import shutil
import argparse
import gradio as gr
from generate import generate_music, get_args
from utils import ZERO, _L, WEIGHTS_DIR, TEMP_DIR


@ZERO
def infer_by_template(dataset: str, v: str, a: str, add_chord: bool):
    status = "Success"
    audio = midi = pdf = xml = mxl = tunes = jpg = None
    try:
        emotion = "Q1"
        if v == _L("Low") and a == _L("High"):
            emotion = "Q2"

        elif v == _L("Low") and a == _L("Low"):
            emotion = "Q3"

        elif v == _L("High") and a == _L("Low"):
            emotion = "Q4"

        if add_chord:
            print("Chord generation comes soon!")

        parser = argparse.ArgumentParser()
        args = get_args(parser)
        args.template = True
        audio, midi, pdf, xml, mxl, tunes, jpg = generate_music(
            args,
            emo=emotion,
            weights=f"{WEIGHTS_DIR}/{dataset.lower()}/weights.pth",
        )

    except Exception as e:
        status = f"{e}"

    return status, audio, midi, pdf, xml, mxl, tunes, jpg


@ZERO
def infer_by_features(
    dataset: str,
    pitch_std: str,
    mode: str,
    tempo: int,
    octave: int,
    rms: int,
    add_chord: bool,
):
    status = "Success"
    audio = midi = pdf = xml = mxl = tunes = jpg = None
    try:
        emotion = "Q1"
        if mode == _L("minor") and pitch_std == _L("High"):
            emotion = "Q2"

        elif mode == _L("minor") and pitch_std == _L("Low"):
            emotion = "Q3"

        elif mode == _L("Major") and pitch_std == _L("Low"):
            emotion = "Q4"

        if add_chord:
            print("Chord generation comes soon!")

        parser = argparse.ArgumentParser()
        args = get_args(parser)
        args.template = False
        audio, midi, pdf, xml, mxl, tunes, jpg = generate_music(
            args,
            emo=emotion,
            weights=f"{WEIGHTS_DIR}/{dataset.lower()}/weights.pth",
            fix_tempo=tempo,
            fix_pitch=octave,
            fix_volume=rms,
        )

    except Exception as e:
        status = f"{e}"

    return status, audio, midi, pdf, xml, mxl, tunes, jpg


def feedback(
    fixed_emo: str,
    source_dir=f"./{TEMP_DIR}/output",
    target_dir=f"./{TEMP_DIR}/feedback",
):
    try:
        if not fixed_emo:
            raise ValueError("Please select feedback before submitting! ")

        os.makedirs(target_dir, exist_ok=True)
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.endswith(".mxl"):
                    prompt_emo = file.split("]")[0][1:]
                    if prompt_emo != fixed_emo:
                        file_path = os.path.join(root, file)
                        target_path = os.path.join(
                            target_dir, file.replace(".mxl", f"_{fixed_emo}.mxl")
                        )
                        shutil.copy(file_path, target_path)
                        return f"Copied {file_path} to {target_path}"

                    else:
                        return "Thanks for your feedback!"

        return "No .mxl files found in the source directory."

    except Exception as e:
        return f"{e}"


def save_template(label: str, pitch_std: str, mode: str, tempo: int, octave: int, rms):
    status = "Success"
    template = None
    try:
        if (
            label
            and pitch_std
            and mode
            and tempo != None
            and octave != None
            and rms != None
        ):
            json_str = json.dumps(
                {
                    "label": label,
                    "pitch_std": pitch_std == _L("High"),
                    "mode": mode == _L("Major"),
                    "tempo": tempo,
                    "octave": octave,
                    "volume": rms,
                }
            )

            with open(
                f"./{TEMP_DIR}/feedback/templates.jsonl",
                "a",
                encoding="utf-8",
            ) as file:
                file.write(json_str + "\n")

            template = f"./{TEMP_DIR}/feedback/templates.jsonl"

        else:
            raise ValueError("Please check features")

    except Exception as e:
        status = f"{e}"

    return status, template


if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                with gr.Accordion(label=_L("Additional info & options"), open=False):
                    gr.Video(f"{WEIGHTS_DIR}/demo.mp4", label=_L("Video demo"))
                    gr.Markdown(f"## {_L('Cite')}" + """
                        ### AIART
                        ```bibtex
                        @inproceedings{11152266,
                            author    = {Zhou, Monan and Li, Xiaobing and Yu, Feng and Li, Wei},
                            booktitle = {2025 IEEE International Conference on Multimedia and Expo Workshops (ICMEW)},
                            title     = {EMelodyGen: Emotion-Conditioned Melody Generation in ABC Notation with the Musical Feature Template},
                            year      = {2025},
                            pages     = {1-6},
                            keywords  = {Correlation;Codes;Conferences;Confusion matrices;Music;Psychology;Data augmentation;Complexity theory;Reliability;Melody generation;controllable music generation;ABC notation;emotional condition},
                            doi       = {10.1109/ICMEW68306.2025.11152266}
                        }
                        ```
                        ### TAI
                        ```bibtex
                        @article{zhou_li_yu_li_2025,
                            title     = {EMelodyGen: Emotion-Conditioned Melody Generation in ABC Notation with Musical Feature Templates},
                            volume    = {1},
                            issn      = {2982-3439},
                            doi       = {10.53941/tai.2025.100013},
                            number    = {1},
                            journal   = {Transactions on Artificial Intelligence},
                            publisher = {Scilight Press},
                            author    = {Zhou, Monan and Li, Xiaobing and Yu, Feng and Li, Wei},
                            year      = {2025},
                            pages     = {199&ndash;211}
                        }
                        ```""")
                    with gr.Row():
                        data_opt = gr.Dropdown(
                            ["VGMIDI", "EMOPIA", "Rough4Q"],
                            label=_L("Dataset"),
                            value="Rough4Q",
                        )
                        chord_chk = gr.Checkbox(
                            label=_L("Generate chords coming soon"),
                            value=False,
                        )

                with gr.Tab(_L("By template")):
                    gr.Image(f"{WEIGHTS_DIR}/4q.jpg", show_label=False)
                    v_radio = gr.Radio(
                        [_L("Low"), _L("High")],
                        label=_L(
                            "Valence: reflects negative-positive levels of emotion"
                        ),
                        value=_L("High"),
                    )
                    a_radio = gr.Radio(
                        [_L("Low"), _L("High")],
                        label=_L(
                            "Arousal: reflects the calmness-intensity of the emotion"
                        ),
                        value=_L("High"),
                    )
                    gen1_btn = gr.Button(_L("Generate"))

                with gr.Tab(_L("By feature control")):
                    std_opt = gr.Radio(
                        [_L("Low"), _L("High")],
                        label=_L("Pitch SD"),
                        value=_L("High"),
                    )
                    mode_opt = gr.Radio(
                        [_L("minor"), _L("Major")],
                        label=_L("Mode"),
                        value=_L("Major"),
                    )
                    tempo_opt = gr.Slider(
                        minimum=40,
                        maximum=228,
                        step=1,
                        value=120,
                        label=_L("BPM tempo"),
                    )
                    octave_opt = gr.Slider(
                        minimum=-24,
                        maximum=24,
                        step=12,
                        value=0,
                        label=_L("±12 octave"),
                    )
                    volume_opt = gr.Slider(
                        minimum=-5,
                        maximum=10,
                        step=5,
                        value=0,
                        label=_L("Volume in dB"),
                    )
                    gen2_btn = gr.Button(_L("Generate"))
                    with gr.Accordion(label=_L("Save template"), open=False):
                        with gr.Row():
                            with gr.Column(min_width=160):
                                save_radio = gr.Radio(
                                    ["Q1", "Q2", "Q3", "Q4"],
                                    label=_L(
                                        "The emotion to which the current template belongs"
                                    ),
                                )
                                save_btn = gr.Button(_L("Save"))

                            with gr.Column(min_width=160):
                                save_file = gr.File(label=_L("Download template"))

            with gr.Column():
                wav_audio = gr.Audio(
                    label=_L("Audio"),
                    type="filepath",
                    buttons=["download"],
                )
                with gr.Accordion(label=_L("Feedback"), open=False):
                    fdb_radio = gr.Radio(
                        ["Q1", "Q2", "Q3", "Q4"],
                        label=_L(
                            "The emotion you believe the generated result should belong to"
                        ),
                    )
                    fdb_btn = gr.Button(_L("Submit"))

                status_bar = gr.Textbox(label=_L("Status"), buttons=["copy"])
                with gr.Row():
                    mid_file = gr.File(label=_L("Download MIDI"), min_width=80)
                    pdf_file = gr.File(label=_L("Download PDF score"), min_width=80)
                    xml_file = gr.File(label=_L("Download MusicXML"), min_width=80)
                    mxl_file = gr.File(label=_L("Download MXL"), min_width=80)

                with gr.Row():
                    abc_txt = gr.TextArea(label=_L("ABC notation"), buttons=["copy"])
                    staff_img = gr.Image(
                        label=_L("Staff"),
                        type="filepath",
                        buttons=["fullscreen", "download"],
                    )

        # actions
        gen1_btn.click(
            fn=infer_by_template,
            inputs=[data_opt, v_radio, a_radio, chord_chk],
            outputs=[
                status_bar,
                wav_audio,
                mid_file,
                pdf_file,
                xml_file,
                mxl_file,
                abc_txt,
                staff_img,
            ],
        )
        gen2_btn.click(
            fn=infer_by_features,
            inputs=[
                data_opt,
                std_opt,
                mode_opt,
                tempo_opt,
                octave_opt,
                volume_opt,
                chord_chk,
            ],
            outputs=[
                status_bar,
                wav_audio,
                mid_file,
                pdf_file,
                xml_file,
                mxl_file,
                abc_txt,
                staff_img,
            ],
        )
        save_btn.click(
            fn=save_template,
            inputs=[save_radio, std_opt, mode_opt, tempo_opt, octave_opt, volume_opt],
            outputs=[status_bar, save_file],
        )
        fdb_btn.click(fn=feedback, inputs=fdb_radio, outputs=status_bar)

    demo.launch(
        theme=gr.themes.Ocean(),
        css="#gradio-share-link-button-0 { display: none; }",
        ssr_mode=False,
    )
