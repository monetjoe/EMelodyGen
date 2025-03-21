import re
import os
import time
import torch
import random
import shutil
import argparse
import warnings
import subprocess
import soundfile as sf
from utils import Patchilizer, TunesFormer, DEVICE, MSCORE
from modelscope import snapshot_download
from transformers import GPT2Config
from music21 import converter, interval, clef, stream
from config import *

EMelodyGen_WEIGHTS_DIR = snapshot_download(f"monetjoe/{DATASET}", cache_dir=TEMP_DIR)


def get_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-num_tunes",
        type=int,
        default=1,
        help="the number of independently computed returned tunes",
    )
    parser.add_argument(
        "-max_patch",
        type=int,
        default=128,
        help="integer to define the maximum length in tokens of each tune",
    )
    parser.add_argument(
        "-top_p",
        type=float,
        default=0.8,
        help="float to define the tokens that are within the sample operation of text generation",
    )
    parser.add_argument(
        "-top_k",
        type=int,
        default=8,
        help="integer to define the tokens that are within the sample operation of text generation",
    )
    parser.add_argument(
        "-temperature",
        type=float,
        default=1.2,
        help="the temperature of the sampling operation",
    )
    parser.add_argument("-seed", type=int, default=None, help="seed for randomstate")
    parser.add_argument(
        "-show_control_code",
        type=bool,
        default=False,
        help="whether to show control code",
    )
    return parser.parse_args()


def get_abc_key_val(text: str, key="K"):
    pattern = re.escape(key) + r":(.*?)\n"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    else:
        return None


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
    for part in score.parts:
        for measure in part.getElementsByClass(stream.Measure):
            # 检查当前小节的谱号
            if measure.clef:
                measure.clef = clef.BassClef()

    octaves_interval = interval.Interval(offset)
    # 遍历每个音符，将其上下移八度
    for note in score.recurse().notes:
        note.transpose(octaves_interval, inPlace=True)

    score.write("musicxml", fp=out_xml_file)
    return xml2abc(out_xml_file), out_xml_file


def adjust_volume(in_audio: str, dB_change: int):
    y, sr = sf.read(in_audio)
    sf.write(in_audio, y * 10 ** (dB_change / 20), sr)


def generate_music(
    args,
    emo: str,
    weights: str,
    outdir=TEMP_DIR,
    fix_tempo=True,
    fix_mode=True,
    fix_pitch=True,
    fix_std=True,
    fix_volume=True,
    clean_score=False,
):
    patchilizer = Patchilizer()
    patch_config = GPT2Config(
        num_hidden_layers=PATCH_NUM_LAYERS,
        max_length=PATCH_LENGTH,
        max_position_embeddings=PATCH_LENGTH,
        vocab_size=1,
    )
    char_config = GPT2Config(
        num_hidden_layers=CHAR_NUM_LAYERS,
        max_length=PATCH_SIZE,
        max_position_embeddings=PATCH_SIZE,
        vocab_size=128,
    )
    model = TunesFormer(patch_config, char_config, share_weights=SHARE_WEIGHTS)
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint["model"])
    model = model.to(DEVICE)
    model.eval()
    prompt = ""
    tunes = ""
    num_tunes = args.num_tunes
    max_patch = args.max_patch
    top_p = args.top_p
    top_k = args.top_k
    temperature = args.temperature
    seed = args.seed
    show_control_code = args.show_control_code
    print(" Hyper parms ".center(60, "#"), "\n")
    args_dict: dict = vars(args)
    for arg in args_dict.keys():
        print(f"{arg}: {str(args_dict[arg])}")

    # fix mode / pitch_std
    if fix_mode and fix_std:
        prompt = f"A:{emo}\n"

    elif fix_mode:
        if emo == "Q1" or emo == "Q4":
            prompt = "A:" + random.choice(["Q1", "Q4"]) + "\n"

        elif emo == "Q2" or emo == "Q3":
            prompt = "A:" + random.choice(["Q2", "Q3"]) + "\n"

    elif fix_std:
        if emo == "Q1" or emo == "Q2":
            prompt = "A:" + random.choice(["Q1", "Q2"]) + "\n"

        elif emo == "Q3" or emo == "Q4":
            prompt = "A:" + random.choice(["Q3", "Q4"]) + "\n"

    print("\n", " Output tunes ".center(60, "#"))
    start_time = time.time()
    for i in range(num_tunes):
        title = f"T:{emo} Fragment\n"
        artist = f"C:Generated by AI\n"
        tune = f"X:{str(i + 1)}\n{title}{artist}{prompt}"
        lines = re.split(r"(\n)", tune)
        tune = ""
        skip = False
        for line in lines:
            if show_control_code or line[:2] not in ["S:", "B:", "E:"]:
                if not skip:
                    print(line, end="")
                    tune += line

                skip = False

            else:
                skip = True

        input_patches = torch.tensor(
            [patchilizer.encode(prompt, add_special_patches=True)[:-1]],
            device=DEVICE,
        )
        if tune == "":
            tokens = None

        else:
            prefix = patchilizer.decode(input_patches[0])
            remaining_tokens = prompt[len(prefix) :]
            tokens = torch.tensor(
                [patchilizer.bos_token_id] + [ord(c) for c in remaining_tokens],
                device=DEVICE,
            )

        while input_patches.shape[1] < max_patch:
            predicted_patch, seed = model.generate(
                input_patches,
                tokens,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                seed=seed,
            )
            tokens = None
            if predicted_patch[0] != patchilizer.eos_token_id:
                next_bar = patchilizer.decode([predicted_patch])
                if show_control_code or next_bar[:2] not in ["S:", "B:", "E:"]:
                    print(next_bar, end="")
                    tune += next_bar

                if next_bar == "":
                    break

                next_bar = remaining_tokens + next_bar
                remaining_tokens = ""
                predicted_patch = torch.tensor(
                    patchilizer.bar2patch(next_bar),
                    device=DEVICE,
                ).unsqueeze(0)
                input_patches = torch.cat(
                    [input_patches, predicted_patch.unsqueeze(0)],
                    dim=1,
                )

            else:
                break

        tunes += f"{tune}\n\n"
        print("\n")

    # fix tempo
    tempo = ""
    if fix_tempo:
        tempo = f"Q:{random.randint(88, 132)}\n"
        if emo == "Q1":
            tempo = f"Q:{random.randint(160, 184)}\n"
        elif emo == "Q2":
            tempo = f"Q:{random.randint(184, 228)}\n"
        elif emo == "Q3":
            tempo = f"Q:{random.randint(40, 69)}\n"
        elif emo == "Q4":
            tempo = f"Q:{random.randint(40, 69)}\n"

        Q_val = get_abc_key_val(tunes, "Q")
        if Q_val:
            tunes = tunes.replace(f"Q:{Q_val}\n", "")

    tunes = tunes.replace(f"A:{emo}\n", tempo)
    # fix mode:major/minor
    mode = "major" if emo == "Q1" or emo == "Q4" else "minor"
    if fix_mode:
        K_val = get_abc_key_val(tunes)
        if mode == "major" and K_val and "m" in K_val:
            tunes = tunes.replace(f"\nK:{K_val}\n", f"\nK:{K_val.split('m')[0]}\n")

        elif mode == "minor" and K_val and not "m" in K_val:
            tunes = tunes.replace(f"\nK:{K_val}\n", f"\nK:{K_val.lower()}min\n")

    print("Generation time: {:.2f} seconds".format(time.time() - start_time))
    timestamp = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
    try:
        # fix avg_pitch (octave)
        if mode == "minor" and fix_pitch:
            offset = -12
            if emo == "Q2":
                offset -= 12

            tunes, xml = transpose_octaves_abc(
                tunes,
                f"{outdir}/{timestamp}.musicxml",
                offset,
            )
            tunes = tunes.replace(title + title, title)
            os.rename(xml, f"{outdir}/[{emo}]{timestamp}.musicxml")
            xml = f"{outdir}/[{emo}]{timestamp}.musicxml"

        else:
            xml = abc2xml(tunes, f"{outdir}/[{emo}]{timestamp}.musicxml")

        audio = xml2(xml, "wav")
        if os.path.exists(xml) and clean_score:
            os.remove(xml)

        if os.path.exists(audio):
            # fix rms vol
            if fix_volume:
                if emo == "Q1":
                    adjust_volume(audio, 5)

                elif emo == "Q2":
                    adjust_volume(audio, 10)

            return audio

        else:
            return ""

    except Exception as e:
        print(f"{e}")
        return ""


def infers(
    dataset: str,
    emotion: str,
    outdir=TEMP_DIR,
    fix_tempo=True,
    fix_mode=True,
    fix_pitch=True,
    fix_std=True,
    fix_volume=True,
):
    os.makedirs(outdir, exist_ok=True)
    parser = argparse.ArgumentParser()
    args = get_args(parser)
    return generate_music(
        args,
        emo=emotion,
        weights=f"{EMelodyGen_WEIGHTS_DIR}/{dataset.lower()}/weights.pth",
        outdir=outdir,
        fix_tempo=fix_tempo,
        fix_mode=fix_mode,
        fix_pitch=fix_pitch,
        fix_std=fix_std,
        fix_volume=fix_volume,
    )


def add_to_log(message: str, log_file_path=f"{EXPERIMENT_DIR}/success_rates.log"):
    print(message)
    with open(log_file_path, "a", encoding="utf-8") as file:
        file.write(message + "\n")


def generate_exps(
    fix_t=True,
    fix_m=True,
    fix_p=True,
    fix_s=True,
    fix_v=True,
    total=100,
    labels=["Q1", "Q2", "Q3", "Q4"],
):
    subdir = "none"
    if not fix_t:
        subdir = "tempo"

    if not fix_m:
        subdir = "mode"

    if not fix_p:
        subdir = "pitch"

    if not fix_s:
        subdir = "std"

    if not fix_v:
        subdir = "volume"

    outdir = f"{EXPERIMENT_DIR}/{subdir}"
    hit_rate = []
    for emo in labels:
        success, fail = 0, 0
        while success < total / len(labels):
            if infers("Rough4Q", emo, outdir, fix_t, fix_m, fix_p, fix_s, fix_v):
                success += 1
            else:
                fail += 1

        hit_rate.append(success / (success + fail))

    add_to_log(f"Rough4Q-{outdir.split('/')[-1]}: {sum(hit_rate) / len(hit_rate)}")


def success_rate(total=100, subset="EMOPIA", labels=["Q1", "Q2", "Q3", "Q4"]):
    hit_rate = []
    outdir = f"{EXPERIMENT_DIR}/{subset.lower()}"
    for emo in labels:
        success, fail = 0, 0
        while success + fail < total / len(labels):
            if infers(subset, emo, outdir):
                success += 1
            else:
                fail += 1

        hit_rate.append(success / (success + fail))

    add_to_log(f"{subset}: {sum(hit_rate) / len(hit_rate)}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    if os.path.exists(EXPERIMENT_DIR):
        shutil.rmtree(EXPERIMENT_DIR)

    generate_exps()  # no ablation
    generate_exps(fix_t=False)  # ablate tempo
    generate_exps(fix_m=False)  # ablate mode
    generate_exps(fix_p=False)  # ablate avg_pitch (octave)
    generate_exps(fix_s=False)  # ablate pitch_std
    generate_exps(fix_v=False)  # ablate volume

    success_rate()  # calc render success rate for EMOPIA
    success_rate(subset="VGMIDI")  # calc render success rate for VGMIDI
