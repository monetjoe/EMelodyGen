import os
import sys
import torch
import shutil
import warnings
import subprocess

warnings.filterwarnings("ignore")

TEMP_DIR = "./__pycache__"
EN_US = os.getenv("LANG") != "zh_CN.UTF-8"
if EN_US:
    import huggingface_hub

    WEIGHTS_DIR = huggingface_hub.snapshot_download(
        "monetjoe/EMelodyGen",
        cache_dir=TEMP_DIR,
    )
else:
    import modelscope

    WEIGHTS_DIR = modelscope.snapshot_download(
        "monetjoe/EMelodyGen",
        cache_dir=TEMP_DIR,
    )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_LENGTH = 128  # Patch Length
PATCH_SIZE = 32  # Patch Size
PATCH_NUM_LAYERS = 9  # Number of layers in the encoder
CHAR_NUM_LAYERS = 3  # Number of layers in the decoder
PATCH_SAMPLING_BATCH_SIZE = 0  # Batch size for training patch, 0 for full context
LOAD_FROM_CHECKPOINT = True  # Whether to load weights from a checkpoint
SHARE_WEIGHTS = False  # Whether to share weights between the encoder and decoder
EN2ZH = {
    "Low": "低",
    "High": "高",
    "Cite": "引用",
    "Save": "保存",
    "Audio": "音频",
    "minor": "小调",
    "Major": "大调",
    "Mode": "大小调",
    "Submit": "提交",
    "Staff": "五线谱",
    "Status": "状态栏",
    "Feedback": "反馈",
    "Generate": "生成",
    "Dataset": "数据集",
    "BPM tempo": "BPM 速度",
    "Pitch SD": "音高标准差",
    "Video demo": "视频教程",
    "ABC notation": "ABC 记谱",
    "Download MXL": "下载 MXL",
    "Save template": "保存模板",
    "Download MIDI": "下载 MIDI",
    "By template": "通过模板生成",
    "Volume in dB": "dB 音量调节",
    "±12 octave": "±12 八度上下移",
    "Download template": "下载模板",
    "Download MusicXML": "下载 MusicXML",
    "Download PDF score": "下载 PDF 乐谱",
    "By feature control": "通过特征控制生成",
    "Additional info & options": "附加信息及选项",
    "Generate chords coming soon": "生成和声控制暂不可用",
    "The emotion to which the current template belongs": "当前模板所属情感",
    "Valence: reflects negative-positive levels of emotion": "愉悦度 反映情绪的 消极-积极 程度",
    "Arousal: reflects the calmness-intensity of the emotion": "唤醒度 反映情绪的 平静-激烈 程度",
    "The emotion you believe the generated result should belong to": "您所认为生成结果应该所属的情感",
}


def _L(en_txt: str):
    return en_txt if EN_US else f"{en_txt} ({EN2ZH[en_txt]})"


if sys.platform.startswith("linux"):
    apkname = "MuseScore.AppImage"
    shutil.move(os.path.realpath(f"{WEIGHTS_DIR}/{apkname}"), f"./{apkname}")
    extra_dir = "squashfs-root"
    if not os.path.exists(extra_dir):
        subprocess.run(["chmod", "+x", f"./{apkname}"])
        subprocess.run([f"./{apkname}", "--appimage-extract"])

    MSCORE = f"./{extra_dir}/AppRun"
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

else:
    MSCORE = os.getenv("mscore")
