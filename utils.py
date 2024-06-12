import os
from tqdm import tqdm
from music21 import converter
from functools import partial
from multiprocessing import Pool

# utils
CPU_ALL_IN = False
LOG_FILE = "./data/log.txt"
MSCORE3 = "D:/Program Files/MuseScore 3/bin/MuseScore3.exe"
TONE_CHOICES = [
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
]


def add_to_log(err_msg: str):
    import time

    timestamp = time.strftime("[%Y-%m-%d_%H-%M-%S]", time.localtime(time.time()))
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(f"{timestamp}{err_msg}\n")


def rm_ext(filename: str):
    last_dot_index = len(filename) - 1
    while last_dot_index >= 0:
        if filename[last_dot_index] == ".":
            break

        last_dot_index -= 1

    # 如果找到了点，则去掉后缀
    if last_dot_index >= 0:
        return filename[:last_dot_index]

    else:
        # 如果没有找到点，则返回原始文件名
        return filename


def change_ext(filename: str, ext: str):
    if ext == "":
        return rm_ext(filename)

    extension = ext if len(ext) > 0 and ext[0] == "." else f".{ext}"
    return rm_ext(filename) + extension


def split_by_cpu(items):
    num_cpus = os.cpu_count()
    if not CPU_ALL_IN:
        num_cpus -= 1

    if num_cpus is None:
        num_cpus = 1  # 如果无法获取CPU数量，则默认为1

    index = 0
    if type(items) == dict:
        split_items = [{} for _ in range(num_cpus)]
        for key, value in items.items():
            split_items[index][key] = value
            index = (index + 1) % num_cpus
    else:
        split_items = [[] for _ in range(num_cpus)]
        for item in items:
            split_items[index].append(item)
            index = (index + 1) % num_cpus

    return split_items, num_cpus


def clean_dir(in_dir: str):
    import shutil

    if os.path.exists(in_dir):
        shutil.rmtree(in_dir)

    os.makedirs(in_dir)


def str2md5(origin_str: str):
    import hashlib

    md5_obj = hashlib.md5()
    # Update the md5 object with the original string encoded as bytes
    md5_obj.update(origin_str.encode("utf-8"))
    # Retrieve the hexadecimal representation of the MD5 hash
    return md5_obj.hexdigest()


def write_jsonl(data: list, out_jsonl: str):
    import json

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for item in tqdm(data, desc=f"Saving to {out_jsonl}..."):
            json.dump(item, f)
            f.write("\n")


def rm_duplicates(in_files_dir: str):
    import hashlib

    file_hashes = {}
    # 遍历文件夹中的所有文件
    for root, _, files in os.walk(in_files_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            # 使用哈希算法计算文件内容的哈希值
            with open(file_path, "rb") as f:
                file_content = f.read()
                file_hash = hashlib.md5(file_content).hexdigest()

            # 如果哈希值已经存在于字典中，则删除重复文件
            if file_hash in file_hashes:
                print(f"Removing duplicate file: {file_path}")
                os.remove(file_path)
            else:
                # 否则，将哈希值和文件路径添加到字典中
                file_hashes[file_hash] = file_path


def emo_label(score_path: str):
    score = converter.parse(score_path, encoding="utf-8")
    mode: str = score.analyze("key").mode
    # 获取所有小节的 tempo 值
    tempos = []
    for part in score.parts:
        for measure in part.getElementsByClass("Measure"):
            measure_tempos = measure.getElementsByClass("MetronomeMark")
            for tempo in measure_tempos:
                tempos.append(tempo.getQuarterBPM())

    tempo = 120
    # 计算平均 tempo
    if tempos:
        average_tempo = sum(tempos) / len(tempos)
        tempo = average_tempo

    if mode == "major" and tempo >= 120:
        return "Q1"
    elif mode == "minor" and tempo >= 120:
        return "Q2"
    elif mode == "minor" and tempo < 120:
        return "Q3"
    else:
        return "Q4"


def batch_rename(in_score_paths, out_scores_dir: str, relabel_split_by: str):
    import shutil

    fail_list = []
    for srcname in tqdm(in_score_paths, desc=f"Renaming scores with labels..."):
        ext = "." + srcname.split(".")[-1]
        dstname = str2md5(srcname) + ext
        try:
            if relabel_split_by:
                label = os.path.basename(srcname).split(relabel_split_by)[0]
            else:
                label = emo_label(srcname)

            shutil.copy2(srcname, f"{out_scores_dir}/{label}_{dstname}")

        except PermissionError as e:
            print(f"Add {srcname} to retry list : {e}")
            fail_list.append(srcname)

        except Exception as e:
            add_to_log(f"[batch_rename]Failed to rename {srcname} : {e}")

    if fail_list:
        import time

        time.sleep(1)
        batch_rename(fail_list, out_scores_dir, relabel_split_by)


def multi_batch_rename(
    in_scores_dir: str, out_scores_dir: str, relabel_split_by="_", multi=True
):
    if not os.path.exists(in_scores_dir):
        print(f"Please put scores into {in_scores_dir} before this!")
        exit()

    clean_dir(out_scores_dir)
    rename_list = []
    for root, _, files in os.walk(in_scores_dir):
        for file in tqdm(files, desc="Loading files..."):
            rename_list.append(os.path.join(root, file))

    if multi:
        batches, num_cpu = split_by_cpu(rename_list)
        fixed_batch_rename = partial(
            batch_rename,
            out_scores_dir=out_scores_dir,
            relabel_split_by=relabel_split_by,
        )
        pool = Pool(processes=num_cpu)
        pool.map(fixed_batch_rename, batches)

    else:
        batch_rename(rename_list, out_scores_dir, relabel_split_by)
