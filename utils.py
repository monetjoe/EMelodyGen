import os
from tqdm import tqdm
from multiprocessing import Pool

# utils
CPU_ALL_IN = False
MIDI_OUTPUT = "./data/mids"
XML_INPUT = "./data/xmls/inputs"
XML_OUTPUT = "./data/xmls/slices"
ABC_INPUT = "./data/abcs/inputs"
ABC_OUTPUT = "./data/abcs/trans"
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


def clean_dir(target_dir: str):
    import shutil

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    os.makedirs(target_dir)


def str2md5(original_string: str):
    import hashlib

    md5_obj = hashlib.md5()
    # Update the md5 object with the original string encoded as bytes
    md5_obj.update(original_string.encode("utf-8"))
    # Retrieve the hexadecimal representation of the MD5 hash
    return md5_obj.hexdigest()


def write_jsonl(data: list, output_file: str):
    import json

    with open(output_file, "w", encoding="utf-8") as f:
        for item in tqdm(data, desc=f"Saving to {output_file}..."):
            json.dump(item, f)
            f.write("\n")


def rm_duplicates(folder_path: str):
    import hashlib

    file_hashes = {}
    # 遍历文件夹中的所有文件
    for root, _, files in os.walk(folder_path):
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


def determine_key_mode(score_path: str):
    from music21 import converter

    return str(converter.parse(score_path, encoding="utf-8").analyze("key").mode)


def batch_rename(rename_list: list[str], outdir: str):
    fail_list = []
    for srcname in tqdm(rename_list, desc=f"Renaming files with mode label..."):
        ext = "." + srcname.split(".")[-1]
        dstname = str2md5(srcname) + ext
        try:
            mode = determine_key_mode(srcname)
            os.renames(srcname, f"{outdir}/{mode}_{dstname}")

        except PermissionError as e:
            print(f"Add {srcname} to retry list : {e}")
            fail_list.append(srcname)

        except Exception as e:
            add_to_log(f"[batch_rename]Failed to rename {srcname} : {e}")

    if fail_list:
        import time

        time.sleep(1)
        batch_rename(fail_list, outdir)


def multi_batch_rename(dirpath: str, outdir=MIDI_OUTPUT, multi=True):
    rename_list = []
    for root, _, files in os.walk(dirpath):
        for file in tqdm(files, desc="Loading files..."):
            rename_list.append(os.path.join(root, file))

    if multi:
        batches, num_cpu = split_by_cpu(rename_list)
        pool = Pool(processes=num_cpu)
        pool.map(lambda rename_list: batch_rename(rename_list, outdir), batches)

    else:
        batch_rename(rename_list, outdir)
