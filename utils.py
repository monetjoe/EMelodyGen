import os
from tqdm import tqdm

# utils
MSCORE3 = "D:/Program Files/MuseScore 3/bin/MuseScore3.exe"
MIDI_OUTPUT = "./data/mids"
XML_INPUT = "./data/xmls"
XML_OUTPUT = "./data/slices"
ABC_INPUT = "./data/abcs"
ABC_OUTPUT = "./data/trans"
LOG_FILE = "./data/log.txt"
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

CPU_ALL_IN = False


def add_to_log(err_msg: str):
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(f"{err_msg}\n")


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


def split_dict_by_cpu(dictionary: dict):
    num_cpus = os.cpu_count()
    if not CPU_ALL_IN:
        num_cpus -= 1

    if num_cpus is None:
        num_cpus = 1  # 如果无法获取CPU数量，则默认为1

    split_dicts = [{} for _ in range(num_cpus)]
    index = 0

    for key, value in dictionary.items():
        split_dicts[index][key] = value
        index = (index + 1) % num_cpus

    return split_dicts, num_cpus


def split_list_by_cpu(lst: list):
    num_cpus = os.cpu_count()
    if not CPU_ALL_IN:
        num_cpus -= 1

    if num_cpus is None:
        num_cpus = 1  # 如果无法获取CPU数量，则默认为1

    split_lists = [[] for _ in range(num_cpus)]
    index = 0

    for item in lst:
        split_lists[index].append(item)
        index = (index + 1) % num_cpus

    return split_lists, num_cpus


def clean_target_dir(target_dir: str):
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

    # 读取MusicXML文件
    score = converter.parse(score_path)
    # 分析乐谱的调性
    key_signature = score.analyze("key")
    # 获取调性模式（大调或小调）
    mode = key_signature.mode
    # 返回调性模式
    return str(mode)


def batch_rename(rename_list: list):
    fail_list = []
    for srcname in tqdm(rename_list, desc=f"Renaming files with mode label..."):
        ext = "." + srcname.split(".")[-1]
        dstname = str2md5(srcname) + ext
        dirpath = os.path.dirname(srcname)
        try:
            mode = determine_key_mode(srcname)
            os.renames(srcname, f"{dirpath}/{mode}_{dstname}")
        except PermissionError as e:
            print(f"\nAdd {srcname} to retry list : {e}")
            fail_list.append(srcname)

        except Exception as e:
            add_to_log(f"\nFailed to rename {srcname} : {e}")

    if fail_list:
        batch_rename(fail_list)


def multi_batch_rename(dirpath: str, multi=True):
    rename_list = []
    for root, _, files in os.walk(dirpath):
        for file in tqdm(files, desc="Loading files..."):
            rename_list.append(os.path.join(root, file))

    if multi:
        from multiprocessing import Pool

        batches, num_cpu = split_list_by_cpu(rename_list)
        pool = Pool(processes=num_cpu)
        pool.map(batch_rename, batches)

    else:
        batch_rename(rename_list)
