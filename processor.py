import re
import random
import subprocess
from unidecode import unidecode
from music21 import converter, stream
from lib.add_control_codes import split_txt, run_filter, add_tokens
from lib.abc_transposition import find_all_abc, transpose_an_abc_text
from utils import *


# midi to xml
def fix_filename_for_mscore(filename: str):
    output_name = re.sub(r'[\\/:*?"<>|]', "", unidecode(filename))
    output_name = output_name.replace("[", "  ").replace("]", "  ")
    return re.sub(r"\s+", " ", output_name).strip()


def midi2xml(input_midi_file: str, output_xml_file: str):
    filename = rm_ext(os.path.basename(input_midi_file))
    fixed_name = fix_filename_for_mscore(filename)
    fixed_mid_name = input_midi_file.replace(filename, fixed_name)
    if input_midi_file != fixed_mid_name:
        os.rename(input_midi_file, fixed_mid_name)

    with open(LOG_FILE, "a", encoding="utf-8") as error_file:
        result = subprocess.run(
            [MSCORE3, "-o", output_xml_file, fixed_mid_name],
            stderr=error_file,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )

    if result.returncode != 0:
        add_to_log(
            f"[midi2xml]Failed to convert {input_midi_file} to {output_xml_file} : {result.returncode}"
        )


def batch_midi2mxl(mids_to_xmls: dict, output_xml_dir=XML_INPUT):
    os.makedirs(output_xml_dir, exist_ok=True)
    for midi in tqdm(mids_to_xmls, desc="Converting mids to xmls..."):
        midi2xml(midi, mids_to_xmls[midi])


def multi_batch_midi2mxl(
    input_midi_dir=MIDI_OUTPUT, output_xml_dir=XML_INPUT, multi=True
):
    if not os.path.exists(input_midi_dir):
        print(f"Please extract mids into {input_midi_dir} before this!")
        exit()
    else:
        rm_duplicates(input_midi_dir)

    clean_target_dir(XML_INPUT)
    mids_to_xmls = {}
    for rel_root, _, filenames in os.walk(input_midi_dir):
        for filename in tqdm(filenames, desc="Converting mids to xmls..."):
            if filename.endswith(".mid"):
                midi_file = os.path.join(rel_root, filename)
                label = filename.split("_")[0] + "_"
                xml_file = label + str2md5(rm_ext(filename)) + ".musicxml"
                mids_to_xmls[midi_file] = os.path.join(output_xml_dir, xml_file)

    if multi:
        batches, num_cpu = split_dict_by_cpu(mids_to_xmls)
        pool = Pool(processes=num_cpu)
        pool.map(batch_midi2mxl, batches)

    else:
        batch_midi2mxl(mids_to_xmls, output_xml_dir)


# xml augumentation
def slice_xml(input_file: str, output_folder: str, measures_per_part=20):
    # Load the .musicxml file
    score = converter.parse(input_file)

    # Initialize variables
    current_measures = stream.Part()
    current_measure_count = 0
    part_index = 1
    filename_no_ext = rm_ext(os.path.basename(input_file))

    for part in score.parts:
        for element in part.getElementsByClass(stream.Measure):
            current_measures.append(element)
            current_measure_count += 1

            # Check if we've reached the desired number of measures
            if current_measure_count == measures_per_part:
                # Export the current set of measures
                xml_stream = stream.Score([current_measures])
                export_path = f"{output_folder}/{filename_no_ext}_{part_index}.musicxml"
                try:
                    xml_stream.write("musicxml", fp=export_path)
                except Exception as e:
                    add_to_log(
                        f"[slice_xml]Failed to slice {input_file} to {export_path} : {e}"
                    )

                part_index += 1
                # Reset for the next batch
                current_measures = stream.Part()
                current_measure_count = 0

    # Check if there are any remaining measures to be saved
    if current_measure_count > 0:
        # Export the remaining measures
        xml_stream = stream.Score([current_measures])
        export_path = f"{output_folder}/{filename_no_ext}_{part_index}.musicxml"
        try:
            xml_stream.write("musicxml", fp=export_path)
        except Exception as e:
            add_to_log(
                f"[slice_xml]Failed to slice {input_file} to {export_path} : {e}"
            )


def slice_xmls(xmls_files: list, output_slice_dir=XML_OUTPUT):
    os.makedirs(output_slice_dir, exist_ok=True)
    for xml_file in tqdm(xmls_files, desc="Slicing xmls..."):
        slice_xml(xml_file, output_slice_dir)


def multi_slice_xmls(input_xml_dir=XML_INPUT, multi=True):
    if not os.path.exists(input_xml_dir):
        print("Please convert mids to xmls before this!")
        exit()

    clean_target_dir(XML_OUTPUT)
    xmls_files = []
    for rel_root, _, filenames in os.walk(input_xml_dir):
        for filename in filenames:
            if (
                filename.endswith(".xml")
                or filename.endswith(".musicxml")
                or filename.endswith(".mxl")
            ):
                xmls_files.append(os.path.join(rel_root, filename))

    if multi:
        batches, num_cpu = split_list_by_cpu(xmls_files)
        pool = Pool(processes=num_cpu)
        pool.map(slice_xmls, batches)

    else:
        slice_xmls(xmls_files)


# xml to abc
def xml2abc(input_xml_file: str, output_abc_file: str):
    with open(LOG_FILE, "a", encoding="utf-8") as error_file:
        result = subprocess.run(
            f"python -Xfrozen_modules=off ./lib/xml2abc.py {input_xml_file}",
            stdout=subprocess.PIPE,
            stderr=error_file,
            text=True,
        )

    if result.returncode == 0:
        output = str(result.stdout).strip()
        if output:
            with open(output_abc_file, "w", encoding="utf-8") as f:
                f.write(output)

        else:
            add_to_log(f"[xml2abc]Convert {input_xml_file} to an empty abc")

    else:
        add_to_log(
            f"[xml2abc]Failed to convert {input_xml_file} to {output_abc_file} : {result.returncode}"
        )


def batch_xml2abc(xmls_to_abcs: dict, output_abc_dir=ABC_INPUT):
    os.makedirs(output_abc_dir, exist_ok=True)
    for xml in tqdm(xmls_to_abcs, desc="Converting xmls to abcs..."):
        try:
            xml2abc(xml, xmls_to_abcs[xml])
        except UnicodeDecodeError as e:
            add_to_log(
                f"[batch_xml2abc]Failed to convert {xml} into {output_abc_dir} : {e}"
            )


def multi_batch_xml2abc(input_xml_dir=XML_OUTPUT, output_abc_dir=ABC_INPUT, multi=True):
    if not os.path.exists(input_xml_dir):
        print("Please slice xmls before this!")
        exit()

    clean_target_dir(output_abc_dir)
    xmls_to_abcs = {}
    for rel_root, _, filenames in os.walk(input_xml_dir):
        for filename in filenames:
            if (
                filename.endswith(".xml")
                or filename.endswith(".musicxml")
                or filename.endswith(".mxl")
            ):
                xml_file = os.path.join(rel_root, filename)
                xmls_to_abcs[xml_file] = os.path.join(
                    output_abc_dir, change_ext(filename, ".abc")
                )

    if multi:
        batch, num_cpu = split_dict_by_cpu(xmls_to_abcs)
        pool = Pool(processes=num_cpu)
        pool.map(batch_xml2abc, batch)

    else:
        batch_xml2abc(xmls_to_abcs, output_abc_dir)


# abc augumentation
def transpose_abc(abc_path: str, transposed_abc_dir=ABC_OUTPUT):
    for tone in TONE_CHOICES:
        with open(abc_path, "r", encoding="utf-8") as f:
            abc_text_lines = f.readlines()

        try:
            transposed_abc_text, _, _ = transpose_an_abc_text(abc_text_lines, tone)
            abc_name = rm_ext(os.path.basename(abc_path))
            output_path = f"{transposed_abc_dir}/{abc_name}_{tone}.abc"
            with open(output_path, "w", encoding="utf-8") as file:
                file.write(transposed_abc_text)

        except Exception as e:
            add_to_log(f"[transpose_abc]Failed to transpose {abc_path} to {tone} : {e}")


def transpose_abcs(abc_files: list, transposed_abc_dir=ABC_OUTPUT):
    os.makedirs(transposed_abc_dir, exist_ok=True)
    for abc in tqdm(abc_files, desc="Transposing abcs..."):
        transpose_abc(abc)


def multi_transpose_abcs(input_abc_dir=ABC_INPUT, multi=True):
    if not os.path.exists(input_abc_dir):
        print("Please convert xml slices to abcs before this!")
        exit()

    clean_target_dir(ABC_OUTPUT)
    abc_files = []
    for abc_path in find_all_abc(input_abc_dir):
        abc_files.append(abc_path)

    if multi:
        batches, num_cpu = split_list_by_cpu(abc_files)
        pool = Pool(processes=num_cpu)
        pool.map(transpose_abcs, batches)

    else:
        transpose_abcs(abc_files)


def split_abc_to_xml(abc_path: str):
    with open(abc_path, "r", encoding="cp437") as file:
        text = file.read()

    text_parts = text.split("\n\n")
    filename = rm_ext(os.path.basename(abc_path))
    for i, part in enumerate(text_parts):
        piece = part.strip()
        if piece:
            outpath = f"{XML_OUTPUT}/{filename}_{i}.musicxml"
            try:
                score = converter.parse(piece, format="abc")
                score.write(fmt="musicxml", fp=outpath, encoding="utf-8")

            except Exception as e:
                add_to_log(
                    f"[split_abc_to_xml]Cannot convert invalid abc to {outpath} : {e}"
                )


def split_abcs_to_xmls(abc_files: list):
    os.makedirs(XML_OUTPUT, exist_ok=True)
    for abc in tqdm(abc_files, desc="Splitting abcs..."):
        split_abc_to_xml(abc)


def multi_split_abcs_to_xmls(input_abc_dir: str, multi=True):
    if not os.path.exists(input_abc_dir):
        print(f"{input_abc_dir} does not exist!")
        exit()

    clean_target_dir(XML_OUTPUT)
    abc_files = []
    for abc_path in find_all_abc(input_abc_dir):
        abc_files.append(abc_path)

    if multi:
        batches, num_cpu = split_list_by_cpu(abc_files)
        pool = Pool(processes=num_cpu)
        pool.map(split_abcs_to_xmls, batches)

    else:
        split_abcs_to_xmls(abc_files)


# generate dataset
def save_dataset(dataset: list, split_on: bool):
    random.shuffle(dataset)
    data_count = len(dataset)

    if split_on:
        p90 = int(data_count * 0.9)
        write_jsonl(dataset[:p90], "./data/train.jsonl")
        write_jsonl(dataset[p90:], "./data/test.jsonl")

    else:
        write_jsonl(dataset, "./data/dataset.jsonl")

    print(f"{data_count} succeeded in total")
    add_to_log(f"[save_dataset]{data_count} succeeded in total")


def create_dataset(transposed_abcs_dir=ABC_OUTPUT, split_on=False):
    if not os.path.exists(transposed_abcs_dir):
        print("Please transpose abcs before this!")
        exit()
    else:
        rm_duplicates(transposed_abcs_dir)

    dataset = []
    empty_count, fail_count = 0, 0
    # Traverse the path
    for dirpath, _, filelist in os.walk(transposed_abcs_dir):
        # Traverse the list of files
        for this_file in tqdm(filelist, desc="Generating dataset..."):
            if this_file.endswith(".abc"):
                filename = os.path.join(dirpath, this_file)
                content = run_filter(filename)
                if content:
                    meta_data, merged_body_data = split_txt(content)
                    control_code, melody = add_tokens(meta_data, merged_body_data)
                    if melody:
                        dataset.append(
                            {
                                "prompt": control_code,
                                "data": f"X:1\n{melody}",
                                "label": os.path.basename(filename)
                                .split("_")[0]
                                .strip(),
                            }
                        )
                    else:
                        fail_count += 1
                        add_to_log(
                            f"[create_dataset]Failed to extract melody from {filename}"
                        )

                else:
                    empty_count += 1
                    add_to_log(
                        f"[create_dataset]Failed to parse content from {filename}"
                    )

    results = f"{fail_count} failed in total\n{empty_count} empty V1 in total"
    add_to_log(f"[create_dataset]{results}")
    print(results)
    save_dataset(dataset, split_on)


if __name__ == "__main__":
    # multi_batch_midi2mxl()
    # multi_slice_xmls()
    # multi_batch_xml2abc()
    multi_split_abcs_to_xmls("./data/nottingham")
    # multi_batch_rename(XML_OUTPUT)
    # multi_batch_xml2abc()
    # multi_transpose_abcs()
    # create_dataset()
