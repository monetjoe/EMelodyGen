import random
import subprocess
from music21 import converter, stream
from lib.add_control_codes import split_txt, run_filter, add_tokens
from lib.abc_transposition import find_all_abc, transpose_an_abc_text
from utils import *


# midi to xml
def midi2xml(in_midi_path: str, out_xml_path: str):
    with open(LOG_FILE, "a", encoding="utf-8") as error_file:
        result = subprocess.run(
            [MSCORE3, "-o", out_xml_path, in_midi_path],
            stderr=error_file,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )

    if result.returncode != 0:
        add_to_log(
            f"[midi2xml]Failed to convert {in_midi_path} to {out_xml_path} : {result.returncode}"
        )


def batch_midi2xml(mids_to_xmls: dict):
    for midi in tqdm(mids_to_xmls, desc="Converting mids to xmls..."):
        midi2xml(midi, mids_to_xmls[midi])


def multi_batch_midi2xml(in_mids_dir: str, out_xmls_dir: str, multi=True):
    if not os.path.exists(in_mids_dir):
        print(f"Please extract mids into {in_mids_dir} before this!")
        exit()

    clean_dir(out_xmls_dir)
    mids_to_xmls = {}
    for rel_root, _, filenames in os.walk(in_mids_dir):
        for filename in tqdm(filenames, desc="Converting mids to xmls..."):
            if filename.endswith(".mid"):
                midi_file = os.path.join(rel_root, filename)
                label = filename.split("_")[0] + "_"
                xml_file = label + str2md5(rm_ext(filename)) + ".musicxml"
                mids_to_xmls[midi_file] = os.path.join(out_xmls_dir, xml_file)

    if multi:
        batches, num_cpu = split_by_cpu(mids_to_xmls)
        with Pool(processes=num_cpu) as pool:
            pool.map(batch_midi2xml, batches)

    else:
        batch_midi2xml(mids_to_xmls)


# xml augumentation
def slice_xml(in_xml_path: str, out_xmls_dir: str, measures_per_part=20):
    score = converter.parse(in_xml_path)
    # Initialize variables
    current_measures = stream.Part()
    current_measure_count = 0
    part_index = 1
    filename_no_ext = rm_ext(os.path.basename(in_xml_path))

    for part in score.parts:
        for element in part.getElementsByClass(stream.Measure):
            current_measures.append(element)
            current_measure_count += 1
            # Check if we've reached the desired number of measures
            if current_measure_count == measures_per_part:
                current_measures[-1].rightBarline = "final"
                # Export the current set of measures
                xml_stream = stream.Score([current_measures])
                export_path = f"{out_xmls_dir}/{filename_no_ext}_{part_index}.musicxml"
                try:
                    xml_stream.write("musicxml", fp=export_path, encoding="utf-8")

                except Exception as e:
                    add_to_log(
                        f"[slice_xml]Failed to slice {in_xml_path} to {export_path} : {e}"
                    )

                part_index += 1
                # Reset for the next batch
                current_measures = stream.Part()
                current_measure_count = 0

    # Check if there are any remaining measures to be saved
    if current_measure_count > 0:
        current_measures[-1].rightBarline = "final"
        # Export the remaining measures
        xml_stream = stream.Score([current_measures])
        export_path = f"{out_xmls_dir}/{filename_no_ext}_{part_index}.musicxml"
        try:
            xml_stream.write("musicxml", fp=export_path, encoding="utf-8")

        except Exception as e:
            add_to_log(
                f"[slice_xml]Failed to slice {in_xml_path} to {export_path} : {e}"
            )


def slice_xmls(in_xml_paths: list, out_xmls_dir: str):
    for xml_file in tqdm(in_xml_paths, desc="Slicing xmls..."):
        slice_xml(xml_file, out_xmls_dir)


def multi_slice_xmls(in_xmls_dir: str, out_xmls_dir: str, multi=True):
    if not os.path.exists(in_xmls_dir):
        print("Please convert mids to xmls before this!")
        exit()

    clean_dir(out_xmls_dir)
    xmls_files = []
    for rel_root, _, filenames in os.walk(in_xmls_dir):
        for filename in filenames:
            if (
                filename.endswith(".xml")
                or filename.endswith(".musicxml")
                or filename.endswith(".mxl")
            ):
                xmls_files.append(os.path.join(rel_root, filename))

    if multi:
        batches, num_cpu = split_by_cpu(xmls_files)
        fixed_slice_xmls = partial(slice_xmls, out_xmls_dir=out_xmls_dir)
        with Pool(processes=num_cpu) as pool:
            pool.map(fixed_slice_xmls, batches)

    else:
        slice_xmls(xmls_files, out_xmls_dir)


# xml to abc
def xml2abc(in_xml_path: str, out_abc_path: str):
    with open(LOG_FILE, "a", encoding="utf-8") as error_file:
        result = subprocess.run(
            ["python", "-Xfrozen_modules=off", "./lib/xml2abc.py", in_xml_path],
            stdout=subprocess.PIPE,
            stderr=error_file,
            text=True,
        )

    if result.returncode == 0:
        output = str(result.stdout).strip()
        if output:
            with open(out_abc_path, "w", encoding="utf-8") as f:
                f.write(output)

        else:
            add_to_log(f"[xml2abc]Convert {in_xml_path} to an empty abc")

    else:
        add_to_log(
            f"[xml2abc]Failed to convert {in_xml_path} to {out_abc_path} : {result.returncode}"
        )


def batch_xml2abc(xmls_to_abcs: dict):
    for xml in tqdm(xmls_to_abcs, desc="Converting xmls to abcs..."):
        try:
            xml2abc(xml, xmls_to_abcs[xml])

        except UnicodeDecodeError as e:
            add_to_log(f"[batch_xml2abc]Failed to convert {xml} into abc : {e}")


def multi_batch_xml2abc(in_xmls_dir: str, out_abcs_dir: str, multi=True):
    if not os.path.exists(in_xmls_dir):
        print("Please slice xmls before this!")
        exit()

    clean_dir(out_abcs_dir)
    xmls_to_abcs = {}
    for rel_root, _, filenames in os.walk(in_xmls_dir):
        for filename in filenames:
            if (
                filename.endswith(".xml")
                or filename.endswith(".musicxml")
                or filename.endswith(".mxl")
            ):
                xml_file = os.path.join(rel_root, filename)
                xmls_to_abcs[xml_file] = os.path.join(
                    out_abcs_dir, change_ext(filename, ".abc")
                )

    if multi:
        batch, num_cpu = split_by_cpu(xmls_to_abcs)
        with Pool(processes=num_cpu) as pool:
            pool.map(batch_xml2abc, batch)

    else:
        batch_xml2abc(xmls_to_abcs)


# abc augumentation
def transpose_abc(in_abc_path: str, out_abcs_dir: str):
    for tone in TONE_CHOICES:
        with open(in_abc_path, "r", encoding="utf-8") as f:
            abc_text_lines = f.readlines()

        try:
            transposed_abc, _, _ = transpose_an_abc_text(abc_text_lines, tone)
            abc_name = rm_ext(os.path.basename(in_abc_path))
            output_path = f"{out_abcs_dir}/{abc_name}_{tone}.abc"
            with open(output_path, "w", encoding="utf-8") as file:
                file.write(transposed_abc)

        except Exception as e:
            add_to_log(
                f"[transpose_abc]Failed to transpose {in_abc_path} to {tone} : {e}"
            )


def transpose_abcs(in_abc_paths: list, out_abcs_dir: str):
    for abc in tqdm(in_abc_paths, desc="Transposing abcs..."):
        transpose_abc(abc, out_abcs_dir)


def multi_transpose_abcs(in_abcs_dir: str, out_abcs_dir: str, multi=True):
    if not os.path.exists(in_abcs_dir):
        print("Please convert xml slices to abcs before this!")
        exit()

    clean_dir(out_abcs_dir)
    abc_files = []
    for abc_path in find_all_abc(in_abcs_dir):
        abc_files.append(abc_path)

    if multi:
        batches, num_cpu = split_by_cpu(abc_files)
        fixed_transpose_abcs = partial(transpose_abcs, out_abcs_dir=out_abcs_dir)
        with Pool(processes=num_cpu) as pool:
            pool.map(fixed_transpose_abcs, batches)

    else:
        transpose_abcs(abc_files, out_abcs_dir)


def split_abc2xml(in_abc_path: str, out_xmls_dir: str):
    with open(in_abc_path, "r", encoding="cp437") as file:
        text = file.read()

    text_parts = text.split("\n\n")
    filename = rm_ext(os.path.basename(in_abc_path))
    for i, part in enumerate(text_parts):
        piece = part.strip()
        if piece:
            outpath = f"{out_xmls_dir}/{filename}_{i}.musicxml"
            try:
                score = converter.parse(piece, format="abc")
                score.parts[0].getElementsByClass(stream.Measure)[
                    -1
                ].rightBarline = "final"
                score.write(fmt="musicxml", fp=outpath, encoding="utf-8")

            except Exception as e:
                add_to_log(
                    f"[split_abc_to_xml]Cannot convert invalid abc to {outpath} : {e}"
                )


def split_abcs2xmls(in_abc_paths: list, out_xmls_dir: str):
    for abc in tqdm(in_abc_paths, desc="Splitting abcs..."):
        split_abc2xml(abc, out_xmls_dir)


def multi_split_abcs2xmls(in_abcs_dir: str, out_xmls_dir: str, multi=True):
    if not os.path.exists(in_abcs_dir):
        print(f"{in_abcs_dir} does not exist!")
        exit()

    clean_dir(out_xmls_dir)
    abc_files = []
    for abc_path in find_all_abc(in_abcs_dir):
        abc_files.append(abc_path)

    if multi:
        batches, num_cpu = split_by_cpu(abc_files)
        fixed_split_abcs2xmls = partial(split_abcs2xmls, out_xmls_dir=out_xmls_dir)
        with Pool(processes=num_cpu) as pool:
            pool.map(fixed_split_abcs2xmls, batches)

    else:
        split_abcs2xmls(abc_files, out_xmls_dir)


# generate dataset
def save_dataset(dataset: list, jsonl_name: str, split_on: bool):
    random.shuffle(dataset)
    data_count = len(dataset)
    if ".jsonl" in jsonl_name:
        jsonl_name = jsonl_name.split(".json")[0]

    if split_on:
        p90 = int(data_count * 0.9)
        write_jsonl(dataset[:p90], f"./data/{jsonl_name}-train.jsonl")
        write_jsonl(dataset[p90:], f"./data/{jsonl_name}-test.jsonl")

    else:
        write_jsonl(dataset, f"./data/{jsonl_name}.jsonl")

    print(f"{data_count} succeeded in total")
    add_to_log(f"[save_dataset]{data_count} succeeded in total")


def create_dataset(in_abcs_dir: str, jsonl_name="dataset", split_on=False):
    if not os.path.exists(in_abcs_dir):
        print("Please transpose abcs before this!")
        exit()

    dataset = []
    empty_count, fail_count = 0, 0
    # Traverse the path
    for dirpath, _, filelist in os.walk(in_abcs_dir):
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
    save_dataset(dataset, jsonl_name, split_on)
