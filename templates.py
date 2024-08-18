from processor import *


def whole():
    rm_duplicates("./data/mids/inputs")
    multi_batch_rename("./data/mids/inputs", "./data/mids/outputs")
    multi_batch_midi2xml("./data/mids/outputs", "./data/xmls/inputs")
    multi_slice_xmls("./data/xmls/inputs", "./data/xmls/outputs")
    multi_batch_xml2abc("./data/xmls/outputs", "./data/abcs/inputs")
    multi_transpose_abcs("./data/abcs/inputs", "./data/abcs/outputs")
    rm_duplicates("./data/abcs/outputs")
    create_dataset("./data/abcs/outputs")


def joint_abcs2xmls():
    rm_duplicates("./data/abcs/joins")
    multi_batch_rename("./data/abcs/joins", "./data/abcs/renames")
    multi_split_abcs2xmls("./data/abcs/renames", "./data/xmls/outputs")


def joint_abcs2dataset():
    joint_abcs2xmls()
    multi_batch_xml2abc("./data/xmls/outputs", "./data/abcs/inputs")
    multi_transpose_abcs("./data/abcs/inputs", "./data/abcs/outputs")
    rm_duplicates("./data/abcs/outputs")
    create_dataset("./data/abcs/outputs")


def xmls2dataset():
    rm_duplicates("./data/xmls/inputs")
    multi_batch_rename(
        "./data/xmls/inputs", "./data/xmls/outputs", relabel_split_by=None
    )
    multi_batch_xml2abc("./data/xmls/outputs", "./data/abcs/inputs")
    multi_transpose_abcs("./data/abcs/inputs", "./data/abcs/outputs")
    rm_duplicates("./data/abcs/outputs")
    create_dataset("./data/abcs/outputs")


def labelled_midi2dataset(jsonl_name="emopia"):
    multi_batch_midi2xml("./data/mids/outputs", "./data/xmls/inputs")
    multi_slice_xmls("./data/xmls/inputs", "./data/xmls/outputs")
    multi_batch_xml2abc("./data/xmls/outputs", "./data/abcs/inputs")
    multi_transpose_abcs("./data/abcs/inputs", "./data/abcs/outputs")
    rm_duplicates("./data/abcs/outputs")
    create_dataset("./data/abcs/outputs", jsonl_name)


if __name__ == "__main__":
    multi_batch_xml2abc("./data/xmls/outputs", "./data/abcs/inputs")
    multi_transpose_abcs("./data/abcs/inputs", "./data/abcs/outputs")
    rm_duplicates("./data/abcs/outputs")
    create_dataset("./data/abcs/outputs", "vgmidi")
