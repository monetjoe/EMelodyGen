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


if __name__ == "__main__":
    whole()
