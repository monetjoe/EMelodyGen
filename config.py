PATCH_LENGTH = 128  # Patch Length
PATCH_SIZE = 32  # Patch Size

PATCH_NUM_LAYERS = 9  # Number of layers in the encoder
CHAR_NUM_LAYERS = 3  # Number of layers in the decoder

# Number of epochs to train for (if early stopping doesn't intervene)
NUM_EPOCHS = 32
LEARNING_RATE = 5e-5  # Learning rate for the optimizer
# Batch size for patch during training, 0 for full context
PATCH_SAMPLING_BATCH_SIZE = 0
LOAD_FROM_CHECKPOINT = True  # Whether to load weights from a checkpoint
# Whether to share weights between the encoder and decoder
SHARE_WEIGHTS = False
OUTPUT_PATH = "./output"
LOG_PATH = f"{OUTPUT_PATH}/logs.jsonl"
DATASET = "emo2music"
SUBSET = "VGMIDI"
MSCORE = "D:/Program Files/MuseScore 3/bin/MuseScore3.exe"
TEMP_DIR = "./__pycache__"
EXPERIMENT_DIR = "./exps"
