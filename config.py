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
WEIGHT_URL = "https://huggingface.co/sander-wood/tunesformer/resolve/main/weights.pth"
WEIGHT_URL_ZH = "https://www.modelscope.cn/models/MuGeminorum/tunesformer/resolve/master/weights.pth"
OUTPUT_PATH = "./output"
WEIGHT_PATH = f"{OUTPUT_PATH}/weights.pth"
LOG_PATH = f"{OUTPUT_PATH}/logs.jsonl"
PROMPT_PATH = "prompt.txt"
DATASET = "emo2music"
SUBSET = "EMOPIA"
