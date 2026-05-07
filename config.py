from pathlib import Path

DATA_DIR  = Path(__file__).parent / "data"
CKPT_DIR  = Path(__file__).parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

# shared
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 1e-3
EMBED_DIM   = 128
DROPOUT     = 0.5

# TextCNN
CNN_FILTERS      = 128
CNN_KERNEL_SIZES = (2, 3, 4, 5)

# BiLSTM
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
