import torch
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = r'C:\Users\dell\Desktop\safe'
TRAIN_LIST = ["set00" , 'set01' , 'set02' , 'set06' , 'set07']
#TRAIN_LIST = ['set00']
TEST_LIST = ["set08"]
VAL_DIR = r'C:\Users\dell\Desktop\safe'
MODEL_LIST = ["ResUnet", "Unet"]
MODEL = MODEL_LIST[0]
LEARNING_RATE = 2e-4
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 10
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 0.5
NUM_EPOCHS = 30
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"

transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 