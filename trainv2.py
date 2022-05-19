import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from costumDataset import Kaisetdepth
#chooses what model to train
if config.MODEL == "ResUnet":
    from Wnet import Generator
else:
    from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from time import localtime
import os
import sys
if not os.path.exists("evaluation"):
    os.mkdir("evaluation")
torch.backends.cudnn.benchmark = True


def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler, writer, epoch=0
):
    loop = tqdm(loader, leave=True)

    for idx, (x,x2, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        x2 = x2.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x,x2)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            writer.add_scalar("L1 train loss",L1.item()/config.L1_LAMBDA,epoch*(len(loop))+idx)
            writer.add_scalar("D_real train loss", torch.sigmoid(D_real).mean().item(), epoch * (len(loop)) + idx)
            writer.add_scalar("D_fake train loss", torch.sigmoid(D_fake).mean().item(), epoch * (len(loop)) + idx)
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
                L1    =L1.item()
            )
def test_fn(
    disc, gen, loader, l1_loss, bce, writer, epoch=0
):
    loop = tqdm(loader, leave=True)
    disc.eval()
    gen.eval()
    with torch.no_grad():
     resultat=[]
     for idx, (x,x2, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        x2 = x2.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x,x2)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2



        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1
            resultat.append(L1.item())



        if idx % 10 == 0:
            writer.add_scalar("L1 test loss",L1.item()/config.L1_LAMBDA,epoch*(len(loop))+idx)
            writer.add_scalar("D_real test loss", torch.sigmoid(D_real).mean().item(), epoch * (len(loop)) + idx)
            writer.add_scalar("D_fake test loss", torch.sigmoid(D_fake).mean().item(), epoch * (len(loop)) + idx)
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
                L1    =L1.item()
            )
    disc.train()
    gen.train()
    return torch.tensor(resultat).mean()

def main():
    writer = SummaryWriter("train{}-{}-{}".format(localtime().tm_mon, localtime().tm_mday, localtime().tm_hour))
    #instancing the models
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    #print(disc)
    gen = Generator().to(config.DEVICE)
    #print(gen)
    #instancing the optims
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    #instancing the Loss-functions
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    #if true loads the checkpoit in the ./
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )


    #training data loading
    train_dataset = Kaisetdepth(depthPath=sys.argv[2],path=sys.argv[1], Listset=config.TRAIN_LIST)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    test_dataset = Kaisetdepth(depthPath=sys.argv[2],path=sys.argv[1],train=False, Listset=config.TRAIN_LIST)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    #enabling MultiPrecision Mode, the optimise performance
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    save_some_examples(gen, test_loader, 0, folder="evaluation")

    best=10000000
    resultat=1
    for epoch in range(config.NUM_EPOCHS):
        print("\n epoch", epoch, "\n")
        train_fn(
           disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler, writer, epoch=epoch
        )
        resultat=test_fn(disc, gen, test_loader,  L1_LOSS, BCE, writer, epoch=epoch)
        if best>resultat:
            best=resultat
            print("improvement of the loss from {} to {}".format(best,resultat))
            save_checkpoint(gen, opt_gen, epoch, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, epoch, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, test_loader, epoch, folder="evaluation")


if __name__ == "__main__":
    main()
