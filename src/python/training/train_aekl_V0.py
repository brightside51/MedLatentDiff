""" Training script for the autoencoder with KL regulization. """
import argparse
import sys
import os
import warnings
from pathlib import Path

import torch
import torch.optim as optim
sys.path.append("../../../../MedLatentDiff/generative_models/generative/losses")
from perceptual import PerceptualLoss
sys.path.append("../../../../MedLatentDiff/generative_models/generative/networks/nets")
from autoencoderkl import AutoencoderKL
from patchgan_discriminator import PatchDiscriminator
#from generative.losses.perceptual import PerceptualLoss
#from generative.networks.nets import AutoencoderKL
#from generative.networks.nets.patchgan_discriminator import PatchDiscriminator
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from training_functions import train_aekl
from util import get_dataloader, log_mlflow

warnings.filterwarnings("ignore")

# Non-Conditional 3D Diffusion Model Parser Initialization
if True:
    ncdiff_parser = argparse.ArgumentParser(
        description = "Non-Conditional 3D Diffusion Model")
    ncdiff_parser.add_argument('--model_type', type = str,            # Chosen Model / Diffusion
                                choices =  {'video_diffusion',
                                            'blackout_diffusion',
                                            'gamma_diffusion',
                                            'latent_med_diff'},
                                default = 'latent_med_diff')
    ncdiff_parser.add_argument('--model_version', type = int,         # Model Version Index
                                default = 0)
    ncdiff_parser.add_argument('--data_version', type = int,          # Dataset Version Index
                                default = 4)
    ncdiff_parser.add_argument('--noise_type', type = str,            # Diffusion Noise Distribution
                                default = 'gaussian')
    settings1 = ncdiff_parser.parse_args("")

    # ============================================================================================

    # Directories and Path Arguments
    ncdiff_parser.add_argument('--reader_folderpath', type = str,         # Path for Dataset Reader Directory
                                default = '../../../../MetaBreast/data/non_cond')

    ncdiff_parser.add_argument('--public_data_folderpath', type = str,    # Path for Private Dataset Directory
                                #default = "X:/nas-ctm01/datasets/public/MEDICAL/Duke-Breast-Cancer-T1")
                                default = "../../../../../../datasets/public/MEDICAL/Duke-Breast-Cancer-T1")
    ncdiff_parser.add_argument('--private_data_folderpath', type = str,   # Path for Private Dataset Directory
                                #default = "X:/nas-ctm01/datasets/private/METABREST/T1W_Breast")
                                default = '../../../../../../datasets/private/METABREST/T1W_Breast')
    ncdiff_parser.add_argument( '--lung_data_folderpath', type = str,     # Path for LUCAS Dataset Directory
                                #default = "X:/nas-ctm01/datasets/private/LUCAS/lidc/TCIA_LIDC-IDRI_20200921/LIDC-IDRI")
                                default = "../../../../../../datasets/private/LUCAS/lidc/TCIA_LIDC-IDRI_20200921/LIDC-IDRI")

    # Directory | Model-Related Path Arguments
    ncdiff_parser.add_argument('--model_folderpath', type = str,          # Path for Model Architecture Directory
                                default = f'.../../../../MetaBreast/models/{settings1.model_type}')
    ncdiff_parser.add_argument('--script_folderpath', type = str,         # Path for Model Training & Testing Scripts Directory
                                default = f'../../scripts/{settings1.model_type}')
    ncdiff_parser.add_argument('--logs_folderpath', type = str,           # Path for Model Saving Directory
                                default = f'../../../../MetaBreast/logs/{settings1.model_type}')
        
    # ============================================================================================

    # Dataset | Dataset General Arguments
    ncdiff_parser.add_argument('--data_format', type = str,           # Chosen Dataset Format for Reading
                                choices =  {'mp4', 'dicom'},
                                default = 'mp4')
    ncdiff_parser.add_argument('--img_size', type = int,              # Generated Image Resolution
                                default = 128)
    ncdiff_parser.add_argument('--num_slice', type = int,             # Number of 2D Slices in MRI
                                default = 30)
    ncdiff_parser.add_argument('--slice_spacing', type = bool,        # Usage of Linspace for Slice Spacing
                                default = True)
    ncdiff_parser.add_argument('--slice_bottom_margin', type = int,   # Number of 2D Slices to be Discarded in Bottom Margin
                                default = 5)
    ncdiff_parser.add_argument('--slice_top_margin', type = int,      # Number of 2D Slices to be Discarded in Top Margin
                                default = 15)
    ncdiff_parser.add_argument('--data_prep', type = bool,            # Usage of Dataset Pre-Processing Control Value
                                default = True)
    ncdiff_parser.add_argument('--h_flip', type = int,                # Percentage of Horizontally Flipped Subjects
                                default = 50)

    # Dataset | Dataset Splitting Arguments
    ncdiff_parser.add_argument('--train_subj', type = int,            # Number of Random Subjects in Training Set
                                default = 0)                          # PS: Input 0 for all Subjects in the Dataset
    ncdiff_parser.add_argument('--val_subj', type = int,              # Number of Random Subjects in Validation Set
                                default = 0)
    ncdiff_parser.add_argument('--test_subj', type = int,             # Number of Random Subjects in Test Set
                                default = 0)

    # Dataset | DataLoader Arguments
    ncdiff_parser.add_argument('--batch_size', type = int,            # DataLoader Batch Size Value
                                default = 1)
    ncdiff_parser.add_argument('--shuffle', type = bool,              # DataLoader Subject Shuffling Control Value
                                default = False)
    ncdiff_parser.add_argument('--num_workers', type = int,           # Number of DataLoader Workers
                                default = 8)
    ncdiff_parser.add_argument('--num_fps', type = int,               # Number of Video Frames per Second
                                default = 4)

    # ============================================================================================

    # Model | Architecture-Defining Arguments
    ncdiff_parser.add_argument('--seed', type = int,                  # Randomised Generational Seed
                                default = 0)
    ncdiff_parser.add_argument('--dim', type = int,                   # Input Dimensionality (Not Necessary)
                                default = 64)
    ncdiff_parser.add_argument('--num_channel', type = int,           # Number of Input Channels for Dataset
                                default = 1)
    ncdiff_parser.add_argument('--mult_dim', type = tuple,            # Dimensionality for all Conditional Layers
                                default = (1, 2, 4, 8))

    # Model | Training & Diffusion Arguments
    #ncdiff_parser.add_argument('--num_epochs', type = int,            # Number of Training Epochs
    #                            default = 30)
    ncdiff_parser.add_argument('--num_ts', type = int,                # Number of Scheduler Timesteps
                                default = 300)
    ncdiff_parser.add_argument('--num_steps', type = int,             # Number of Diffusion Training Steps
                                default = 500000)
    ncdiff_parser.add_argument('--lr_base', type = float,             # Base Learning Rate Value
                                default = 1e-4)
    ncdiff_parser.add_argument('--save_interval', type = int,         # Number of Training Step Interval inbetween Image Saving
                                default = 1000)
    ncdiff_parser.add_argument('--save_img', type = int,              # Square Root of Number of Images Saved for Manual Evaluation
                                default = 2)

    # ============================================================================================

    settings = ncdiff_parser.parse_args("")
    settings.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--run_dir", help="Location of model to resume.")
    parser.add_argument("--training_ids", help="Location of file with training ids.")
    parser.add_argument("--validation_ids", help="Location of file with validation ids.")
    parser.add_argument("--config_file", default = "../../../configs/stage1/aekl_v0.yaml", help="Location of file with validation ids.")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--n_epochs", type=int, default=25, help="Number of epochs to train.")
    parser.add_argument("--adv_start", type=int, default=25, help="Epoch when the adversarial training starts.")
    parser.add_argument("--eval_freq", type=int, default=10, help="Number of epochs to between evaluations.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    parser.add_argument("--experiment", help="Mlflow experiment name.")

    args = parser.parse_args()
    return args


def main(args):
    set_determinism(seed=args.seed)
    print_config()

    output_dir = Path("../../project/outputs/runs/")
    output_dir.mkdir(exist_ok=True, parents=True)

    run_dir = Path(f"{output_dir}/{args.run_dir}")
    if run_dir.exists() and (run_dir / "checkpoint.pth").exists():
        resume = True
    else:
        resume = False
        run_dir.mkdir(exist_ok=True)

    print(f"Run directory: {str(run_dir)}")
    print(f"Arguments: {str(args)}")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    writer_train = SummaryWriter(log_dir=str(run_dir / "train"))
    writer_val = SummaryWriter(log_dir=str(run_dir / "val"))

    print("Getting data...")
    cache_dir = output_dir / "cached_data_aekl"
    cache_dir.mkdir(exist_ok=True)

    train_loader, val_loader = get_dataloader(settings)

    print("Creating model...")
    config = OmegaConf.load(args.config_file)
    model = AutoencoderKL(**config["stage1"]["params"])
    discriminator = PatchDiscriminator(**config["discriminator"]["params"])
    perceptual_loss = PerceptualLoss(**config["perceptual_network"]["params"])

    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    device = torch.device("cuda")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        discriminator = torch.nn.DataParallel(discriminator)
        perceptual_loss = torch.nn.DataParallel(perceptual_loss)

    model = model.to(device)
    perceptual_loss = perceptual_loss.to(device)
    discriminator = discriminator.to(device)

    # Optimizers
    optimizer_g = optim.Adam(model.parameters(), lr=config["stage1"]["base_lr"])
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config["stage1"]["disc_lr"])

    # Get Checkpoint
    best_loss = float("inf")
    start_epoch = 0
    if resume:
        print(f"Using checkpoint!")
        checkpoint = torch.load(str(run_dir / "checkpoint.pth"))
        model.load_state_dict(checkpoint["state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
    else:
        print(f"No checkpoint found.")

    # Train model
    print(f"Starting Training")
    val_loss = train_aekl(
        model=model,
        discriminator=discriminator,
        perceptual_loss=perceptual_loss,
        start_epoch=start_epoch,
        best_loss=best_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        writer_train=writer_train,
        writer_val=writer_val,
        device=device,
        run_dir=run_dir,
        kl_weight=config["stage1"]["kl_weight"],
        adv_weight=config["stage1"]["adv_weight"],
        perceptual_weight=config["stage1"]["perceptual_weight"],
        adv_start=args.adv_start,
    )

    log_mlflow(
        model=model,
        config=config,
        args=args,
        experiment=args.experiment,
        run_dir=run_dir,
        val_loss=val_loss,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
