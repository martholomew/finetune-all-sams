from statistics import mean
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataloader import DatasetSegmentation, collate_fn
from utils.processor import Samprocessor
import argparse
import monai
import torch
import yaml

from segment_anything import build_sam_vit_b, build_sam_vit_l, build_sam_vit_h
#from segment_anything_fast import build_sam_vit_b, build_sam_vit_l, build_sam_vit_h
from mobile_sam import build_sam_vit_t
from utils.lora import LoRA_sam
from utils.lora_mobilesam import LoRA_sam as LoRA_mobilesam

"""
This file is used to train a LoRA_sam model. I use that monai DiceLoss for the training. The batch size and number of epochs are taken from the configuration file.
The model is saved at the end as a safetensor.
"""

parser = argparse.ArgumentParser(description="SAM-fine-tune Training")
parser.add_argument("load" nargs='?', default=None, help="Load LoRA weights.")
parser.add_argument("-d", "--device", choices=["cuda", "cpu"], default="cuda", help="What device to run the training on.")
parser.add_argument("-s", "--sam", choices=["sam", "samfast", "mobilesam", "mobilesamv2"], default="sam", help="What version of SAM to use.")
parser.add_argument("-w", "--weights", choices=["b", "l", "h"], default="b", help="Which SAM weights to use, does not change if using MobileSAM.")
parser.add_argument("-l", "--lora", action="store_true", help="Whether to use LoRA.")
args = parser.parse_args()

# Load the config file
with open("./config.yaml", "r") as ymlfile:
   config_file = yaml.load(ymlfile, Loader=yaml.Loader)

# Take dataset path
train_dataset_path = config_file["DATASET"]["PATH"]

# Load SAM model
if args.sam == "mobilesam":
  sam = build_sam_vit_t(checkpoint=config_file["SAM"]["MOBILESAM_VIT"])
elif args.weights == "b":
  sam = build_sam_vit_b(checkpoint=config_file["SAM"]["SAM_VIT_B"])
elif args.weights == "l":
  sam = build_sam_vit_l(checkpoint=config_file["SAM"]["SAM_VIT_L"])
elif args.weights == "h":
  sam = build_sam_vit_h(checkpoint=config_file["SAM"]["SAM_VIT_H"])

#Create SAM LoRA
rank = config_file["LORA"]["RANK"]
if args.sam == "mobilesam" and args.lora:
  sam_lora = LoRA_mobilesam(sam, rank)
elif args.lora:
  sam_lora = LoRA_sam(sam, rank)

if args.load:
  sam_lora.load_lora_parameters(args.load)

model = sam_lora.sam

# Process the dataset
processor = Samprocessor(model)
train_ds = DatasetSegmentation(config_file, processor, mode="train")

# Create a dataloader
train_dataloader = DataLoader(train_ds, batch_size=config_file["TRAIN"]["BATCH_SIZE"], shuffle=True, collate_fn=collate_fn)

# Initialize optimize and Loss
optimizer = Adam(model.image_encoder.parameters(), lr=config_file["TRAIN"]["LEARNING_RATE"], weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
num_epochs = config_file["TRAIN"]["NUM_EPOCHS"]

if args.device == "cuda":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = "cpu"

# Set model to train and into the device
model.train()
model.to(device)

total_loss = []

for epoch in range(num_epochs):
    epoch_losses = []

    for i, batch in enumerate(tqdm(train_dataloader)):
      
      outputs = model(batched_input=batch,
                      multimask_output=False)

      stk_out = torch.stack([out["low_res_logits"] for out in outputs], dim=0)
      stk_gt = torch.stack([b["ground_truth_mask"] for b in batch], dim=0)
      stk_out = stk_out.squeeze(1)
      stk_gt = stk_gt.unsqueeze(1) # We need to get the [B, C, H, W] starting from [H, W]
      loss = seg_loss(stk_out, stk_gt.float().to(device))
      
      optimizer.zero_grad()
      
      # if args.sam == "samfast":
      #   loss.requires_grad = True
      loss.backward()
      # optimize
      optimizer.step()
      epoch_losses.append(loss.item())

    print(f'EPOCH: {epoch}')
    print(f'Mean loss training: {mean(epoch_losses)}')
    if not epoch % 10:
      if args.lora:
        sam_lora.save_lora_parameters(f"lora_rank{rank}_{epoch}.safetensors")
      else:
        torch.save(model.state_dict(), f"model_{epoch}.pt")
      print("Saved!")

# Save the parameters of the model in safetensors format
if args.lora:
  sam_lora.save_lora_parameters(f"lora_rank{rank}.safetensors")
else:
  torch.save(model.state_dict(), f"model_final.pt")