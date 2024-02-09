from statistics import mean
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataloader import DatasetSegmentation, collate_fn
from utils.lora import LoRA_sam
from utils.processor import Samprocessor
import argparse
import monai
import torch
import yaml

"""
This file is used to train a LoRA_sam model. I use that monai DiceLoss for the training. The batch size and number of epochs are taken from the configuration file.
The model is saved at the end as a safetensor.
"""

parser = argparse.ArgumentParser(description="SAM-fine-tune Training")
parser.add_argument("-d", "--device", choices=["cuda", "cpu"], default="cuda", help="What device to run the training on.")
parser.add_argument("-s", "--sam", choices=["sam", "samfast", "mobilesam", "mobilesamv2"], default="sam", help="What version of SAM to use.")

args = parser.parse_args()

if args.sam == "sam":
  from segment_anything import build_sam_vit_b
elif args.sam == "samfast":
   from segment_anything_fast import build_sam_vit_b

# Load the config file
with open("./config.yaml", "r") as ymlfile:
   config_file = yaml.load(ymlfile, Loader=yaml.Loader)

# Take dataset path
train_dataset_path = config_file["DATASET"]["PATH"]
# Load SAM model
sam = build_sam_vit_b(checkpoint=config_file["SAM"]["SAM_VIT_B"])
#Create SAM LoRA
rank = config_file["LORA"]["RANK"]
sam_lora = LoRA_sam(sam, rank)  
model = sam_lora.sam
# Process the dataset
processor = Samprocessor(model, args.sam)
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
      sam_lora.save_lora_parameters(f"lora_rank{rank}_{epoch}.safetensors")
      print("Saved!")

# Save the parameters of the model in safetensors format
rank = config_file["SAM"]["RANK"]
sam_lora.save_lora_parameters(f"lora_rank{rank}.safetensors")
