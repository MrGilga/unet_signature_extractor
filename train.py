from pathlib import Path
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from models.unet import UNet  

# dataset with paired images
class SigPairDataset(Dataset):
    def __init__(self, root: str | Path, size: tuple[int,int] = (512, 512)):
        self.root = Path(root)
        self.in_dir  = self.root / "input"
        self.gt_dir  = self.root / "target"
        self.files   = sorted([f for f in self.in_dir.iterdir() if f.suffix.lower() == ".jpg"])
        self.size    = size
        self.to_tensor = transforms.ToTensor()     # just converts to [0-1] tensor

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        f_in  = self.files[idx]
        f_gt  = self.gt_dir / f_in.name

        x_img = Image.open(f_in).convert("L")
        y_img = Image.open(f_gt).convert("L")

        # same geometric transform for both images
        x_img = TF.resize(x_img, self.size, antialias=True)
        y_img = TF.resize(y_img, self.size, antialias=True)

        x = self.to_tensor(x_img)
        y = self.to_tensor(y_img)
        return x, y


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

dataset = SigPairDataset("data/train")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = UNet(in_channels=1, out_channels=1).to(DEVICE)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

running_loss = 0.0
for x, y in tqdm(loader, desc="Training (1 epoch)"):
    x, y = x.to(DEVICE), y.to(DEVICE)
    optimizer.zero_grad()
    preds = model(x)
    loss = loss_fn(preds, y)
    loss.backward()
    optimizer.step()
    running_loss += loss.item() * x.size(0)

epoch_loss = running_loss / len(dataset)
print(f"Loss: {epoch_loss:.4f}")

Path("checkpoints").mkdir(exist_ok=True)
torch.save(model.state_dict(), "checkpoints/unet_simple.pth")
print("Pesos salvati in checkpoints/unet_simple.pth")
