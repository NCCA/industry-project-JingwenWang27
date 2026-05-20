import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class GreenScreenDataset(Dataset):
    """
    loads green screen dataset with image / fg / alpha triplets


    folder structure expected:
         root/
          image/<seq>/<frame>.png
          fg/<seq>/<frame>.png
          alpha/<seq>/<frame>.png

    returns a dict with keys: image, fg, alpha
    """
    def __init__(self, root, size=(512, 512)):
        self.size    = size
        self.samples = []

        image_root = os.path.join(root, 'image')
        fg_root    = os.path.join(root, 'fg')
        alpha_root = os.path.join(root, 'alpha')

        for seq in sorted(os.listdir(image_root)):
            img_dir   = os.path.join(image_root, seq)
            fg_dir    = os.path.join(fg_root,    seq)
            alpha_dir = os.path.join(alpha_root, seq)

            if not os.path.isdir(img_dir):
                continue

            for fname in sorted(os.listdir(img_dir)):
                self.samples.append((
                    os.path.join(img_dir,   fname),
                    os.path.join(fg_dir,    fname),
                    os.path.join(alpha_dir, fname),
                ))

        self.to_tensor = T.ToTensor()
        self.resize    = T.Resize(size, antialias=True)   # antialias avoids checkerboard on downscale  

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, fg_path, alpha_path = self.samples[idx]

        image = self.to_tensor(self.resize(Image.open(img_path).convert('RGB')))
        fg    = self.to_tensor(self.resize(Image.open(fg_path).convert('RGB')))
        alpha = self.to_tensor(self.resize(Image.open(alpha_path).convert('L')))

        # return dict so callers can access by name, easier to extend later
 
        return {'image': image, 'fg': fg, 'alpha': alpha}
