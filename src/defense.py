import torch
from torchvision import transforms
from io import BytesIO
from PIL import Image

class JPEGDefense:
    def __init__(self, quality=75, device='cuda'):
        """
        Simulates JPEG compression/decompression to remove adversarial noise.
        """
        self.quality = quality
        self.device = device
        # Normalization (must match model training)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        # We need to un-normalize first to save as JPEG, then re-normalize
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    def __call__(self, images):
        """
        Args:
            images: Tensor (N, 3, H, W) normalized.
        Returns:
            purified: Tensor (N, 3, H, W) re-normalized.
        """
        purified_batch = []
        
        # JPEG operation is non-differentiable, so we do it on CPU per image
        for img_tensor in images:
            # 1. Denormalize (Approximate reconstruction of original image)
            img_unnorm = img_tensor * self.std + self.mean
            img_unnorm = torch.clamp(img_unnorm, 0, 1)
            
            # 2. Convert to PIL
            pil_img = transforms.ToPILImage()(img_unnorm.cpu())
            
            # 3. Compress -> Decompress via memory buffer
            buffer = BytesIO()
            pil_img.save(buffer, "JPEG", quality=self.quality)
            buffer.seek(0)
            img_jpeg = Image.open(buffer)
            
            # 4. Re-transform
            t_img = transforms.ToTensor()(img_jpeg).to(self.device)
            t_img = self.normalize(t_img)
            purified_batch.append(t_img)
            
        return torch.stack(purified_batch)