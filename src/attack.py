import torch
import torch.nn.functional as F

class BIMAttack:
    def __init__(self, model, epsilon=0.05, alpha=0.01, steps=10, device='cuda'):
        """
        Args:
            model: The trained PyTorch model.
            epsilon: Maximum perturbation (L-infinity norm).
            alpha: Step size per iteration.
            steps: Number of iterations.
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.device = device

    def __call__(self, images, target_label_idx, target_values):
        """
        Args:
            images: Batch of original images (N, C, H, W).
            target_label_idx: Integer index of the label to attack (0-39).
            target_values: Tensor of shape (N,) containing the GOAL values (0 or 1).
                           (Inverse of ground truth).
        """
        # 1. Clone images and enable gradients
        # We need to detach first to ensure we don't mess up existing graphs
        adv_images = images.clone().detach().to(self.device)
        adv_images.requires_grad = True
        
        for i in range(self.steps):
            # Forward pass
            logits = self.model(adv_images)
            
            # 2. Calculate Loss ONLY for the target label
            # We slice the logits to get just the column we care about
            target_logits = logits[:, target_label_idx]
            
            # BCE Loss between current prediction and the TARGET (inverse truth)
            # We want to MINIMIZE the distance to the target.
            loss = F.binary_cross_entropy_with_logits(target_logits, target_values)
            
            # 3. Compute Gradients
            self.model.zero_grad()
            loss.backward()
            
            # 4. Update Images (Targeted = Move TOWARDS negative gradient)
            # Math: x_new = x - alpha * sign(grad)
            with torch.no_grad():
                grad_sign = adv_images.grad.sign()
                adv_images = adv_images - self.alpha * grad_sign
                
                # 5. Projection (Clip)
                # Ensure we don't modify more than epsilon from original
                delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
                adv_images = torch.clamp(images + delta, min=0, max=1) # Valid image range
                
                # Reset gradients for next step
                adv_images.grad = None
                
        return adv_images.detach()