import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class=None):
        self.model.eval()
        output = self.model(input_image)

        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()

        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = torch.mean(gradients, dim=(1,2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i,w in enumerate(weights):
            cam += w*activations[i]

        cam = F.relu(cam)

        cam = cam-cam.min()
        cam = cam / cam.max()

        return cam.cpu().numpy(), output.softmax(dim=1)[0].cpu().detach().numpy()

    def visualize(self, original_image, cam, alpha=0.5):
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)

        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam,(w,h))

        heatmap = cv2.applyColorMap(np.uint8(255*cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

        overlayed = heatmap*alpha+original_image*(1-alpha)
        overlayed = np.uint8(overlayed)

        return overlayed

def visualize_gradcam(model, image_path, transform, classes, device='cuda'):
    original_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(original_image).unsqueeze(0).to(device)

    if hasattr(model.backbone, 'layer4'):
        target_layer = model.backbone.layer4[-1]
    elif hasattr(model.backbone, 'features'):
        target_layer = model.backbone.features[-1]
    else:
        raise ValueError("Could not find target layer")

    gradcam = GradCAM(model, target_layer)
    cam, predictions = gradcam.generate_cam(input_tensor)

    pred_class = predictions.argmax()
    confidence = predictions[pred_class] * 100

    overlayed = gradcam.visualize(np.array(original_image), cam, alpha=0.4)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')

    axes[2].imshow(overlayed)
    axes[2].set_title(f'Prediction: {classes[pred_class]}\nConfidence: {confidence:.2f}%')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    return predictions, cam, overlayed