import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms

# Load the MiDaS model from PyTorch Hub
model_type = "DPT_Large"  # Can also try "DPT_Hybrid" or "MiDaS_small"
model = torch.hub.load("intel-isl/MiDaS", model_type)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load the MiDaS transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

# Choose the correct transform for the model type
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# Load and preprocess a sample image
image_path = '/home/thales1/ODS all Dataset/ODS15Kdataset/test/images/043_1631639435-400000072_png.rf.77540e6c3b4efd9cabbcc58c49cd8432.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply the transforms to the image
input_batch = transform(image).to(device)

# Run the image through the model
with torch.no_grad():
    prediction = model(input_batch)

# Resize prediction to match the input image size
prediction = torch.nn.functional.interpolate(
    prediction.unsqueeze(1),
    size=image.shape[:2],
    mode="bicubic",
    align_corners=False,
).squeeze()

# Normalize the output for visualization
prediction = prediction.cpu().numpy()
prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())

# Display the pixel height map
plt.imshow(prediction, cmap='hot')
plt.colorbar(label='Pixel Height')
plt.title('Pixel Height Estimation')
plt.show()
