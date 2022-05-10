from PIL import Image
import torch
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json

# read the category labels of the imagenet dataset
json_path = './cam/labels.json'
with open(json_path, 'r') as load_f:
    load_json = json.load(load_f)
classes = {int(key): value for (key, value)
           in load_json.items()}


img_path = './cam/9933031-large.jpg'
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Image pre-processing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

img_pil = Image.open(img_path)
img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))


# loading pre-trained model
model_id = 1
if model_id == 1:
    net = models.squeezenet1_1(pretrained=False)
    pthfile = r'./pretrained/squeezenet1_1-f364aa15.pth'
    net.load_state_dict(torch.load(pthfile))
    finalconv_name = 'features'
elif model_id == 2:
    net = models.resnet18(pretrained=False)
    finalconv_name = 'layer4'
elif model_id == 3:
    net = models.densenet161(pretrained=False)
    finalconv_name = 'features'
net.eval()
print(net)

# print(net._modules.get(finalconv_name))


features_blobs = []     # it is used to store the feature map

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

# Get the output of the features module
net._modules.get(finalconv_name).register_forward_hook(hook_feature)


# get conv weights
params = list(net.parameters())
print(len(params))		# 52
weight_softmax = np.squeeze(params[-2].data.numpy())  # shape:(1000, 512)


logit = net(img_variable)				# Calculate the output value of the input image after passing through the network
print(logit.shape)						# torch.Size([1, 1000])
print(params[-2].data.numpy().shape)    # the size of weights is 1000 (1000, 512, 1, 1)
print(features_blobs[0].shape)			# the size of feature map　(1, 512, 13, 13)

# The result has 1000 classes, sorted, and gets sorted index
h_x = F.softmax(logit, dim=1).data.squeeze()
print(h_x.shape)						# torch.Size([1000])
probs, idx = h_x.sort(0, True)
probs = probs.numpy()					# Probability value sorting
idx = idx.numpy()						# Sort by category index, the higher the probability value, the higher the index

# Take the category with the top 5 probability values and look at the category names and probability values
for i in range(0, 5):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
'''
0.678 -> mountain bike, all-terrain bike, off-roader
0.088 -> bicycle-built-for-two, tandem bicycle, tandem
0.042 -> unicycle, monocycle
0.038 -> horse cart, horse-cart
0.019 -> lakeside, lakeshore

'''


# Define the function to calculate CAM
def returnCAM(feature_conv, weight_softmax, class_idx):
    # The class activation map is up sampled to 256 x 256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape  # 1,512,13,13
    output_cam = []

    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
    print(cam.shape)
    cam = cam.reshape(h, w)
    # All elements on the feature map are normalized to 0-1
    cam_img = (cam - cam.min()) / (cam.max() - cam.min())
    # resize to　0-255
    cam_img = np.uint8(255 * cam_img)
    output_cam.append(cv2.resize(cam_img, size_upsample))

    activate = cv2.resize(cam_img, (380, 216))
    cv2.imwrite('CAM1.jpg', activate)

    return output_cam


# Generate class activation maps for the highest probability class
CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
# Fusion class activation map and original image
img = cv2.imread(img_path)
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
# Response of the feature map is superimposed on the original map
result = heatmap * 0.3 + img * 0.7
cv2.imwrite('CAM0.jpg', result)
