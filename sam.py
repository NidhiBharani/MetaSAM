from segment_anything import SamPredictor, sam_model_registry
import cv2

#load model checkpoint
sam = sam_model_registry["vit_h"](checkpoint="/home/nidhi/code/Meta-SAM/sam_vit_h_4b8939.pth")
sam.to(device="cuda")

#input image and prompt.
img_path = "/home/nidhi/code/Meta-SAM/ioana-ye-5EkUELLjYEI-unsplash.jpg"
prompt = "hand"

#image as np array
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

predictor = SamPredictor(sam)
predictor.set_image(img)
masks, _, _ = predictor.predict("hand")