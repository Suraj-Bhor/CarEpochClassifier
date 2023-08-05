try:
    import detectron2
except:
    import os 
    os.system('pip install git+https://github.com/facebookresearch/detectron2.git')
import torch, detectron2
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import numpy as np
import os, json, cv2, random
import PIL.Image as Image
import matplotlib.pyplot as plt
import gradio as gr
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from torchvision import transforms
import torch
from torchvision import models
from sklearn.metrics.pairwise import cosine_similarity
import pickle


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = './models/model_viewpoint_0.006_10k.pt'
num_classes = 9
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, num_classes) # replace the fully connected layer
model.load_state_dict(torch.load(model_path,  map_location=torch.device('cpu')))
model = model.to(device)
model.eval() 


# load model for body year estimation
num_classes_year = 5
model_year = models.resnet18(pretrained=False)
num_features_year = model_year.fc.in_features
model_year.fc = torch.nn.Linear(num_features_year, num_classes_year)
model_year.load_state_dict(torch.load('./models/model_2_body_year_est.pt',  map_location=torch.device('cpu')))
model_year = model_year.to(device)
model_year.eval()

#load model for the body type estimation
num_classes_body_type = 5
model_body_type = models.resnet50(pretrained=False)
num_features_body_type = model_body_type.fc.in_features
model_body_type.fc = torch.nn.Linear(num_features_body_type, num_classes_body_type)
model_body_type.load_state_dict(torch.load('./models/best_model_body_type.pth', map_location=torch.device('cpu')))
model_body_type = model_body_type.to(device)
model_body_type.eval()

def calculate_modernity_score(probabilities):

    year_category_weights = [0,1,2,3,4]
    modernity_score = sum(p * w for p, w in zip(probabilities, year_category_weights))

    return modernity_score

def category_to_year_range(category):
    if category == 0:
        return "2000-2003"
    elif category == 1:
        return "2006-2008"
    elif category == 2:
        return "2009-2011"
    elif category == 3:
        return "2012-2014"
    elif category == 4:
        return "2015-2018"
    else:
        return "Unknown"
    
    
def get_typicality_level(similarity_score):
    if similarity_score > 0.66:
        return "High"
    elif similarity_score > 0.33:
        return "Medium"
    else:
        return "Low"


relevant_categories = ["Hatchback", "SUV", "MPV", "Saloon", "Convertible"]

def category_to_label(category):
    return relevant_categories[category]


def detectron_segmentation(image):
    # Convert PIL Image to OpenCV Image (numpy array)
    if image is None or not image.size:
        return "No image provided.", np.zeros((100, 100, 3), dtype=np.uint8) # return placeholder black image
    image = np.array(image)

    # Configure the model
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    predictor = DefaultPredictor(cfg)
    outputs = predictor(image)

    instances = outputs["instances"].to("cpu")
    scores = instances.scores
    sorted_scores, indices = torch.sort(scores, descending=True)
    instances = instances[indices]

    car_instance = None
    for i in range(len(instances)):
        if instances.pred_classes[i] == 2:
            car_instance = instances[i]
            break

    if car_instance is None:
        return "No automobiles were found in the image.", np.zeros((100, 100, 3), dtype=np.uint8) # return placeholder black image
    else:
        box = car_instance.pred_boxes.tensor.numpy()[0].astype(int)
        x1, y1, x2, y2 = box
        cropped_im = image[y1:y2, x1:x2, :]

        # Preprocess the cropped image and convert it to a PyTorch tensor
        inputs = transforms.ToTensor()(cropped_im)
        inputs = inputs.unsqueeze(0).to(device)

        # Predict the viewpoint

        outputs_viewpoint = model(inputs)
        _, preds_viewpoint = torch.max(outputs_viewpoint, 1)
        viewpoint = preds_viewpoint.item()

        result_text = "Automobile found in the image. "
        if viewpoint != 1:  # If the viewpoint is not frontal, return early.
            result_text += "But, the viewpoint is not frontal."
            return result_text, cropped_im

        result_text += "The viewpoint is frontal. "

        # Predict the year category
        outputs_year = model_year(inputs)
        probs = torch.nn.functional.softmax(outputs_year, dim=1)
        _, preds_year = torch.max(probs, 1)
        year_category = preds_year.item()
        
        modernity_score = calculate_modernity_score(probs.squeeze().detach().cpu().numpy())
        
        year_range = category_to_year_range(year_category)
        
        # Predict the body type using model_3
        outputs_body_type = model_body_type(inputs)
        probs_body_type = torch.nn.functional.softmax(outputs_body_type, dim=1)
        _, preds_body_type = torch.max(probs_body_type, 1)
        body_type = preds_body_type.item()
        body_type_label = category_to_label(preds_body_type.item())
        
        # Load the group_morphs dictionary
        with open('./group_morphs.pickle', 'rb') as handle:
            group_morphs = pickle.load(handle)

        # Remove the final fully connected layer to get the feature extractor
        feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
        #feature_extractor.cuda()
        features = feature_extractor(inputs)
        features = features.view(features.size(0), -1)  # Flatten the features

        # Calculate typicality
        group = (body_type, year_category)
        if group in group_morphs:
            morph = group_morphs[group]
            morph = morph.reshape(1, -1)  # Make sure morph is a 2-D array
        else:
            morph = None
            print(f"Group {group} not found in the dictionary.")
        print(group_morphs.keys()   )
        if morph is not None:
            typicality = cosine_similarity(features.cpu().detach().numpy(), morph)
        else:
            typicality = "N/A"  # or some default value when the group is not recognized
        
        typicality_level = get_typicality_level(typicality)

        result_text = f"Automobile found in the image.\n"\
          f"Viewpoint: Frontal\n"\
          f"Estimated year range: {year_range}\n"\
          f"Modernity score: {modernity_score:.2f}\n"\
          f"Body type: {body_type_label}\n"\
          f"Typicality score: {typicality_level} ({typicality[0][0]:.2f})"

        return result_text, cropped_im
        
# Create title, description, and article strings
title = "CarTypology 🚗🚘🚙🏎️"
description = """
A deep learning model that classifies cars based on their visual appearance and provides additional insights like body type, model year, and modernity score. Additionally, it offers a 'typicality' score indicating how typical the car's appearance is for its determined body type and model year. This application utilizes a combined approach of Object Detection using Detectron2 and Feature Extraction using a Pre-trained CNN.
[Dataset Used](https://deepvisualmarketing.github.io/)
Currently running with high precision and recall.
"""
article = """
This project uses Object Detection, Image Classification, and Feature Extraction to classify cars and provide detailed insights. The application first uses Detectron2, a model that helps identify the car in an image. After successful identification, the image of the car is passed through a pre-trained Convolutional Neural Network (CNN) to extract features. 

These features are then used to classify the car and provide details about its body type and model year. The application also calculates a 'modernity' score based on the model year of the car. 

Additionally, the application calculates a 'typicality' score, which indicates how typical the appearance of the car is for its determined body type and model year. This score is based on the cosine similarity between the extracted features and the average features (morph) for that body type and model year.
"""
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Define Gradio interface
iface = gr.Interface(
    fn=detectron_segmentation, 
    inputs=gr.inputs.Image(), 
    outputs=[gr.outputs.Textbox(), gr.outputs.Image(type='numpy')],
    title=title,
    description=description,
    article=article,
    examples=example_list
)

iface.launch()
