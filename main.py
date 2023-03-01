import streamlit as st
import torch
from vgg_face import VGG_16
from torchvision import transforms
from dataset import dataset
from dataset import dataset
from torch.utils.data import DataLoader
import io
from PIL import Image
from inference import inference
import os

app = st.container()

@st.cache_resource
def initiate_model():
    transform_img = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.double),
    transforms.Resize((224,224)),
    ])
    model =  VGG_16().double().to('cuda')
    model.load_weights()
    model.eval()
    return model,transform_img

@st.cache_data
def initiate_data():
    data = dataset('data')
    data =  DataLoader(data, batch_size = 8)
    return data

model,transform_img = initiate_model()
data = initiate_data()

def show_photo(name,img):
    st.image(img)
    dir = f'{os.getcwd()}/data/Images/{name}.jpg'
    st.image(dir,caption = name)

with app:
    with st.form("app_form"):
        img = st.file_uploader("Upload Image")
        submitted = st.form_submit_button("Submit")
        if submitted:
            img = img.getvalue()
            img = io.BytesIO(img)
            img = Image.open(img)
            tensor_img = transform_img(img).unsqueeze(0).to('cuda')
            tensor_img = model(tensor_img)
            output = inference(data,model,tensor_img)
            show_photo(output,img)
