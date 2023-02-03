import streamlit as st
from streamlit_option_menu import option_menu
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import copy
import matplotlib.pyplot as plt
import io
from streamlit_echarts import st_echarts
import json
from src import *
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import *

def menu():

    with st.sidebar:
        
        menu = option_menu("Menu", ["Regression", "Naive Bayes", "Decision Threes", "K-nearest neighbors", "K-means clustering", "Support Vector Machine", "Neural Networks", "Convolutional Neural Networks", "Generative Neural Networks"], icons=["plus"]*9, menu_icon="list", default_index=0, styles={
            "nav-link-selected": {"background-color": "#0c0b09"}}) # 9468fc

        st.session_state.sidebar_selected = menu

def regression():
    
    linear_regression_theory()
    
    linear_regression_visualization()

    multiple_linear_regression_theory()

    multiple_linear_regression_visualization()

    polynomial_regression_theory()

    polynomial_regression_visualization()

    logistic_regression_theory()

    logistic_regression_visualization()
              
def naive_bayes():

    pass

def k_nearest_neighbors():

    k_nearest_neighbors_visualization()

def k_means_clustering():

    pass

def support_vector_machine():

    pass

def neural_networks():

    st.selectbox("Choose optimizer", ["SGD", "SGD+Momentum", "Nesterov Momentum", "RMSProp", "RMSProp+Momentum", "Adam"])

def convolutional_neural_networks():

    st.markdown("# Fundamentals") 
    st.markdown("## Metrics")
    metrics()



def generative_neural_networks():

    pass

def check_states():

    if "sidebar_selected" not in st.session_state:
        st.session_state.sidebar_selected = None
    if "marker_function" not in st.session_state:
        st.session_state.marker_function = None
    if "canvas_length" not in st.session_state:
        st.session_state.canvas_length = 0 
    if "canvas_nms" not in st.session_state:
        st.session_state.canvas_nms = []
    if "compare" not in st.session_state:
        st.session_state.compare = []
        
def upload_image(name):

    image = st.file_uploader(name, type=["png", "jpg"])

    if image is not None:
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)

    return image

def create_canvas(ID: str, image: list, label_color: str, drawing_mode: str, point_display_radius: int=4, stroke_width: int=2):
        
    canvas = st_canvas(fill_color=label_color, width=660, height=440, stroke_width=stroke_width, stroke_color="#eee", background_image=Image.open(image).resize((660, 420)), drawing_mode=drawing_mode, key=ID, point_display_radius=point_display_radius if drawing_mode == "point" else 0)

    return canvas

if __name__ == "__main__":
    
    check_states()
    menu()

    if st.session_state.sidebar_selected == "Regression":
        regression()
    elif st.session_state.sidebar_selected == "K-nearest neighbors":
        k_nearest_neighbors()
    elif st.session_state.sidebar_selected == "Convolutional Neural Networks":
        convolutional_neural_networks()
    
    elif st.session_state.sidebar_selected == "Neural Networks":
        neural_networks()
      
