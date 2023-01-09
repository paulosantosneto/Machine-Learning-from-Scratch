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

def hex_to_rgb(hex):
  rgb = []
  for i in (0, 2, 4):
    decimal = int(hex[i:i+2], 16)
    rgb.append(decimal)
  
  return tuple(rgb)

def menu():

    with st.sidebar:
        
        menu = option_menu("Menu", ["Linear Regression", "Logistic Regression", "General Linear Model", "Naive Bayes", "K-nearest neighbors", "K-means clustering", "Support Vector Machine", "Neural Networks", "Convolutional Neural Networks", "Generative Neural Networks"], icons=["plus"]*10, menu_icon="list", default_index=0)

        st.session_state.sidebar_selected = menu

def linear_regression():
    
    st.markdown("# Simple Linear Regression")
    with st.expander("Theory"):

        st.markdown("## Algorithm steps:")
        st.markdown("1. calculate the error;")
        st.markdown("2. calculate the new slope and linear coefficients;")
        st.markdown("3. update slope and linear coefficient;")
        st.markdown("4. returns to step 1 if the number os epochs has not yet been completed.")
        st.markdown("### 1. Calculate the error")
        st.latex(r"Error function: \dfrac{1}{n} \sum_{i=0}^{n} (y - (mx_i + b))^2")
        st.markdown("### 2. Calculate the new slople and linear coefficients")
        st.markdown("Derived from the error function with respect to the slope:")
        st.latex(r"\dfrac{\partial E}{\partial m} = -\dfrac{2}{n} \sum_{i=0}^{n} x_i(y - (mx_i + b))")
        st.markdown("Derived from the error function with respect to the linear coefficient:")
        st.latex(r"\dfrac{\partial E}{\partial b} = -\dfrac{2}{n} \sum_{i=0}^{n}(y - (mx_i + b))")
        st.markdown("### 3. update slope and linear coefficient")
        st.latex(r"m_{i+1} = m_{i} - r\dfrac{\partial E}{\partial m}")
        st.latex(r"b_{i+1} = b_{i} - r.\dfrac{\partial E}{\partial b}")
    
    with st.expander("Visualization"): 
        csv_file = st.file_uploader("Upload your data", type=["csv"])

        if csv_file is not None:
          learning_rate = float(st.selectbox("Learning rate", ["1.0", "0.1", "0.01", "0.001", "0.0001"], index=2))
          epochs = int(st.selectbox("Epochs", ["1", "10", "20", "50", "100", "500", "1000", "10000"], index=4))
          data = pd.read_csv(csv_file, header=0)
          columns = st.multiselect("Select two columns", data.columns)
          if len(columns) == 2: 
              xs = data.iloc[1:, data.columns.get_loc(columns[0])].values
              ys = data.iloc[1:, data.columns.get_loc(columns[1])].values

              json_data = {"data": []}
              for x, y in zip(xs, ys):
                  json_data["data"].append([x, y])
              options = {"xAxis": {"name": columns[0], "min": 0}, "yAxis": {"name": columns[1], "min": 0}, "series": [{"type": "scatter", "symbolSize":10, "data": json_data["data"]}]} 
              st_echarts(options)

              apply_lr = st.button("Apply Linear Regression")
          
              if apply_lr:

                  lr = LinearRegression(xs=xs, ys=ys, learning_rate=learning_rate, epochs=epochs)
                  result = lr.fit()
                  cords = []
                  for i in np.arange(0, int(max(xs)), 0.1):
                      predicted = lr.predict(i)
                      cords.append((i, predicted))

                  options_lr = {"xAxis": {"name": columns[0], "min": 0}, "yAxis": {"name": columns[1], "min": 0}, "series": [{"type": "scatter", "symbolSize": 10, "data": json_data["data"]}, {"type": "line", "showSymbol": False, "data": cords, "markPoint": {"itemStyle": {"color": "transparent"}, "label": {"show": True, "position": "left", "formatter": "y = {:.1f}x + {:.1f}".format(result["coef"], result["intercept"]), "color": "#333", "fontSize": 14}, "data": [{"coord": cords[-1]}]}}]}
                  st_echarts(options_lr)
                  options_error = {"xAxis": {"name": "epochs", "min": 0, "max": len(result["errors"])}, "yAxis": {"name": "error"}, "series": [{"type": "line", "showSymbol": False, "data": result["errors"], "smooth": True}]}

                  st_echarts(options_error)

    st.markdown("# Multiple Linear Regression")

    with st.expander("Theory "):
      st.write("in progress") 

    with st.expander("Visualization "):
        csv_file = st.file_uploader("Upload your data", type=["csv"], key="multiple linear regression")

        if csv_file is not None:
          learning_rate_mlr = float(st.selectbox("Learning rate", ["1.0", "0.1", "0.01", "0.001", "0.0001"], index=2, key="mlr lr"))
          epochs_mlr = int(st.selectbox("Epochs", ["1", "10", "20", "50", "100", "500", "1000", "10000"], index=4, key="mlr epochs"))
          data_mlr = pd.read_csv(csv_file, header=0)
          columns = st.multiselect("Select two columns", data_mlr.columns)
          
          if len(columns) == 3:
              
              df = data_mlr[[columns[0], columns[1], columns[2]]]
          
              for column in df.columns:
                df[column] = df[column]  / df[column].abs().max()
                
              xs = df.iloc[1:, df.columns.get_loc(columns[0])].values.tolist()
              ys = df.iloc[1:, df.columns.get_loc(columns[1])].values.tolist()
              zs = df.iloc[1:, df.columns.get_loc(columns[2])].values.tolist()
              xss = []
              zss = []
              
              for x, y in zip(xs, ys):
                xss.append([x, y])

              for z in zs:
                zss.append(z)
              
           
              fig = px.scatter_3d(df, x=columns[0], y=columns[1], z=columns[2], template="plotly_white")
              fig.update_traces(marker_size = 2)
              st.plotly_chart(fig)
              apply_mlr = st.button("Apply Multiple Linear Regression")

              if apply_mlr:
                mlr = MultipleLinearRegression(xss, zss, learning_rate=learning_rate_mlr, epochs=epochs_mlr)
                results = mlr.fit()
                
                
                light_yellow = [[0, '#b6d7a8'], [1, '#b6d7a8']]
                x = np.linspace(0, 1.0, 100)
                y = np.linspace(0, 1.0, 100)
                cont = 0
                z= np.ones((100,100))
                
                for j in range(100):
                  for i in range(100):
                    z[j][i] = mlr.predict([x[i], y[i]])
                
                
                
                fig_r = px.scatter_3d(df, x=columns[0], y=columns[1], z=columns[2], template="plotly_white")
                fig_r.update_traces(marker_size = 2)
                fig_r.add_trace(go.Surface(x=x, y=y, z=z, colorscale=light_yellow,  showscale=False, opacity=0.5))
                st.plotly_chart(fig_r)
              

                            
def logistic_regression():

    pass

def general_linear_model():

    pass

def naive_bayes():

    pass

def k_nearest_neighbors():

    pass

def k_means_clustering():

    pass

def support_vector_machine():

    pass

def neural_networks():

    pass

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

    if st.session_state.sidebar_selected == "Linear Regression":
        linear_regression()
    elif st.session_state.sidebar_selected == "Convolutional Neural Networks":
        convolutional_neural_networks()
