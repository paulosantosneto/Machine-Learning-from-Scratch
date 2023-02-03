import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from src import *
from streamlit_echarts import st_echarts
import random

def hex_to_rgb(hex):
  rgb = []
  for i in (0, 2, 4):
    decimal = int(hex[i:i+2], 16)
    rgb.append(decimal)
  
  return tuple(rgb)

def linear_regression_theory():

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
        st.markdown("Calculating the gradient descent with respect to the slope (m):")
        st.latex(r"\dfrac{\partial E}{\partial m} = -\dfrac{2}{n} \sum_{i=0}^{n} x_i(y - (mx_i + b))")
        st.markdown("Calculating the gradient descent in relation to the linear coefficient (b):")
        st.latex(r"\dfrac{\partial E}{\partial b} = -\dfrac{2}{n} \sum_{i=0}^{n}(y - (mx_i + b))")
        st.markdown("### 3. update slope and linear coefficient")
        st.latex(r"m_{t+1} = m_{t} - r\dfrac{\partial E}{\partial m}")
        st.latex(r"b_{t+1} = b_{t} - r.\dfrac{\partial E}{\partial b}")

def linear_regression_visualization():
    
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
              new_xs = [[x] for x in xs]
              new_ys = [[y] for y in ys]
              json_data = {"data": []}
              for x, y in zip(xs, ys):
                  json_data["data"].append([x, y])
              options = {"xAxis": {"name": columns[0], "min": 0}, "yAxis": {"name": columns[1], "min": 0}, "series": [{"type": "scatter", "symbolSize":10, "data": json_data["data"]}]} 
              st_echarts(options)

              apply_lr = st.button("Apply Linear Regression")
          
              if apply_lr:

                  lr = LinearRegression(xs=new_xs, ys=new_ys, learning_rate=learning_rate, epochs=epochs)
                  result = lr.fit()
                  cords = []
                  for i in np.arange(0, int(max(xs)), 0.1):
                      predicted = lr.predict(i)
                      cords.append((i, predicted))

                  options_lr = {"xAxis": {"name": columns[0], "min": 0}, "yAxis": {"name": columns[1], "min": 0}, "series": [{"type": "scatter", "symbolSize": 10, "data": json_data["data"]}, {"type": "line", "showSymbol": False, "data": cords, "markPoint": {"itemStyle": {"color": "transparent"}, "label": {"show": True, "position": "left", "formatter": "y = {:.1f}x + {:.1f}".format(result["coef"], result["intercept"]), "color": "#333", "fontSize": 14}, "data": [{"coord": cords[-1]}]}}]}
                  st_echarts(options_lr)
                  max_errors = max(result["errors"], key=lambda x: x[1])[1]
                  print(max_errors)
                  result["errors"] = [(epoch, error / max_errors) for epoch, error in result["errors"]]
                  print(result["errors"])
                  options_error = {"xAxis": {"name": "epochs", "min": 0, "max": len(result["errors"])}, "yAxis": {"name": "error", "min": 0, "max": 1}, "series": [{"type": "line", "showSymbol": False, "data": result["errors"], "smooth": True}]}

                  st_echarts(options_error)

def multiple_linear_regression_theory():

    st.markdown("# Multiple Linear Regression")

    with st.expander("Theory "):
        st.markdown("First, we define our input computation function. As the number of inputs (or features) is variable, the function is defined as follows.")
        st.latex(r"\hat{y_{i}} = x_0 + x_1{\theta}_1 + x_2{\theta}_2 + ... + x_n{\theta}_n")
        st.markdown(r"- $\hat{y_{i}}$: dependent variable for an example i;")
        st.markdown(r"- $x_0$: y-intercept;")
        st.markdown(r"- $\theta_{n}$: slope coefficients for a specific variable;")
        st.markdown(r"- $x_n$: variable (or feature).")
        st.markdown("")
        st.markdown("The next step is to define an error function. For this, we will use the mean squared error function (MSE).")
        st.latex(r"E = MSE: \dfrac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2")
        st.markdown(r"After computing the error for all inputs, we need to calculate the gradient and back-propagate the error to the parameters. For this, and taking into account that $\textbf{W}$ represents a vector of slope coefficients, we first differentiate with respect to any $w_j$.")
        #st.latex(r"\dfrac{d(E)}{d \textbf{w}} = \begin{bmatrix} \dfrac{d(E)}{d\textbf{w}_{1}} & \dfrac{d(E)}{d\textbf{w}_{2}} \cdots \dfrac{d(E)}{d\textbf{w}_{||\textbf{w}||}} \end{bmatrix}") 
        st.latex(r"\dfrac{d(E)}{d\textbf{w}_{j}^{t}} = \dfrac{2}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)x_{j}")
        st.markdown(r"We also derive with respect to the linear coefficient (b).") 
        st.latex(r"\dfrac{d(E)}{db^{t}} = \dfrac{2}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)")
        st.markdown("At the end of every process, it is necessary to update the weights. For this, we used the concept of gradient descent and the learning rate (r).")
        st.latex(r"w_{j}^{t+1} = w^{t} - r\dfrac{d(E)}{dw_{j}^{t}}")
        st.latex(r"b^{t+1} = b^{t} - r\dfrac{d(E)}{db^{t}}")

def multiple_linear_regression_visualization():

    with st.expander("Visualization "):
        csv_file = st.file_uploader("Upload your data", type=["csv"], key="multiple linear regression")

        if csv_file is not None:
          learning_rate_mlr = float(st.selectbox("Learning rate", ["1.0", "0.1", "0.01", "0.001", "0.0001"], index=2, key="mlr lr"))
          epochs_mlr = int(st.selectbox("Epochs", ["1", "10", "20", "50", "100", "500", "1000", "10000"], index=4, key="mlr epochs"))
          data_mlr = pd.read_csv(csv_file, header=0)
          columns = st.multiselect("Select two columns", data_mlr.columns)
          
          if len(columns) == 3:
              
              df = data_mlr[[columns[0], columns[1], columns[2]]][:100]
          
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

def logistic_regression_theory():

    st.markdown("# Logistic Regression")

    with st.expander("Theory   "):

      st.markdown("The logistic function is very similar to the ones previously seen. However, as the objective is to classify and no longer predict a number, two significant changes are made: non-linear function and cross-entropy loss function.")
      st.latex(r"\textit{\textbf{Cost function}}: J(x, w) \rightarrow \frac{1}{N} \sum_{i=0}^{N} L(x; \theta) \quad (1)")
      st.markdown("The cost function remains the same (1), however we will use a new loss function and a non-linear function.")
      st.latex(r"\textit{\textbf{Loss function}}(L): -y_i ln(\hat{y_i}) - (1 - y_i)ln(1 - \hat{y_i}) \quad (2)")
      st.markdown("The binary cross-entropy loss function represents a partial activation of the error according to the predicted and real labels. For example, when the true label is equal to zero, only the fraction of the equation equal to $(1 - y_i) ln(1 - \hat{y_i}) $ will contribute to the loss. In this sense, the closer $\hat{y_i}$ is to zero, the more the loss will also be close to zero. On the other hand, the closer to one the negative Neperian logarithm makes the error tend to infinity.For the other case, that is, when $y_i$ is equal to 1, only the fraction of the equation $-y_i ln(\hat{y_i})$ will contribute to the loss. In this case, when $\hat{y_i}$ is equal to one, log will tend to zero. However, if $\hat{y_i} = 0$, the value of the loss will tend to infinity.")
      

def logistic_regression_visualization():

    with st.expander("Visualization  "):
        csv_file = st.file_uploader("Upload your data", type=["csv"], key="logistic regression")

        if csv_file is not None:
          learning_rate_logistic = float(st.selectbox("Learning rate", ["1.0", "0.1", "0.01", "0.001", "0.0001"], index=2, key="logistic regression learning rate"))
          epochs_logistic = int(st.selectbox("Epochs", ["1", "10", "20", "50", "100", "500", "1000", "10000"], index=4, key="logistic regression epochs"))
          data_logistic_reg = pd.read_csv(csv_file, header=0)
          columns = st.multiselect("Select two columns", data_logistic_reg.columns, key="logistic regression columns")
          
          if len(columns) == 2:

              df = data_logistic_reg[[columns[0], columns[1]]]
              
              xs = data_logistic_reg.iloc[1:, data_logistic_reg.columns.get_loc(columns[0])].values
              ys = data_logistic_reg.iloc[1:, data_logistic_reg.columns.get_loc(columns[1])].values

              json_data = {"data": []}
              for x, y in zip(xs, ys):
                  json_data["data"].append([float(x), float(y)])
              
              options = {"xAxis": {"name": columns[0], "min": 0}, "yAxis": {"name": columns[1], "min": 0}, "series": [{"type": "scatter", "symbolSize":10, "data": json_data["data"]}]} 
              st_echarts(options)
              
              apply_lr = st.button("Apply Logistic Regression")
          
              if apply_lr:
                  new_xs = [[x] for x in xs]
                  
                  lr = LogisticRegression(xs=new_xs, ys=ys, learning_rate=learning_rate_logistic, epochs=epochs_logistic)
                  result = lr.fit()
                  print(result)
                  cords_x = []
                  cords_y = []
                  for i in np.arange(0, int(max(xs)), 0.1):
                      predicted = lr.predict(i)
                      cords_x.append(i)
                      cords_y.append(predicted)
                      #cords.append((i, predicted))

                  #options_lr = {"xAxis": {"name": columns[0], "min": 0}, "yAxis": {"name": columns[1], "min": 0}, "series": [{"type": "scatter", "symbolSize": 10, "data": json_data["data"]}, {"type": "line", "showSymbol": False, "data": cords, "markPoint": {"itemStyle": {"color": "transparent"}, "label": {"show": True, "position": "left","color": "#333", "fontSize": 14}, "data": [{"coord": cords[-1]}]}}]}
                  #st_echarts(options_lr)
                  #options_error = {"xAxis": {"name": "epochs", "min": 0, "max": len(result["errors"])}, "yAxis": {"name": "error"}, "series": [{"type": "line", "showSymbol": False, "data": result["errors"], "smooth": True}]}

                  #st_echarts(options_error)
                  
                  #fig = plt.plot(cords_x, cords_y)
                  #st.plotly_chart(fig)

                  #fig_log = px.scatter(df, x=columns[0], y=columns[1], template="plotly_white")
                  #fig_log.update_traces(marker_size = 10)
                  
                  #fig_log.set_size_inches(400, 400)
                  #fig_r.add_trace(go.Surface(x=x, y=y, z=z, colorscale=light_yellow,  showscale=False, opacity=0.5))
                  fig_log = plt.figure()
                  plt.plot(cords_x, cords_y)
                  st.plotly_chart(fig_log)

def polynomial_regression_theory():

  pass

def polynomial_regression_visualization():

  pass

def k_nearest_neighbors_visualization():
  
  with st.expander("Visualization   "):
        csv_file = st.file_uploader("Upload your data", type=["csv"], key="knn")

        if csv_file is not None:
          k_knn = int(st.selectbox("K", ["1", "3", "5", "7", "9", "11"], index=1, key="knn epochs"))
          data_knn = pd.read_csv(csv_file, header=0)
          columns = st.multiselect("Select three columns", data_knn.columns, key="knn columns")
          
          if len(columns) == 3:

              df = data_knn[[columns[0], columns[1]]]
              
              xs = data_knn.iloc[1:, data_knn.columns.get_loc(columns[0])].values
              ys = data_knn.iloc[1:, data_knn.columns.get_loc(columns[1])].values
              zs = data_knn.iloc[1:, data_knn.columns.get_loc(columns[2])].values
              
              set_zs = list(set(zs.tolist()))
              
              
              colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(set_zs))]
              
              json_data = {"data": []}
              
              for i, (x, y) in enumerate(zip(xs, ys)):
                  
                  json_data["data"].append({"value": [float(x), float(y)], "itemStyle": {"color": colors[set_zs.index(zs.tolist()[i])]}})

              options = {"xAxis": {"name": columns[0], "min": 0}, "yAxis": {"name": columns[1], "min": 0}, "series": [{"type": "scatter", "symbolSize":10, "data": json_data["data"]}]} 
              st_echarts(options)

              x_input = st.number_input(label="X",step=1.,format="%.2f")
              
              y_input = st.number_input(label="Y",step=1.,format="%.2f")
              
              
              predicted_knn = st.button("Predict")

              if predicted_knn:
                
                
                features = []
                for x, y in zip(xs.tolist(), ys.tolist()):
                  features.append([x, y])
                
                knn = KNearestNeighbor(n_neighbors=k_knn)
                knn.fit(features, zs.tolist())
                answer = knn.predict([float(x_input), float(y_input)])

                st.write(answer)
              
            
          
              
