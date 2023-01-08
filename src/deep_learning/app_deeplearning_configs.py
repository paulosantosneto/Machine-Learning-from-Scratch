from .computer_vision import *
from streamlit_option_menu import option_menu
from streamlit_drawable_canvas import st_canvas
import streamlit as st
import numpy as np
from PIL import Image

def metrics():
    with st.expander("Introduction"):
        
        st.write(r"<div style='text-align': justify; '>Validating how good your object detection model predicts is critical, and for that you need to have good validation metrics in your toolkit. In this sense, four metrics commonly used in most articles in the field of computer vision are presented below.</div>", unsafe_allow_html=True)  

        st.markdown("")
        
        st.markdown("- Intersection Over Union (IoU)")
        st.markdown("- Non Max Suppression (NMS)")
        st.markdown("- Mean Average Precision (mAP)")
        st.markdown("- Average Recall (AR)")

    with st.expander("Intersection Over Union"):
        iou_plot()
    with st.expander("Non Max Suppression"):
        nms_plot()
    with st.expander("Mean Average Precision"):
        mAP_plot()
    with st.expander("Average Recall"):
        AR_plot()

def iou_plot():

    st.markdown("# Intersection Over Union (IoU)")
    st.write("The IoU metric indicates how close the predicted bounding box is to the true label. The numeric indicator varies between 0 and 1, and can be calculated from the formula below.")
    st.latex(r'''IoU = \frac{pred \cup true}{pred \cap true}''')
        
    ID = ""
    image = upload_image(ID)
    if image != None:

        label_color = (st.color_picker("Choose a color: ", "#FFA500") + "77")
        canvas = create_canvas(ID, image, label_color, "rect")
        drawing_iou(canvas, ID)

def nms_plot():

    st.markdown("# Non Max Suppression")
    st.markdown("<div style='text-align: justify; '>NMS is an algorithm for removing bounding boxes of the same class that are overlapping. In short, the bounding box with the highest confidence value is chosen. It is worth noting that the value compares only bounding boxes of the same class and that have intersect. In this way, it is possible to suppress two or more detections of the same class that identify two identical objects but in different positions. </div>", unsafe_allow_html=True)
    
    ID = " "
    image = upload_image(ID)
        
    if image != None:
            
        prob_threshold = st.slider("Probability threshold:", 0.0, 1.0)
        iou_threshold = st.slider("IoU threshold:", 0.0, 1.0)
        label_prob = st.text_input("Enter the probability:", "0.5")
        label_class = st.text_input("Enter the class:", "myclass")
        label_color = (st.color_picker("Choose a color:", "#FFA500") + "77") 
        canvas = create_canvas(ID, image, label_color, "rect")
        if len(canvas.json_data["objects"]) == 0:
            st.session_state.compare = []
            st.session_state.canvas_nms = []
        for i, rect in enumerate(canvas.json_data["objects"]):
            if rect not in st.session_state.compare:
                st.session_state.compare.append(copy.deepcopy(rect))
                aux = copy.deepcopy(rect)
                aux["probability"] = float(label_prob)
                aux["class"] = label_class
                aux["label_color"] = label_color
                st.session_state.canvas_nms.append(aux)
        boxes = drawing_nms(canvas, ID, prob_threshold, iou_threshold)
        image_nms = np.array(Image.open(image).resize((660, 440)))
        shape_image = image_nms.copy()
        output_image = image_nms.copy()
        if len(boxes) > 0:
            for box in boxes:
                cv2.rectangle(image_nms, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color=hex_to_rgb(box[3][1:len(box[3])-2]), thickness=-1) 
        output_image = cv2.addWeighted(image_nms, 0.3, output_image, 1 - 0.3, gamma=0) 

        if len(boxes) > 0:
            for box in boxes:
                cv2.rectangle(output_image, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color=(255, 255, 255), thickness=1)

        output_image = cv2.addWeighted(output_image, 0.5, output_image, 1 - 0.5, gamma=0)
        if len(canvas.json_data["objects"]) > 0:
            st.image(output_image)
    else:
        st.session_state.canvas_nms = []
        st.session_state.compare = []

def mAP_plot():
    ### Mean Average Precision ###
    st.markdown("# Mean Average Precision")

    
    ID = "  "
    image = upload_image(ID)

    if image != None:
        drawing_bboxes(ID, image)

def AR_plot():

    pass

def drawing_iou(canvas, ID):

    if len(canvas.json_data["objects"]) == 2: 
        button = st.button("Calculate IoU")
        if button:
            print(st.session_state.canvas_nms)
            bboxes = []

            for rect in canvas.json_data["objects"]:
                box = [rect["left"], rect["top"], rect["width"]+rect["left"], rect["height"]+rect["top"]]
                bboxes.append(box)
            
            iou_value, box = iou(bboxes[0], bboxes[1])
            iou_value = round(iou_value, 4) 
            if iou_value == 1.:
                st.balloons()
            st.info("Iou: {:.2f}%".format(iou_value*100))

def drawing_nms(canvas: dict, ID: str, prob_threshold: float, iou_threshold: float):
    box = []
    if len(canvas.json_data["objects"]) > 0:
        button = st.button("Apply NMS")
        print(st.session_state.canvas_nms)
        if button:
            bboxes = []

            for rect in st.session_state.canvas_nms:
                box = [0, rect["class"], float(rect["probability"]), rect["label_color"], rect["left"], rect["top"], rect["width"]+rect["left"], rect["height"]+rect["top"]]
                bboxes.append(box)
            print(bboxes)            
            box = nms(bboxes, prob_threshold=prob_threshold, iou_threshold=iou_threshold)
    
    return box

    st.markdown("# Average Recall")

def upload_image(name):

    image = st.file_uploader(name, type=["png", "jpg"])

    if image is not None:
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)

    return image

def create_canvas(ID: str, image: list, label_color: str, drawing_mode: str, point_display_radius: int=4, stroke_width: int=2):
        
    canvas = st_canvas(fill_color=label_color, width=660, height=440, stroke_width=stroke_width, stroke_color="#eee", background_image=Image.open(image).resize((660, 420)), drawing_mode=drawing_mode, key=ID, point_display_radius=point_display_radius if drawing_mode == "point" else 0)

    return canvas

def hex_to_rgb(hex):
  rgb = []
  for i in (0, 2, 4):
    decimal = int(hex[i:i+2], 16)
    rgb.append(decimal)
  
  return tuple(rgb)
