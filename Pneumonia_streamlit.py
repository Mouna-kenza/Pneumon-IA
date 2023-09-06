from PIL import Image
import streamlit as st  # data web app development
import time
import numpy as np  # np mean, np random
import cv2
from keras.models import load_model
import tensorflow as tf
import joblib


def resize (img, scale):
    dim = (scale, scale)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def get_image(path:str)->Image:
    image = Image.open(path)
    return image

def model(option,loaded_image):
    loaded_image = np.array(loaded_image) 
    loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)
    test_image=Image.fromarray(loaded_image)
    if option == 'CNN': 
        
        loaded_model = load_model("./cnn.h5")        
        if test_image is not None:
            test_image = tf.keras.preprocessing.image.img_to_array(test_image)
            test_image = tf.image.resize(test_image, size=(200,200))
            test_image = np.expand_dims(test_image, axis = 0)
            result = loaded_model.predict(test_image)        
                
    elif option == "Decision Tree": 

        loaded_model = joblib.load("./Tree_model.joblib") 
        if test_image is not None:
            test_image = np.array(test_image)
            test_image =  cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            test_image = cv2.resize(test_image, (200,200), interpolation = cv2.INTER_AREA)
            test_image = test_image.flatten()
            test_image = test_image.reshape(1,40000)
            result = loaded_model.predict_proba(test_image)

    elif option == "KNN (doesnt work with cloud version)":

        loaded_model = joblib.load("./KNN_model.joblib") 
        
        if test_image is not None:
            test_image = np.array(test_image)
            test_image =  cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            test_image = cv2.resize(test_image, (200,200), interpolation = cv2.INTER_AREA)
            filters.threshold_mean( test_image)
            out = test_image > filters.threshold_mean( test_image)
            st.image(out, width=200)
            out = out.reshape(1,40000)
            result = loaded_model.predict_proba(out)
    elif option == "LDA":
        loaded_model = joblib.load("./LDA_model.joblib") 

        if test_image is not None:            
            test_image = np.array(test_image)
            test_image =  cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            test_image = cv2.resize(test_image, (200,200), interpolation = cv2.INTER_AREA)
            test_image = test_image.flatten()
            test_image = test_image.reshape(1,40000)
            result = loaded_model.predict_proba(test_image)
    else:  
        loaded_model = joblib.load("./Bayes_model.joblib") 

        if test_image is not None:
            test_image = np.array(test_image)
            test_image =  cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            test_image = cv2.resize(test_image, (200,200), interpolation = cv2.INTER_AREA)
            test_image = test_image.flatten()
            test_image = test_image.reshape(1,40000)
            result = loaded_model.predict_proba(test_image)
            
    if np.argmax(result) == 0:
        prediction_ = 'Non Pneumonia'
    else:
        prediction_ = 'Pneumonia'

    return prediction_ , result

image = get_image("./PneumonIA.png") # path of the file
image1 = get_image("./Frame_5.png") # path of the file
st.set_page_config(layout="centered", page_icon= image1, page_title="Pneumon-IA")


st.header("Pneumon-IA")
st.sidebar.image(image, use_column_width=True)

tab1, tab2, tab3, tab4,tab5 = st.tabs(["Pneumonia Detection","Understanding Pneumonia","Mechanism of Detection", "Model Accuracy","About the App"])
with tab1:
    
    option = st.selectbox(
        'Choose a model : ',
        ('CNN', 'Decision Tree', 'Bayes', 'LDA', 'KNN (doesnt work with cloud version)'))
    loaded_image = st.file_uploader("X-ray : ", type="jpeg")
    if loaded_image is not None:
        st.success('Your picture is conform!', icon="✅")
        col1, col2 = st.columns(2)
        file_details = {"filename":loaded_image.name, "filetype":loaded_image.type,
                                "filesize":loaded_image.size}
        col1.write(file_details)
        loaded_image = Image.open(loaded_image)
        col2.image(loaded_image, width= 200)
        
        my_bar = st.progress(0)
        for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1)


        prediction_, result= model(option, loaded_image)
        col1, col2, col3 = st.columns(3)

        if prediction_ == 'Non Pneumonia':
            col1.write("Results : **<span style='color:green;'>Non Pneumonia</span>**", unsafe_allow_html=True)
        else:
            col1.write("Results : **<span style='color:red;'>Pneumonia</span>**", unsafe_allow_html=True)
        var_result =round(np.max(result)*100, 2)
        col2.write("Probability : "+ str(var_result) +"%")
        if var_result < 70:
            col3.write("Confidence level : **<span style='color:red;'>Low</span>**", unsafe_allow_html=True)
        elif var_result > 70 and var_result < 90:
            col3.write("Confidence level : **<span style='color:orange;'>Medium</span>**", unsafe_allow_html=True)
        else:
            col3.write("Confidence level : **<span style='color:green;'>High</span>**", unsafe_allow_html=True)

        st.warning( "⚠️ Important Notice : This application is intended solely for professional use. If you are a patient seeking medical assistance, please ensure you consult a qualified doctor for proper diagnosis and guidance.")
        st.markdown('''
        <a href="https://www.doctolib.fr/">
            <h5>➡️ Consult a Medical Professional ⬅️</h5>
        </a>''',
        unsafe_allow_html=True)
with tab2:


    # Understanding Pneumonia
    expander = st.expander("Understanding Pneumonia")
    expander.write("Pneumonia is a serious respiratory infection that affects the lungs. It can be caused by various factors, including bacteria, viruses, fungi, or other microorganisms. Pneumonia leads to inflammation in the lung's air sacs, causing them to fill with fluid or pus. This can result in difficulty breathing, coughing, fever, and other symptoms.")

    # Common Symptoms of Pneumonia
    expander = st.expander("Common Symptoms of Pneumonia")
    expander.write("Common symptoms of pneumonia include:")
    expander.write("- High fever with chills")
    expander.write("- Persistent cough, often with phlegm")
    expander.write("- Shortness of breath or rapid breathing")
    expander.write("- Chest pain, especially when breathing deeply or coughing")
    expander.write("- Fatigue and weakness")
    expander.write("- Confusion or changes in mental awareness (especially in older adults)")
    expander.write("- Bluish lips or nail beds due to lack of oxygen")

    # Risk Factors
    expander = st.expander("Risk Factors")
    expander.write("Certain factors can increase the risk of developing pneumonia:")
    expander.write("- Age: Very young children and the elderly are more vulnerable.")
    expander.write("- Weakened Immune System: Conditions like HIV/AIDS, cancer, or organ transplantation can weaken the immune system's ability to fight infections.")
    expander.write("- Smoking: Smoking damages the lungs and impairs the body's natural defense mechanisms.")
    expander.write("- Chronic Lung Diseases: Conditions like asthma, COPD, or bronchiectasis make the lungs more susceptible to infections.")
    expander.write("- Recent Respiratory Infections: Having had a cold, flu, or other respiratory infection can leave the body more susceptible to pneumonia.")

    # Detection and Diagnosis
    expander = st.expander("Detection and Diagnosis")
    expander.write("Pneumonia can be diagnosed through medical evaluation, physical examination, and imaging tests. The most common imaging method used is a chest X-ray. On an X-ray image, pneumonia appears as dense white spots or infiltrates in the lung areas, indicating infection or inflammation.")

    # Types of Pneumonia
    expander = st.expander("Types of Pneumonia")
    expander.write("Different types of pneumonia exist:")
    expander.write("- Community-Acquired Pneumonia: Acquired outside of healthcare settings.")
    expander.write("- Hospital-Acquired (Nosocomial) Pneumonia: Develops during hospitalization.")
    expander.write("- Aspiration Pneumonia: Caused by inhaling foreign substances.")
    expander.write("- Atypical Pneumonia: Caused by atypical pathogens like Mycoplasma or Legionella.")

    # Treatment and Prevention
    expander = st.expander("Treatment and Prevention")
    expander.write("Treatment for pneumonia depends on the cause, severity, and overall health of the patient. Common approaches include antibiotics for bacterial pneumonia, antiviral medications for viral pneumonia, and supportive care. To prevent pneumonia:")
    expander.write("- Get vaccinated: Flu and pneumococcal vaccines can reduce the risk.")
    expander.write("- Practice good hygiene: Regular handwashing and covering your mouth when coughing or sneezing.")
    expander.write("- Avoid smoking: Smoking damages the lungs' defense mechanisms.")
    expander.write("- Stay away from sick individuals: Reduce exposure to contagious respiratory infections.")
        
    image_pneu = get_image("./normal_chest-x-ray-annotated_watermark-scaled.jpg")
    st.image(image_pneu, use_column_width=True)

with tab3:

    # Create a table
    algorithm_names = ["CNN", "Decision Tree", "Bayes", "LDA", "KNN"]
    hyperparameters = ["Resized: 64x64, 2 Convolutional Layers, 2 Dense Layers, 2 Outputs", 
                    "Gini Criterion, Resized: 200x200px", "Resized: 200x200px", 
                    "Resized: 200x200px", "Resized: 200x200px, 5 Nearest Neighbors"]
    model_info = [
        "A Convolutional Neural Network (CNN) that resizes the image to 64x64 before flattening. It has 2 convolutional layers and 2 dense layers. The model provides 2 outputs for class prediction.",
        "Decision Tree model working with the Gini criterion. It operates without any additional filters and resizes images to 200x200px.",
        "Bayesian model that operates without additional filters and resizes images to 200x200px.",
        "Linear Discriminant Analysis (LDA) model operating without filters, resizing images to 200x200px.",
        "K-Nearest Neighbors (KNN) model with a mean filter, working with 5 nearest neighbors. Images are resized to 200x200px."
    ]

    # Display the table
    table_data = {
        "Algorithm": algorithm_names,
        "Hyperparameters": hyperparameters,
        "Model Information": model_info
    }
    st.table(table_data)


with tab4:
    image2 = get_image("Acc_res.png")
    image3 = get_image("Acc_filter.png")
    image4 = get_image("AccMod.png")
    st.image(image4)
    st.image(image2)
    st.image(image3)
    
with tab5 : 
    st.info("Pneumon-IA is an advanced Streamlit application designed to analyze chest X-ray images of your lungs and determine the presence of pneumonia. This powerful tool leverages a variety of cutting-edge algorithms for accurate and efficient detection.", icon="ℹ️")        
    expander = st.expander("Owner of the application:")
    expander.write("Mouna Kholassi")


   