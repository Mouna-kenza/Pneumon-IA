# PneumoDetect

What is pneumodetect
Penumodetect is an application that will detect from an Xray radio if there is a pneumonia.

How to setup
You can access to the app by this link https://mushurisen-pneumodetect-pneumonia-streamlit-0xxay4.streamlitapp.com/
But the KNN model is too heavy to be deployed on streamlit so you can't use it on cloud.
If you want to use it locally, you can clone the repo (be sure to be added to the repo github).
Before you start, you need to install the following dependencies:
- Python
- numpy
- tensorflow
- streamlit
- PIL
- cv2
- sklearn
- time
Then download the KNN file and put it in the folder of the project.
Link for KNN : https://efrei365net-my.sharepoint.com/:f:/g/personal/liora_chemla_efrei_net/EkkbBzlKQq1OtAx3uf5QCPIBNvjV5qznJBHcu9IcT1_qiw?e=LTczty
Then you can run the app by typing the following command in your terminal:
Streamlit run Pneumonia_streamlit.py

How to use Pneumodetect
To use our app, you have to click on the url. Then you can choose your model to try several options. CNN is the better one so far.
You can drag and drop an image on the website and the app will gives you score of prediction.

What are in Pneumodetect
We use several models and filters that you can reuse.
Filters tested : Yen, Mean, Otsu
Models tested : KNN, Tree, LDA, Bayes, CNN

We resized all images in 200x200px format. We found that is the best ratio time/accuracy.
After several tests we choose to apply a mean filter on our KNN model and no filter on the others model (exept CNN). We found that the mean filter was the best one for KNN.
For the CNN model, the resizing was 64x64px.
The most efficient model is the CNN model.
