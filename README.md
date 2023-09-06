# PneumoDetect

What is Pneumon-IA
Pneumon-IA is an application that will detect from an Xray radio if there is a pneumonia.

How to use Pneumon-IA
To use our app, you have to click on the url. Then you can choose your model to try several options. CNN is the better one so far.
You can drag and drop an image on the website and the app will gives you score of prediction.

What are in Pneumon-IA
We use several models and filters that you can reuse.
Filters tested : Yen, Mean, Otsu
Models tested : KNN, Tree, LDA, Bayes, CNN

I resized all images in 200x200px format. We found that is the best ratio time/accuracy.
After several tests I choose to apply a mean filter on our KNN model and no filter on the others model (exept CNN). I found that the mean filter was the best one for KNN.
For the CNN model, the resizing was 64x64px.
The most efficient model is the CNN model.
