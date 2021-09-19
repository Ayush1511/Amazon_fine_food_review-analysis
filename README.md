# Review Analysis
## Description
This webpage illustrates the application of natural language processing.This application will classify the review entered into different classes like-positive,negative and neutral.
The web application was developed using flask framework and deployed using heroku and the link for the application is https://fine-food-review.herokuapp.com/

## How to use the application
<ul>
<li>The user has to input a review in the text box provided</li>
<li>Then press the predict button</li>
<li>After which the machine learning model at the backend will classifiy the review into positive,negative or neutral review.</li>
</ul>

## Methodology
<ul>
<li>The dataset used for training the model is https://www.kaggle.com/snap/amazon-fine-food-reviews.</li>
<li>Reviews divided into positive ,negative and neutral based on the score in datset.Score=0-2(Negative),Score=3(Neutral),Score=4-5(Positive)
<li>Tokenization,Stopwards Removal,Lemmatization ,Count Vectorization used for data preprocessing.</li>
<li>Random Forest Classifier model is used for classifying the review.</li>
</ul>

### Flowchart of Methodology
![](Flowchart.JPG)

## I/O screenshots
### Input
![](/Screenshots/i2.JPG)
### Output
![](/Screenshots/o1.JPG)
### Input
![](/Screenshots/i1.JPG)
### Output
![](/Screenshots/o2.JPG)

## Authors
Ayush Goel

## License
[MIT](https://choosealicense.com/licenses/mit/)
