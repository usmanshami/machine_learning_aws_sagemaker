# Trying out AWS SageMaker Studio for a simple machine learning task

## Overview
Let’s look at how to accomplish a simple machine learning task on AWS SageMaker

We’ll take a movie ratings dataset comprising of user ratings for different movies and the movie metadata. Based on these existing user ratings of different movies, we’ll try to predict what the user’s rating would be for a movie that they haven’t rated yet.  

The following two documents are the primary references used in creating this doc - so feel free to refer to them in case there are any issues.

1. Build, Train, and Deploy a Machine Learning Model (https://aws.amazon.com/getting-started/hands-on/build-train-deploy-machine-learning-model-sagemaker/)
2. Machine Learning Project – Data Science Movie Recommendation System Project in R (https://data-flair.training/blogs/data-science-r-movie-recommendation/)

We’d be using the MovieLens data from GroupLens Research.

3. MovieLens | GroupLens (https://grouplens.org/datasets/movielens/)

## Steps
1. Log into the AWS console and select Amazon SageMaker from the services to be redirected to the SageMaker Dashboard.
2. Select Amazon SageMaker Studio from the navigation bar on the left and select quick start to start a new instance of Amazon SageMaker Studio. Consider leaving the default name as is, select “Create a new role” in execution role and specify the S3 buckets you’d be using (Leaving these defaults should be okay as well) and click “Create Role”. Once the execution role has been created, click on “Submit” - this will create a new Amazon SageMaker instance.
3. Once the Amazon SageMaker Studio instance is created, click on Open Studio link to launch the Amazon SageMaker Studio IDE.
4. Create a new Jupyter notebook using the Data Science as the Kernel and the latest python (Python 3) notebook.
5. Import the python libraries we’d be using in this task. After writing the following code in the Jupyter notebook cell, look for a play button in the controls bar on top - click it should run the currently active cell and execute its code. Here is what each imported library does:    
  * boto3 is the python library which is used for making AWS requests, 
  * sagemaker is the sagemaker library 
  * urllib.request is the library to make url requests such as HTTP GET etc to download csv files stored on S3 and elsewhere
  * numpy is a scientific computing python library 
  * pandas is a python data analysis library. 
```python
import boto3, sagemaker, urllib.request
from sagemaker import get_execution_role
import numpy as np                                
import pandas as pd                              
from sagemaker.predictor import csv_serializer 
```
6. Once the imports have been completed, lets add some standard code to create execution role, define region settings and initialize boto for the region and xgboost
```python
# Define IAM role
role = get_execution_role()
prefix = ‘sagemaker/movielens’
containers = {'us-west-2’: '433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest’,
             'us-east-1’: '811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest’,
             'us-east-2’: '825641698319.dkr.ecr.us-east-2.amazonaws.com/xgboost:latest’,
             'eu-west-1’: '685385470294.dkr.ecr.eu-west-1.amazonaws.com/xgboost:latest’} # each region has its XGBoost container
my_region = boto3.session.Session().region_name # set the region of the instance
print(“Success - the MySageMakerInstance is in the ” + my_region + “ region. You will use the ” + containers[my_region] + “ container for your SageMaker endpoint.”)
```
7. Create an S3 bucket that will contain our dataset files as well as training, test data, the computed machine learning models and the results. 
```python
bucket_name = ’<BUCKET_NAME_HERE>’
s3 = boto3.resource('s3’)
try:
   if  my_region == 'us-east-1’:
     s3.create_bucket(Bucket=bucket_name)
   else:
     s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={ 'LocationConstraint’: my_region })
   print('S3 bucket created successfully’)
except Exception as e:
   print('S3 error: ’,e)
```

8. Once the bucket has been created, upload the movie lens data files to the bucket by using the S3 console UI. You’ll need to download the zip from https://grouplens.org/datasets/movielens/, extract it on the local machine and then upload the extracted files on S3. We used the Small dataset which has 100,000 ratings applied to 9,000 movies by 600 users. Be sure to read the MovieLens README file to make sure you understand the conditions on the data usage.  

9. After the files have been uploaded, add the following code to the notebook to download the csv files and convert them to pandas data format
```python
try:
 urllib.request.urlretrieve (“https://{}.s3.{}.amazonaws.com/ratings.csv”.format(bucket_name, my_region), “ratings.csv”)
 print('Success: downloaded ratings.csv.’)
except Exception as e:
 print('Data load error: ’,e)
try:
 urllib.request.urlretrieve (“https://{}.s3.{}.amazonaws.com/movies.csv”.format(bucket_name, my_region), “movies.csv”)
 print('Success: downloaded ratings.csv.’)
except Exception as e:
 print('Data load error: ’,e)

try:
 model_data = pd.read_csv(’./ratings.csv’)
 print('Success: Data loaded into dataframe.’)
except Exception as e:
   print('Data load error: ’,e)
try:
 movie_data = pd.read_csv(’./movies.csv’)
 print('Success: Data loaded into dataframe.’)
except Exception as e:
   print('Data load error: ’,e)
```

10. Now we create training and test datasets from the ratings data by splitting 70-30% the data randomly
```python
train_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data))])
print(train_data.shape, test_data.shape)
print(train_data.info, test_data.info)
```

11. We will need to normalize the training and test datasets to include boolean genre membership columns for each genre. To do this, we first process the movie dataset to create a movie to list of its genres map. Once the map has been created, we iterate the ratings data and update each row with the boolean genre membership columns. The following code creates the movie genre maps
```python
movie_id_genre_map=dict()
for movie_row in movie_data.itertuples():
   # print (movie_row)
   genres=movie_row[3].split(’|’)
   # print(movie_row[1])
   if movie_row[1] in movie_id_genre_map:
       raise
   movie_id_genre_map[movie_row[1]] = genres
print(len(movie_id_genre_map))
```
12. We now normalize the training data as mentioned above. Note that the rating column which is the column we are trying to predict in this model (and would be classifying into different rating classes) needs to be the first column in the training dataset. Also note that we multiplied the rating with 2 to convert it from 0.5-5 range of 0.5 rating increments to the range of 1-10. This is necessary since the new integer rating value becomes the rating class for the model classification.  
```python
normalized_train_data = list()
for tuple in train_data.itertuples():
   userId = tuple[1]
   movieId = tuple[2]
   rating = tuple[3]/0.5
   timestamp = tuple[4]
   curr_row_normalized=dict()
   curr_row_normalized['rating’] = rating
   curr_row_normalized['userId’] = userId
   curr_row_normalized['movieId’] = movieId
   curr_row_normalized['timestamp’] = timestamp
   curr_genres = {'Action’: 0,
    'Adventure’: 0,
    'Animation’: 0,
    'Children’: 0,
    'Comedy’: 0,
    'Crime’: 0,
    'Documentary’: 0,
    'Drama’: 0,
    'Fantasy’: 0,
    'Film-Noir’: 0,
    'Horror’: 0,
    'Musical’: 0,
    'Mystery’: 0,
    'Romance’: 0,
    'Sci-Fi’: 0,
    'Thriller’: 0,
    'War’: 0,
    'Western’: 0
   }
   curr_movie = movie_id_genre_map[movieId]
   for genre in curr_movie:
       curr_genres[genre]=1

   curr_row_normalized.update(curr_genres)
   normalized_train_data.append(curr_row_normalized)
   #print(curr_row_normalized)
   #print(normalized_train_data)

print(len(normalized_train_data))
normalized_train_data_pd = pd.DataFrame(data=normalized_train_data)
print(normalized_train_data_pd.columns)
```
13. We do the same for test data, with a small difference - we create two different normalized data arrays - one with the user’s rating and one without the user’s rating. The one without the user’s rating would be used in ML predictions - and these predictions would then be compared with the array with user’s ratings to determine the accuracy of the predictions.
```python
normalized_test_data = list()
normalized_test_data_array = list()
for tuple in test_data.itertuples():
   userId = tuple[1]
   movieId = tuple[2]
   rating = tuple[3]/0.5
   timestamp = tuple[4]
   curr_array_row_normalized=dict()
   curr_row_normalized=dict()
   curr_row_normalized['rating’] = rating
   curr_row_normalized['userId’] = userId
   curr_array_row_normalized['userId’] = userId
   curr_row_normalized['movieId’] = movieId
   curr_array_row_normalized['movieId’] = movieId
   curr_row_normalized['timestamp’] = timestamp
   curr_array_row_normalized['timestamp’] = timestamp
   curr_genres = {'Action’: 0,
    'Adventure’: 0,
    'Animation’: 0,
    'Children’: 0,
    'Comedy’: 0,
    'Crime’: 0,
    'Documentary’: 0,
    'Drama’: 0,
    'Fantasy’: 0,
    'Film-Noir’: 0,
    'Horror’: 0,
    'Musical’: 0,
    'Mystery’: 0,
    'Romance’: 0,
    'Sci-Fi’: 0,
    'Thriller’: 0,
    'War’: 0,
    'Western’: 0
   }
   curr_movie = movie_id_genre_map[movieId]
   for genre in curr_movie:
       curr_genres[genre]=1

   curr_row_normalized.update(curr_genres)
   curr_array_row_normalized.update(curr_genres)
   normalized_test_data.append(curr_row_normalized)
   normalized_test_data_array.append(curr_array_row_normalized)
   #print(curr_row_normalized)
   #print(normalized_test_data)
print(len(normalized_test_data))
print(len(normalized_test_data_array))
normalized_test_data_pd = pd.DataFrame(data=normalized_test_data)
print(normalized_test_data_pd.columns)
normalized_test_data_array_pd = pd.DataFrame(data=normalized_test_data_array)
print(normalized_test_data_array_pd.columns)
```

14. Now lets run the predictions
```python
xgb_predictor.content_type = 'text/csv’
xgb_predictor.serializer = csv_serializer
predictions = xgb_predictor.predict(normalized_test_data_array_pd.values).decode('utf-8’)
predictions_array = np.fromstring(predictions[1:], sep=’,’) # and turn the prediction into an array
print(predictions_array)
```
15. Now create a frequency histogram of the rating classes for the predictions array - pandas crosstab utility does the trick here
```python
cm = pd.crosstab(index=normalized_test_data_pd['rating’], columns=np.round(predictions_array), rownames=['Observed’], colnames=['Predicted’])
```
16. Now compute the accuracy of the prediction - we define the following classes - zero distance i.e. the prediction was accurate, one distance i.e. the predicted score differed from the actual by 1, two distance i.e. the predicted score differed from the actual by 2 and remaining i.e. the predicted score differed from actual by > 2
```python
zero_distance = 0
one_distance = 0
two_distance = 0
remaining = 0
total = 0
for tuple in cm.itertuples():
   total += tuple[0]
   total += tuple[1]
   total += tuple[2]
   total += tuple[3]
   total += tuple[4]
   total += tuple[5]
   total += tuple[6]
   total += tuple[7]
   total += tuple[8]
   total += tuple[9]
   total += tuple[10]

   actual = tuple[0]
   if actual == 1.0:
       zero_distance += tuple[1]
       one_distance += tuple[2]
       two_distance += tuple[3]
       remaining += tuple[4]
       remaining += tuple[5]
       remaining += tuple[6]
       remaining += tuple[7]
       remaining += tuple[8]
       remaining += tuple[9]
       remaining += tuple[10]

   if actual == 2.0:
       zero_distance += tuple[2]
       one_distance += tuple[1]
       one_distance += tuple[3]
       two_distance += tuple[4]
       remaining += tuple[5]
       remaining += tuple[6]
       remaining += tuple[7]
       remaining += tuple[8]
       remaining += tuple[9]
       remaining += tuple[10]

   if actual == 3.0:
       zero_distance += tuple[3]
       one_distance += tuple[2]
       one_distance += tuple[4]
       two_distance += tuple[1]
       two_distance += tuple[5]
       remaining += tuple[6]
       remaining += tuple[7]
       remaining += tuple[8]
       remaining += tuple[9]
       remaining += tuple[10]

   if actual == 4.0:
       zero_distance += tuple[4]
       one_distance += tuple[3]
       one_distance += tuple[5]
       two_distance += tuple[2]
       two_distance += tuple[6]
       remaining += tuple[1]
       remaining += tuple[7]
       remaining += tuple[8]
       remaining += tuple[9]
       remaining += tuple[10]

   if actual == 5.0:
       zero_distance += tuple[5]
       one_distance += tuple[4]
       one_distance += tuple[6]
       two_distance += tuple[3]
       two_distance += tuple[7]
       remaining += tuple[1]
       remaining += tuple[2]
       remaining += tuple[8]
       remaining += tuple[9]
       remaining += tuple[10]

   if actual == 6.0:
       zero_distance += tuple[6]
       one_distance += tuple[5]
       one_distance += tuple[7]
       two_distance += tuple[4]
       two_distance += tuple[8]
       remaining += tuple[1]
       remaining += tuple[2]
       remaining += tuple[3]
       remaining += tuple[9]
       remaining += tuple[10]

   if actual == 7.0:
       zero_distance += tuple[7]
       one_distance += tuple[6]
       one_distance += tuple[8]
       two_distance += tuple[5]
       two_distance += tuple[9]
       remaining += tuple[1]
       remaining += tuple[2]
       remaining += tuple[3]
       remaining += tuple[4]
       remaining += tuple[10]

   if actual == 8.0:
       zero_distance += tuple[8]
       one_distance += tuple[7]
       one_distance += tuple[9]
       two_distance += tuple[6]
       two_distance += tuple[10]
       remaining += tuple[1]
       remaining += tuple[2]
       remaining += tuple[3]
       remaining += tuple[4]
       remaining += tuple[5]

   if actual == 9.0:
       zero_distance += tuple[9]
       one_distance += tuple[8]
       one_distance += tuple[10]
       two_distance += tuple[7]
       remaining += tuple[1]
       remaining += tuple[2]
       remaining += tuple[3]
       remaining += tuple[4]
       remaining += tuple[5]
       remaining += tuple[6]

   if actual == 10.0:
       zero_distance += tuple[10]
       one_distance += tuple[9]
       two_distance += tuple[8]
       remaining += tuple[1]
       remaining += tuple[2]
       remaining += tuple[3]
       remaining += tuple[4]
       remaining += tuple[5]
       remaining += tuple[6]
       remaining += tuple[7]

zero_distance_percent = 100*(zero_distance/total)
one_distance_percent = 100*(one_distance/total)
two_distance_percent = 100*(two_distance/total)
remaining_percent = 100*(remaining/total)
print(“zero distance percentage: ”+str(zero_distance_percent))
print(“one distance percentage: ”+str(one_distance_percent))
print(“two distance percentage: ”+str(two_distance_percent))
print(“remaining distance percentage: ”+str(remaining_percent))
```
17. The results show that our model was 100% accurate for 11.4% predictions, had a deviation of 1 for 47.64% predictions, deviation of 2 for 13.51% predictions and deviation of > 2 for the remaining 27.24%

**zero distance percentage:** 11.403445800430726

**one distance percentage:** 47.641780330222545

**two distance percentage:** 13.513998564249821

**remaining distance percentage:** 27.243359655419958

18. Now terminate the sagemaker instance and free the allocated resources.
```python
sagemaker.Session().delete_endpoint(xgb_predictor.endpoint)
bucket_to_delete = boto3.resource('s3’).Bucket(bucket_name)
bucket_to_delete.objects.all().delete()
```
