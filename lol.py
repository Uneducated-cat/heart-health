import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



import seaborn as sns
df = pd.read_csv("heart.csv")


from sklearn.preprocessing import LabelEncoder

label_encoder_sex = LabelEncoder()
label_encoder_resting_ecg = LabelEncoder()
label_encoder_exercise_angina = LabelEncoder()
label_encoder_chest_pain_type = LabelEncoder()
label_encoder_st_slope = LabelEncoder()



# Label Encoding
df['Sex_encoded'] = label_encoder_sex.fit_transform(df['Sex'])
df['RestingECG_encoded'] = label_encoder_resting_ecg.fit_transform(df['RestingECG'])
df['ExerciseAngina_encoded'] = label_encoder_exercise_angina.fit_transform(df['ExerciseAngina'])
df['ChestPainType_encoded'] = label_encoder_chest_pain_type.fit_transform(df['ChestPainType'])
df['ST_Slope_encoded'] = label_encoder_st_slope.fit_transform(df['ST_Slope'])

numeric_columns = df.select_dtypes(include=['number'])


# Here we are removing Heart HEalth column because it is the output we need so we keep the
#output in another dataset for validation.
X = numeric_columns.drop('HeartDisease', axis=1)
y = df['HeartDisease']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#random_state is nothing but creates a set seed to recreate the same random ness in the /n
#program we can keep any value but the value should be same for the model.


# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
# n_estimator creates 100 decision trees to get the output
# random_state is set same as the training randomnes to create the same order.


# Train the model
model.fit(X_train, y_train)
#this will train the model with x and y training data.


# Make predictions on the test set
predictions = model.predict(X_test)
print(X_test)
print(predictions)
#here we are checking the trained model with testing data.
# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
# accuracy_score creates a score of model comparing the prediction and y_test.
#classification report i will post in whatsapp group.


# Display the evaluation metrics
#print(f'Accuracy: {accuracy:.2f}')
#print('Classification Report:')
#print(report)

import pickle
pickle.dump(model,open('model.sav','wb'))
model = pickle.load(open('model.sav','rb'))
result = model.score(X_test,y_test)
print(result)