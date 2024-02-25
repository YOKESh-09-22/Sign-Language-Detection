import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.preprocessing.sequence import pad_sequences

# Load the data from the pickle file
with open('./data.pickle', 'rb') as file:
    data_dict = pickle.load(file)

# Extract data and labels
data = data_dict['data']
labels = data_dict['labels']

# Pad sequences to ensure consistent length
data_padded = pad_sequences(data, padding='post', dtype='float32')

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data_padded, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions on the test set
y_predict = model.predict(x_test)

# Calculate and print the accuracy score
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model to a pickle file
with open('model.p', 'wb') as file:
    pickle.dump({'model': model}, file)
