# Loan-default
Create a model that predicts whether or not a loan will be default using the historical data.   
```python
import pandas as pd

# Load the dataset
data = pd.read_csv('loan_data.csv')

# Perform one-hot encoding on the "purpose" column
data = pd.get_dummies(data, columns=['purpose'])

from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('loan_data.csv')

# Perform label encoding on the "purpose" column
le = LabelEncoder()
data['purpose'] = le.fit_transform(data['purpose'])

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("loan_data.csv")
plt.hist(df["credit.policy"])
plt.xlabel("Credit Policy")
plt.ylabel("Count")
plt.show()

num_features = ["int.rate", "installment", "log.annual.inc", "dti", "fico", "days.with.cr.line", "revol.bal", "revol.util", "inq.last.6mths", "delinq.2yrs", "pub.rec"]
for feature in num_features:
    plt.hist(df[feature])
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.show()

for feature in num_features:
    plt.boxplot(df[feature])
    plt.xlabel(feature)
    plt.show()

corr_matrix = df[num_features].corr()
print(corr_matrix)

import seaborn as sns

sns.heatmap(corr_matrix, cmap="YlGnBu")
plt.show()

# Load the dataset
loans = pd.read_csv('loan_data.csv')

# Convert categorical variables into numerical variables
cat_columns = ['purpose']
loans = pd.get_dummies(loans, columns=cat_columns, drop_first=True)

# Create a correlation matrix
corr_matrix = loans.corr()

# Set figure size
fig, ax = plt.subplots(figsize=(17, 13))

# Plot the correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Drop highly correlated features
loans.drop(['installment', 'fico'], axis=1, inplace=True)


from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# Load the dataset
loans = pd.read_csv('loan_data.csv')

# Convert categorical variables into numerical variables
cat_columns = ['purpose']
loans = pd.get_dummies(loans, columns=cat_columns, drop_first=True)

# Split the data into train and test sets
X = loans.drop('not.fully.paid', axis=1)
y = loans['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=120000)

# Evaluate the model on the test set
score = model.evaluate(X_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



