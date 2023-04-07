#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
# Read the provided CSV file ‘data.csv’
data = pd.read_csv('data.csv')


# In[2]:


print(data.describe(), '\n')


# In[3]:


# Check if the data has null values.
print("Null values in the data: \n", data.isnull().sum(), '\n')


# In[4]:


# a. Replace the null values with the mean
data.fillna(data.mean(), inplace=True)
print("Null values in the data after replacing with mean: \n", data.isnull().sum(), '\n')
print(data, '\n')


# In[5]:


# Select at least two columns and aggregate the data using: min, max, count, mean
# selecting two columns pulse and calories
print("Aggregating the data using min, max, count, mean: \n", data[['Pulse', 'Calories']].agg(['min', 'max', 'count', 'mean']), '\n')


# In[6]:


# Filter the dataframe to select the rows with calories values between 500 and 1000
print("Filtering the dataframe to select the rows with calories values between 500 and 1000: \n", data[(data['Calories'] > 500) & (data['Calories'] < 1000)], '\n')


# In[7]:


# Filter the dataframe to select the rows with calories values > 500 and pulse < 100
print("Filtering the dataframe to select the rows with calories values > 500 and pulse < 100: \n", data[(data['Calories'] > 500) & (data['Pulse'] < 100)], '\n')


# In[8]:


# Create a new “df_modified” dataframe that contains all the columns from df except for “Maxpulse”
df_modified = data.drop('Maxpulse', axis=1)
print("New dataframe after dropping Maxpulse column: \n", df_modified, '\n')


# In[9]:


# Delete the “Maxpulse” column from the main df dataframe
data.drop('Maxpulse', axis=1, inplace=True)
print("Dataframe after dropping Maxpulse column: \n", data, '\n')


# In[10]:


# Convert the datatype of Calories column to int datatype
data['Calories'] = data['Calories'].astype(int)
print("Data types of all columns after converting Calories to int: \n", data.dtypes, '\n')


# In[12]:


from matplotlib import pyplot as plt
# Using pandas create a scatter plot for the two columns (Duration and Calories)
data.plot.scatter(x='Duration', y='Calories', title='Scatter plot for Duration and Calories')
plt.show()


# In[ ]:





# In[ ]:





# In[13]:


import pandas as pd
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt

df=pd.read_csv("train.csv")
df.head()


# In[14]:


le = preprocessing.LabelEncoder()
df['Sex'] = le.fit_transform(df.Sex.values)
df['Survived'].corr(df['Sex'])


# In[15]:


df = df.drop(['Name', 'Sex','Ticket','Cabin','Embarked'], axis=1)


# In[16]:


matrix = df.corr()
print(matrix)


# In[17]:


df.corr().style.background_gradient(cmap="Greens")


# In[18]:


sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plt.show()


# In[19]:


import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv("train.csv")

# Select features and target
features = ['Age', 'Embarked', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp']
target = 'Survived'

# Preprocess categorical variables
df['Sex'] = df['Sex'].replace(["female", "male"], [0, 1])
df['Embarked'] = df['Embarked'].replace(['S', 'C', 'Q'], [1, 2, 3])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Train the Naive Bayes model
model = GaussianNB()
model.fit(X_train_imputed, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_imputed)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))


# In[20]:


glass=pd.read_csv("glass.csv")
glass.head()


# In[21]:


glass.corr().style.background_gradient(cmap="Greens")


# In[22]:


sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plt.show()


# In[23]:


#Naïve Bayes method of Glass Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

# Load the dataset
glass_data = pd.read_csv('glass.csv')

# Separate the target variable
X = glass_data.drop(['Type'], axis=1)
y = glass_data['Type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
score = model.score(X_test, y_test)
report = classification_report(y_test, y_pred)

print("Accuracy Score: {:.2f}%".format(score * 100))
print("\nClassification Report:\n", report)


# In[24]:


#Linear SVM method of Glass Dataset
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
#To avoid warnings
warnings.filterwarnings("ignore")

# Load the dataset
glass_data = pd.read_csv('glass.csv')

# Separate the target variable
X = glass_data.drop(['Type'], axis=1)
y = glass_data['Type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear SVM model
model = LinearSVC(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
score = model.score(X_test, y_test)
report = classification_report(y_test, y_pred)

print("Accuracy Score: {:.2f}%".format(score * 100))
print("\nClassification Report:\n", report)


# In[ ]:




