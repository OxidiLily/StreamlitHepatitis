import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
import warnings
import streamlit as st
warnings.filterwarnings("ignore")


st.title('Data mining Klasifikasi Hepatitis')
# Load data
hepatitis_data = pd.read_csv("hepatitis.csv", na_values="?")
hepatitis_data = hepatitis_data.dropna(subset=['target'])

# Check target distribution
# print("Target distribution:\n", hepatitis_data['target'].value_counts(normalize=True))

# Define numeric and categorical columns
num_cols = ["age", "bili", "alk", "sgot", "albu", "protime"]
cat_cols = ['gender', 'steroid', 'antivirals', 'fatigue', 'malaise', 'anorexia', 'liverBig', 
            'liverFirm', 'spleen', 'spiders', 'ascites', 'varices', 'histology']

# Split the data into features and target
X = hepatitis_data.drop(["target"], axis=1)
y = hepatitis_data["target"]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

# Impute missing values in categorical columns
df_cat_train = X_train[cat_cols]
df_cat_test = X_test[cat_cols]

cat_imputer = SimpleImputer(strategy='most_frequent')
df_cat_train = pd.DataFrame(cat_imputer.fit_transform(df_cat_train), columns=cat_cols)
df_cat_test = pd.DataFrame(cat_imputer.transform(df_cat_test), columns=cat_cols)

# Impute missing values in numeric columns
df_num_train = X_train[num_cols]
df_num_test = X_test[num_cols]

num_imputer = SimpleImputer(strategy='median')
df_num_train = pd.DataFrame(num_imputer.fit_transform(df_num_train), columns=num_cols)
df_num_test = pd.DataFrame(num_imputer.transform(df_num_test), columns=num_cols)

# Combine numeric and categorical columns in train and test sets
X_train = pd.concat([df_num_train, df_cat_train], axis=1)
X_test = pd.concat([df_num_test, df_cat_test], axis=1)

# Convert categorical columns to integers
X_train[cat_cols] = X_train[cat_cols].astype('int')
X_test[cat_cols] = X_test[cat_cols].astype('int')

# Convert categorical columns to dummies
X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

# Ensure the same columns in both train and test data
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Scale numeric features
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Train the SVC model
svc = SVC(kernel='rbf', random_state=0, gamma=0.01, C=1, class_weight='balanced')
svc.fit(X_train, y_train)

# Define the prediction function
def predict_hepatitis(age, gender, steroid, antivirals, fatigue, malaise, anorexia, liverBig, liverFirm, spleen, spiders, ascites, varices, histology, bili, alk, sgot, albu, protime):
    # Creating a DataFrame for the new input
    new_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'steroid': [steroid],
        'antivirals': [antivirals],
        'fatigue': [fatigue],
        'malaise': [malaise],
        'anorexia': [anorexia],
        'liverBig': [liverBig],
        'liverFirm': [liverFirm],
        'spleen': [spleen],
        'spiders': [spiders],
        'ascites': [ascites],
        'varices': [varices],
        'histology': [histology],
        'bili': [bili],
        'alk': [alk],
        'sgot': [sgot],
        'albu': [albu],
        'protime': [protime]
    })

    # Preprocess the new data
    new_data[cat_cols] = new_data[cat_cols].astype('int')
    new_data = pd.get_dummies(new_data, columns=cat_cols, drop_first=True)

    # Align new data columns with training data columns
    new_data = new_data.reindex(columns=X_train.columns, fill_value=0)

    # Scale the numeric features
    new_data[num_cols] = scaler.transform(new_data[num_cols])

    # Print the new_data for debugging purposes
    # print("Processed new data:\n", new_data)

    # Make the prediction
    prediction = svc.predict(new_data)
    
    # Print prediction result for debugging purposes
    # print("Prediction result:", prediction)

    # Interpret the prediction
    result = "Anda terdiagnosa hepatitis" if prediction == 1 else "Tidak terdiagnosa hepatitis"
    return result

# Example usage:
# Using extreme values that are more likely to indicate a diagnosis of hepatitis
# result = predict_hepatitis(age=60, gender=1, steroid=1, antivirals=2, fatigue=1, malaise=1, anorexia=1, liverBig=1, liverFirm=1, spleen=1, spiders=1, ascites=1, varices=1, histology=1, bili=4.0, alk=250, sgot=500, albu=2.1, protime=80)
# print(result)

# # Example usage:
# # Using values that are more likely to indicate no diagnosis of hepatitis
# result = predict_hepatitis(age=40, gender=2, steroid=2, antivirals=2, fatigue=2, malaise=2, anorexia=2, liverBig=2, liverFirm=2, spleen=2, spiders=2, ascites=2, varices=2, histology=2, bili=0.39, alk=33, sgot=13, albu=6.0, protime=10)
# print(result)

age = st.number_input('Masukkan umur',min_value=1,max_value=100)

col1, col2 = st.columns(2)
with col1:
    gender = st.radio(
        "Gender",
        ["Laki-Laki","Perempuan"])
    if gender == "Laki-Laki":
            gender=1
    else:
            gender=2

    antivirals = st.radio(
        "antivirals",
        ["Yes","No"])
    if antivirals == "Laki-Laki":
            antivirals=1
    else:
            antivirals=2

    malaise = st.radio(
        "malaise",
        ["Yes","No"])
    if malaise == "Laki-Laki":
            malaise=1
    else:
            malaise=2

    liverBig = st.radio(
        "liverBig",
        ["Yes","No"])
    if liverBig == "Laki-Laki":
            liverBig=1
    else:
            liverBig=2

    spiders = st.radio(
        "spiders",
        ["Yes","No"])
    if spiders == "Yes":
        spiders=1
    else:
        spiders=2

    varices = st.radio(
        "varices",
        ["Yes","No"])
    if varices == "Yes":
        varices=1
    else:
        varices=2
    
    histology = st.radio(
        "histology",
        ["Yes","No"])
    if histology == "Yes":
        histology=1
    else:
        histology=2
with col2:
    steroid = st.radio(
        "Steroid",
        ["Yes","No"])
    if steroid == "Yes":
        steroid=1
    else:
        steroid=2

    fatigue = st.radio(
        "fatigue",
        ["Yes","No"])
    if fatigue == "Yes":
        fatigue=1
    else:
        fatigue=2

    anorexia = st.radio(
        "anorexia",
        ["Yes","No"])
    if anorexia == "Yes":
        anorexia=1
    else:
        anorexia=2
    
    liverFirm = st.radio(
        "liverFirm",
        ["Yes","No"])
    if liverFirm == "Yes":
        liverFirm=1
    else:
        liverFirm=2

    spleen = st.radio(
        "spleen",
        ["Yes","No"])
    if spleen == "Yes":
        spleen=1
    else:
        spleen=2

    ascites = st.radio(
        "ascites",
        ["Yes","No"])
    if ascites == "Yes":
        ascites=1
    else:
        ascites=2

bili = st.slider(
    "Select a range of bili",
    0.0, 100.0, (5.0))


alk = st.slider(
    "Select a range of alk",
    0.0, 1000.0, (30.0))


sgot = st.slider(
    "Select a range of sgot",
    0.0, 1000.0, (225.0))
 

albu = st.slider(
    "Select a range of albu",
    0.0, 10.0, (5.0))
 

protime = st.slider(
    "Select a range of protime",
    0, 1000, (30))
 
if st.button("Cek Hasil"):
    result = predict_hepatitis(age, gender, steroid, antivirals, fatigue, malaise, anorexia, liverBig, liverFirm, spleen, spiders, ascites, varices, histology, bili, alk, sgot, albu, protime)
    st.write(result)
