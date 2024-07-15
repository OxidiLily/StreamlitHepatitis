import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from sklearn.model_selection import GridSearchCV
import pickle


st.title('Data mining Klasifikasi Hepatitis')
st.subheader("Masukkan Dataset Hepatitis")

uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
        df= pd.read_csv(uploaded_file, na_values="?")
        st.markdown('**Dataset**')
        hepatitis_data = df

        st.write('')
        st.write('1. membaca data set hepatitis')
        hepatitis_data.shape
        st.write(hepatitis_data.head())
        st.write('')
        st.write('2. Memeriksa statistik ringkasan dasar dari data')
        st.write(hepatitis_data.describe())
        st.write('')
        st.write('3.Check for value counts in target variabel')
        st.write(hepatitis_data.target.value_counts())
        st.write('')
        st.write('4.Periksa tipe data setiap variabel')
        st.write(hepatitis_data.dtypes)
        cat_cols = hepatitis_data.columns[hepatitis_data.nunique() < 5]
        num_cols = hepatitis_data.columns[hepatitis_data.nunique() >= 5]
        st.write('')
        st.write('5. hapus kolom yang tidak signifikan')
        hepatitis_data.drop(["ID"], axis = 1, inplace=True)
        num_cols = hepatitis_data.columns[hepatitis_data.nunique() >= 5]
        st.write(hepatitis_data.head())

        st.write('6. Identifikasi Kolom Kategorikal dan simpan dalam variabel cat_cols dan numerik ke dalam num_cols')
        num_cols = ["age", "bili", "alk", "sgot", "albu", "protime"]
        cat_cols = ['gender', 'steroid', 'antivirals', 'fatigue', 'malaise', 'anorexia', 'liverBig', 
                    'liverFirm', 'spleen', 'spiders', 'ascites', 'varices', 'histology']
        st.write(num_cols)
        st.write(cat_cols)
        st.write('')
        st.write('7. memeriksa nilai nol ')
        st.write(hepatitis_data.isnull().sum())
        st.write('')
        st.write('8. Membagi data menjadi x dan y')
        x = hepatitis_data.drop(["target"], axis = 1)
        y = hepatitis_data["target"]
        st.write(x.shape, y.shape)
        st.write('')
        st.write('9. Pisahkan data menjadi X_train, X_test, y_train, y_test dengan test_size = 0.20 menggunakan sklearn')
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 123)
        st.write(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
        st.write('')
        st.write('10. periksa nilai null pada train dan test, periksa value_count pada y_train dan y_test')
        st.write(y_train.value_counts()/X_train.shape[0])
        st.write('nilai null data train')
        st.write(X_train.isna().sum())
        st.write('nilai null data test')
        st.write(X_test.isna().sum())
        st.write('')
        st.write('11. Menghitung Kolom Kategorikal dengan modus dan kolom Numerik dengan rata-rata')
        df_cat_train = X_train[cat_cols]
        df_cat_test = X_test[cat_cols]
        cat_imputer = SimpleImputer(strategy='most_frequent')
        st.write(cat_imputer.fit(df_cat_train))
        df_cat_train = pd.DataFrame(cat_imputer.transform(df_cat_train), columns=cat_cols)
        df_cat_test = pd.DataFrame(cat_imputer.transform(df_cat_test), columns=cat_cols)
        df_num_train = X_train[num_cols]
        df_num_test = X_test[num_cols]
        num_imputer = SimpleImputer(strategy='median')
        st.write(num_imputer.fit(df_num_train[num_cols]))
        df_num_train = pd.DataFrame ( num_imputer.transform(df_num_train), columns= num_cols)
        df_num_test =  pd.DataFrame(num_imputer.transform(df_num_test), columns=num_cols)
        # Combine numeric and categorical in train
        X_train = pd.concat([df_num_train, df_cat_train], axis = 1)

        # Combine numeric and categorical in test
        X_test = pd.concat([df_num_test, df_cat_test], axis = 1)
        st.write('X train')
        st.write(X_train.isna().sum())
        st.write('X test')
        st.write(X_test.isna().sum())
        st.write('Mengonversi semua kolom kategorikal ke Format Bilangan Bulat sebelum dummifikasi (2,0 sebagai 2, dst.)')
        # Train
        X_train[cat_cols] = X_train[cat_cols].astype('int')
        # Test
        X_test[cat_cols] = X_test[cat_cols].astype('int')
        st.write('')
        st.write('12. mengecilkan kolom katergorikal')
        ## Convert Categorical Columns to Dummies
        # Train
        X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
        # Test
        X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
        st.write('X train')
        st.write(X_train.columns)
        st.write('X test')
        st.write(X_test.columns)
        st.write('')
        st.write('13.Scale the numeric attributes ["age", "bili", "alk", "sgot", "albu", "protime"]')
        #num_cols = ["age", "bili", "alk", "sgot", "albu", "protime"]
        scaler = StandardScaler()
        scaler.fit(X_train.loc[:,num_cols])
        # scale on train
        X_train.loc[:,num_cols] = scaler.transform(X_train.loc[:,num_cols])
        #X_train[num_cols] = scaler.transform(X_train[num_cols])
        # scale on test
        X_test.loc[:,num_cols] = scaler.transform(X_test.loc[:,num_cols])

        st.write('METODE KLASIFIKASI SVM')
        # Create a SVC classifier using a linear kernel
        linear_svm = SVC(kernel='linear', C=1, random_state=0)
        # Train the classifier
        st.write(linear_svm.fit(X=X_train, y= y_train))

        ## Predict
        train_predictions = linear_svm.predict(X_train)
        test_predictions = linear_svm.predict(X_test)
        ### Train data accuracy
        st.write("TRAIN Conf Matrix : \n", confusion_matrix(y_train, train_predictions))
        st.write("\nTRAIN DATA ACCURACY",accuracy_score(y_train,train_predictions))
        st.write("\nTrain data f1-score for class '1'",f1_score(y_train,train_predictions,pos_label=1))
        st.write("\nTrain data f1-score for class '2'",f1_score(y_train,train_predictions,pos_label=2))
        ### Test data accuracy
        st.write("\n\n--------------------------------------\n\n")
        st.write("TEST Conf Matrix : \n", confusion_matrix(y_test, test_predictions))
        st.write("\nTEST DATA ACCURACY",accuracy_score(y_test,test_predictions))
        st.write("\nTest data f1-score for class '1'",f1_score(y_test,test_predictions,pos_label=1))
        st.write("\nTest data f1-score for class '2'",f1_score(y_test,test_predictions,pos_label=2))
        st.write('Create an SVC object and see the arguments') 
        svc = SVC(kernel='rbf', random_state=0, gamma=0.01, C=1)
        st.write(svc)
        st.write('Train the model')
        st.write(svc.fit(X=X_train, y= y_train))


        ## Predict
        train_predictions = svc.predict(X_train)
        test_predictions = svc.predict(X_test)
        ### Train data accuracy
        st.write("TRAIN Conf Matrix : \n", confusion_matrix(y_train, train_predictions))
        st.write("\nTRAIN DATA ACCURACY",accuracy_score(y_train,train_predictions))
        st.write("\nTrain data f1-score for class '1'",f1_score(y_train,train_predictions,pos_label=1))
        st.write("\nTrain data f1-score for class '2'",f1_score(y_train,train_predictions,pos_label=2))
        ### Test data accuracy
        st.write("\n\n--------------------------------------\n\n")
        st.write("TEST Conf Matrix : \n", confusion_matrix(y_test, test_predictions))
        st.write("\nTEST DATA ACCURACY",accuracy_score(y_test,test_predictions))
        st.write("\nTest data f1-score for class '1'",f1_score(y_test,test_predictions,pos_label=1))
        st.write("\nTest data f1-score for class '2'",f1_score(y_test,test_predictions,pos_label=2))
        st.write('')
        st.write('14. SVM dengan Pencarian Grid untuk Penyetelan Paramater')
        svc_grid = SVC()
        param_grid = { 
                        'C': [0.001, 0.01, 0.1, 1, 10, 100 ],
                        'gamma': [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100], 
                        'kernel':['linear', 'rbf', 'poly' ]
                    }

        svc_cv_grid = GridSearchCV(estimator = svc_grid, param_grid = param_grid, cv = 5, verbose=3)
        st.write('Fit the grid search model')
        st.write(svc_cv_grid.fit(X=X_train, y=y_train))
        st.write('Get the best parameters')
        st.write(svc_cv_grid.best_params_)
        svc_best = svc_cv_grid.best_estimator_
        ## Predict
        train_predictions = svc_best.predict(X_train)
        test_predictions = svc_best.predict(X_test)

        st.write("TRAIN DATA ACCURACY",accuracy_score(y_train,train_predictions))
        st.write("\nTrain data f1-score for class '1'",f1_score(y_train,train_predictions,pos_label=1))
        st.write("\nTrain data f1-score for class '2'",f1_score(y_train,train_predictions,pos_label=2))

        ### Test data accuracy
        st.write("\n\n--------------------------------------\n\n")
        st.write("TEST DATA ACCURACY",accuracy_score(y_test,test_predictions))
        st.write("\nTest data f1-score for class '1'",f1_score(y_test,test_predictions,pos_label=1))
        st.write("\nTest data f1-score for class '2'",f1_score(y_test,test_predictions,pos_label=2))

        st.write('')
        st.write('15. Simpan model')
        filename = 'model_hepatitis.sav'
        pickle.dump(linear_svm, open(filename, 'wb'))
        with open("model_hepatitis.sav", "rb") as file:
            btn = st.download_button(
                    label="Download Model Hepatitis",
                    data=file,
                    file_name=filename,
                    mime="sav"
                )
