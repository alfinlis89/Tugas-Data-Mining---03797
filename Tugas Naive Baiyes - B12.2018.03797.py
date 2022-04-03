import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix  

# model untuk classifier
cNB = GaussianNB()

df = pd.read_csv("Data.csv")
X = df.drop('target', axis=1)  
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=109)
# latih classifier
cNB = cNB.fit(X_train, y_train)

# prediksi data test
Y_NB = cNB.predict(X_test)
# print akurasi
print("Akurasi Naive Bayes : ", accuracy_score(y_test, Y_NB))

print("Confusion Matrix SVM")
print(confusion_matrix(y_test,Y_NB)) 
print(classification_report(y_test,Y_NB)) 