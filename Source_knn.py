import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix  
import matplotlib.pyplot as plt

# model untuk classifier
knn = KNeighborsClassifier (n_neighbors=5)

df = pd.read_csv("Data.csv")
X = df.drop('target', axis=1)  
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# latih classifier
knn = knn.fit(X_train, y_train)

# prediksi data test
Y_knn = knn.predict(X_test)
# print akurasi
print("Akurasi KNN : ", accuracy_score(y_test, Y_knn))

print("Confusion Matrix KNN")
print(confusion_matrix(y_test,Y_knn)) 
print(classification_report(y_test,Y_knn)) 


liat=df.groupby('sex')['target'].sum()
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
nama = ['Laki-laki', 'Perempuan']
ax.bar(nama,liat.values)
ax.set_xlabel('jenis kelamin')
ax.set_ylabel('jumlah')
ax.set_title('Perbandingan Jenis kelamin yang terkena penyakit jantung')
plt.show()