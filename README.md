### Apa itu Clustering dan Algoritma K-Prototypes
Clustering adalah proses pembagian objek-objek ke dalam beberapa kelompok (cluster) berdasarkan tingkat kemiripan antara satu objek dengan yang lain.

Terdapat beberapa algoritma untuk melakukan clustering ini. Salah satu yang populer adalah k-means.

K-means itu sendiri biasa nya hanya digunakan untuk data-data yang bersifat numerik. Sedangkan untuk yang bersifat kategorikal saja, kita bisa menggunakan k-modes.

### Mencari Jumlah Cluster yang Optimal
Buatlah elbow plot dengan jumlah cluster 2 sampai 9 dan tentukan jumlah cluster yang optimal.

source :
from kmodes.kmodes import KModes  
from kmodes.kprototypes import KPrototypes  
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  
df_model = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/df-customer-segmentation.csv')

# Melakukan Iterasi untuk Mendapatkan nilai Cost  
cost = {}  
for k in range(2,10):  
    kproto = KPrototypes(n_clusters = k,random_state=75)  
    kproto.fit_predict(df_model, categorical=[0,1,2])  
    cost[k]= kproto.cost_  
  
# Memvisualisasikan Elbow Plot  
sns.pointplot(x=list(cost.keys()), y=list(cost.values()))  
plt.show()  

out :
download.png


### Membuat Model
Selanjutnya kamu dapat melakukan pembuatan model dengan jumlah kluster yang sudah di dapat pada tahap sebelumnya yaitu 5 dan menyimpan hasilnya sebagai pickle file.

Tugas:
Buatlah model Kprototypes dengan nilai k = 5 dan random state 75. Kemudian simpan hasilnya dalam bentuk pickle.

sou :
import pickle  
from kmodes.kmodes import KModes  
from kmodes.kprototypes import KPrototypes    
kproto = KPrototypes(n_clusters=5, random_state = 75)  
kproto = kproto.fit(df_model, categorical=[0,1,2])  
#Save Model  
pickle.dump(kproto, open('cluster.pkl', 'wb'))


### Menggunakan Model
Model yang sudah kamu buat dapat di gunakan untuk menentukan setiap pelanggan masuk ke dalam cluster yang mana. Kali ini kamu akan menggunakan model tersebut untuk menentukan segmen pelanggan yang ada di data set.

Tugas:
Tentukan cluster setiap pelanggan yang ada di dataset menggunakan model kproto yang sudah di buat sebelumnya. Kemudian gabungkan hasil prediksi tersebut dengan data awal (df) sehingga kita mendapatkan data pelanggan beserta nama cluster nya.

sou :
import pandas as pd
df = pd.read_csv("https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/customer_segments.txt", sep="\t") 
# Menentukan segmen tiap pelanggan    
clusters =  kproto.predict(df_model, categorical=[0,1,2])    
print('segmen pelanggan: {}\n'.format(clusters))       
# Menggabungkan data awal dan segmen pelanggan    
df_final = df.copy()    
df_final['cluster'] = clusters    
print(df_final.head())

out :
download (1).png

### Menampilkan Cluster Tiap Pelanggan
Setelah kamu menggabungkan data awal pelanggan dan clusternya, kamu perlu untuk menampilkan dan mengobservasi hasilnya untuk membantu kamu dalam memberi nama tiap cluster berdasarkan karakteristiknya.

Tugas:
Tampilkan data pelanggan yang di kelompokkan berdasarkan nomor clusternya. Jika dilakukan dengan benar, makan kamu akan mendapatkan hasil sebagai berikut :

sou :
# Menampilkan data pelanggan berdasarkan cluster nya  
for i in range (0,5):  
    print('\nPelanggan cluster: {}\n'.format(i))  
    print(df_final[df_final['cluster']== i])
    
    
 ou :
 download (2).png
 download (3).png
 download (4).png
 
 
 ### Visualisasi Hasil Clustering - Box Plot
Kamu juga membuat visualiasi hasil clustering untuk dapat memudahkan kamu melakukan penamaan di tiap-tiap cluster.

Tugas:

Buatlah boxlplot untuk memvisualisasikan setiap variabel tiap pelanggan yang dibagi berdasarkan nama clusternya.


sou :
import matplotlib.pyplot as plt
# Data Numerical
kolom_numerik = ['Umur','NilaiBelanjaSetahun']    
for i in kolom_numerik:  
    plt.figure(figsize=(6,4))  
    ax = sns.boxplot(x = 'cluster',y = i, data = df_final)  
    plt.title('\nBox Plot {}\n'.format(i), fontsize=12)  
    plt.show()  

 ou :
 download (5).png
 
 ### Visualisasi Hasil Clustering - Count Plot
Kamu juga membuat visualiasi hasil clustering untuk dapat memudahkan kamu melakukan penamaan di tiap-tiap cluster.

Tugas:

Buatlah countplot untuk memvisualisasikan setiap variabel tiap pelanggan yang dibagi berdasarkan nama clusternya.

sou : 
import matplotlib.pyplot as plt  
# Data Kategorikal  
kolom_categorical = ['Jenis Kelamin','Profesi','Tipe Residen']  
  
for i in kolom_categorical:  
    plt.figure(figsize=(6,4))  
    ax = sns.countplot(data = df_final, x = 'cluster', hue = i )  
    plt.title('\nCount Plot {}\n'.format(i), fontsize=12)  
    ax.legend(loc="upper center")  
    for p in ax.patches:  
        ax.annotate(format(p.get_height(), '.0f'),  
                    (p.get_x() + p.get_width() / 2., p.get_height()),  
                     ha = 'center',  
                     va = 'center',  
                     xytext = (0, 10),  
                     textcoords = 'offset points')  
      
    sns.despine(right=True,top = True, left = True)  
    ax.axes.yaxis.set_visible(False)  
    plt.show()  
    
     ou :
 download (6).png
 
 ### Menamakan Cluster
Dari hasil observasi yang dilakukan kamu dapat memberikan nama segmen dari tiap tiap nomor kluster nya. Yaitu:

Cluster 0: Diamond Young Entrepreneur, isi cluster ini adalah para wiraswasta yang memiliki nilai transaksi rata-rata mendekati 10 juta. Selain itu isi dari cluster ini memiliki umur sekitar 18 - 41 tahun dengan rata-ratanya adalah 29 tahun.
Cluster 1: Diamond Senior Entrepreneur, isi cluster ini adalah para wiraswata yang memiliki nilai transaksi rata-rata mendekati 10 juta. Isi dari cluster ini memiliki umur sekitar 45 - 64 tahun dengan rata-ratanya adalah 55 tahun.
Cluster 2: Silver Students, isi cluster ini adalah para pelajar dan mahasiswa dengan rata-rata umur mereka adalah 16 tahun dan nilai belanja setahun mendekati 3 juta.
Cluster 3: Gold Young Member, isi cluster ini adalah para professional dan ibu rumah tangga yang berusia muda dengan rentang umur sekitar 20 - 40 tahun dan dengan rata-rata 30 tahun dan nilai belanja setahun nya mendekati 6 juta.
Cluster 4: Gold Senior Member, isi cluster ini adalah para professional dan ibu rumah tangga yang berusia tua dengan rentang umur 46 - 63 tahun dan dengan rata-rata 53 tahun dan nilai belanja setahun nya mendekati 6 juta.

sou :
# Mapping nama kolom  
df_final['segmen'] = df_final['cluster'].map({  
    0: 'Diamond Young Member',  
    1: 'Diamond Senior Member',  
    2: 'Silver Member',  
    3: 'Gold Young Member',  
    4: 'Gold Senior Member'  
})  

print(df_final.info())
print(df_final.head())

     ou :
 download (7).png
 
 ### Mempersiapkan Data Baru
Disini kamu membuat contoh data baru untuk di prediksi dengan model yang sudah di buat. Hal ini kamu lakukan dengan membuat satu buah dataframe yang berisi informasi pelanggan.

Tugas:

Buatlah satu dataframe yang berisi informasi pelanggan dan tampilkan hasilnya.


sou :
# Data Baru  
data = [{  
    'Customer_ID': 'CUST-100' ,  
    'Nama Pelanggan': 'Joko' ,  
    'Jenis Kelamin': 'Pria',  
    'Umur': 45,  
    'Profesi': 'Wiraswasta',  
    'Tipe Residen': 'Cluster' ,  
    'NilaiBelanjaSetahun': 8230000  
      
}]  
  
# Membuat Data Frame  
new_df = pd.DataFrame(data)  
  
# Melihat Data  
print(new_df)  

  ou :
 download (8).png
 
 
 ### Membuat Fungsi Data Pemrosesan
Selanjutnya kamu perlu membuat fungsi untuk melakukan pemrosesan data berdasarkan paramater yang sama pada saat kita melakukan permodelan dan kita panggil dengan data baru kita.

Jadi fungsi ini nantinya akan bisa di gunakan untuk:

Melakukan konversi data kategorikal menjadi numerik

Dari proses sebelumnya kita tau representasi tiap kode dan maksudnya yaitu:

Jenis Kelamin

0 : Pria
1 : Wanita
Profesi

0 : Ibu Rumah Tangga
1 : Mahasiswa
2 : Pelajar
3 : Professional
4 : Wiraswasta
Tipe Residen

1 : Sector
0 : Cluster
Selanjutnya kita harus membuat fungsi untuk merubah data kategorikal menjadi numerik berdasarkan referensi tersebut.

Melakukan standardisasi kolom numerikal

Untuk melakukan standardisasi dengan variable yang sama pada saat permodelan kita perlu menggunakan nilai rata-rata dan standard deviasi dari tiap variabel pada saat kita melakukan permodelan. Yaitu:

Umur

Rata - rata: 37.5
Standard Deviasi: 14.7
NilaiBelanjaSetahun

Rata - rata: 7069874.8
Standard Deviasi: 2590619.0
Dari nilai-nilai tersebut kita dapat menghitung nilai standardisasi (z) dengan menggunakan rumus Z = (x - u)/s dengan x adalah tiap nilai, u adalah rata-rata dan s adalah standard deviasi.

Menggabungkan hasil dua proses sebelumnya menjadi satu data frame

Selanjutnya kamu perlu menggabungkan kedua perintah tersebut dan menjadi data frame yang siap untuk dilakukan permodelan.


sou :
def data_preprocess(data):  
    # Konversi Kategorikal data  
    kolom_kategorikal = ['Jenis Kelamin','Profesi','Tipe Residen']        
    df_encode = data[kolom_kategorikal].copy()    
    ## Jenis Kelamin   
    df_encode['Jenis Kelamin'] = df_encode['Jenis Kelamin'].map({  
        'Pria': 0,  
        'Wanita' : 1  
    })      
    ## Profesi  
    df_encode['Profesi'] = df_encode['Profesi'].map({  
        'Ibu Rumah Tangga': 0,  
        'Mahasiswa' : 1,  
        'Pelajar': 2,  
        'Professional': 3,  
        'Wiraswasta': 4  
    })       
    ## Tipe Residen  
    df_encode['Tipe Residen'] = df_encode['Tipe Residen'].map({  
        'Cluster': 0,  
        'Sector' : 1  
    })        
    # Standardisasi Numerical Data  
    kolom_numerik = ['Umur','NilaiBelanjaSetahun']  
    df_std = data[kolom_numerik].copy()        
    ## Standardisasi Kolom Umur  
    df_std['Umur'] = (df_std['Umur'] - 37.5)/14.7       
    ## Standardisasi Kolom Nilai Belanja Setahun  
    df_std['NilaiBelanjaSetahun'] = (df_std['NilaiBelanjaSetahun'] - 7069874.8)/2590619.0        
    # Menggabungkan Kategorikal dan numerikal data  
    df_model = df_encode.merge(df_std, left_index = True,  
                           right_index=True, how = 'left')        
    return df_model    
# Menjalankan fungsi  
new_df_model = data_preprocess(new_df)   
print(new_df_model) 



### Memanggil Model dan Melakukan Prediksi
Setelah kamu memiliki data yang siap di gunakan, saatnya memanggil model yang sudah di simpan sebelumnya dan dilanjutkan dengan melakukan prediksi.

Untuk melakukan hal tersebut, kamu perlu membuat prosesnya menjadi dalam satu fungsi yang bernama modelling dengan menggunakan data baru sebagai input nya.

Tugas:

Buatlah fungsi yang bisa di gunakan untuk memanggil model dan melakukan prediksi serta menyimpan hasilnya nya kedalam satu dataframe. Jika berhasil di panggil, kita akan mendapatkan cluster dari data yang kita masukan.

sou :
def modelling (data):       
    # Memanggil Model  
    kpoto = pickle.load(open('cluster.pkl', 'rb'))        
    # Melakukan Prediksi  
    clusters = kpoto.predict(data,categorical=[0,1,2])        
    return clusters    
# Menjalankan Fungsi  
clusters = modelling(new_df_model)    
print(clusters)  


### Menamakan Segmen
Sama dengan sebelumnya, kamu perlu membuat fungsi untuk melakukan proses ini. Nama cluster yang sudah didapat di tahap sebelumnya perlu di rubah menjadi nama segmen agar lebih mudah diidentifikasi.

Tugas:

Disini kamu harus membuat fungsi yang bernama menamakan_segmen dengan data asli dan clusters sebagai inputnya.

sou :
def menamakan_segmen (data_asli, clusters):       
    # Menggabungkan cluster dan data asli  
    final_df = data_asli.copy()  
    final_df['cluster'] = clusters        
    # Menamakan segmen  
    final_df['segmen'] = final_df['cluster'].map({  
        0: 'Diamond Young Member',  
        1: 'Diamond Senior Member',  
        2: 'Silver Students',  
        3: 'Gold Young Member',  
        4: 'Gold Senior Member'  
    })        
    return final_df    
# Menjalankan Fungsi  
new_final_df = menamakan_segmen(new_df,clusters)    
print(new_final_df)   
    
    
    out: download (9).png
    
 

    
 
