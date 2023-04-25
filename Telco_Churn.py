import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from pycaret.datasets import get_data
from pycaret.classification import *
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 999)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


##################################
# GÖREV 1: KEŞİFCİ VERİ ANALİZİ
##################################

##################################
# GENEL RESİM
##################################

df = pd.read_csv("datasets/Telco-Customer-Churn.csv")
df.head()


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df, head=2)
df.describe().T


##################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


# "TotalCharges" sütunundaki boş değerleri NaN olarak değiştir
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# "TotalCharges" sütununun veri tipini "float64" olarak değiştir
df["TotalCharges"] = df["TotalCharges"].astype("float64")


cat_cols, num_cols, cat_but_car = grab_col_names(df)


##################################
# KATEGORİK DEĞİŞKENLERİN ANALİZİ
##################################


def cat_summary_l(dataframe, cat_cols, plot=False):
    for col_name in cat_cols:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")
        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show()


cat_summary_l(df, cat_cols)


##################################
# NUMERİK DEĞİŞKENLERİN ANALİZİ
##################################

def num_summary(dataframe, numerical_col, plot=False):
    for numerical_col in num_cols:
        quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
        print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

num_summary(df,num_cols)


##################################
# DEĞİŞKENLERİN TARGET'A GÖRE ANALİZİ
##################################

# Numerik
def target_summary_with_num(dataframe, target, col):
    print(dataframe.groupby(target).agg({col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Churn", col)



##################################
# GÖREV 2: FEATURE ENGINEERING
##################################

##################################
# EKSİK DEĞER ANALİZİ
##################################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)



# Eksik Değerlerin Doldurulması
for col in na_columns:
    df.loc[df[col].isnull(), col] = df[col].median()

df.isnull().sum()



##################################
# AYKIRI DEĞER ANALİZİ
##################################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit



# Aykırı Değer Analizi ve Baskılama İşlemi

for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))



##################################
# ENCODING
##################################

# Binary encoding

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

df.head()


# One-hot Encoding

cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, cat_cols, drop_first=True)


##################################
# ÖZELLİK ÇIKARIMI
##################################

df["NewTenure*TotalCharges"] = df["tenure"] * df["TotalCharges"]


# # Partner ve dependents indeksini bir arada düşünerek kategorik değişken oluşturma
df.loc[(df["Partner"] == 1) & (df["Dependents"] == 1), "New_Partner_Dependents"] = "BigFamily"
df.loc[(df["Partner"] == 0) & (df["Dependents"] == 1), "New_Partner_Dependents"] = "LittleFamily"
df.loc[(df["Partner"] == 1) & (df["Dependents"] == 0), "New_Partner_Dependents"] = "Lovely"
df.loc[(df["Partner"] == 0) & (df["Dependents"] == 0), "New_Partner_Dependents"] = "Lonely"

# # Tenure değişkenini kategorilere ayırma
df.loc[(df['tenure'] < 6), 'tenureCat'] = 'UpTo6'
df.loc[(df['tenure'] >= 6) & (df['tenure'] < 20), 'tenureCat'] = '6to20'
df.loc[(df['tenure'] >= 20) & (df['tenure'] < 40), 'tenureCat'] = '20to40'
df.loc[(df['tenure'] >= 40), 'tenureCat'] = '40plus'



#### Encoding işlemleri tekrarlanır.


##################################
# STANDARTLAŞTIRMA
##################################

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

##################################
# MODELLEME
##################################

# 1.Yol

from pycaret.datasets import get_data
from pycaret.classification import *
exp1 = setup(df, target = 'Churn')
compare_models()

#                                    Model  Accuracy   AUC  Recall  Prec.    F1  Kappa   MCC  TT (Sec)
#ada                  Ada Boost Classifier     0.807 0.848   0.557  0.663 0.605  0.479 0.483     0.045
#lr                    Logistic Regression     0.805 0.852   0.536  0.668 0.593  0.468 0.473     0.377
#lda          Linear Discriminant Analysis     0.803 0.848   0.585  0.646 0.610  0.480 0.483     0.039
#ridge                    Ridge Classifier     0.803 0.000   0.478  0.684 0.561  0.440 0.452     0.027
#gbc          Gradient Boosting Classifier     0.799 0.850   0.532  0.647 0.583  0.452 0.456     0.048
#rf               Random Forest Classifier     0.795 0.832   0.507  0.647 0.567  0.435 0.442     0.064
#lightgbm  Light Gradient Boosting Machine     0.793 0.836   0.541  0.629 0.581  0.445 0.448     0.116
#svm                   SVM - Linear Kernel     0.790 0.000   0.505  0.648 0.555  0.422 0.435     0.032
#et                 Extra Trees Classifier     0.786 0.818   0.504  0.618 0.554  0.416 0.420     0.077
#knn                K Neighbors Classifier     0.775 0.788   0.554  0.581 0.566  0.414 0.415     0.063
#dummy                    Dummy Classifier     0.735 0.500   0.000  0.000 0.000  0.000 0.000     0.032
#dt               Decision Tree Classifier     0.726 0.655   0.501  0.484 0.491  0.304 0.305     0.038
#qda       Quadratic Discriminant Analysis     0.723 0.775   0.640  0.486 0.545  0.353 0.367     0.035
#nb                            Naive Bayes     0.700 0.824   0.840  0.465 0.598  0.389 0.433     0.035

#AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,  n_estimators=50, random_state=5196)



# 2.Yol
df.set_index('customerID', drop=True, inplace=True)
X = df.drop('Churn',axis=1)
y = df[['Churn']]


