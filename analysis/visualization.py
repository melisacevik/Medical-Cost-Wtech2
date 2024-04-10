import numpy as np
import missingno as msno
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dataset/insurance.csv')

df.head()

sns.pairplot(df[['age', 'bmi', 'children', 'charges']])
plt.show()

# Cinsiyetin sağlık harcama üzerindeki etkisi +
sns.boxplot(x='sex', y='charges', data=df)
plt.show()

# Yaş grafiği +
sns.histplot(df['age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Bölgeye göre toplam sağlık harcamalarının hesaplanması
total_charges_by_region = df.groupby('region')['charges'].sum()

# Pasta grafiği
plt.figure(figsize=(8, 6))
plt.pie(total_charges_by_region, labels=total_charges_by_region.index, autopct='%1.1f%%', startangle=140)
plt.title('Healthcare Charges Distribution by Region')
plt.axis('equal')  # Daireyi daire olarak ayarlar
plt.show()

# Sigara içenler ve içmeyenler arasındaki sağlık harcamaları +

sns.violinplot(x='smoker', y='charges', data=df)
plt.title('Healthcare Charges: Smokers vs Non-Smokers')
plt.xlabel('Smoker')
plt.ylabel('Healthcare Charges')
plt.show()

sns.catplot(x="smoker", kind="count",hue = 'sex', palette="pink", data=df)
plt.show()

#Cinsiyete göre sağlık harcamaları +

# Cinsiyete göre sağlık harcamalarının ortalama değerlerini hesapla +
avg_charges_by_sex = df.groupby('sex')['charges'].mean().reset_index()

# Çubuk grafiği
sns.barplot(x='sex', y='charges', data=avg_charges_by_sex, palette='pastel')
plt.title('Average Healthcare Charges by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Healthcare Charges')
plt.show()

# Çocuk sayısına göre sağlık harcamalarının ortalama değerleri +
avg_charges_by_children = df.groupby('children')['charges'].mean().reset_index()

# Çizgi grafiği +
plt.plot(avg_charges_by_children['children'], avg_charges_by_children['charges'], marker='o', linestyle='-')
plt.title('Average Healthcare Charges by Number of Children')
plt.xlabel('Number of Children')
plt.ylabel('Average Healthcare Charges')
plt.grid(True)
plt.show()