import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the dataset
df = pd.read_csv("Titanic-Dataset.csv")

#Basic Info
print("Data Preview:")
print(df.head())

print("\nData Info:")
df.info()

print("\n Missing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())

#Histograms
print("\n Plotting Histograms...")
df.hist(bins=30, figsize=(15, 10))
plt.suptitle("Histograms of Numerical Features", fontsize=16)
plt.tight_layout()
plt.show()

#STEP 3: Boxplots
print("\n Boxplot: Age vs Survived")
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Age Distribution by Survival Status')
plt.show()

print("\n Boxplot: Fare vs Survived")
sns.boxplot(x='Survived', y='Fare', data=df)
plt.title('Fare Distribution by Survival Status')
plt.show()

#Correlation Matrix
print("\n Correlation Heatmap")
corr_matrix = df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

#Pairplot
print("\n Pairplot of Selected Features")
sns.pairplot(df[['Survived', 'Age', 'Fare', 'Pclass']], hue='Survived')
plt.show()

# Barplots for Categorical Data
print("\n Survival Rate by Gender")
sns.barplot(x='Sex', y='Survived', data=df)
plt.title("Survival Rate by Gender")
plt.show()

print("\n Survival Rate by Passenger Class")
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title("Survival Rate by Class")
plt.show()

#Interactive Plot with Plotly 
print("\n Interactive Plot: Age Distribution by Survival")
fig = px.histogram(df, x="Age", color="Survived", nbins=30, title="Age Distribution by Survival")
fig.show()
