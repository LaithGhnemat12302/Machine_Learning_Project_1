import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

path = "Diabetes.csv"
inputData = pd.read_csv(path)

# ___________________________________________ Replacement ___________________________________________________________

def replaceByMedian(column):
    Q1 = column[column > 0].quantile(0.25)
    Q3 = column[column > 0].quantile(0.75)
    IQR = Q3 - Q1
    lowerLimit = Q1 - (1.5 * IQR)
    upperLimit = Q3 + (1.5 * IQR)
    columnCopy = column.copy()
    columnCopy[(columnCopy == 0) | (columnCopy < lowerLimit) | (columnCopy > upperLimit)] = column.median()
    return columnCopy

replacedColumns = ["PGL", "DIA", "TSF", "INS", "BMI", "DPF", "AGE"]

for co in replacedColumns:
    inputData[co] = replaceByMedian(inputData[co])

# ____________________________________________________ Parts ___________________________________________________

print("Please choose one of these options:")
print("1- Part1")
print("2- Part2")
print("3- Part3")

userInput = input()
partNumber = int(userInput)

# _____________________________________________ Part One ___________________________________________________

if partNumber == 1:
    print("Part1:\n")

    print("Enter one of these choices:")
    print("1- Print the summary statistics of all attributes in the dataset")
    print("2- Show the distribution of the class label 'Diabetic' and indicate any highlights in the distribution")
    print("3- Draw a histogram detailing the amount of diabetics in each subgroup")
    print("4- Show the density plot for the age")
    print("5- Show the density plot for the BMI")
    print("6- Visualise the correlation between all features")
    print("7- Split the data set into training(80%) and test(20%)")
    print("8- Exit Part1")

    partOneNumber = int(input())

    # __________________________________________ First Option ______________________________________________

    if partOneNumber == 1:
        TheSummary = inputData.describe()
        print(TheSummary)

    # __________________________________________ Second Option _____________________________________________

    elif partOneNumber == 2:
        Counts = inputData['Diabetic'].value_counts()

        plt.figure(figsize=(6, 4))
        sns.countplot(x="Diabetic", data=inputData, palette='viridis', hue="Diabetic", legend=False)

        plt.title("Distribution Of Diabetic ")
        plt.xlabel("Diabetic")
        plt.ylabel("Frequency")

        for i, freq in enumerate(Counts):
            plt.text(i, freq + 0.1, str(freq), ha="center")  # Fix the typo here

        plt.show()
    # __________________________________________ Third Option ______________________________________________

    elif partOneNumber == 3:
        plt.figure(figsize=(10, 6))

        sns.histplot(inputData[inputData['Diabetic'] == 1]["AGE"], bins=20, kde=False, color='yellow', label='Diabetic')
        sns.histplot(inputData[inputData['Diabetic'] == 0]["AGE"], bins=20, kde=False, color='blue', label='Non Diabetic')

        plt.title('Diabetics And Non Diabetics')
        plt.xlabel('AGE')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()
    # __________________________________________ Fourth Option _____________________________________________

    elif partOneNumber == 4:
        plt.figure(figsize=(18, 6))

        sns.kdeplot(inputData['AGE'], fill=True, color='red')

        plt.title("Age Density Plot")
        plt.xlabel("Age")
        plt.ylabel("Density")
        plt.show()
    # __________________________________________ Fifth Option ______________________________________________

    elif partOneNumber == 5:
        plt.figure(figsize=(11, 6))

        sns.kdeplot(inputData["BMI"], fill=True, color="blue")

        plt.title("BMI Density Plot")
        plt.xlabel("BMI")
        plt.ylabel("Density")
        plt.show()
    # __________________________________________ Sixth Option ______________________________________________

    elif partOneNumber == 6:
        TheCorrelation = inputData.corr()

        plt.figure(figsize=(13, 7))

        sns.heatmap(TheCorrelation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title("Correlation Between all Features")
        plt.show()
    # __________________________________________ Seventh Option ____________________________________________

    elif partOneNumber == 7:
        X = inputData.iloc[:, :-1]
        y = inputData.iloc[:, -1]

        np.random.seed(42)

        indices = np.random.permutation(len(inputData))
        split = int(0.8 * len(inputData))
        X_train, X_test = X.iloc[indices[:split]], X.iloc[indices[split:]]
        y_train, y_test = y.iloc[indices[:split]], y.iloc[indices[split:]]

        print(X_test)
        print(X_train)

# _____________________________________________ Part Two ___________________________________________________

elif partNumber == 2:
    print("Enter one of these choices:")
    print("1- Apply Linear Regression to learn the attribute 'Age' using all independent attributes")
    print("2- Apply Linear Regression using the most important feature")
    print("3- Apply Linear Regression using the set of 3-most important features")
    print("4- Exit Part2")

    partTwoNumber = int(input())

    # __________________________________________ First Option __________________________________________

    if partTwoNumber == 1:
        X = inputData.drop('AGE', axis=1)
        y = inputData['AGE']

        np.random.seed(42)
        indices = np.random.permutation(len(inputData))
        split = int(0.8 * len(inputData))

        X_train, X_test = X.iloc[indices[:split]], X.iloc[indices[split:]]
        y_train, y_test = y.iloc[indices[:split]], y.iloc[indices[split:]]

        leniar_Model = LinearRegression()
        leniar_Model.fit(X_train, y_train)

        Y_Predict = leniar_Model.predict(X_test)

        MeanSE = mean_squared_error(y_test, Y_Predict)
        print(f'Mean Squared Error = ', MeanSE)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=Y_Predict)

        plt.xlabel('Actual Age')
        plt.ylabel('Predicted Age')
        plt.title('Actual vs Predicted Age')
        plt.show()
    # __________________________________________ Second Option _________________________________________

    elif partTwoNumber == 2:
        Thecorrelation = inputData.corr()
        Age_Correlations = Thecorrelation['AGE'].abs().sort_values(ascending=False)
        important_feature = Age_Correlations.index[1]

        # print(important_feature)
        X = inputData[[important_feature]]
        y = inputData['AGE']

        np.random.seed(42)
        indices = np.random.permutation(len(inputData))
        split = int(0.8 * len(inputData))

        X_train, X_test = X.iloc[indices[:split]], X.iloc[indices[split:]]
        y_train, y_test = y.iloc[indices[:split]], y.iloc[indices[split:]]

        leniar_Model = LinearRegression()
        leniar_Model.fit(X_train, y_train)

        Y_Predict = leniar_Model.predict(X_test)

        MeanSE = mean_squared_error(y_test, Y_Predict)
        print(f'Mean Squared Error = ', MeanSE)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=Y_Predict)

        plt.xlabel('Actual Age')
        plt.ylabel('Predicted Age')
        plt.title('Actual vs Predicted Age')
        plt.show()
    # __________________________________________ Third Option __________________________________________

    elif partTwoNumber == 3:
        Thecorrelation = inputData.corr()
        mostImportantFeatre = Thecorrelation['AGE'].abs().sort_values(ascending=False).index[1:4]

        print(inputData[mostImportantFeatre].describe())

        X = inputData[mostImportantFeatre]
        y = inputData['AGE']

        np.random.seed(42)
        indices = np.random.permutation(len(inputData))
        split = int(0.8 * len(inputData))

        X_train, X_test = X.iloc[indices[:split]], X.iloc[indices[split:]]
        y_train, y_test = y.iloc[indices[:split]], y.iloc[indices[split:]]

        leniar_Model = LinearRegression()
        leniar_Model.fit(X_train, y_train)

        Y_Predict = leniar_Model.predict(X_test)

        MeanSE = mean_squared_error(y_test, Y_Predict)
        print(f'Mean Squared Error = ', MeanSE)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=Y_Predict)

        plt.xlabel('Actual Age')
        plt.ylabel('Predicted Age')
        plt.title('Actual vs Predicted Age')
        plt.show()

# _____________________________________________ Part Three _________________________________________________

elif partNumber == 3:
    print("Enter one of these choices:")
    print("1- K-Nearest-Neighbours using the testing set")
    print("2- K-Nearest-Neighbours with different values of K")
    print("3- Exit Part3")

    partThreeNumber = int(input())

    # __________________________________________ First Option __________________________________________

    if partThreeNumber == 1:
        X = inputData.drop('Diabetic', axis=1)
        y = inputData['Diabetic']

        np.random.seed(42)
        indices = np.random.permutation(len(inputData))
        split = int(0.8 * len(inputData))

        X_train, X_test = X.iloc[indices[:split]], X.iloc[indices[split:]]
        y_train, y_test = y.iloc[indices[:split]], y.iloc[indices[split:]]

        knn_model = KNeighborsClassifier(n_neighbors=9)
        knn_model.fit(X_train, y_train)

        Y_Prediction = knn_model.predict(X_test)

        accuracy = accuracy_score(y_test, Y_Prediction)
        accuracy *= 100
        accuracy = int(accuracy)
        print(f"The accuracy = {accuracy}%")
    # __________________________________________ Second Option _________________________________________

    elif partThreeNumber == 2:
        X = inputData.drop('Diabetic', axis=1)
        y = inputData['Diabetic']

        np.random.seed(42)
        indices = np.random.permutation(len(inputData))
        split = int(0.8 * len(inputData))

        X_train, X_test = X.iloc[indices[:split]], X.iloc[indices[split:]]
        y_train, y_test = y.iloc[indices[:split]], y.iloc[indices[split:]]

        k_values = [1, 4, 5, 8]
        k_models = {}

        for i in k_values:
            model = KNeighborsClassifier(n_neighbors=i)
            model.fit(X_train, y_train)
            k_models[i] = model

        for k, model in k_models.items():
            Y_Prediction = model.predict(X_test)

            accuracy = accuracy_score(y_test, Y_Prediction)
            accuracy *= 100
            auc = roc_auc_score(y_test, Y_Prediction)
            confusion_matrix_result = confusion_matrix(y_test, Y_Prediction)
            print(f"K : {k} Accuracy : {accuracy} AUC : {auc} and Confusion Matrix : {confusion_matrix_result}")