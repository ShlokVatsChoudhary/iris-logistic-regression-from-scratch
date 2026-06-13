from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#scikit learn used only for leading iris data, not for training model

iris = load_iris()
X = iris.data
y_ini = iris.target

# X is [x1,x2,x3,x4] 
# x1 = sepal length
# x2 = sepal width
# x3 = petal length
# x4 = petal width

# Y is currently 0/1/2 for Setosa, Versicolor, and Virginica, but we need to convert to 0/1 i.e. not setosa vs setosa for our model here.

y_binary = (y_ini == 0).astype(int)

#print(X.shape) # (150, 4) => 150 examples of X

#splitting the data into test, cv and train


#Separate Train (70%) and Temporary/Remaining (30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y_binary, test_size=0.3, random_state=42, stratify=y_binary) 

#Split the remaining 30% equally (50/50) into CV and Test
X_cv, X_test, y_cv, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# now we are calculating Mean, Standard daviation and scaled X for Z scale normalization