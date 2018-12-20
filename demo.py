from sklearn import tree
from sklearn import gaussian_process
from sklearn import svm
from sklearn import discriminant_analysis
from sklearn.metrics import accuracy_score

data = []
test_data = [

    ]
prediction_true = []

# Inputs
data.append(float(input("Height(cm): ")))
data.append(float(input("Weight(kg): ")))
data.append(int(input("Shoe-size: ")))
#prediction_true = [input("Gender('male' or 'female'): ")]

test_data.append(data)

# Classifiers:
clf1 = tree.DecisionTreeClassifier()
clf2 = gaussian_process.GaussianProcessClassifier()
clf3 = svm.SVC(gamma='auto')
clf4 = discriminant_analysis.QuadraticDiscriminantAnalysis()

# [height, weight, shoe-size]
X = [
    [190, 70, 43], [165.1, 65, 44], [167.64, 62, 42], [181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [
        166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]
]

Y = ['male', 'male', 'male', 'male', 'male', 'female', 'female',
     'male', 'male', 'female', 'female', 'female', 'male', 'male']

# Training classifiers
clf1 = clf1.fit(X, Y)
clf2 = clf2.fit(X, Y)
clf3 = clf3.fit(X, Y)
clf4 = clf4.fit(X, Y)

prediction1 = clf1.predict(test_data)
prediction2 = clf2.predict(test_data)
prediction3 = clf3.predict(test_data)
prediction4 = clf4.predict(test_data)

# Calculating accuracy

#clf1_score = accuracy_score(prediction_true, prediction1)
#clf2_score = accuracy_score(prediction_true, prediction2)
#clf3_score = accuracy_score(prediction_true, prediction3)
#clf4_score = accuracy_score(prediction_true, prediction4)

#print('')
#print('Decision Tree Classifier: ' + str(prediction1) + '       Accuracy Score: ' + str(clf1_score))
#print('')
#print('Gaussian Process Classifier: ' + str(prediction1) + '        Accuracy Score: ' + str(clf1_score))
#print('')
#print('SVC: ' + str(prediction1) + '        Accuracy Score: ' + str(clf1_score))
#print('')
#print('Quadratic Discriminant Analysis: ' + str(prediction1) + '        Accuracy Score: ' + str(clf1_score))
#print('')

print('')
print('The predictions are:')
print('')
print('Decision Tree Classifier: ' + str(prediction1))
print('')
print('Gaussian Process Classifier: ' + str(prediction1))
print('')
print('SVC: ' + str(prediction1))
print('')
print('Quadratic Discriminant Analysis: ' + str(prediction1))
print('')

#if input("Save data? (Y/N): ")=='Y':
#    X.append(test_data)
#    Y.append(prediction_true)

input("Press enter to exit")
