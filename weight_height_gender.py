from sklearn import tree
from sklearn import gaussian_process
from sklearn import svm
from sklearn import discriminant_analysis
from sklearn.metrics import accuracy_score
from mpl_toolkits import mplot3d
import numpy as np 
import matplotlib.pyplot as plt


data = []
test_data = []

# Inputs
data.append(float(input("Height(cm): ")))
data.append(float(input("Weight(kg): ")))

test_data.append(data)

# Classifiers:
clf1 = tree.DecisionTreeClassifier()
clf2 = gaussian_process.GaussianProcessClassifier()
clf3 = svm.SVC(gamma='auto')
clf4 = discriminant_analysis.QuadraticDiscriminantAnalysis()

# [height, weight]
X = [
    [174, 96], [189, 87], [185, 110], [195, 104], [149, 61], [189, 104], [147, 92], [154, 111], [174, 90], [169, 103], [195, 81], [159, 80], [192, 101], [155, 51], [191, 79], [153, 107], [157, 110], [140, 129], [144, 145], [172, 139], [157, 110], [153, 149], [169, 97], [185, 139], [172, 67], [151, 64], [190, 95], [187, 62], [163, 159], [179, 152], [153, 121], [178, 52], [195, 65], [160, 131], [157, 153], [189, 132], [197, 114], [144, 80], [171, 152], [185, 81], [175, 120], [149, 108], [157, 56], [161, 118], [182, 126], [185, 76], [188, 122], [181, 111], [161, 72], [140, 152], [168, 135], [176, 54], [163, 110], [172, 105], [196, 116], [187, 89], [172, 92], [178, 127], [164, 70], [143, 88], [191, 54], [141, 143], [193, 54], [190, 83], [175, 135], [179, 158], [172, 96], [168, 59], [164, 82], [194, 136], [153, 51], [178, 117], [141, 80], [180, 75], [185, 100], [197, 154], [165, 104], [168, 90], [176, 122], [181, 51], [164, 75], [166, 140], [190, 105], [186, 118], [168, 123], [198, 50], [175, 141], [145, 117], [159, 104], [185, 140], [178, 154], [183, 96], [194, 111], [177, 61], [197, 119], [170, 156], [142, 69], [160, 139], [195, 69], [190, 50], [199, 156], [154, 105], [161, 155], [198, 145], [192, 140], [195, 126], [166, 160], [159, 154], [181, 106], [149, 66], [150, 70], [146, 157], [190, 135], [192, 90], [177, 96], [148, 60], [165, 57], [146, 104], [144, 108], [176, 156], [168, 87], [187, 122], [187, 138], [184, 160], [158, 149], [158, 96], [194, 115], [145, 79], [182, 151], [154, 54], [168, 139], [187, 70], [158, 153], [167, 110], [171, 155], [183, 150], [190, 156], [194, 108], [171, 147], [159, 124], [169, 54], [167, 85], [180, 149], [163, 123], [140, 79], [197, 125], [194, 106], [140, 146], [195, 98], [168, 115], [196, 50], [140, 52], [150, 60], [168, 140], [155, 111], [179, 103], [182, 84], [168, 160], [187, 102], [181, 105], [199, 99], [184, 76], [192, 101], [182, 143], [172, 111], [181, 78], [176, 109], [156, 106], [151, 67], [188, 80], [187, 136], [174, 138], [167, 151], [196, 131], [197, 149], [185, 119], [170, 102], [181, 94], [166, 126], [188, 100], [162, 74], [177, 117], [162, 97], [180, 73], [192, 108], [165, 80], [167, 135], [182, 84], [161, 134], [158, 95], [141, 85], [154, 100], [165, 105], [142, 137], [141, 94], [145, 108], [157, 74], [177, 117], [166, 144], [193, 151], [184, 57], [179, 93], [156, 89], [182, 104], [145, 160], [150, 87], [145, 99], [196, 122], [191, 96], [148, 67], [150, 84], [148, 155], [153, 146], [196, 159], [185, 52], [171, 131], [143, 118], [142, 86], [141, 126], [159, 109], [173, 82], [183, 138], [152, 90], [178, 140], [188, 54], [155, 144], [166, 70], [188, 123], [171, 120], [179, 130], [186, 137], [153, 78], [184, 86], [177, 81], [145, 78], [170, 81], [181, 141], [165, 155], [174, 65], [146, 110], [178, 85], [166, 61], [191, 62], [177, 155], [183, 50], [151, 114], [182, 98], [142, 159], [188, 90], [161, 89], [153, 70], [140, 143], [169, 141], [162, 159], [183, 147], [162, 58], [172, 109], [150, 119], [169, 145], [184, 132], [159, 104], [163, 131], [156, 137], [157, 52], [147, 84], [141, 86], [173, 139], [154, 145], [168, 148], [168, 50], [145, 130], [152, 103], [187, 121], [163, 57], [178, 83], [187, 94], [179, 114], [190, 80], [172, 75], [188, 57], [193, 65], [147, 126], [147, 94], [166, 107], [192, 139], [181, 139], [150, 74], [178, 160], [156, 52], [149, 100], [156, 74], [183, 105], [162, 68], [165, 83], [168, 143], [160, 156], [169, 88], [140, 76], [187, 92], [151, 82], [186, 140], [182, 108], [188, 81], [179, 110], [156, 126], [188, 114], [183, 153], [144, 88], [196, 69], [171, 141], [171, 147], [180, 156], [191, 146], [179, 67], [180, 60], [154, 132], [188, 99], [142, 135], [170, 95], [152, 141], [190, 118], [181, 111], [153, 104], [187, 140], [144, 66], [148, 54], [199, 92], [167, 85], [164, 71], [185, 102], [164, 160], [142, 71], [165, 68], [172, 62], [157, 56], [155, 57], [167, 153], [164, 126], [189, 125], [161, 145], [155, 71], [171, 118], [154, 92], [179, 83], [170, 115], [184, 106], [191, 68], [162, 58], [178, 138], [157, 60], [184, 83], [197, 88], [160, 51], [184, 153], [190, 50], [174, 90], [189, 124], [186, 143], [180, 58], [186, 148], [193, 61], [161, 103], [151, 158], [195, 147], [184, 152], [141, 80], [185, 94], [186, 127], [142, 131], [147, 67], [151, 62], [160, 124], [185, 60], [163, 63], [174, 95], [150, 144], [142, 91], [178, 142], [154, 96], [176, 87], [159, 120], [191, 62], [177, 117], [151, 154], [182, 149], [197, 72], [146, 138], [160, 83], [157, 66], [150, 50], [167, 58], [180, 70], [183, 76], [183, 87], [152, 154], [164, 71], [187, 96], [169, 136], [149, 61], [163, 137], [195, 104], [174, 107], [182, 70], [169, 110], [193, 130], [148, 141], [186, 68], [165, 143], [146, 123], [166, 133], [179, 56], [177, 101], [181, 154], [161, 154], [157, 103], [169, 98], [152, 114], [162, 64], [162, 130], [177, 61], [195, 61], [140, 146], [186, 146], [178, 107], [174, 54], [180, 59], [188, 141], [187, 130], [153, 77], [165, 95], [178, 79], [163, 154], [150, 97], [179, 127], [165, 62], [168, 158], [153, 133], [184, 157], [188, 65], [166, 153], [172, 116], [182, 73], [143, 149], [152, 146], [186, 128], [159, 140], [146, 70], [176, 121], [146, 101], [159, 145], [162, 157], [172, 90], [169, 121], [182, 50], [183, 79], [176, 77], [188, 128], [175, 83], [154, 81], [184, 147], [179, 123], [152, 132], [179, 56], [145, 141], [181, 80], [158, 127], [188, 99], [145, 142], [161, 115], [198, 109], [147, 142], [154, 112], [178, 65], [195, 153], [167, 79], [183, 131], [164, 142], [167, 64], [151, 55], [147, 107], [155, 115], [172, 108], [142, 86], [146, 85], [188, 115], [173, 111], [160, 109], [187, 80], [198, 136], [179, 150], [164, 59], [146, 147], [198, 50], [170, 53], [152, 98], [150, 153], [184, 121], [141, 136], [150, 95], [173, 131]
]

Y = [
    'male', 'male', 'female', 'female', 'male', 'male', 'male', 'male', 'male', 'female', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'male', 'male', 'male', 'male', 'female', 'female', 'male', 'female', 'female', 'male', 'male', 'female', 'male', 'male', 'male', 'female', 'female', 'female', 'female', 'female', 'male', 'female', 'female', 'female', 'female', 'male', 'male', 'female', 'male', 'female', 'male', 'male', 'male', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'male', 'female', 'male', 'female', 'female', 'male', 'male', 'male', 'female', 'female', 'female', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'female', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'female', 'female', 'female', 'female', 'female', 'male', 'male', 'female', 'male', 'male', 'male', 'female', 'male', 'male', 'male', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'female', 'female', 'male', 'female', 'female', 'male', 'female', 'female', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'male', 'male', 'female', 'male', 'male', 'female', 'female', 'female', 'female', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'female', 'male', 'male', 'male', 'male', 'male', 'female', 'male', 'female', 'female', 'male', 'female', 'female', 'female', 'female', 'female', 'male', 'female', 'male', 'male', 'female', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'female', 'female', 'female', 'female', 'male', 'female', 'male', 'male', 'male', 'female', 'male', 'female', 'female', 'female', 'male', 'male', 'male', 'male', 'female', 'male', 'male', 'male', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'male', 'female', 'male', 'female', 'male', 'female', 'female', 'male', 'female', 'female', 'female', 'female', 'female', 'female', 'female', 'male', 'female', 'male', 'female', 'male', 'male', 'female', 'male', 'male', 'female', 'male', 'female', 'female', 'female', 'female', 'male', 'male', 'male', 'male', 'female', 'female', 'male', 'male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'male',
     'male', 'male', 'male', 'female', 'male', 'female', 'female', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'male', 'male', 'male', 'male', 'male', 'male', 'male', 'female', 'female', 'male', 'female', 'female', 'male', 'male', 'male', 'male', 'female', 'female', 'male', 'female', 'male', 'male', 'male', 'female', 'male', 'male', 'male', 'female', 'female', 'female', 'male', 'female', 'female', 'female', 'male', 'female', 'male', 'male', 'male', 'female', 'male', 'male', 'male', 'male', 'male', 'male', 'female', 'male', 'female', 'female', 'female', 'male', 'male', 'male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'female', 'female', 'female', 'female', 'male', 'male', 'female', 'female', 'male', 'female', 'female', 'female', 'female', 'female', 'female', 'female', 'male', 'male', 'female', 'female', 'male', 'male', 'female', 'male', 'male', 'female', 'male', 'male', 'male', 'female', 'female', 'female', 'female', 'female', 'male', 'female', 'female', 'female', 'male', 'female', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'male', 'female', 'male', 'male', 'male', 'male', 'male', 'female', 'female', 'male', 'female', 'female', 'female', 'female', 'female', 'female', 'female', 'female', 'female', 'male', 'male', 'female', 'male', 'female', 'male', 'male', 'male', 'male', 'male', 'male', 'male', 'female', 'female', 'male', 'female', 'male', 'female', 'female', 'female', 'female', 'female', 'male', 'female', 'female', 'male', 'female', 'female', 'male', 'female', 'male', 'female', 'female', 'female', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'male', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'male', 'female', 'male', 'female', 'female', 'male', 'female', 'male', 'male', 'male', 'female', 'female', 'male', 'female', 'male', 'male', 'male', 'male', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'female', 'female', 'female', 'female', 'male', 'female', 'male', 'female', 'male', 'male', 'female', 'female', 'female', 'female', 'female', 'male', 'female', 'female', 'female', 'male', 'male'
]

# Training classifiers
clf1 = clf1.fit(X, Y)
clf2 = clf2.fit(X, Y)
clf3 = clf3.fit(X, Y)
clf4 = clf4.fit(X, Y)

prediction1 = clf1.predict(test_data)
prediction2 = clf2.predict(test_data)
prediction3 = clf3.predict(test_data)
prediction4 = clf4.predict(test_data)


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

input("Press enter to exit")
