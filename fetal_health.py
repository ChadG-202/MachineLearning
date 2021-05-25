import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Read in csv removing row 0
data = pd.read_csv("fetal_health.csv", skiprows=0)
X = data.values[:, 0:20]
y = data.values[:,21]

print(data)

#Creating classifers and setting parameters
clf1 = KNeighborsClassifier(n_neighbors=3)
clf2 = RandomForestClassifier(n_estimators=200)
clf3 = DecisionTreeClassifier()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=LogisticRegression())
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='adam',
alpha=0.05, learning_rate='adaptive', max_iter=1000)
ehclf = VotingClassifier(estimators=[('KNN', clf1), ('rf', clf2), ('dt', clf3)],
voting='hard')
esclf = VotingClassifier(estimators=[('KNN', clf1), ('rf', clf2), ('dt', clf3)],
voting='soft')

#Creating 5-fold and 20% test set
cv = ShuffleSplit(n_splits=5, test_size=0.2)

#Global variables to store best classifier
bestTotal = 0
bestClassifier = ""

print("|||||||||||||||||||||||||||||||||||||||||||||||||| \n")
print("5-fold cross validation:")
print("--------------------------------------------------")

#For loop that runs each algorithm and displays results
for clf, label in zip([clf1, clf2, clf3, mlp, sclf, ehclf, esclf], 
['KNN', 'Random Forest', 'Decision Tree', 'MLP', 'StackingClassifier',
'Hard Voting', 'Soft Voting']):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    clf = clf.fit(X_train, y_train)
    Y_prediction = clf.predict(X_test)
    print("[%s] Train/test accuracy: %0.4f \n" % (label, accuracy_score(y_test,Y_prediction)))
    scoresA = model_selection.cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    scoresP = model_selection.cross_val_score(clf, X, y, cv=cv, scoring='precision_macro')
    scoresF = model_selection.cross_val_score(clf, X, y, cv=cv, scoring='f1_macro')
    print("[%s] 5-fold cross accuracy scores: %s" % (label, scoresA))
    print("[%s] Accuracy mean score: %0.4f (+/- %0.2f) \n" % (label, scoresA.mean(), scoresA.std()))
    print("[%s] 5-fold cross precision scores: %s" % (label, scoresP))
    print("[%s] Precision mean scores: %0.4f (+/- %0.2f) \n" % (label, scoresP.mean(), scoresP.std()))
    print("[%s] 5-fold cross F1 scores: %s" % (label, scoresF))
    print("[%s] F1 mean scores: %0.4f (+/- %0.2f)" % (label, scoresF.mean(), scoresF.std()))
    print("--------------------------------------------------")
    #Find algorithm with best total average
    total = (scoresA.mean() + scoresP.mean() + scoresF.mean())/3
    if(total > bestTotal):
        bestTotal = total
        bestClassifier = label

#Displays best classifer
print("THE BEST CLASSIFIER WAS: " + bestClassifier.upper() + "\n")
print("||||||||||||||||||||||||||||||||||||||||||||||||||")

