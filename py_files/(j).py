"""=============================================================================================================="""
""" (j) """
# Multiclass Classification
print("=== Part (j) ===")
print("Multiclass Classification")

categories=["comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "misc.forsale", "soc.religion.christian"]
multi_train, multi_test = fetch_data(categories)
X_train_counts_m, X_test_counts_m = Counter(2, multi_train, multi_test)
X_train_tfidf_m, X_test_tfidf_m = tfidf(X_train_counts_m, X_test_counts_m)
W_train_svd_m, W_test_svd_m = SVD(X_train_tfidf_m, X_test_tfidf_m)
W_train_nmf_m, W_test_nmf_m = NMF(X_train_tfidf_m, X_test_tfidf_m)

train_Y_m = multi_train.target
test_Y_m = multi_test.target



""" Define Measurement """
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn import svm
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB


def Multi_Measurement(test_Y, predict_Y):

    print(confusion_matrix(test_Y, predict_Y))
    print("accuracy = ", accuracy_score(test_Y, predict_Y))
    print("recall = ", recall_score(test_Y, predict_Y, average=None))
    print("precision = ", precision_score(test_Y, predict_Y, average=None))
    
    
    
""" Apply classifications """
def Multi_SVM(train_X, train_Y, test_X, test_Y):

    #one VS one
    print("One vs One")
    clf = OneVsOneClassifier(svm.LinearSVC(), -1)
    #clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(train_X, train_Y)
    predict_Y = clf.predict(test_X)
    Multi_Measurement(test_Y, predict_Y)
    print('-'*20)
    
    #one VS rest
    print("One vs Rest")
    clf = OneVsRestClassifier(svm.LinearSVC(), -1)
    #clf = svm.SVC(decision_function_shape='ovr')
    clf.fit(train_X, train_Y)
    predict_Y_ovr = clf.predict(test_X)
    Multi_Measurement(test_Y, predict_Y_ovr)
    print('-'*20)
    
print("===SVM===")
print("LSI, min_df=2")
Multi_SVM(W_train_svd_m, train_Y_m, W_test_svd_m, test_Y_m)
print("NMF, min_df=2")
Multi_SVM(W_train_nmf_m, train_Y_m, W_test_nmf_m, test_Y_m)


def Multi_NB(train_X, train_Y, test_X, test_Y):
    clf = MultinomialNB()
    clf.fit(train_X, train_Y)
    predict_Y = clf.predict(test_X)
    Multi_Measurement(test_Y, predict_Y)
    print('-'*20)

print("===Naive Bayes===")
Multi_NB(W_train_nmf_m, train_Y_m, W_test_nmf_m, test_Y_m)



print("===== Completed =====")


