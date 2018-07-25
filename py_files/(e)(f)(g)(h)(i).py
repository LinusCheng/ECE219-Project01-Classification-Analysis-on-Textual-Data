"""=============================================================================================================="""
""" (e) """
# Soft and hard SVM
print("=== Part (e) ===")
print("Soft and hard SVM")
from sklearn import svm

def SVM(train_X, train_Y, test_X, test_Y):

    print("Hard margin SVM")
    #clf = svm.SVC(kernel='linear', C=1000, probability=True)
    clf = svm.LinearSVC(C=1000)
    clf.fit(train_X, train_Y)
    predict_Y = clf.predict(test_X)
    prob_score = np.vstack((np.zeros(len(test_X)), clf.decision_function(test_X))).T
    #prob_score = clf.predict_proba(test_X)
    measurement(prob_score, test_Y, predict_Y)
    print('-'*20)
    print("Soft margin SVM")
    #clf = svm.SVC(kernel='linear', C=0.001, probability=True)
    clf = svm.LinearSVC(C=0.001)
    clf.fit(train_X, train_Y)
    predict_Y = clf.predict(test_X)
    prob_score = np.vstack((np.zeros(len(test_X)), clf.decision_function(test_X))).T
    #prob_score = clf.predict_proba(test_X)
    measurement(prob_score, test_Y, predict_Y)
    print('-'*20)

print("LSI, min_df=2")
SVM(W_train_svd_2, train_Y, W_test_svd_2, test_Y)
print("LSI, min_df=5")
SVM(W_train_svd_5, train_Y, W_test_svd_5, test_Y)
print("NMF, min_df=2")
SVM(W_train_nmf_2, train_Y, W_test_nmf_2, test_Y)


"""=============================================================================================================="""
""" (f) """
# SVM with Cross validation
print("=== Part (f) ===")
print("SVM with Cross validation")

from sklearn import svm
from sklearn.model_selection import cross_val_score

def cross_val(train_X, train_Y, test_X, test_Y):

    max_score = 0
    K = 0
    for k in range(-3, 4):
        clf = svm.LinearSVC(C=10 ** k)
        #clf = svm.SVC(kernel='linear', C=10**k, probability=True)
        scores = cross_val_score(clf, train_X, train_Y, cv=5)
        print(k, sum(scores)/5)
        if sum(scores)/5 > max_score:
            max_score = sum(scores)/5
            K = k
    print("K = ", K)
    print("max_scores = ", max_score)
    clf = svm.LinearSVC(C=10 ** K).fit(train_X, train_Y)
    predict_Y = clf.predict(test_X)
    prob_score = np.vstack((np.zeros(len(test_X)), clf.decision_function(test_X))).T
    measurement(prob_score, test_Y, predict_Y)
    print('-' * 20)

print("LSI, min_df=2")
cross_val(W_train_svd_2, train_Y, W_test_svd_2, test_Y)
print("LSI, min_df=5")
cross_val(W_train_svd_5, train_Y, W_test_svd_5, test_Y)
print("NMF, min_df=2")
cross_val(W_train_nmf_2, train_Y, W_test_nmf_2, test_Y)


"""=============================================================================================================="""
""" (g) """
# Naive Bayes
print("=== Part (g) ===")

from sklearn.naive_bayes import MultinomialNB
    
def Naive_Bayes(train_X, train_Y, test_X, test_Y):

    clf = MultinomialNB()
    clf.fit(train_X, train_Y)
    predict_Y = clf.predict(test_X)
    prob_score = clf.predict_proba(test_X)
    measurement(prob_score, test_Y, predict_Y)
    
Naive_Bayes(W_train_nmf_2, train_Y, W_test_nmf_2, test_Y)



"""=============================================================================================================="""
""" (h) """
# Logistic regression
print("=== Part (h) ===")
print("Logistic regression")


from sklearn.linear_model import LogisticRegression
def Logistic(train_X, train_Y, test_X, test_Y):

    clf = LogisticRegression()
    clf.fit(train_X, train_Y)
    predict_Y = clf.predict(test_X)
    prob_score = clf.predict_proba(test_X)
    measurement(prob_score, test_Y, predict_Y)
    print('-'*20)
    
print("LSI, min_df=2")
Logistic(W_train_svd_2, train_Y, W_test_svd_2, test_Y)
print("LSI, min_df=5")
Logistic(W_train_svd_5, train_Y, W_test_svd_5, test_Y)
print("NMF, min_df=2")
Logistic(W_train_nmf_2, train_Y, W_test_nmf_2, test_Y)






"""=============================================================================================================="""
""" (i) """
# Logistic regression with regulation L1 L2
print("=== Part (i) ===")
print("With regulation L1 L2")

from sklearn.linear_model import LogisticRegression
def sub_log(pen, k, train_X, train_Y, test_X, test_Y):
    
    clf = LogisticRegression(penalty=pen, C=10 ** k)
    clf.fit(train_X, train_Y)
    predict_Y = clf.predict(test_X)
    prob_score = clf.predict_proba(test_X)
    measurement(prob_score, test_Y, predict_Y)
    
def Logistic_with_reg(train_X, train_Y, test_X, test_Y):
    
    for k in range(-2, 3):
        print("LogisticRegression(penalty='l1'), C={}".format(10 ** k))
        sub_log("l1", k, train_X, train_Y, test_X, test_Y)
        print("LogisticRegression(penalty='l2'), C={}".format(10 ** k))
        sub_log("l2", k, train_X, train_Y, test_X, test_Y)


print("LSI, min_df=2")
Logistic_with_reg(W_train_svd_2, train_Y, W_test_svd_2, test_Y)

print("LSI, min_df=5")
Logistic_with_reg(W_train_svd_5, train_Y, W_test_svd_5, test_Y)

print("NMF, min_df=2")
Logistic_with_reg(W_train_nmf_2, train_Y, W_test_nmf_2, test_Y)









