"""=============================================================================================================="""
""" (d) """
print("=== Part (d) ===")
# Dimension reduction


def SVD(X_train_tfidf, X_test_tfidf):
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=50)
    W_train_svd = svd.fit_transform(X_train_tfidf)
    W_test_svd = svd.transform(X_test_tfidf)
    return W_train_svd, W_test_svd

W_train_svd_2, W_test_svd_2 = SVD(X_train_tfidf_2, X_test_tfidf_2)    
W_train_svd_5, W_test_svd_5 = SVD(X_train_tfidf_5, X_test_tfidf_5)    

def NMF(X_train_tfidf, X_test_tfidf):
    from sklearn.decomposition import NMF
    nmf = NMF(n_components=50, init='random', random_state=0)
    W_train_nmf = nmf.fit_transform(X_train_tfidf)
    W_test_nmf = nmf.transform(X_test_tfidf)
    return W_train_nmf, W_test_nmf

W_train_nmf_2, W_test_nmf_2 = NMF(X_train_tfidf_2, X_test_tfidf_2)    
W_train_nmf_5, W_test_nmf_5 = NMF(X_train_tfidf_5, X_test_tfidf_5)  

print("Dim of Train SVD, min_df=2 =",W_train_svd_2.shape)
print("Dim of Test SVD, min_df=2 =",W_test_svd_2.shape)
print("Dim of Train SVD, min_df=5 =",W_train_svd_5.shape)
print("Dim of Test SVD, min_df=5 =",W_test_svd_5.shape)
print("Dim of Train NMF =",W_train_nmf_2.shape)
print("Dim of Test NMF =",W_test_nmf_2.shape)

print ("======")

print("train")
print("SVD")
print(W_train_svd_2[0])
print("NMF")
print(W_train_nmf_2[0])
print("test")
print("SVD")
print(W_test_svd_2[0])
print("NMF")
print(W_test_nmf_2[0])