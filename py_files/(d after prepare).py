"""=============================================================================================================="""
# Prepare Y
print("Prepare Y")

Com_Tech = set(('comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware'))

train_Y = np.array([twenty_train.target_names[_] not in Com_Tech for _ in twenty_train.target]).astype(int)
ones, zeros = np.count_nonzero(train_Y), len(train_Y) - np.count_nonzero(train_Y)

print("# of 0 : 1 = ", zeros, ones)

test_Y = np.array([twenty_test.target_names[_] not in Com_Tech for _ in twenty_test.target]).astype(int)
ones, zeros = np.count_nonzero(test_Y), len(test_Y) - np.count_nonzero(test_Y)

print("# of 0 : 1 = ", zeros, ones)

print(train_Y[:10],twenty_train.target[:10],[twenty_train.target_names[i] for i in twenty_train.target[:10]])




"""=============================================================================================================="""
# Measurement
# Define the function that plot ROC curves, generate confusion matrices, accuracies, recalls, and precisions
print("Measurement")
print("ROC curves, confusion matrices, accuracies, recalls, and precisions")


def plot_roc(test_Y, prob_score):
    from sklearn.metrics import auc
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(test_Y, prob_score[:,1])

    fig, ax = plt.subplots()

    roc_auc = auc(fpr,tpr)

    ax.plot(fpr, tpr, lw=2, label= 'area under curve = %0.4f' % roc_auc)

    ax.grid(color='0.7', linestyle='--', linewidth=1)

    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate',fontsize=15)
    ax.set_ylabel('True Positive Rate',fontsize=15)

    ax.legend(loc="lower right")

    for label in ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(15)
    
def confusion_matrix(test_Y, predict_Y):
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(test_Y, predict_Y)

def accuracy(test_Y, predict_Y):
    from sklearn.metrics import accuracy_score
    return accuracy_score(test_Y, predict_Y)

def recall(test_Y, predict_Y):
    from sklearn.metrics import recall_score
    return recall_score(test_Y, predict_Y, average='binary')

def precision(test_Y, predict_Y):
    from sklearn.metrics import precision_score
    return precision_score(test_Y, predict_Y, average='binary')

def measurement(prob_score, test_Y, predict_Y):
    plot_roc(test_Y, prob_score)
    print(confusion_matrix(test_Y, predict_Y))
    print("accuracy = ", accuracy(test_Y, predict_Y))
    print("recall = ", recall(test_Y, predict_Y))
    print("precision = ", precision(test_Y, predict_Y))