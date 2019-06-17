from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report

valid_labels = [0,1,2,3]

def get_f1_score(true, pred):

  results = dict()
  results['f1-macro'] = f1_score(true, pred, labels=valid_labels, average="macro")
  results['classification_report'] = classification_report(true, pred, labels=valid_labels, digits=3)

  return results
