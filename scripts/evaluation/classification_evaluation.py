from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(predict, label):
    res = {}
    res['acc'] = accuracy_score(label, predict)
    res['pre'] = precision_score(label, predict, average='macro')
    res['rec'] = recall_score(label, predict, average='macro')
    res['f1'] = f1_score(label, predict, average='macro')
    res['positive_pre'] = precision_score(label, predict, average='binary')
    res['positive_rec'] = recall_score(label, predict, average='binary')
    res['positive_f1'] = f1_score(label, predict, average='binary')
    return res