from sklearn.metrics import roc_auc_score

def get_auc(y_true, y_pred):
    """get auc metric score
    
    """
    score = roc_auc_score(y_true, y_pred)
    return score