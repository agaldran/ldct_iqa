from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score, matthews_corrcoef
import numpy as np

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None, text_file=None):
    """
    pretty print for confusion matrixes
    https://gist.github.com/zachguo/10296432
    """
    if text_file is None: print("\n", end=" ")
    else: print("\n", end=" ", file=open(text_file, "a"))

    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    fst_empty_cell = (columnwidth - 3) // 2 * " " + "t/p" + (columnwidth - 3) // 2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    if text_file is None: print("    " + fst_empty_cell, end=" ")
    else: print("    " + fst_empty_cell, end=" ", file = open(text_file, "a"))

    for label in labels:
        if text_file is None: print("%{0}s".format(columnwidth) % label, end=" ")
        else: print("%{0}s".format(columnwidth) % label, end=" ", file = open(text_file, "a"))
    if text_file is None: print()
    else: print(' ', file = open(text_file, "a"))
    # Print rows
    for i, label1 in enumerate(labels):
        if text_file is None: print("    %{0}s".format(columnwidth) % label1, end=" ")
        else: print("    %{0}s".format(columnwidth) % label1, end=" ", file = open(text_file, "a"))
        for j in range(len(labels)):
            cell = "%{}d".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            if text_file is None: print(cell, end=" ")
            else: print(cell, end=" ", file = open(text_file, "a"))
        if text_file is None: print()
        else: print(' ', file = open(text_file, "a"))

def get_one_hot_np(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def evaluate_cls(y_true, y_pred, y_proba, print_conf=True, text_file=None, class_names=None, loss=0):
    # preds
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    # probs - handle case of not every label in y_true
    present_classes, _ = np.unique(y_true, return_counts=True)
    present_classes = list(present_classes)

    y_true_ohe = get_one_hot_np(np.array(y_true), len(present_classes)).astype(int)
    y_pred_ohe = get_one_hot_np(np.array(y_pred), len(present_classes)).astype(int)
    auc_all = [roc_auc_score(y_true_ohe[:, i], y_proba[:, i]) for i in range(len(present_classes))]

    if len(present_classes)==2:
        mean_auc = roc_auc_score(y_true, y_proba[:,1])
    else:
        mean_auc = roc_auc_score(y_true, y_proba, multi_class='ovr') # equivalent to np.mean(auc_all)

    if class_names is None:
        class_names = [str(n) for n in present_classes]
    if text_file is not None:
        print("ACC={:.2f} - MCC={:.2f} - AUC={:.2f} - Loss={:.4f}".format(100 * acc, 100 * mcc, 100 * mean_auc, loss),
              end=" ", file=open(text_file, "a"))
    if print_conf:
        cm = confusion_matrix(y_true, y_pred, labels=present_classes)
        print_cm(cm, class_names, text_file=text_file)
    return mean_auc, mcc, acc, auc_all

