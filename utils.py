import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os

def plot_confusion_matrix(y_true, y_pred, classes, save_path, title='Confusion Matrix', normalize=True, cmap='Blues'):

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)


    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f' 
        print("Generated Normalized Confusion Matrix")
    else:
        fmt = 'd'   
        print("Generated Count Confusion Matrix")


    plt.figure(figsize=(10, 8))
    

    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, square=True,
                xticklabels=classes, yticklabels=classes,
                cbar_kws={"shrink": 0.8}) # 

    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45) 
    

    plt.tight_layout()
    

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion Matrix saved to: {save_path}")

def calculate_metrics_report(y_true, y_pred, classes):

    return classification_report(y_true, y_pred, target_names=classes, digits=4)