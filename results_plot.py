import json
import numpy as np
import matplotlib.pyplot as plt

# Load the data from 'results.json'
with open('results.json', 'r') as file:
    data = json.load(file)

# Extract the confusion matrices as lists of lists
confusion_matrices = data['Confusion Matrix']
print(confusion_matrices)

# Define a function to plot a confusion matrix
def plot_confusion_matrix(confusion_matrix, class_names):
    # Convert the confusion matrix to a NumPy array
    confusion_matrix = np.array(confusion_matrix)
    
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(confusion_matrix.shape[1]),
           yticks=np.arange(confusion_matrix.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    plt.show()

# Example usage for one of the models (you can repeat this for other models)
for i in range(6):
    model_index = str(i)  # Change this index to select a different model
    confusion_matrix = confusion_matrices[model_index]
    class_names = ['Class 0', 'Class 1', 'Class 2']  # Replace with your actual class names

plot_confusion_matrix(confusion_matrix, class_names)

# Extract the metrics
models = data['Model']
accuracy = data['Accuracy']
precision = data['Precision']
recall = data['Recall']
f1_score = data['F1-Score']
roc_auc = data['ROC-AUC']

# Define a function to plot a metric with data values
def plot_metric_with_values(metric, metric_name):
    fig, ax = plt.subplots()
    bars = ax.bar(models, metric)
    
    ax.set(xlabel='Model', ylabel=metric_name,
           title=f'{metric_name} for Different Models')
    ax.set_xticklabels(models, rotation=45)
    
    # Display data values on the bars
    for bar, val in zip(bars, metric):
        ax.text(bar.get_x() + bar.get_width()/2 - 0.1, val + 0.01, f'{val:.2f}', fontsize=9)
    
    plt.show()

# Plot various metrics with data values
plot_metric_with_values(accuracy, 'Accuracy')
plot_metric_with_values(precision, 'Precision')
plot_metric_with_values(recall, 'Recall')
plot_metric_with_values(f1_score, 'F1-Score')
plot_metric_with_values(roc_auc, 'ROC-AUC')