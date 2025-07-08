import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import joblib

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay

# Load training and test datasets from CSV files
dtrain = pd.read_csv(r"C:\Users\islin\Documents\Islin Kamdem\Python Projects\Handwriting Recognition Application\data\mnist_train.csv")
dtest  = pd.read_csv(r"C:\Users\islin\Documents\Islin Kamdem\Python Projects\Handwriting Recognition Application\data\mnist_test.csv")

# Check the shape of the dataframes
print(dtrain.shape)
print(dtest.shape)

# Function to display an image from the training set
def limg(row):
    image = dtrain.iloc[row, 1:].values.reshape(28, 28)
    plt.imshow(image, cmap="gray")
    plt.title(f"Label: {dtrain.iloc[row, 0]}")
    plt.axis('off')
    plt.show()

# Separate features and labels
x_train = dtrain.iloc[:, 1:]
y_train = dtrain.iloc[:, 0]

x_test  = dtest.iloc[:, 1:]
y_test  = dtest.iloc[:, 0]

# Normalize pixel values to the range [0, 1]
x_train = x_train / 255.0
x_test  = x_test / 255.0

# Create and train a support vector machine model with scaling
clf = make_pipeline(
    StandardScaler(),  # scale features to zero mean and unit variance
    SVC(gamma="auto")  
)
clf.fit(x_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(x_test)

# Example of predicting a single sample
clf.predict(x_test.iloc[[2]])

# Compute accuracy and macro precision
accuracy  = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average="macro")
print(f"Accuracy: {accuracy:.4f} | Macro Precision: {precision:.4f}")

# Compute the confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

# Display raw and normalized confusion matrices
titles_options = [
    ("Confusion matrix (raw counts)", None),
    ("Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        clf,
        x_test,
        y_test,
        display_labels=list(range(10)),
        cmap=plt.cm.Blues,
        normalize=normalize,
        values_format=".2f" if normalize else None,
    )
    disp.ax_.set_title(title)
plt.show()

# Identify frequent misclassifications (more than five errors)
confusion_dict = {}
for i in range(len(confusion_matrix)):
    for j in range(len(confusion_matrix)):
        if i != j:
            if confusion_matrix[i][j] > 5:
                if i not in confusion_dict:
                    confusion_dict[i] = [j]
                else:
                    confusion_dict[i].append(j)
print(confusion_dict)

# Build and visualize a directed graph of frequent confusions
G = nx.DiGraph()
for source, targets in confusion_dict.items():
    for target in targets:
        G.add_edge(source, target, weight=1)
pos = nx.kamada_kawai_layout(G)
plt.figure(figsize=(8, 6))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color='lightblue',
    edge_color='red',
    arrows=True
)
plt.title("Frequent confusions (> 5)")
plt.show()

# Display the distribution of training labels
print(y_train.value_counts())

# Save the trained model and artifacts
joblib.dump(clf, "handwriting_recognition_model.joblib")
np.save("confusion_matrix.npy", confusion_matrix)
joblib.dump(confusion_dict, "dico_erreurs.joblib")
joblib.dump(G, "graphe_confusion.joblib")

# Sanity check with a single test example
test_pred = clf.predict(x_test.iloc[[42]])
print(f"True label: {y_test.iloc[42]} | Predicted label: {test_pred[0]}")
