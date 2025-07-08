# Handwritten Digit Recognition
This repository contains my project on handwritten digit recognition using the MNIST dataset. It is the first that I build without external assistance, relying solely on documentation and self-study. 

## Project Overview
- **Data:** MNIST training and test datasets (CSV format) stored in the `data/` folder.
- **Model:** Support Vector Machine
- **Analysis:** Confusion Matrix, Graph of Frequent Confusions

## Key Findings from Confusion Analysis
The confusion analysis reveals several structural weaknesses in the model’s predictions:
- **Persistent Misclassifications:** Some pairs such as [2,7], [3,5] and [4,9] exhibit bilateral confusion, suggesting that the model struggles to distinguish visually similar shapes.
- **Stable Classes:** Stable Classes: Digits 0, 1, and 6 show strong separability with minimal confusion.
- **Bias Indication:** The model tends to predict 1 and 7 more frequently than other classes, hinting at a possible bias.
The confusion graph provides a clear visualization of these relationships and will guide future improvements.

## Results

- **Accuracy:** 0.96  
- **Macro Precision:** 0.96 

## Next Steps
I plan to revisit it in order to deploy the model on an interactive platform. It could be interesting to see if the model give the same result in another environment than Visual Studio Code. 
