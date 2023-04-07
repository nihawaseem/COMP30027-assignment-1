Required libraries: pandas, math, matplotlib, sklearn, collections

Naive Bayes implementation:
    - def preprocess(csv_name): reads in a csv file and drops 'filename' (unused attributes)
    - def train(data): takes the training dataset and returns a list of unique labels, prior probabilities for all labels, and parameters (mean and variance)
    - def predict(test_data, labels, priors, parameters): returns predicted labels
    - def evaluate(predicted_labels, test_data, positive_class): prints accuracy, classification report, and confusion matrix 
        - Note: For multiclass evaluation, set positive_class = None
        
Question 4
    - To test different k values, modify the k_values list
    - To test different number of runs, modify num_runs variable
    
Question 5
    - To test different percentages of data deleted, modify the list p 
    - To test different number of runs, modify num_runs variable
    - To print precision and recall scores for all attributes, use the provided code but only when num_runs = 1