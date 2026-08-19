import numpy as np
from ucimlrepo import fetch_ucirepo 

iris = fetch_ucirepo(id=53) 

X = iris.data.features.to_numpy() 
species_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y = iris.data.targets.replace(species_map).to_numpy().astype(int).reshape(-1, 1)

def prepare_data(X_raw, y_raw, normalize=True): 
    X, y = X_raw.copy(), y_raw.copy()
    if normalize: 
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X = (X - mean) / np.where(std != 0, std, 1.0) 
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    return X, y

def custom_train_test_split(X, y, test_size=0.2, random_state=None, stratify=None): 
    if random_state is not None: 
        np.random.seed(random_state) 
        
    M = X.shape[0] 
    indices = np.arange(M) 
    
    if stratify is not None: 
        train_indices = [] 
        test_indices = [] 
        unique_classes = np.unique(stratify) 
        
        for c in unique_classes: 
            class_indices = indices[stratify.flatten() == c] 
            n_class = len(class_indices)
            n_test_class = int(n_class * test_size) 
            
            np.random.shuffle(class_indices) 
            
            test_indices.extend(class_indices[:n_test_class]) 
            train_indices.extend(class_indices[n_test_class:])
            
        X_train = X[train_indices]
        X_test = X[test_indices] 
        y_train = y[train_indices] 
        y_test = y[test_indices] 

    else:
        np.random.shuffle(indices) 
        split_index = int(M * (1 - test_size)) 
        
        X_train = X[indices[:split_index]] 
        X_test = X[indices[split_index:]] 
        y_train = y[indices[:split_index]] 
        y_test = y[indices[split_index:]] 
        
    return X_train, X_test, y_train, y_test

def linear_model(X, W):
    return W.T @ X.T

def multiclass_softmax_cost(W, X, y, lam): 
    M = X.shape[0]
    all_evals = linear_model(X, W) 
    max_evals = np.max(all_evals, axis=0, keepdims=True)
    a = np.log(np.sum(np.exp(all_evals - max_evals), axis=0)) + max_evals 
    b = all_evals[y.astype(int).flatten(), np.arange(M)] 
    cost = np.sum(a - b) 
    reg = lam * np.linalg.norm(W[1:, :])**2 
    return (cost + reg) / M

def multiclass_softmax_grad(W, X, y, lam): 
    M = X.shape[0]
    all_evals = linear_model(X, W) 
    exp_scores = np.exp(all_evals - np.max(all_evals, axis=0, keepdims=True)) 
    probs = exp_scores / np.sum(exp_scores, axis=0, keepdims=True) 
    I = np.zeros_like(probs) 
    I[y.astype(int).flatten(), np.arange(M)] = 1
    grad_loss = X.T @ (probs - I).T 
    grad_reg = lam * W 
    grad_reg[0, :] = 0 
    return (grad_loss + grad_reg) / M

def gradient_descent(cost_fun, grad_fun, W_init, X, y, lam, max_its, alpha):
    W = W_init.copy()
    cost_history = []
    for _ in range(max_its):
        cost = cost_fun(W, X, y, lam)
        cost_history.append(cost)
        grad = grad_fun(W, X, y, lam)
        W = W - alpha * grad
    return W, cost_history

def predict_softmax(W, X): 
    return np.argmax(linear_model(X, W), axis=0)

def custom_confusion_matrix(y_true, y_pred, num_classes=3): 
    conf_mat = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true)):
        true_label = y_true[i]
        predicted_label = y_pred[i]
        conf_mat[true_label, predicted_label] += 1
    return conf_mat

def evaluate_results(y_true, y_pred, name="Model"): 
    y_true = y_true.flatten()
    conf_mat = custom_confusion_matrix(y_true, y_pred)
    acc_overall = np.mean(y_true == y_pred)
    
    print(f"\nResults for {name}")
    print("Confusion Matrix (Rows=Actual, Columns=Predicted):")
    print(conf_mat)
    print(f"\nOverall Accuracy: {acc_overall:.4f}")
    
    acc_per_class = conf_mat.diagonal() / conf_mat.sum(axis=1)
    print("\nAccuracy per class (Setosa, Versicolor, Virginica):")
    for i, acc in enumerate(acc_per_class):
        print(f"  Class {i}: {acc:.4f} ({conf_mat[i, i]}/{conf_mat[i].sum()})")
        
    return acc_overall

def run_classification(X_raw, y_raw, normalize, alpha): 
    X, y = prepare_data(X_raw, y_raw, normalize=normalize)
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    N = X.shape[1] - 1 
    K = 3              
    lam = 1e-4         
    max_its = 5000   
    
    W_init = 0.1 * np.random.randn(N + 1, K)
    W_final, _ = gradient_descent(multiclass_softmax_cost, multiclass_softmax_grad, W_init, X_train, y_train, lam, max_its, alpha)
  
    y_pred = predict_softmax(W_final, X_test)
    model_name = f"Softmax ({'Normalized' if normalize else 'NOT Normalized'}, LR={alpha})"
    
    return evaluate_results(y_test, y_pred, model_name)


X_raw, y_raw = X, y

print("Comparison of Normalization and Learning Rate")

results = {}
configs = [(False, 1e-3), (False, 1e-1), (True, 1e-3), (True, 1e-1)]

for normalize, alpha in configs:
    results[(normalize, alpha)] = run_classification(X_raw, y_raw, normalize, alpha)

best_config = max(results, key=results.get)
print(f"\nBest Overall Accuracy: {results[best_config]:.4f}")
print(f"Configuration: Normalized={best_config[0]}, Learning Rate={best_config[1]}")
