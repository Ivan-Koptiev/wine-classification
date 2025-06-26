import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=== Wine Classification with K-Nearest Neighbors ===\n")

# Load dataset
data = datasets.load_wine(as_frame=True)
X = data.data
y = data.target
names = data.target_names

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(names)}")
print(f"Class names: {names}")
print(f"Features: {list(data.feature_names)}")

# Create DataFrame for analysis
df = pd.DataFrame(X, columns=data.feature_names)
df['wine class'] = data.target.replace([0, 1, 2], names)

# Display basic dataset info
print(f"\nClass distribution:")
print(df['wine class'].value_counts())
print(f"\nFeature statistics:")
print(df.describe())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# 1. Baseline KNN without scaling
print("\n=== Baseline KNN (No Scaling) ===")
knn_baseline = KNeighborsClassifier(n_neighbors=7)
knn_baseline.fit(X_train, y_train)
pred_baseline = knn_baseline.predict(X_test)
accuracy_baseline = metrics.accuracy_score(y_test, pred_baseline)
print(f"Accuracy (baseline KNN): {accuracy_baseline:.4f}")

# 2. KNN with scaling
print("\n=== KNN with Standard Scaling ===")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_scaled = KNeighborsClassifier(n_neighbors=7)
knn_scaled.fit(X_train_scaled, y_train)
pred_scaled = knn_scaled.predict(X_test_scaled)
accuracy_scaled = metrics.accuracy_score(y_test, pred_scaled)
print(f"Accuracy (scaled KNN): {accuracy_scaled:.4f}")

# 3. Test different K values
print("\n=== Testing Different K Values ===")
k_values = range(1, 21)
accuracies_unscaled = []
accuracies_scaled = []

for k in k_values:
    # Unscaled
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    accuracies_unscaled.append(metrics.accuracy_score(y_test, pred))
    
    # Scaled
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    pred = knn.predict(X_test_scaled)
    accuracies_scaled.append(metrics.accuracy_score(y_test, pred))

# Find best K values
best_k_unscaled = k_values[np.argmax(accuracies_unscaled)]
best_k_scaled = k_values[np.argmax(accuracies_scaled)]

print(f"Best K (unscaled): {best_k_unscaled} (accuracy: {max(accuracies_unscaled):.4f})")
print(f"Best K (scaled): {best_k_scaled} (accuracy: {max(accuracies_scaled):.4f})")

# 4. Test different distance metrics
print("\n=== Testing Different Distance Metrics ===")
metrics_list = ['euclidean', 'manhattan', 'chebyshev']
best_metric = None
best_accuracy = 0

for metric in metrics_list:
    knn = KNeighborsClassifier(n_neighbors=best_k_scaled, metric=metric)
    knn.fit(X_train_scaled, y_train)
    pred = knn.predict(X_test_scaled)
    accuracy = metrics.accuracy_score(y_test, pred)
    print(f"Accuracy with {metric}: {accuracy:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_metric = metric

print(f"Best distance metric: {best_metric}")

# 5. Cross-validation
print("\n=== Cross-Validation Results ===")
best_knn = KNeighborsClassifier(n_neighbors=best_k_scaled, metric=best_metric)
cv_scores = cross_val_score(best_knn, X_train_scaled, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 6. Final model with best parameters
print(f"\n=== Final Model (K={best_k_scaled}, metric={best_metric}) ===")
final_knn = KNeighborsClassifier(n_neighbors=best_k_scaled, metric=best_metric)
final_knn.fit(X_train_scaled, y_train)
final_pred = final_knn.predict(X_test_scaled)
final_accuracy = metrics.accuracy_score(y_test, final_pred)
print(f"Final test accuracy: {final_accuracy:.4f}")

# Generate classification report
classification_report = metrics.classification_report(y_test, final_pred, target_names=names, output_dict=True)

# Save classification report
with open('classification_report.json', 'w') as f:
    json.dump(classification_report, f, indent=2)

print(f"\nClassification report saved to 'classification_report.json'")

# ===== VISUALIZATIONS =====

# 1. K values comparison
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(k_values, accuracies_unscaled, 'o-', label='Unscaled', linewidth=2, markersize=6)
plt.plot(k_values, accuracies_scaled, 's-', label='Scaled', linewidth=2, markersize=6)
plt.axvline(x=best_k_unscaled, color='red', linestyle='--', alpha=0.7, label=f'Best K (unscaled): {best_k_unscaled}')
plt.axvline(x=best_k_scaled, color='blue', linestyle='--', alpha=0.7, label=f'Best K (scaled): {best_k_scaled}')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy vs K Value')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Feature importance (using feature correlation with target)
plt.subplot(1, 2, 2)
correlations = []
for feature in data.feature_names:
    correlation = np.corrcoef(X[feature], y)[0, 1]
    correlations.append(abs(correlation))

feature_importance = pd.DataFrame({
    'Feature': data.feature_names,
    'Correlation': correlations
}).sort_values('Correlation', ascending=True)

plt.barh(range(len(feature_importance)), feature_importance['Correlation'])
plt.yticks(range(len(feature_importance)), feature_importance['Feature'])
plt.xlabel('Absolute Correlation with Target')
plt.title('Feature Importance (Correlation with Target)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('knn_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Confusion matrix
plt.figure(figsize=(10, 8))
cm = metrics.confusion_matrix(y_test, final_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=names, yticklabels=names)
plt.title(f'Confusion Matrix (K={best_k_scaled}, {best_metric})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Class distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
df['wine class'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Overall Class Distribution')
plt.xlabel('Wine Class')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.subplot(1, 3, 2)
train_dist = pd.Series(y_train).map({0: names[0], 1: names[1], 2: names[2]}).value_counts()
train_dist.plot(kind='bar', color='lightgreen')
plt.title('Training Set Distribution')
plt.xlabel('Wine Class')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.subplot(1, 3, 3)
test_dist = pd.Series(y_test).map({0: names[0], 1: names[1], 2: names[2]}).value_counts()
test_dist.plot(kind='bar', color='lightcoral')
plt.title('Test Set Distribution')
plt.xlabel('Wine Class')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Feature pairplot (sample of most important features)
print("\nGenerating feature pairplot...")
top_features = feature_importance.tail(6)['Feature'].tolist()
df_subset = df[top_features + ['wine class']]

sns.pairplot(df_subset, hue='wine class', diag_kind='kde', height=2)
plt.savefig('feature_pairplot.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Scaling comparison
plt.figure(figsize=(15, 5))

# Before scaling
plt.subplot(1, 3, 1)
df[data.feature_names].boxplot(figsize=(12, 6))
plt.title('Feature Distributions (Before Scaling)')
plt.xticks(rotation=45)
plt.ylabel('Feature Values')

# After scaling
plt.subplot(1, 3, 2)
df_scaled = pd.DataFrame(X_train_scaled, columns=data.feature_names)
df_scaled.boxplot(figsize=(12, 6))
plt.title('Feature Distributions (After Scaling)')
plt.xticks(rotation=45)
plt.ylabel('Scaled Feature Values')

# Accuracy comparison
plt.subplot(1, 3, 3)
comparison_data = ['Unscaled', 'Scaled']
comparison_accuracies = [accuracy_baseline, accuracy_scaled]
colors = ['lightcoral', 'lightgreen']

bars = plt.bar(comparison_data, comparison_accuracies, color=colors)
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# Add value labels on bars
for bar, acc in zip(bars, comparison_accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('scaling_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== Summary ===")
print(f"Best model configuration:")
print(f"- K value: {best_k_scaled}")
print(f"- Distance metric: {best_metric}")
print(f"- Scaling: Yes")
print(f"- Final accuracy: {final_accuracy:.4f}")
print(f"- Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

print(f"\nAll visualizations saved:")
print("- knn_analysis.png: K value analysis and feature importance")
print("- confusion_matrix.png: Confusion matrix for best model")
print("- class_distribution.png: Class distribution across datasets")
print("- feature_pairplot.png: Feature relationships")
print("- scaling_comparison.png: Scaling effect comparison")
print("- classification_report.json: Detailed classification metrics")