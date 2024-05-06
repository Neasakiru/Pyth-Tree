import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('data.csv', encoding='latin1', sep=';')

# Preprocessing
# Convert categorical variables into numerical representations
data['day'] = data['day'].astype('category').cat.codes
data['weather'] = data['weather'].astype('category').cat.codes
data['well-being'] = data['well-being'].astype('category').cat.codes

# Separate features and target variable
X = data.drop(columns=['conclusion'])
y = data['conclusion']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Adjust the test input
test_input = pd.DataFrame({'age': [24], 'day': ['Wtorek'], 'weather': ['slonce'], 'amount_of_money': [0], 'well-being': ['zle']})

# Convert categorical variables into numerical representations
test_input['day'] = test_input['day'].astype('category').cat.codes
test_input['weather'] = test_input['weather'].astype('category').cat.codes
test_input['well-being'] = test_input['well-being'].astype('category').cat.codes

# Reorder columns to match the training data
test_input = test_input[X_train.columns]

# Predict for the test input
prediction = model.predict(test_input)

print("According to the test data, you should spend the evening:", prediction[0])

# Create decision tree classifier
tree_clf = DecisionTreeClassifier(random_state=42)

# Train the model
tree_clf.fit(X_train, y_train)

# Plot the decision tree
plt.figure(figsize=(20,10))
plot_tree(tree_clf, feature_names=X_train.columns, class_names=tree_clf.classes_, filled=True)
plt.show()

# Predict for the test input
prediction = tree_clf.predict(test_input)

print("According to the test data, you should spend the evening:", prediction[0])