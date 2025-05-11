import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load and prepare data
# Dataset link: https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning
data = pd.read_csv('Training.csv')
test_data = pd.read_csv('Testing.csv')

# Extract features and target
X = data.drop(['prognosis', 'Unnamed: 133'], axis=1, errors='ignore')
y = data['prognosis']
X_test_final = test_data.drop(['prognosis', 'Unnamed: 133'], axis=1, errors='ignore')
y_test_final = test_data['prognosis']

# Encode categorical target variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_test_encoded = le.transform(y_test_final)
num_classes = len(le.classes_)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_test_final_scaled = scaler.transform(X_test_final)

# Build a more complex model for multi-class classification
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))  # Use softmax for multi-class

# Compile model with appropriate loss function for multi-class
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Add early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train model
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test accuracy: {accuracy:.4f}")

# Predict on test data
y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)

# Generate classification report
print(classification_report(y_test, y_pred_classes, target_names=le.classes_))

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Save the model
model.save('disease_prediction_model.h5')
print("Model saved successfully.")

# Add function for sample predictions with test data
def predict_disease(symptoms_dict, model, scaler, label_encoder):
    """
    Make disease predictions based on input symptoms.
    
    Args:
        symptoms_dict: Dictionary with symptom names as keys and 0/1 as values
        model: Trained disease prediction model
        scaler: Fitted StandardScaler
        label_encoder: Fitted LabelEncoder for disease classes
    
    Returns:
        Dictionary containing predicted disease and probability
    """
    # Create a DataFrame with all symptoms set to 0
    sample = pd.DataFrame(0, index=[0], columns=X.columns)
    
    # Set the provided symptoms to 1
    for symptom, value in symptoms_dict.items():
        if symptom in sample.columns:
            sample[symptom] = value
        else:
            print(f"Warning: Symptom '{symptom}' not recognized")
    
    # Scale the features
    sample_scaled = scaler.transform(sample)
    
    # Make prediction
    prediction = model.predict(sample_scaled)[0]
    predicted_class = np.argmax(prediction)
    predicted_disease = label_encoder.inverse_transform([predicted_class])[0]
    confidence = float(prediction[predicted_class])
    
    return {
        "disease": predicted_disease,
        "confidence": confidence,
        "top_3_diseases": [
            (label_encoder.inverse_transform([idx])[0], float(prediction[idx]))
            for idx in prediction.argsort()[-3:][::-1]
        ]
    }

# Sample test cases
print("\n==== Sample Test Cases ====")

# Test Case 1: Symptoms of Common Cold
cold_symptoms = {
    "continuous_sneezing": 1,
    "runny_nose": 1,
    "chills": 1,
    "fatigue": 1
}
result = predict_disease(cold_symptoms, model, scaler, le)
print("\nTest Case 1 - Common Cold Symptoms:")
print(f"Predicted Disease: {result['disease']}")
print(f"Confidence: {result['confidence']:.4f}")
print("Top 3 possibilities:")
for disease, prob in result['top_3_diseases']:
    print(f"  - {disease}: {prob:.4f}")

# Test Case 2: Symptoms of Diabetes
diabetes_symptoms = {
    "excessive_hunger": 1,
    "increased_appetite": 1,
    "polyuria": 1,
    "weight_loss": 1,
    "fatigue": 1
}
result = predict_disease(diabetes_symptoms, model, scaler, le)
print("\nTest Case 2 - Diabetes Symptoms:")
print(f"Predicted Disease: {result['disease']}")
print(f"Confidence: {result['confidence']:.4f}")
print("Top 3 possibilities:")
for disease, prob in result['top_3_diseases']:
    print(f"  - {disease}: {prob:.4f}")

# Test Case 3: Symptoms of Heart Disease
heart_symptoms = {
    "chest_pain": 1,
    "breathlessness": 1,
    "sweating": 1,
    "fatigue": 1
}
result = predict_disease(heart_symptoms, model, scaler, le)
print("\nTest Case 3 - Heart Disease Symptoms:")
print(f"Predicted Disease: {result['disease']}")
print(f"Confidence: {result['confidence']:.4f}")
print("Top 3 possibilities:")
for disease, prob in result['top_3_diseases']:
    print(f"  - {disease}: {prob:.4f}")

# Interactive prediction
print("\n==== Interactive Disease Prediction ====")
print("Enter 1 if symptom is present, 0 if absent.")
print("Type 'done' when finished entering symptoms.")

user_symptoms = {}
while True:
    symptom = input("Enter symptom name (or 'done' to finish): ")
    if symptom.lower() == 'done':
        break
    
    if symptom not in X.columns:
        print(f"Warning: '{symptom}' is not a recognized symptom. Available symptoms include:")
        print(", ".join(X.columns[:10]) + ", ...")
        continue
        
    value = int(input(f"Is {symptom} present (1) or absent (0)? "))
    user_symptoms[symptom] = value

if user_symptoms:
    result = predict_disease(user_symptoms, model, scaler, le)
    print("\nPrediction Results:")
    print(f"Predicted Disease: {result['disease']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("Top 3 possibilities:")
    for disease, prob in result['top_3_diseases']:
        print(f"  - {disease}: {prob:.4f}")
