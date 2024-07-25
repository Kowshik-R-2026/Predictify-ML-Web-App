import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Set page configuration to start content at the top
st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 90%;
    }
</style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "trained_model" not in st.session_state:
    st.session_state["trained_model"] = None

# Function to perform Linear Regression with Gradient Descent
def perform_linear_regression(X, y, test_size, random_state, num_iterations, learning_rate):
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize theta for gradient descent
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train).reshape(-1, 1)
    X_train_np = np.insert(X_train_np, 0, 1, axis=1)
    theta = np.zeros((X_train_np.shape[1], 1))

    # Perform gradient descent
    theta, J_history = gradient_descent(X_train_np, y_train_np, theta, learning_rate, num_iterations)

    # Prepare X_test for prediction
    X_test_np = np.array(X_test)
    X_test_np = np.insert(X_test_np, 0, 1, axis=1)

    # Calculate predictions
    y_pred = X_test_np.dot(theta)

    # Calculate and display metrics
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse}")
    st.write("Model Coefficients:")
    st.write(dict(zip(X.columns, theta.flatten())))

    # Plot gradient descent
    plot_gradient_descent(J_history)

    # Save trained model
    st.session_state["trained_model"] = {"model": "linear_regression", "theta": theta}

def plot_gradient_descent(J_history):
    """
    Function to plot the cost function over iterations during gradient descent.
    """
    fig, ax = plt.subplots()
    ax.plot(range(len(J_history)), J_history, '-b', linewidth=2)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Gradient Descent: Cost vs Iterations')
    st.pyplot(fig)

def gradient_descent(X, y, theta, alpha, num_iterations):
    """
    Gradient Descent function to minimize the cost function.
    """
    m = len(y)
    J_history = []

    for i in range(num_iterations):
        theta = theta - (alpha / m) * (X.T.dot(X.dot(theta) - y))
        cost = (1 / (2 * m)) * np.sum((X.dot(theta) - y) ** 2)
        J_history.append(cost)

    return theta, J_history

def perform_random_forest_regression(X, y, test_size, random_state):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize Random Forest Regressor
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    st.write("R2 Score: ", r2_score(y_test, y_pred))
    st.write("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
    st.write("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))
    st.write("Score: ", model.score(X_test, y_test))

    # Display feature importances
    st.write("Feature Importances:")
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.write(feature_importances)

    # Plot feature importances
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=feature_importances.index, ax=ax)
    ax.set_xlabel('Feature Importance Score')
    ax.set_ylabel('Features')
    ax.set_title('Feature Importances')
    st.pyplot(fig)

    # Save trained model
    st.session_state["trained_model"] = {"model": "random_forest", "model_instance": model}

def perform_logistic_regression(X, y, test_size, random_state, label_encoder):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Decode predictions
    y_pred_decoded = label_encoder.inverse_transform(y_pred)

    # Calculate and display metrics
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy}")
    st.write("Confusion Matrix and Classification Report:")
    # Layout for confusion matrix and classification report
    col1, col2 = st.columns(2)

    # Display classification report
    
    col1.text(classification_report(y_test, y_pred))

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(2, 2))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', cbar=False)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    col2.pyplot(fig)



    # Save trained model
    st.session_state["trained_model"] = {"model": "logistic_regression", "model_instance": model, "label_encoder": label_encoder}

# Function to make predictions using user input
def make_predictions(user_data, features):
    trained_model = st.session_state["trained_model"]

    if trained_model is None:
        st.error("No model has been trained yet. Please train a model first.")
        return

    if trained_model["model"] == "linear_regression":
        theta = trained_model["theta"]
        user_data = np.insert(np.array(user_data), 0, 1).reshape(1, -1)
        prediction = user_data.dot(theta)
        st.write("Prediction:", prediction.flatten()[0])

    elif trained_model["model"] == "random_forest":
        model = trained_model["model_instance"]
        prediction = model.predict(np.array(user_data).reshape(1, -1))
        st.write("Prediction:", prediction[0])

    elif trained_model["model"] == "logistic_regression":
        model = trained_model["model_instance"]
        prediction = model.predict(np.array(user_data).reshape(1, -1))
        label_encoder = trained_model["label_encoder"]
        prediction_decoded = label_encoder.inverse_transform(prediction)
        st.write("Prediction:", prediction_decoded[0])

# Streamlit UI
st.title("Machine Learning Algorithms Tester")
st.header("Select Options:")

uploaded_file = st.file_uploader("Drag and drop a CSV or Excel file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df, height=210)

    st.write("Dataset Summary:")
    st.write(df.describe().T)

    df.dropna()

    # Label Encoding for categorical variables
    label_encoder = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col])

    # Display pre-processed data
    st.write("Sample Pre-processed Data:")
    st.write(df.head())

    # Select features and output column for ML
    st.subheader("Select Features and Output Column:")
    features = st.multiselect("Select Features:", df.columns.tolist())
    output_column = st.selectbox("Select Output Column:", df.columns.tolist())

    # Algorithm selection
    st.subheader("Select Algorithm:")
    algorithm = st.selectbox("Select Algorithm:", ["Linear Regression", "Random Forest Regression", "Logistic Regression"])

    test_size = st.slider("Test Size:", min_value=0.1, max_value=0.5, step=0.1)
    random_state = st.number_input("Random State:", value=42)

    if algorithm == "Linear Regression":
        learning_rate = st.number_input("Learning Rate:", value=0.01, min_value=0.001, max_value=10.00, step=0.01)
        num_iterations = st.number_input("Number of Iterations:", value=1000)
        if st.button("Run Algorithm"):
            with st.spinner('Running Linear Regression...'):
                perform_linear_regression(df[features], df[output_column], test_size, random_state, num_iterations, learning_rate)

    elif algorithm == "Random Forest Regression":
        if st.button("Run Algorithm"):
            with st.spinner('Running Random Forest Regression...'):
                perform_random_forest_regression(df[features], df[output_column], test_size, random_state)

    elif algorithm == "Logistic Regression":
        if st.button("Run Algorithm"):
            with st.spinner('Running Logistic Regression...'):
                perform_logistic_regression(df[features], df[output_column], test_size, random_state, label_encoder)

    # User input for predictions
    st.subheader("Test the Trained Model")
    st.write("Enter values for the features to make a prediction:")
    user_input = []
    for feature in features:
        user_input.append(st.number_input(f"Input value for {feature}:", value=0.0))

    if st.button("Predict"):
        with st.spinner('Making prediction...'):
            make_predictions(user_input, features)
