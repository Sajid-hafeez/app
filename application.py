# -*- coding: utf-8 -*-
"""
Created on Tue May  9 23:11:21 2023

@author: Sajid
"""


import streamlit as st
from streamlit import components
# ... other imports and code ...
import streamlit as st
from streamlit import components
import matplotlib.pyplot as plt

from PIL import Image
# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Load and preprocess data (use your data loading and preprocessing steps here)
import pandas as pd
import chardet

file_path = 'apt for sale2.csv'

# Detect the encoding of the file
with open(file_path, 'rb') as f:
    result = chardet.detect(f.read())

# Read the CSV file with the detected encoding
df = pd.read_csv(file_path, encoding=result['encoding'])

print(df)
# Extract bedrooms and bathrooms from the Summary column
df[['Bedrooms', 'Bathrooms']] = df['Summary'].str.extract('Bedroom\(s\): (\d+) \| Bathroom\(s\): (\d+)').astype(float)
df.info()
# Remove ' SQM' from the Area column and convert it to a float
df['Area'] = df['Area'].str.replace(' SQM', '').str.replace(',', '').astype(float)
df['Price'] =df[' Price '].str.replace('$', '').str.replace(',', '').astype(float)
df['Furnished'] = df['Furnished?'].apply(lambda x: 0 if x == 'Unfurnished' else 1)
# Replace 'No Data' in 'Floor Number' column with NaN and convert it to float
df['Floor Number'] = df['Floor Number'].replace('Basement', 0).replace('No Data', np.nan).astype(float)

# Calculate the mean of 'Floor Number' and fill missing values with it
mean_floor_number = df['Floor Number'].median()
df['Floor Number'].fillna(mean_floor_number, inplace=True)
df['Bedrooms'].fillna(df['Bedrooms'].median(), inplace=True)
df['Bathrooms'].fillna(df['Bathrooms'].median(), inplace=True)
df = df.drop(columns=['Title','Furnished?','Kaza','Floor Number',' Price ', 'Summary','Village', 'Details', 'Website'])
# df = pd.get_dummies(df, columns=[ 'Kaza'])
# df = df.drop(columns=['Kaza_No Data'])
# ... your data loading and preprocessing code ...
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Split the data into training and testing sets
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a dictionary of models
models = {
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    'Random Forest Regression': RandomForestRegressor(random_state=42),
    'Gradient Boosting Regression': GradientBoostingRegressor(random_state=42),
#    'XGBoost Regression': XGBRegressor(random_state=42),
    'LightGBM Regression': LGBMRegressor(random_state=42),
    'CatBoost Regression': CatBoostRegressor(random_state=42, silent=True),
}



pipelines = {}

# Create a function to evaluate each model
def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ('scaler', RobustScaler()), # Using RobustScaler to scale the data as it is less sensitive to outliers
        (model_name, model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, r2, pipeline  # Return the fitted pipeline


# Evaluate each model and report MSE and R-squared values, and store fitted pipelines
for model_name, model in models.items():
    mse, r2, pipeline = evaluate_model(model_name, model, X_train, X_test, y_train, y_test)
    print(f"{model_name}:")
    print(f"  Mean Squared Error: {mse:.2f}")
    print(f"  R-squared: {r2:.2f}")
    pipelines[model_name] = pipeline  # Store the fitted pipeline

# Define a function for the home page
def home_page():
    # Set the title and subheader
    st.title("Lebanon Property Finder")
    st.subheader("Welcome to our property analysis and prediction app!")
    
    # Provide a brief description of the app and its pages using markdown
    st.markdown("""
    This app is designed to help you explore and analyze properties listed on OLX in Lebanon. It consists of four main pages:

    1. **Home**: An overview of the app and its features.
    2. **Map**: An interactive map displaying properties listed on OLX in Lebanon. You can navigate and explore different locations to find your desired property.
    3. **Prediction**: A machine learning-based price prediction tool that estimates property prices based on various independent variables. You can input relevant details about a property and receive an estimated price.
    4. **Contact Us**: Get in touch with us for any inquiries, suggestions, or feedback. You can find our address, phone number, and email on this page.

    Use the buttons to navigate between the different pages and make the most out of the available tools!
    """)

# Define a function for the map page
def map_page():
    # Set the title
    st.title('Map')
    
    # Add the HTML code for the map using markdown
    st.markdown("""
    <style>.embed-container {position: relative; padding-bottom: 80%; height: 0; max-width: 100%;} .embed-container iframe, .embed-container object, .embed-container iframe{position: absolute; top: 0; left: 0; width: 100%; height: 100%;} small{position: absolute; z-index: 40; bottom: 0; margin-bottom: -15px;}</style><div class="embed-container"><small><a href="//www.arcgis.com/apps/Embed/index.html?webmap=703511eeac1642ec9fc46bd5186241f2&extent=34.3821,33.1522,37.4144,34.7717&home=true&zoom=true&scale=true&search=true&searchextent=true&disable_scroll=true&theme=light" style="color:#0000FF;text-align:left" target="_blank">View larger map</a></small><br><iframe width="500" height="400" frameborder="0" scrolling="no" marginheight="0" marginwidth="0" title="OLX2" src="//www.arcgis.com/apps/Embed/index.html?webmap=703511eeac1642ec9fc46bd5186241f2&extent=34.3821,33.1522,37.4144,34.7717&home=true&zoom=true&previewImage=false&scale=true&search=true&searchextent=true&disable_scroll=true&theme=light"></iframe></div>
    """, unsafe_allow_html=True)

# Define a function for the prediction page
def prediction_page():
    # Set the title
    st.title('House Price Predictions with Machine Learning Models')

    # Initialize the input features dictionary in session state if it doesn't exist
    if "input_features" not in st.session_state:
        st.session_state["input_features"] = {column: 0 for column in X.columns}

    # Create a number input widget for each input feature
    for column in X.columns:
        st.session_state["input_features"][column]= st.number_input(f"{column}:", value=st.session_state["input_features"][column])
    # Initialize the predictions dictionary in session state if it doesn't exist
    if "predictions" not in st.session_state:
        st.session_state["predictions"] = None

    # Create a button for generating predictions
    if st.button('Predict'):
        # Get the input data from the input features dictionary in session state and create a dataframe
        input_data = pd.DataFrame([st.session_state["input_features"]])

        # Generate predictions from each machine learning model using the fitted pipelines
        predictions = {}
        for model_name, pipeline in pipelines.items():
            y_pred = pipeline.predict(input_data)
            predictions[model_name] = y_pred[0]

        # Update the predictions dictionary in session state
        st.session_state["predictions"] = predictions

    # Display the predictions if they exist
    if st.session_state["predictions"]:
        st.subheader("Predictions:")
        for model_name, prediction in st.session_state["predictions"].items():
            st.write(f"{model_name}: {prediction:.2f}")

    # Create scatter plots for each machine learning model using the X_test and y_test data
    st.subheader("Scatter Plots:")
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, model in models.items():
        y_pred = pipelines[model_name].predict(X_test)
        ax.scatter(y_test, y_pred, label=model_name, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label="Ideal Line")
    ax.set_xlabel('True Prices')
    ax.set_ylabel('Predicted Prices')
    ax.legend(loc='best')
    ax.grid(True)

    # Display the scatter plots
    st.pyplot(fig)
    
import streamlit as st
from fontawesome import icons

import streamlit as st
import streamlit as st

# Define a function for the contact page
def contact_page():
    # Set the title
    st.title("Contact Us")

    # Add custom font style to the page using markdown
    st.markdown('<style>body { font-family: Arial; }</style>', unsafe_allow_html=True)
    
    # Add Font Awesome icons to the page using markdown
    fontawesome_cdn = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css"
    st.markdown(f'<link rel="stylesheet" href="{fontawesome_cdn}">', unsafe_allow_html=True)
    st.markdown('<i class="fas fa-envelope"></i> Email: chidiacnora@gmail.com', unsafe_allow_html=True)
    st.markdown('<i class="fas fa-phone"></i> Phone Number: +961 76 118 558', unsafe_allow_html=True)
    st.markdown('<i class="fas fa-map-marker-alt"></i> Address: Beirut, Lebanon, Street 2, Building 809', unsafe_allow_html=True)

# Set the page configuration to wide
st.set_page_config(layout="wide")

# Load an image and display it on the top right corner
image = Image.open("olx.png")
image_placeholder = st.empty()
image_placeholder.image(image, use_column_width=False, width=None, output_format='auto')

# Create a horizontal bar of tiles for each page
col1, col2, col3, col4 = st.columns(4)
with col1:
    home_tile = st.button("Home")
with col2:
    map_tile = st.button("Map")
with col3:
    prediction_tile = st.button("Predictions")
with col4:
    contact_tile = st.button("Contact Us")

# Initialize the current_page session state variable if it doesn't exist
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Home"

# Change the current page if a tile is clicked
if home_tile:
    st.session_state["current_page"] = "Home"
elif map_tile:
    st.session_state["current_page"] = "Map"
elif prediction_tile:
    st.session_state["current_page"] = "Predictions"
elif contact_tile:
    st.session_state["current_page"] = "Contact Us"

# Display the appropriate page based on the current_page session state variable
if st.session_state["current_page"] == "Home":
    home_page()
elif st.session_state["current_page"] == "Map":
    map_page()
elif st.session_state["current_page"] == "Predictions":
    prediction_page()
elif st.session_state["current_page"] == "Contact Us":
    contact_page()