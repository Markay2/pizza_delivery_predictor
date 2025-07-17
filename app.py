import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Streamlit app setup
try:
    st.set_page_config(page_title="Delivery Time Predictor", layout="centered")
except st.errors.StreamlitAPIException:
    pass  # Already configured

st.title("🍕 Pizza Delivery Duration Predictor")
st.markdown("Fill out the form to estimate delivery time (in minutes).")

# Load model and encoders
@st.cache_resource
def load_model_and_encoders():
    try:
        model = joblib.load('lgbm_tuned_model.pkl')
        encoders = joblib.load('label_encoders.pkl')
        return model, encoders
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Please ensure 'lgbm_tuned_model.pkl' and 'label_encoders.pkl' are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load the model and encoders
model, encoders = load_model_and_encoders()

# Define the exact feature list used during training
feature_names = [
    'Pizza Size', 'Pizza Type', 'Toppings Count', 'Distance (km)', 'Traffic Level',
    'Payment Method', 'Is_Peak_Hour', 'Is Weekend', 'Topping Density', 'Order Month',
    'Payment Category', 'Pizza Complexity', 'Traffic Impact', 'Order Hour',
    'Restaurant Avg Time', 'order_day'
]

# Preprocessing function
def preprocess_input(input_df):
    """
    Preprocess input data by applying label encoders to categorical columns
    """
    processed_df = input_df.copy()
    
    # Handle columns with encoders
    for col in encoders.keys():
        if col in processed_df.columns:
            try:
                value = processed_df[col].iloc[0]
                value_str = str(value)
                
                if value_str in encoders[col].classes_:
                    processed_df[col] = encoders[col].transform([value_str])[0]
                else:
                    # Handle unknown categories - use the first class as default
                    st.warning(f"Unknown value '{value_str}' for {col}. Using default value.")
                    processed_df[col] = encoders[col].transform([encoders[col].classes_[0]])[0]
                    
            except Exception as e:
                st.error(f"Error encoding column {col}: {str(e)}")
                return None
    
    # Handle columns without encoders (if any)
    if 'Pizza Complexity' in processed_df.columns and 'Pizza Complexity' not in encoders:
        complexity_map = {'Low': 0, 'Medium': 1, 'High': 2}
        processed_df['Pizza Complexity'] = complexity_map.get(processed_df['Pizza Complexity'].iloc[0], 1)
    
    if 'order_day' in processed_df.columns and 'order_day' not in encoders:
        day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                   'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        processed_df['order_day'] = day_map.get(processed_df['order_day'].iloc[0], 0)
    
    return processed_df

# Get options from encoders or use defaults
def get_encoder_options(column_name, default_options):
    if column_name in encoders:
        return list(encoders[column_name].classes_)
    return default_options

# Input form
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        pizza_size = st.selectbox("Pizza Size", 
                                get_encoder_options('Pizza Size', ['Large', 'Medium', 'Small', 'XL']))
        
        pizza_type = st.selectbox("Pizza Type", 
                                get_encoder_options('Pizza Type', ['BBQ Chicken', 'Cheese Burst', 'Deep Dish', 'Gluten-Free', 'Margarita', 'Non-Veg', 'Sicilian', 'Stuffed Crust', 'Thai Chicken', 'Thin Crust', 'Veg', 'Vegan']))
        
        toppings_count = st.slider("Toppings Count", 0, 10, 2)
        
        distance = st.number_input("Distance (km)", min_value=0.0, value=3.0, step=0.1)
        
        traffic_level = st.selectbox("Traffic Level", 
                                   get_encoder_options('Traffic Level', ['High', 'Low', 'Medium']))
        
        payment_method = st.selectbox("Payment Method", 
                                    get_encoder_options('Payment Method', ['Card', 'Cash', "Domino's Cash", 'Hut Points', 'UPI', 'Wallet']))
        
        is_peak = st.selectbox("Is Peak Hour?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
        is_weekend = st.selectbox("Is Weekend?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    with col2:
        topping_density = st.number_input("Topping Density", min_value=0.0, value=1.5, step=0.1)
        
        order_month = st.selectbox("Order Month", 
                                 get_encoder_options('Order Month', ['April', 'August', 'December', 'February', 'January', 'July', 'June', 'March', 'May', 'November', 'October', 'September']))
        
        payment_category = st.selectbox("Payment Category", 
                                      get_encoder_options('Payment Category', ['Offline', 'Online']))
        
        pizza_complexity = st.selectbox("Pizza Complexity", 
                                      get_encoder_options('Pizza Complexity', ['Low', 'Medium', 'High']))
        
        traffic_impact = st.slider("Traffic Impact", 0, 5, 2)
        
        order_hour = st.slider("Order Hour", 0, 23, 12)
        
        restaurant_avg_time = st.number_input("Restaurant Avg Time (min)", min_value=1.0, value=15.0, step=0.5)
        
        order_day = st.selectbox("Order Day", 
                               get_encoder_options('order_day', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']))
    
    submitted = st.form_submit_button("🔮 Predict Delivery Time", use_container_width=True)

# Prediction
if submitted:
    try:
        # Create DataFrame from inputs
        input_data = pd.DataFrame([{
            'Pizza Size': pizza_size,
            'Pizza Type': pizza_type,
            'Toppings Count': toppings_count,
            'Distance (km)': distance,
            'Traffic Level': traffic_level,
            'Payment Method': payment_method,
            'Is_Peak_Hour': is_peak,
            'Is Weekend': is_weekend,
            'Topping Density': topping_density,
            'Order Month': order_month,
            'Payment Category': payment_category,
            'Pizza Complexity': pizza_complexity,
            'Traffic Impact': traffic_impact,
            'Order Hour': order_hour,
            'Restaurant Avg Time': restaurant_avg_time,
            'order_day': order_day
        }])
        
        # Ensure columns are in the correct order
        input_data = input_data[feature_names]
        
        # Preprocess the input
        processed_input = preprocess_input(input_data)
        
        if processed_input is not None:
            # Make prediction
            prediction = model.predict(processed_input)[0]
            
            # Display result
            st.success(f"🕒 **Estimated Delivery Duration: {prediction:.1f} minutes**")
            
            # Additional insights
            if prediction < 20:
                st.info("⚡ Fast delivery expected!")
            elif prediction < 35:
                st.info("🚚 Normal delivery time")
            else:
                st.warning("⏰ Longer delivery time expected")
                
        else:
            st.error("❌ Failed to preprocess input data")
            
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")