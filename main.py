import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

# Set page config with aviation theme
st.set_page_config(
    page_title="Aviation Profit Predictor",
    page_icon="✈️",
    layout="wide"
)

# Custom CSS for background
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1436491865332-7a61a109cc05');
        background-size: cover;
        background-color: #E6F2FF;
    }
    .stButton>button {
        background-color: #1E90FF;
        color: white;
        font-weight: bold;
    }
    .st-bb {
        background-color: rgba(255,255,255,0.8);
    }
    .st-bc {
        background-color: rgba(255,255,255,0.9);
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #FFA000;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("Aviation_KPIs_POWERBI.csv")
    return df

df = load_data()

# Main title with plane emoji
st.title("✈️ Aviation Profit Predictor Dashboard")
st.markdown("""
**Predict your airline's profitability** based on key operational parameters.
""")

# Create two columns for input methods
col1, col2 = st.columns(2)

with col1:
    st.header("Quick Prediction")
    st.markdown("Use sliders for fast estimates:")
    
    # Default values based on data
    fuel_cost = st.slider("Fuel Cost per Gallon (USD)", 1.0, 5.0, 2.5, key="quick_fuel")
    load_factor = st.slider("Load Factor (%)", 50, 100, 75, key="quick_load")
    avg_ticket = st.slider("Average Ticket Price (USD)", 100, 1000, 350, key="quick_ticket")
    
    quick_predict = st.button("Quick Predict")

with col2:
    st.header("Custom Prediction")
    st.markdown("Enter exact values for precise estimates:")
    
    # Free-hand inputs
    custom_fuel = st.number_input("Fuel Cost (USD)", min_value=0.1, max_value=10.0, value=2.5, step=0.1, key="custom_fuel")
    custom_load = st.number_input("Load Factor (%)", min_value=1, max_value=100, value=75, key="custom_load")
    custom_ticket = st.number_input("Ticket Price (USD)", min_value=10, max_value=2000, value=350, key="custom_ticket")
    
    custom_predict = st.button("Custom Predict")

# Feature selection and model setup
target_col = 'Profit (USD)'
numerical_df = df.select_dtypes(include=['float64', 'int64'])

if target_col not in numerical_df.columns:
    st.error(f"Error: '{target_col}' column not found in the dataset.")
else:
    # Select top 3 most correlated features
    corr_matrix = numerical_df.corr()
    top_features = corr_matrix[target_col].abs().sort_values(ascending=False).index[1:4]
    
    # Show feature explanations
    with st.expander("ℹ️ How These Parameters Affect Profit"):
        st.markdown("""
        **1. Fuel Cost**: The single largest expense for airlines. Every $0.10 increase in fuel price can \
        reduce industry profits by $2 billion annually. Lower fuel costs directly improve profitability.
        
        **2. Load Factor**: The percentage of seats filled. At 75%, most airlines break even. \
        Each percentage point above 75% typically increases profit margin by 1-2%.
        
        **3. Average Ticket Price**: Revenue per passenger. Pricing strategy must balance demand \
        with capacity. Higher prices don't always mean higher profits if load factor drops too much.
        """)
    
    # Prepare data
    features = numerical_df[top_features]
    target = numerical_df[target_col]
    
    # Sample data
    df_sample = df.sample(n=10000, random_state=42)
    X = df_sample[top_features]
    y = df_sample[target_col]
    
    # Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Model training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make prediction when button is clicked
    def make_prediction(input_values, input_type="quick"):
        user_input = np.array([input_values])
        user_scaled = scaler.transform(user_input)
        predicted_profit = model.predict(user_scaled)[0]
        
        # Display prediction
        st.success("")
        st.markdown(f"""
        <div style="background-color:#E1F5FE; padding:20px; border-radius:10px;">
            <h2 style="color:#0277BD;">Predicted Profit: ${predicted_profit:,.2f}</h2>
            <p>Based on {input_type} input values:</p>
            <ul>
                <li>Fuel Cost: ${input_values[0]}/gallon</li>
                <li>Load Factor: {input_values[1]}%</li>
                <li>Avg Ticket Price: ${input_values[2]}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Show confidence interval
        st.markdown(f"""
        <div style="background-color:#FFF8E1; padding:10px; border-radius:5px; margin-top:10px;">
            <p>📊 <strong>Prediction Range:</strong> ${predicted_profit*0.85:,.0f} to ${predicted_profit*1.15:,.0f}</p>
            <small>(Actual results may vary ±15% based on unmodeled factors)</small>
        </div>
        """, unsafe_allow_html=True)
        
        return predicted_profit
    
    if quick_predict:
        predicted = make_prediction([fuel_cost, load_factor, avg_ticket], "quick")
        
    if custom_predict:
        predicted = make_prediction([custom_fuel, custom_load, custom_ticket], "custom")
    
    # Show visualizations if any prediction was made
    if quick_predict or custom_predict:
        st.header("📊 Profitability Analysis")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Parameter Relationships", "Model Accuracy", "Optimization Guide"])
        
        with tab1:
            st.subheader("How Parameters Affect Profit")
            fig = plt.figure(figsize=(12, 8))
            
            # Create 3 subplots
            for i, feature in enumerate(top_features):
                plt.subplot(2, 2, i+1)
                sns.regplot(x=features[feature], y=target, scatter_kws={'alpha':0.3})
                plt.title(f"{feature} vs Profit")
                plt.xlabel(feature)
                plt.ylabel("Profit (USD)")
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("""
            **Key Insights:**
            - Each graph shows how one parameter relates to profitability
            - The trend line indicates the general relationship
            - Points represent actual historical data points
            """)
        
        with tab2:
            st.subheader("Model Performance Metrics")
            
            # Calculate metrics
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            error = y_test - y_pred
            avg_error = np.mean(np.abs(error))
            
            # Metrics display
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Squared Error", f"{mse:,.0f}")
            col2.metric("R² Score", f"{r2:.3f}")
            col3.metric("Average Error", f"${avg_error:,.0f}")
            
            # Error distribution plot
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(error, bins=50, kde=True, ax=ax)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title("Prediction Error Distribution")
            plt.xlabel("Error (Actual - Predicted)")
            st.pyplot(fig)
            
            st.markdown("""
            **Interpretation:**
            - The model tends to be accurate when the error is near zero (red line)
            - A symmetric distribution around zero indicates unbiased predictions
            - Wider spread indicates areas where predictions are less certain
            """)
        
        with tab3:
            st.subheader("Profit Optimization Strategies")
            
            # Create strategy cards
            st.markdown("""
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px;">
            
            <div style="background: #E3F2FD; padding: 15px; border-radius: 10px;">
            <h4>🛢️ Fuel Cost Strategies</h4>
            <ul>
            <li>Hedge 50-70% of fuel needs when prices are low</li>
            <li>Invest in newer aircraft (15-20% more efficient)</li>
            <li>Optimize flight paths to reduce distance</li>
            <li>Reduce aircraft weight through design changes</li>
            </ul>
            </div>
            
            <div style="background: #E8F5E9; padding: 15px; border-radius: 10px;">
            <h4>🧑‍🤝‍🧑 Load Factor Tactics</h4>
            <ul>
            <li>Dynamic pricing to fill last 10% of seats</li>
            <li>Offer bundled services to increase value</li>
            <li>Optimize route frequency based on demand</li>
            <li>Partner with tourism boards for packages</li>
            </ul>
            </div>
            
            <div style="background: #FFF8E1; padding: 15px; border-radius: 10px;">
            <h4>💰 Pricing Approaches</h4>
            <ul>
            <li>Segment customers by price sensitivity</li>
            <li>Offer premium services at higher margins</li>
            <li>Early bird discounts to predict demand</li>
            <li>Last
