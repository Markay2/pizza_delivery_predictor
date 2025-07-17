# 🍕 Pizza Delivery Time Predictor

A machine learning web application that predicts pizza delivery duration based on various factors like pizza type, distance, traffic conditions, and more.

### Live Demo

**[View Live Application](https://pizzadeliverypredictor.streamlit.app/)**

## 📁 Project Structure


pizza-delivery-predictor/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── lgbm_tuned_model.pkl       # Trained LightGBM model
├── label_encoders.pkl         # Label encoders for categorical features
├── README.md                  # This file
└── .gitignore                 # Git ignore file


### Features

- **Interactive Web Interface**: User-friendly form to input delivery parameters
- **Real-time Predictions**: Instant delivery time estimates using trained ML model
- **Comprehensive Inputs**: Considers 16 different factors affecting delivery time
- **Smart Encoding**: Handles categorical variables with proper label encoding
- **Responsive Design**: Works on desktop and mobile devices

### Technologies Used

- **Streamlit**: Web application framework
- **LightGBM**: Machine learning model for predictions
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning utilities
- **Joblib**: Model serialization

### Model Features

The model uses the following features to predict delivery time:

1. **Pizza Details**: Size, Type, Toppings Count, Complexity
2. **Location**: Distance (km), Traffic Level, Traffic Impact
3. **Timing**: Order Hour, Order Month, Order Day, Peak Hour, Weekend
4. **Payment**: Payment Method, Payment Category
5. **Restaurant**: Restaurant Average Time
6. **Derived**: Topping Density

### How to Use

1. Fill out the form with your pizza order details
2. Click "Predict Delivery Time" to get an estimate
3. View the estimated delivery time in minutes
4. Get additional insights about delivery speed

### Live Demo

https://pizzadeliverypredictor.streamlit.app/

### Local Development

## Prerequisites

- Python 3.8+
- pip

## Installation

1. Clone the repository:
  bash
git clone https://github.com/Markay2/pizza-delivery-predictor.git
cd pizza-delivery-predictor


2. Install dependencies:
  bash
pip install -r requirements.txt


3. Run the application:
  bash
streamlit run app.py


4. Open your browser and go to `http://localhost:8501`

### Deployment

This app is deployed on Streamlit Cloud. To deploy your own version:

1. Fork this repository
2. Sign up at [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select your forked repository
5. Choose `app.py` as your main file
6. Click "Deploy"

### Requirements

See `requirements.txt` for a complete list of dependencies.

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


### Acknowledgments

- Thanks to the open-source community for the amazing tools
- Dataset and model training inspiration from pizza delivery optimization research

### Contact

Mark Ata Adu - mark-ata.adu@epita.fr
