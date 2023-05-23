# Import relevant libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from sklearn.feature_selection import SelectKBest, f_classif


 # Load dataset
data = pd.read_csv('data/winequality-red.csv')
# Check for missing values
data.isna().sum()
# Remove duplicate data
data.drop_duplicates(keep='first')
# Calculate the correlation matrix
corr_matrix = data.corr()
# Label quality into Good (1) and Bad (0)
data['quality'] = data['quality'].apply(lambda x: 1 if x >= 6.0 else 0)
    # Drop the target variable
X = data.drop('quality', axis=1)
# Set the target variable as the label
y = data['quality']
# == = = = = == 

# Select the top 8 features using SelectKBest
kbest = SelectKBest(score_func=f_classif, k=8)
X_new = kbest.fit_transform(X, y)

# Print the selected features
print(X.columns[kbest.get_support()])

# Reassign the selected features to X
X = X[X.columns[kbest.get_support()]]
# = = == = = =

# Split the dat a into training and testing sets (80% training, 40% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)
# Create an instance of the logistic regression model
logreg_model = LogisticRegression()
# Fit the model to the training data
logreg_model.fit(X_train, y_train)

# Predict the labels of the test set
# y_pred = logreg_model.predict(X_test)


# Create the Dash app
app = dash.Dash(__name__)
server = app.server

# Define the layout of the dashboard
app.layout = html.Div(
    children=[
    
    html.H1('CO544-2023 Lab 3: Wine Quality Prediction', style={'color': '#333', 'text-align': 'center', 'margin-bottom': '20px'}),
    
    html.Div([
        html.H3('Exploratory Data Analysis', style={'color': '#666', 'margin-bottom': '10px'}),
        html.Label('Feature 1 (X-axis)', style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='x_feature',
            options=[{'label': col, 'value': col} for col in data.columns],
            value=data.columns[0],
            style={'width': '100%', 'margin-bottom': '10px'}
        )
    ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
    
    html.Div([
        html.Label('Feature 2 (Y-axis)', style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='y_feature',
            options=[{'label': col, 'value': col} for col in data.columns],
            value=data.columns[1],
            style={'width': '100%', 'margin-bottom': '10px'}
        )
    ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
    
    dcc.Graph(id='correlation_plot'),
    
    # Wine quality prediction based on input feature values
    html.H3("Wine Quality Prediction", style={'color': '#666', 'margin-top': '40px'}),
    html.Div([
        html.Label("Fixed Acidity", style={'font-weight': 'bold'}),
        dcc.Input(id='fixed_acidity', type='number', required=True, style={'margin-bottom': '10px'}),    
        html.Label("Volatile Acidity", style={'font-weight': 'bold'}),
        dcc.Input(id='volatile_acidity', type='number', required=True, style={'margin-bottom': '10px'}), 
        html.Label("Citric Acid", style={'font-weight': 'bold'}),
        dcc.Input(id='citric_acid', type='number', required=True, style={'margin-bottom': '10px'}),
        html.Br(),
        
        html.Label("Residual Sugar", style={'font-weight': 'bold'}),
        dcc.Input(id='residual_sugar', type='number', required=True, style={'margin-bottom': '10px'}),  
        html.Label("Chlorides", style={'font-weight': 'bold'}),
        dcc.Input(id='chlorides', type='number', required=True, style={'margin-bottom': '10px'}), 
        html.Label("Free Sulfur Dioxide", style={'font-weight': 'bold'}),
        dcc.Input(id='free_sulfur_dioxide', type='number', required=True, style={'margin-bottom': '10px'}),
        html.Br(),
        
        html.Label("Total Sulfur Dioxide", style={'font-weight': 'bold'}),
        dcc.Input(id='total_sulfur_dioxide', type='number', required=True, style={'margin-bottom': '10px'}),
        html.Label("Density", style={'font-weight': 'bold'}),
        dcc.Input(id='density', type='number', required=True, style={'margin-bottom': '10px'}),
        html.Label("pH", style={'font-weight': 'bold'}),
        dcc.Input(id='ph', type='number', required=True, style={'margin-bottom': '10px'}),
        html.Br(),
        
        html.Label("Sulphates",style={'font-weight': 'bold'}),
        dcc.Input(id='sulphates', type='number', required=True, style={'margin-bottom': '10px'}),
        html.Label("Alcohol", style={'font-weight': 'bold'}),
        dcc.Input(id='alcohol', type='number', required=True, style={'margin-bottom': '10px'}),
        html.Br(),
    ]),

    html.Div([
        html.Button('Predict', id='predict-button', n_clicks=0, style={'background-color': '#4CAF50', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'text-align': 'center', 'text-decoration': 'none', 'display': 'inline-block', 'font-size': '16px', 'margin-bottom': '20px'}),
    ]),

    html.Div([
        html.H4("Predicted Quality", style={'color': '#666', 'margin-top': '20px'}),
        html.Div(id='prediction-output', style={'font-size': '20px', 'font-weight': 'bold'})
    ])
])


# Define the callback to update the correlation plot
@app.callback(
    dash.dependencies.Output('correlation_plot', 'figure'),
    [dash.dependencies.Input('x_feature', 'value'),
     dash.dependencies.Input('y_feature', 'value')]
)
def update_correlation_plot(x_feature, y_feature):
    fig = px.scatter(data, x=x_feature, y=y_feature, color='quality')
    fig.update_layout(title=f"Correlation between {x_feature} and {y_feature}")
    return fig

# Define the callback function to predict wine quality
@app.callback(
    Output(component_id='prediction-output', component_property='children'),
    [Input('predict-button', 'n_clicks')],
    [State('fixed_acidity', 'value'),
     State('volatile_acidity', 'value'),
     State('citric_acid', 'value'),
     State('residual_sugar', 'value'),
     State('chlorides', 'value'),
     State('free_sulfur_dioxide', 'value'),
     State('total_sulfur_dioxide', 'value'),
     State('density', 'value'),
     State('ph', 'value'),
     State('sulphates', 'value'),
     State('alcohol', 'value')]
)
def predict_quality(n_clicks, fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                     chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol):
    # Create input features array for prediction
    input_features = np.array([fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, 
                               free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol]).reshape(1, -1)

    # Predict the wine quality (0 = bad, 1 = good)
    prediction = logreg_model.predict(input_features)[0]

    # Return the prediction
    if prediction == 1:
        return 'This wine is predicted to be good quality.'
    else:
        return 'This wine is predicted to be bad quality.'


if __name__ == '__main__':
    app.run_server(debug=False)