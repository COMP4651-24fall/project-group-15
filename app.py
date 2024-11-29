from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import random
import requests  # For making HTTP requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv

# Define your model class
class ModelClass(nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        self.fc1 = nn.Linear(in_features=10, out_features=50)  # Adjust input/output features
        self.fc2 = nn.Linear(in_features=50, out_features=20)
        self.fc3 = nn.Linear(in_features=20, out_features=6)  # Adjust for your output size

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation for the output layer 
        return x

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

def load_model(model_data):
    model = ModelClass()
    
    # Load the state dictionary into the model
    model.load_state_dict({k: torch.tensor(np.array(v)) for k, v in model_data.items()})
    
    model.eval()  # Set the model to evaluation mode
    return model

# Load items from CSV file
def load_items():
    df = pd.read_csv('items.csv')
    items = df.to_dict(orient='records')  # Convert DataFrame to a list of dictionaries
    return items

# Load user ratings from dummy_data.csv
def load_ratings():
    df = pd.read_csv('dummy_data.csv')
    ratings = df.to_dict(orient='records')  # Convert DataFrame to a list of dictionaries
    return ratings

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    user_list = list(pd.read_csv("dummy_data.csv")["user_id"].unique())
    user_id = session.get('user_id')

    # if not user_id:
    #    return redirect(url_for('index'))  # Redirect if user is not logged in

    # uncomment them when the http request is avaliable
    """
    lambda_url = 'https://sample.HTTP'  # Update accordingly
    response = requests.post(lambda_url)
    response_state = response.status_code
    """ 
    response_state = 200  # remove this when http request is avaliable
    if response_state == 200:
        """
        model_data = response.json()  # Get the model data from the Lambda response
        
        # Load the model with the retrieved data
        model = load_model(model_data)

        # Prepare input data for the model (modify according to your specific needs)
        input_data = torch.tensor([user_id], dtype=torch.float32)  # Adjust based on actual input shape

        # Run inference
        with torch.no_grad():
            recommendations = model(input_data) # Inference

        # Process recommendations (convert to a list)
        recommendations_list = recommendations.numpy().tolist()  # Adjust as necessary
        """
        # Assume the model can return array of product_ids sorted from most relevant to least relevant
        recommendations_list = np.array([5,2,3,1,7])[:3] 
        # Fetch item details based on recommendations
        selected_items = fetch_items_by_recommendations(recommendations_list)
        
        return render_template('recommendation.html', selected_items=selected_items)
    else:
        flash('Failed to get model from Lambda, please try again later.', 'error')
        return redirect(url_for('personal'))  # Redirect back to the personal page

@app.route('/login', methods=['POST'])
def login():
    # Load valid client IDs from clientlist.csv
    try:
        user_list = pd.read_csv("clientlist.csv")["clientid"].unique().tolist()
    except Exception as e:
        flash('Error loading client list. Please try again later.', 'error')
        return redirect(url_for('index'))

    user_id = request.form.get('user_id')
    if user_id:
        try:
            user_id = int(user_id)
            if user_id not in user_list:
                flash(f'Please enter a valid USER ID (User ID {user_id} does not exist!).', 'error')
                return redirect(url_for('index'))
            session['user_id'] = user_id
            return redirect(url_for('personal'))  # Redirect to personal.html
        except ValueError:
            flash('Please enter a valid user ID (non-negative integer only).', 'error')
            return redirect(url_for('index'))
    return redirect(url_for('index'))


@app.route('/contribute')
def contribute():
    # Load items from items.csv
    items = pd.read_csv('items.csv').to_dict(orient='records')  # Load items if needed
    # Load client data from clientdata.csv
    client_data = pd.read_csv('clientdata.csv').to_dict(orient='records')
    return render_template('contribute.html', items=items, client_data=client_data)

@app.route('/home')
def home():
    session.pop('user_id', None)  # Clear the user_id from the session
    return redirect(url_for('index'))

@app.route('/personal')
def personal():
    all_items = load_items()  # Load all items from the dataset
    featured_items = random.sample(all_items, min(3, len(all_items)))  # Select 3 random items
    return render_template('personal.html', featured_items=featured_items)


def fetch_items_by_recommendations(recommendations):
    items = load_items()  # Load all items
    selected_items = [item for item in items if item['item_id'] in recommendations]
    return selected_items


@app.route('/submit_contribution', methods=['POST'])
def submit_contribution():
    user_ids = request.form.getlist('user_id[]')
    event_times = request.form.getlist('event_time[]')
    event_types = request.form.getlist('event_type[]')
    product_ids = request.form.getlist('product_id[]')
    category_ids = request.form.getlist('category_id[]')
    category_codes = request.form.getlist('category_code[]')
    brands = request.form.getlist('brand[]')
    prices = request.form.getlist('price[]')
    user_sessions = request.form.getlist('user_session[]')

    contributions = []
    for user_id, event_time, event_type, product_id, category_id, category_code, brand, price, user_session in zip(
            user_ids, event_times, event_types, product_ids, category_ids, category_codes, brands, prices, user_sessions):
        contributions.append({
            'user_id': int(user_id),  # Ensure user_id is an integer
            'event_time': event_time.strip(), 
            'event_type': event_type.strip(), 
            'product_id': int(product_id),  # Convert to integer
            'category_id': int(category_id),  # Convert to integer
            'category_code': category_code.strip(), 
            'brand': brand.strip(),  
            'price': float(price),  
            'user_session': int(user_session)  # Ensure user_session is an integer
        })

    # Append contributions to dummy_data.csv
    with open('dummy_data.csv', 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=contributions[0].keys())
        if f.tell() == 0:  # Check if file is empty and write header
            writer.writeheader()
        writer.writerows(contributions)

    # Remove the submitted data from clientdata.csv
    try:
        existing_data = pd.read_csv('clientdata.csv')

        # Print the types of the existing data
        print("Existing data types:")
        print(existing_data.dtypes)

        # Ensure correct data types in existing_data
        existing_data['user_id'] = existing_data['user_id'].astype(int)
        existing_data['price'] = existing_data['price'].astype(float)

        # Create a DataFrame for the submitted contributions
        contributions_df = pd.DataFrame(contributions)

        # Create a mask to filter out contributions
        mask = pd.Series([False] * len(existing_data))

        for _, contribution in contributions_df.iterrows():
            print(f"Checking contribution: {contribution.to_dict()}")

            current_mask = (
                (existing_data['user_id'] == contribution['user_id']) &
                (existing_data['event_time'] == contribution['event_time']) &
                (existing_data['event_type'] == contribution['event_type']) &
                (existing_data['product_id'] == contribution['product_id']) &
                (existing_data['category_id'] == contribution['category_id']) &
                (existing_data['category_code'] == contribution['category_code']) &
                (existing_data['brand'] == contribution['brand']) &
                (existing_data['price'] == contribution['price']) &
                (existing_data['user_session'] == contribution['user_session'])
            )

            # Debugging prints
            print(f"Current mask for this contribution: {current_mask.values}")

            mask |= current_mask

        # Filter the existing data to keep rows not in the contributions
        updated_data = existing_data[~mask]

        # Write the updated data back to clientdata.csv
        updated_data.to_csv('clientdata.csv', index=False)

    except Exception as e:
        flash(f'Error processing contributions: {e}', 'error')

    flash('Contributions submitted successfully!', 'success')
    return redirect(url_for('personal'))  # Redirect back to the personal dashboard

if __name__ == '__main__':
    app.run(debug=True)