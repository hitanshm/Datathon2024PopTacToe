from flask import Flask, request, render_template, redirect, url_for
import os
from createUnique import create_unique_csv_from_order

app = Flask(__name__)

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    # Process the file to create unique orders CSV
    create_unique_csv_from_order(file_path)
    
    return redirect(url_for('orders', month=file.filename.split('_')[0]))

@app.route('/orders/<month>')
def orders(month):
    # Logic to read the generated unique orders CSV and prepare data for the graph
    unique_orders_file = f"{month}_unique_orders.csv"
    orders_data = []
    
    # Read the unique orders CSV
    with open(unique_orders_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            orders_data.append(row)
    
    return render_template('orders.html', month=month, orders=orders_data)

if __name__ == '__main__':
    app.run(debug=True)