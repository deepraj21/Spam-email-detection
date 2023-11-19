from flask import Flask, render_template, url_for, request, redirect, url_for, session
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask_sqlalchemy import SQLAlchemy
from flask import Flask

app=Flask(__name__)
app.secret_key = 'MYSECRETKEY'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

def create_tables():
    with app.app_context():
        db.create_all()

# @app.route('/home')
# def home():
#     return render_template('home.html')

@app.route('/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('Signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['user_id'] = user.id
            return redirect(url_for('home'))
    return render_template('Login.html')
   
@app.route('/home', methods=['GET', 'POST'])
def home():
    
    my_prediction = None
    #Loading the data from csv file to a pandas Dataframe
    raw_mail_data = pd.read_csv('mail_data.csv')
        
    #Replace the null values with a null string
    mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')
    
    #Label spam mail as 0;  ham mail as 1;
    mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
    mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1
    
    #Separating the data as texts and label
    #X-input
    #Y-Output/target
    X = mail_data['Message']
    Y = mail_data['Category']
    
    #Splitting the data into training data & test data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
    
    #Transform the text data to feature vectors that can be used as input to the Logistic regression
    feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)

    #Splited X has string values, those need to be fit & converted to integer
    X_train_features = feature_extraction.fit_transform(X_train)
    X_test_features = feature_extraction.transform(X_test)

    #Convert Y_train and Y_test values as integers [convert object type to int]
    Y_train = Y_train.astype('int')
    Y_test = Y_test.astype('int')
    
    #Training the Model
    model = LogisticRegression()

    #Training the Logistic Regression model with the training data
    model.fit(X_train_features, Y_train)

    #Prediction on training data
    prediction_on_training_data = model.predict(X_train_features)
    accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

    #Prediction on test data
    prediction_on_test_data = model.predict(X_test_features)
    accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

    if request.method=='POST':
        comment=request.form['comment']
        data=[comment]
            
        #Convert text to feature vectors
        input_data_features = feature_extraction.transform(data).toarray()
        
        #Making prediction
        my_prediction = model.predict(input_data_features)

    return render_template('home.html', prediction=my_prediction)

if __name__== '__main__':
    create_tables()
    app.run(debug=True)
