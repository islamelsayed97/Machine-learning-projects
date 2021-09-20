from flask import Flask, request, render_template
from IndustryClassification import IndustryClassification

app = Flask(__name__)
industry  = IndustryClassification()
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/train')
def train():
    industry.train()
    return render_template('result.html', message="Model was Trained Successfully")

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method=='GET':
        return render_template('predict.html')
    else:
        text = request.form['text']
        return render_template('result.html', message=industry.predict(text))

@app.route('/evaluate')
def evaluate():
    return render_template('result.html', message="F1 score is "+str(industry.evaluate()))




app.run(debug=True)