from flask import Flask, request, jsonify, render_template
from mrac_qa_v1 import MRAC_QA

bot = None

app = Flask(__name__)

bot = MRAC_QA()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    queries = [str(x) for x in request.form.values()]
    q = queries[0]
    n = int(queries[1])
    numcontext = int(queries[2])
    print(q, n, numcontext)
    prediction, context = bot.query(q, n, numcontext)

    return render_template('index.html', prediction_text=prediction, context=context)

@app.route('/load_docs',methods=['POST'])
def load_docs():
    queries = [str(x) for x in request.form.values()]
    q = queries[0]
    n = 5
    numcontext = 1
    print(q, n, numcontext)
    prediction, context = bot.query(q, n, numcontext)

    return render_template('documents.html', prediction_text=prediction, context=context)

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    n = int(request.args.get('num'))
    q = str(userText)
    numcontext = 1
    print(q, n, numcontext)
    prediction, context = bot.query(q, n, numcontext)
    result = '<ol style="padding-left:14px;">'
    wgtint = 900
    for i in range(n):
        if (i == n - 1):
            wgt  = str(wgtint) + '"'
            html = '<li style="font-weight:' + str(wgt) + ';>'
            result = result + html + prediction[i].capitalize() + '</li>'
            break
        wgt = str(wgtint) + '"'
        html = '<li style="font-weight:' + str(wgt) + ';>'
        result = result + html + prediction[i].capitalize() + '</li><br>'
        wgtint = wgtint - 200

    result = result + '</ol>'

    return str(result)

@app.route('/retrieve',methods=['POST'])
def retrieve():
    queries = [str(x) for x in request.form.values()]
    q = queries[0]
    n = int(queries[1])
    print(q, n)
    doc = bot.doc_retrieve(q, n)

    return render_template('index.html', document=doc)

if __name__ == "__main__":
    app.run(debug=True)
