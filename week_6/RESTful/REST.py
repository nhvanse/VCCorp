from flask import Flask, render_template, request, jsonify, json
from flask_restful import Api, Resource
import findContent

app = Flask(__name__)

@app.route('/')
def student():
   return render_template('input.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
      form = request.form
      text = form['TEXT']
      result = findContent.find(text)
      return render_template("result.html",result = result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
