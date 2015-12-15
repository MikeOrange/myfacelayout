from flask import Flask, url_for, render_template
app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    print url_for('static', filename='mustache.png')
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def picture():
    if request.method == 'POST':
        print "it's a post!"
    return 'Hello World!'

if __name__ == '__main__':
    app.run(debug=True)