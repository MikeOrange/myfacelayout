from flask import Flask, url_for, render_template, redirect, send_from_directory
from flask.ext.wtf import Form
from wtforms import SelectField, FileField
from wtforms.validators import DataRequired, regexp
import os
from werkzeug import secure_filename

app = Flask(__name__)
app.secret_key = 'Por la mega donde sea'

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

class FileUploadForm(Form):
    profession = SelectField('Profesion',
                             validators=[DataRequired()],
                             choices=[('', 'Seleccione'),
                                      ('1', 'Cocinero')])
    image = FileField(u'Image File', [DataRequired()])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def picture():
    form = FileUploadForm()

    if form.validate_on_submit() and allowed_file(form.image.data.filename):
        filename = secure_filename(form.image.data.filename)
        form.image.data.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('picture'))
    print form.errors
    return render_template('index.html', form=form)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')