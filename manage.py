#!/usr/bin/env python
# -*- coding: utf-8 -*-

from faceoverlay import ChristmasPromo, CookPromo, FashionPromo, HairdresserPromo
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


THEME_CHOICES = [('', 'Select a filter'),
                 ('1', 'Cook'),
                 ('2', 'Fashion'),
                 ('3', 'Stylist'),
                 ('4', 'Merry Christmas')]


@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)


def dated_url_for(endpoint, **values):
    if endpoint == 'uploaded_file':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     "uploads", filename)
            values['q'] = int(os.stat(file_path).st_mtime)

    return url_for(endpoint, **values)


class FileUploadForm(Form):
    profession = SelectField('Profesion',
                             validators=[DataRequired()],
                             choices=THEME_CHOICES)
    image = FileField(u'Image File', [DataRequired()])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def overlay_picture(pre_filename, promo):
    filename = 'uploads/' + pre_filename
    my_image = None

    if promo == 'Cook':
        my_image = CookPromo(filename)
    elif promo == 'Fashion':
        my_image = FashionPromo(filename)
    elif promo == 'Stylist':
        my_image = HairdresserPromo(filename)
    elif promo == 'Merry Christmas':
        my_image = ChristmasPromo(filename)
    else:
        return None

    my_image.put_hat()
    my_image.put_message()
    my_image.background.save_image()


@app.route('/', methods=['GET', 'POST'])
def picture():
    form = FileUploadForm()

    if form.validate_on_submit() and allowed_file(form.image.data.filename):
        filename = secure_filename(form.image.data.filename)
        form.image.data.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        overlay_picture(filename,
                        dict(THEME_CHOICES).get(form.profession.data))
        return redirect(url_for('specific_picture', filename=filename))

    return render_template('index.html', form=form, picture=None)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/picture/<filename>')
def specific_picture(filename):
    form = FileUploadForm()
    return render_template('index.html', form=form, picture=filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')