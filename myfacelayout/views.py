# -*- coding: utf-8 -*-
import os
import string, random
from flask import url_for, render_template, redirect, send_from_directory
from myfacelayout import app
from myfacelayout.faceoverlay import (ChristmasPromo, CookPromo, FashionPromo,
                                      HairdresserPromo)
from myfacelayout.forms import FileUploadForm, THEME_CHOICES

ALLOWED_TYPES = {'png': 'image/png',
                 'jpg': 'image/jpeg',
                 'jpeg': 'image/jpeg'}


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


def extract_file_extension(filename):
    """
    Returns the extension used for the file
    :param filename: String containing a filename (i.e. file.txt)
    :return: String containing file extension (i.e. 'txt')
    """
    return filename.rsplit('.', 1)[1]


def is_file_allowed(checked_file):
    """
    Takes a file object and checks if its safe to save it on
    our filesystem and process it
    :param checked_file: uploaded file by the user
    :return: boolean value, True if the file can be saved, False otherwise
    """
    # check if file has extension
    if '.' not in checked_file.filename:
        return False

    # Check if file extension is in whitelist
    file_extension = extract_file_extension(checked_file.filename)
    if file_extension not in ALLOWED_TYPES:
        return False

    # Check if file has valid MIME type
    if checked_file.mimetype != ALLOWED_TYPES[file_extension]:
        return False

    return True


def overlay_picture(filename, promo):
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


def random_filename(extension):
    """
    Generates a random filename of 8 chars for our file
    :param extension: extension to be used by the file (i.e. jpg)
    :return: random name including extension
    """
    return ''.join(random.sample(string.ascii_lowercase, 8)) + '.' + extension


@app.route('/', methods=['GET', 'POST'])
def picture():
    form = FileUploadForm()
    if form.validate_on_submit() and is_file_allowed(form.image.data):
        filename = random_filename(extract_file_extension(form.image.data.filename))
        absolute_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        form.image.data.save(absolute_filename)
        overlay_picture(absolute_filename,
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