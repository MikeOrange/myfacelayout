#!/usr/bin/env python
from flask.ext.wtf import Form
from wtforms import SelectField, FileField
from wtforms.validators import DataRequired

THEME_CHOICES = [('', 'Select a filter'),
                 ('1', 'Cook'),
                 ('2', 'Fashion'),
                 ('3', 'Stylist'),
                 ('4', 'Merry Christmas')]


class FileUploadForm(Form):
    profession = SelectField('Profession',
                             validators=[DataRequired()],
                             choices=THEME_CHOICES)
    image = FileField(u'Image File', [DataRequired()])