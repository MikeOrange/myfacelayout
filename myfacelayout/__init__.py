#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask


app = Flask(__name__)
app.secret_key = 'Por la mega donde sea'


UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

import myfacelayout.views



