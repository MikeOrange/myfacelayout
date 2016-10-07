#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask
import os

app = Flask(__name__)
app.secret_key = os.environ.get('MYFACEOVERLAY_SECRET_KEY')


UPLOAD_FOLDER = os.environ.get('MYFACEOVERLAY_UPLOADS_FOLDER')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

import myfacelayout.views



