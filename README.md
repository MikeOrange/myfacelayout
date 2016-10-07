# MyFaceOverlay

Webpage to demonstrate the use of OpenCV with Python to put images over selected regions of pictures based on face detection.
Code based on the tutorials and examples provided by [OpenCV docs](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) and the great example at [Sublime Robots](http://sublimerobots.com/2015/02/dancing-mustaches/)


### Install and run
 - Install and activate virtualenv
 - Install OpenCV on the virtualenv. Use this very detailed [tutorial](http://manuganji.com/deployment/install-opencv-numpy-scipy-virtualenv-ubuntu-server/) as a guide if you are working in Ubuntu.
 - Install backend requirements on the virtualenv
```sh
$ pip install -r requirements.txt
```

 - Run project
 - Run project
```sh
$ python runserver.py
```

### Environment variables
```sh
export MYFACEOVERLAY_SECRET_KEY=Random secret key for csrf usage
export MYFACEOVERLAY_UPLOADS_FOLDER=Absolute path of the uploads folder to be used (preferably outside the projects root)
```

### Links
Template based on [freelancer by startbootstrap](https://startbootstrap.com/template-overviews/freelancer/)

### Preview
![](res/example.png?raw=true)