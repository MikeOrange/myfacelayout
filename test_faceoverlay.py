from faceoverlay import ChristmasImage, ChristmasPromo


if __name__ == '__main__':
     #my_image = ChristmasImage("larger_image.jpg")
     #my_image.get_complete_image()
     #my_image.show_original_image()
     my_image = ChristmasPromo("larger_image.jpg")
     my_image.put_hat()
     my_image.background.show()