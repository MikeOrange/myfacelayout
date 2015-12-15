from faceoverlay import ChristmasPromo, CookPromo, FashionPromo, HairdresserPromo


if __name__ == '__main__':
     #my_image = ChristmasPromo("ninho.jpg")
     my_image = HairdresserPromo("yomesmo.jpg")
     my_image.put_hat()
     my_image.put_scissors()
     my_image.put_message()
     my_image.background.show()