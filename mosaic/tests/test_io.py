from mosaic import image_io


def test_load_image(rgb_image_data):
    image_dir, image_list = rgb_image_data
    image = image_io.load_image(image_list[0],
                                image_dir=image_dir,
                                as_image=True)


def test_load_images(rgb_image_data):
    image_dir, image_list = rgb_image_data
    images = image_io.load_images(image_list,
                                  image_dir=image_dir,
                                  as_image=True)

    assert len(images) == len(image_list)


def test_load_from_directory(rgb_image_data):
    image_dir, image_list = rgb_image_data
    images = image_io.load_from_directory(image_dir,
                                          as_image=True)

    assert len(images) == len(image_list)
