import image_vis


def test_image_dir():
    assert image_vis.get_image_dir() == ''

    with image_vis.image_dir('test'):
        assert image_vis.get_image_dir() == 'test'

    assert image_vis.get_image_dir() == ''
