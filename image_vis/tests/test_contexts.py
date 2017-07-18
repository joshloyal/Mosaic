import image_vis as ivs

from image_vis import contexts


def test_image_dir():
    assert contexts.get_image_dir() == ''
    assert contexts.get_image_col() == None

    with ivs.plotting_context(image_dir='test'):
        assert contexts.get_image_dir() == 'test'
        assert contexts.get_image_col() == None

    assert contexts.get_image_dir() == ''
    assert contexts.get_image_col() == None


def test_image_col():
    assert contexts.get_image_dir() == ''
    assert contexts.get_image_col() == None

    with ivs.plotting_context(image_col='test'):
        assert contexts.get_image_dir() == ''
        assert contexts.get_image_col() == 'test'

    assert contexts.get_image_dir() == ''
    assert contexts.get_image_col() == None


def test_multiple_context():
    assert contexts.get_image_dir() == ''
    assert contexts.get_image_col() == None

    with ivs.plotting_context(image_col='test_col', image_dir='test_dir'):
        assert contexts.get_image_dir() == 'test_dir'
        assert contexts.get_image_col() == 'test_col'

    assert contexts.get_image_dir() == ''
    assert contexts.get_image_col() == None
