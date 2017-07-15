class ColorFeatures(object):
    HUE = '__hue__'
    SATURATION = '__saturation__'
    VALUE = '__value__'

    index_map = {
        HUE: 0, SATURATION: 1, VALUE: 2
    }

    @classmethod
    def all_features(cls):
        return (cls.HUE, cls.SATURATION, cls.VALUE)

    @classmethod
    def validate(cls, value):
        if value not in cls.all_features():
            raise ValueError('`value` = {} not a valid color feature.')

    @classmethod
    def feature_index(cls, value):
        return cls.index_map[value]
