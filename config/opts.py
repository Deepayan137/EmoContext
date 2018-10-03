from pprint import pprint

class Config:
    # data
    path = 'data/'
    lr = 1e-3
    _in = 50
    hidden = 256
    out = 4
    depth = 2
    epochs = 20
    save_dir = 'saves'
    save_file =  'emo_model.t7'
    lmap = {'Happy': 0,
            'Sad': 1,
            'Angry': 2,
            'Others': 3
            }
    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


# opt = Config()