from pprint import pprint

class Config:
    # data
    path = 'data/'
    lr = 1e-3
    inp = 25
    hidden = 256
    out = 4
    depth = 2
    epochs = 20
    filters = 100
    save_dir = 'saves'
    save_file =  '%d_emo_model.t7'%inp
    
    lmap = {'happy': 0,
            'sad': 1,
            'angry': 2,
            'others': 3
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