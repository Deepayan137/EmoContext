class Config:
    # data
    path = 'data/'
    lr = 1e-3
    hidden_size = 256
    # visualization
    plot_every = 40  # vis every N iter
    epoch = 14
    # debug
    debug_file = '/tmp/debugf'

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


opt = Config()