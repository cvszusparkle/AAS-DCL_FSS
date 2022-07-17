import visdom
import numpy as np
import time

class Visualizer(object):
    """
    It encapsulates the basic operations of visdom, but you can still call the native visdom interface through `self.vis.function` or `self.function`
    such as:
    self.text('hello visdom')
    self.histogram(t.randn(1000))
    self.line(t.arange(0, 10),t.arange(1, 11))
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, port=2233, **kwargs)

        # Abscissa
        # such as ('loss',23) That is, the 23rd point of loss
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):

        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        plot multiple graphs at a time
        @params d: dict (name, value) i.e. ('loss', 0.11)
        """
        for k, v in d.iteritems():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.iteritems():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img', t.Tensor(64, 64))
        self.img('input_imgs', t.Tensor(3, 64, 64))
        self.img('input_imgs', t.Tensor(100, 1, 64, 64))
        self.img('input_imgs', t.Tensor(100, 3, 64, 64), nrows=10)
        """
        self.vis.images(img_,
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1, 'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'), info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        """
        self.function  ==  self.vis.function
        Except for custom plot, image, log, plot_many, etc.
        """
        return getattr(self.vis, name)
