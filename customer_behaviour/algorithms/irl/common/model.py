import chainer
import chainer.functions as F
import chainer.links as L


def pass_fn(x):
    return x


class MLP(chainer.ChainList):
    def __init__(self, n_layer, n_units, n_out, n_in, activation=F.leaky_relu, out_activation=pass_fn, hook=None, hook_params=None):
        super().__init__()
        self.add_link(L.Linear(n_in, n_units))
        #for _ in range(n_layer):
        #    self.add_link(L.Linear(None, n_units))
        self.add_link(L.Linear(n_units, n_units))
        self.add_link(L.Linear(n_units, n_units))
        self.add_link(L.Linear(n_units, n_out))
        self.activations = [activation] * (3) + [out_activation]

        if hook:
            hook_params = dict() if hook_params is None else hook_params
            for link in self.children():
                link.add_hook(hook(**hook_params))

    def forward(self, x):
        for link, act in zip(self.children(), self.activations):
            x = act(link(x))
        return x

    def __call__(self, x):
        return self.forward(x)