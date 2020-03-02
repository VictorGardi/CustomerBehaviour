import chainer
import chainer.functions as F
from customer_behaviour.algorithms.irl.common.model import MLP
from chainerrl import links


class Discriminator2:
    def __init__(self, observation_dim, action_dim, hidden_sizes=(32, 32), loss_type='gan', gpu=-1):

        self.model = links.MLP(observation_dim + action_dim, 1, hidden_sizes=hidden_sizes, nonlinearity=F.relu)

        if gpu >= 0: self.model.to_gpu(gpu)

        self.optimizer = chainer.optimizers.Adam(alpha=1e-5, eps=1e-5)  # should alpha be somewhat higher?
        self.optimizer.setup(self.model)
        self.loss_type = loss_type
        self.loss = None

    def __call__(self, x):
        # x is state and actions converted to "discriminator format"
        # x is a matrix where each row matrix where each row is state followed by action
        # (action is for example [1, 0] or [0, 1] but with some noise)

        # state + action
        logits = self.model(x)
        return F.sigmoid(logits)  # prob(s) of x being expert data

    def train(self, expert_data, fake_data, xp):
        self.model.cleargrads()  # model, not optimizer?

        # Compute loss
        if self.loss_type == 'gan':

            x = F.concat((fake_data, expert_data), axis=0)
            logits = self.model(x)

            mb_size = expert_data.shape[0]
            t = xp.zeros((2*mb_size, 1), dtype=xp.int32)
            t[mb_size:] = 1

            self.loss = F.sigmoid_cross_entropy(logits, t, normalize=True, reduce='mean')


        elif self.loss_type == 'wgangp':
            raise NotImplementedError
        else:
            raise NotImplementedError

        self.loss.backward()
        self.optimizer.update()

        return self.loss

    def get_rewards(self, x):
        # - log p(fake|x) == - (log 1 - p(expert|x)) is more stable than log(1 - p(fake|x)) and log(p(expert|x))
        if self.loss_type == 'gan':
            # reward = -log(1 - D(s, a)) (nosyndicate)
            return - F.log(1 - self(x))
        return self.model(x)


