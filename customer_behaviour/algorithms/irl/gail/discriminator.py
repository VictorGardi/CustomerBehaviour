import chainer
import chainer.functions as F
from chainerrl import links


class Discriminator:
    def __init__(self, input_dim, hidden_sizes=(64,64,64), loss_type='wgangp', PAC_k=1, PAC_eps=1, gpu=-1):
        self.model = links.MLP(input_dim, 1, hidden_sizes=hidden_sizes)

        if gpu >= 0:
            self.model.to_gpu(gpu)
        if PAC_k > 1:
            self.optimizer = chainer.optimizers.Adam(alpha=1e-5, eps=PAC_eps) #PACGAIL needs a larger epsilon to prevent divison by zero when gradient is small
        else:
            self.optimizer = chainer.optimizers.Adam(alpha=1e-5, eps=1e-5)
        self.optimizer.setup(self.model)
        self.loss_type = loss_type
        self.loss = None

    def __call__(self, x):
        return F.sigmoid(self.model(x))

    def train(self, expert_data, fake_data):
        self.model.cleargrads()

        if self.loss_type == 'gan':
            # d_expert = self.model(expert_data)
            # d_fake = self.model(fake_data)
            # discriminator is trained to predict a p(expert|x)

            # loss = dw [ E_agent(log(D)) + E_expert(log(1-D)) ]
            # D = MLP ---> should apply sigmoid afterwards

            self.loss = F.mean(F.log(self(fake_data)))
            self.loss += F.mean(F.log(1-self(expert_data)))


            # self.loss = F.mean(F.softplus(-d_expert))
            # self.loss += F.mean(F.softplus(d_fake))
        elif self.loss_type == 'wgangp':
            # sampling along straight lines
            xp = chainer.cuda.get_array_module(expert_data)
            e = xp.random.uniform(0., 1., len(expert_data))[:, None].astype(xp.float32)

            x_hat = chainer.Variable((e * expert_data + (1 - e) * fake_data).array, requires_grad=True)
            grad, = chainer.grad([self.model(x_hat)], [x_hat], enable_double_backprop=True)
            grad = F.sqrt(F.batch_l2_norm_squared(grad))
 
            loss_grad = 1 * F.mean_squared_error(grad, xp.ones_like(grad.data))
            loss_gan = F.mean(self.model(fake_data) - self.model(expert_data))
            # discriminator is trained to predict a p(expert|x)
            self.loss = loss_gan + loss_grad

        else:
            raise NotImplementedError

        self.loss.backward()
        self.optimizer.update()

        return self.loss

    def get_rewards(self, x):
        # - log p(fake|x) == - (log 1 - p(expert|x)) is more stable than log(1 - p(fake|x)) and log(p(expert|x))
        if self.loss_type == 'gan':
            return - F.log(1 - self(x))
        return self.model(x)

