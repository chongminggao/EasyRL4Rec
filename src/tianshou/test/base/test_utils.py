import numpy as np
import torch

from tianshou.exploration import GaussianNoise, OUNoise
from tianshou.utils import MovAvg, MultipleLRSchedulers, RunningMeanStd
from tianshou.utils.net.common import MLP, Net
from tianshou.utils.net.continuous import RecurrentActorProb, RecurrentCritic


def test_noise():
    noise = GaussianNoise()
    size = (3, 4, 5)
    assert np.allclose(noise(size).shape, size)
    noise = OUNoise()
    noise.reset()
    assert np.allclose(noise(size).shape, size)


def test_moving_average():
    stat = MovAvg(10)
    assert np.allclose(stat.get(), 0)
    assert np.allclose(stat.mean(), 0)
    assert np.allclose(stat.std()**2, 0)
    stat.add(torch.tensor([1]))
    stat.add(np.array([2]))
    stat.add([3, 4])
    stat.add(5.)
    assert np.allclose(stat.get(), 3)
    assert np.allclose(stat.mean(), 3)
    assert np.allclose(stat.std()**2, 2)


def test_rms():
    rms = RunningMeanStd()
    assert np.allclose(rms.mean, 0)
    assert np.allclose(rms.var, 1)
    rms.update(np.array([[[1, 2], [3, 5]]]))
    rms.update(np.array([[[1, 2], [3, 4]], [[1, 2], [0, 0]]]))
    assert np.allclose(rms.mean, np.array([[1, 2], [2, 3]]), atol=1e-3)
    assert np.allclose(rms.var, np.array([[0, 0], [2, 14 / 3.]]), atol=1e-3)


def test_net():
    # here test the networks that does not appear in the other script
    bsz = 64
    # MLP
    data = torch.rand([bsz, 3])
    mlp = MLP(3, 6, hidden_sizes=[128])
    assert list(mlp(data).shape) == [bsz, 6]
    # output == 0 and len(hidden_sizes) == 0 means identity model
    mlp = MLP(6, 0)
    assert data.shape == mlp(data).shape
    # common net
    state_shape = (10, 2)
    action_shape = (5, )
    data = torch.rand([bsz, *state_shape])
    expect_output_shape = [bsz, *action_shape]
    net = Net(
        state_shape,
        action_shape,
        hidden_sizes=[128, 128],
        norm_layer=torch.nn.LayerNorm,
        activation=None
    )
    assert list(net(data)[0].shape) == expect_output_shape
    assert str(net).count("LayerNorm") == 2
    assert str(net).count("ReLU") == 0
    Q_param = V_param = {"hidden_sizes": [128, 128]}
    net = Net(
        state_shape,
        action_shape,
        hidden_sizes=[128, 128],
        dueling_param=(Q_param, V_param)
    )
    assert list(net(data)[0].shape) == expect_output_shape
    # concat
    net = Net(state_shape, action_shape, hidden_sizes=[128], concat=True)
    data = torch.rand([bsz, np.prod(state_shape) + np.prod(action_shape)])
    expect_output_shape = [bsz, 128]
    assert list(net(data)[0].shape) == expect_output_shape
    net = Net(
        state_shape,
        action_shape,
        hidden_sizes=[128],
        concat=True,
        dueling_param=(Q_param, V_param)
    )
    assert list(net(data)[0].shape) == expect_output_shape
    # recurrent actor/critic
    data = torch.rand([bsz, *state_shape]).flatten(1)
    expect_output_shape = [bsz, *action_shape]
    net = RecurrentActorProb(3, state_shape, action_shape)
    mu, sigma = net(data)[0]
    assert mu.shape == sigma.shape
    assert list(mu.shape) == [bsz, 5]
    net = RecurrentCritic(3, state_shape, action_shape)
    data = torch.rand([bsz, 8, np.prod(state_shape)])
    act = torch.rand(expect_output_shape)
    assert list(net(data, act).shape) == [bsz, 1]


def test_lr_schedulers():
    initial_lr_1 = 10.0
    step_size_1 = 1
    gamma_1 = 0.5
    net_1 = torch.nn.Linear(2, 3)
    optim_1 = torch.optim.Adam(net_1.parameters(), lr=initial_lr_1)
    sched_1 = torch.optim.lr_scheduler.StepLR(
        optim_1, step_size=step_size_1, gamma=gamma_1
    )

    initial_lr_2 = 5.0
    step_size_2 = 2
    gamma_2 = 0.3
    net_2 = torch.nn.Linear(3, 2)
    optim_2 = torch.optim.Adam(net_2.parameters(), lr=initial_lr_2)
    sched_2 = torch.optim.lr_scheduler.StepLR(
        optim_2, step_size=step_size_2, gamma=gamma_2
    )
    schedulers = MultipleLRSchedulers(sched_1, sched_2)
    for _ in range(10):
        loss_1 = (torch.ones((1, 3)) - net_1(torch.ones((1, 2)))).sum()
        optim_1.zero_grad()
        loss_1.backward()
        optim_1.step()
        loss_2 = (torch.ones((1, 2)) - net_2(torch.ones((1, 3)))).sum()
        optim_2.zero_grad()
        loss_2.backward()
        optim_2.step()
        schedulers.step()
    assert (
        optim_1.state_dict()["param_groups"][0]["lr"] ==
        (initial_lr_1 * gamma_1**(10 // step_size_1))
    )
    assert (
        optim_2.state_dict()["param_groups"][0]["lr"] ==
        (initial_lr_2 * gamma_2**(10 // step_size_2))
    )


if __name__ == '__main__':
    test_noise()
    test_moving_average()
    test_rms()
    test_net()
    test_lr_schedulers()
