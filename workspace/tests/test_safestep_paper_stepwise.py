import torch
from fire_evac.safestep_paper_stepwise import selected_td_loss


def test_both_exit_and_timeout_have_zero_bootstrap():
    identity=torch.eye(3)
    policy=torch.nn.Linear(3,4,bias=False)
    target=torch.nn.Linear(3,4,bias=False)
    with torch.no_grad():target.weight.fill_(100.)
    # Invalid next-state indices are deliberate: terminal observations are never queried.
    loss,q,y=selected_td_loss(policy,target,identity,torch.tensor([0,1,2]),torch.tensor([0,1,2]),
        torch.tensor([999,998,0]),torch.tensor([1.,-2.,3.]),torch.tensor([True,True,False]),.95)
    assert y.tolist()==[1.,-2.,98.]
    assert not y.requires_grad


def test_only_sampled_action_predictions_receive_loss_gradients():
    policy=torch.nn.Linear(2,4,bias=False);target=torch.nn.Linear(2,4,bias=False)
    with torch.no_grad():policy.weight.zero_()
    loss,q,y=selected_td_loss(policy,target,torch.eye(2),torch.tensor([0,1]),torch.tensor([1,3]),
        torch.tensor([999,999]),torch.tensor([2.,4.]),torch.tensor([True,True]),.95)
    loss.backward()
    assert loss.item()==10.
    assert policy.weight.grad.tolist()==[[0.,0.],[-2.,0.],[0.,0.],[0.,-4.]]
    assert all(p.grad is None for p in target.parameters())


def test_nonterminal_target_matches_equation10():
    policy=torch.nn.Linear(2,4,bias=False);target=torch.nn.Linear(2,4,bias=False)
    with torch.no_grad():target.weight.copy_(torch.tensor([[1.,2.],[3.,4.],[5.,6.],[7.,8.]]))
    _,_,y=selected_td_loss(policy,target,torch.eye(2),torch.tensor([0]),torch.tensor([1]),
        torch.tensor([1]),torch.tensor([-.5]),torch.tensor([False]),.95)
    assert torch.allclose(y,torch.tensor([7.1]))
