"""Reproduction harness for Table 3 of the BLOB paper (KDD 2020): a copy of the authors' simulate_abtest_with_bandit.py
restricted to ONE repetition and to the agents needed (BLO, BLOB-MNQ, BLOB-NQ, logistic regression, random), with the
README's Table 3 arguments. The agents, their arguments and the evaluation calls are copied unchanged."""
import sys
from recogym import env_1_args
from models.models_organic_bandit import RecoModelRTAEWithBanditTF, RecoModelRTAEWithBanditTF_Full
from models.models_organic import RecoModelRTAE
from recogym.agents import organic_user_count_args
from recogym.agents import RandomAgent, random_args
from models.models_bandit import PyTorchMLRAgent, pytorch_mlr_args
from models.model_based_agents import ModelBasedAgent
from utils.utils_agents import eval_against_session_pop, first_element
import tensorflow as tf
import pandas as pd

P = int(sys.argv[1]) if len(sys.argv) > 1 else 100
flips_list = [int(f) for f in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, P // 2]
dict_args = dict(algorithm='rt/ae', batch_size=1024, lr=0.000001, K=20, rcatk=5, P=P, cudavis=None, units=None,
                 batch_size_test=5, reg_Psi=0.1, reg_shift=0.0, reg_rho=1, test_freq=1, neg_samples=1,
                 num_sessions=1000, num_sessions_organic=20000, num_users_to_score=4000, organic_epochs=1000,
                 bandit_epochs=800 if P <= 100 else 1200)
import os
if os.environ.get("QUICK"):  # a smoke test of the harness, not a reproduction
    dict_args.update(num_sessions=100, num_sessions_organic=500, num_users_to_score=200, organic_epochs=3, bandit_epochs=3)
num_sessions, num_sessions_organic = dict_args['num_sessions'], dict_args['num_sessions_organic']
num_users_to_score = dict_args['num_users_to_score']
sig_omega, seed, latent_factor, log_eps = 0., 0, 10, 0.3
eval_fn = eval_against_session_pop
r = []
for num_flips in flips_list:
    res = []
    dict_args.update(lr=0.0001, num_epochs=dict_args['organic_epochs'], use_em=False, organic=True, weight_path='weights/linAE')
    parameters = {'recomodel': RecoModelRTAE, 'modelargs': dict_args, 'num_products': P, 'K': dict_args['K']}
    sc = eval_fn(P, num_sessions_organic, num_sessions, num_users_to_score, seed, latent_factor, num_flips, log_eps, sig_omega, ModelBasedAgent, parameters, str(parameters['recomodel']), True)
    res.append(first_element(sc, 'LVM'))
    for name, cls in (('LVM Bandit MVN-Q', RecoModelRTAEWithBanditTF), ('LVM Bandit NQ', RecoModelRTAEWithBanditTF_Full)):
        dict_args.update(lr=0.001, num_epochs=dict_args['organic_epochs'], num_epochs_bandit=dict_args['bandit_epochs'],
                         use_em=False, organic=False, norm=True, organic_weights='weights/linAE',
                         wa_m=-1.0, wb_m=-6.0, wc_m=-4.5, wa_s=1., wb_s=1., wc_s=10., kappa_s=0.01)
        parameters = {'recomodel': cls, 'modelargs': dict_args, 'num_products': P, 'K': dict_args['K']}
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            sc = eval_fn(P, num_sessions_organic, num_sessions, num_users_to_score, seed, latent_factor, num_flips, log_eps, sig_omega, ModelBasedAgent, parameters, str(parameters['recomodel']), True)
            res.append(first_element(sc, name))
    pytorch_mlr_args.update(n_epochs=dict_args['bandit_epochs'], learning_rate=0.01, ll_IPS=False, alpha=1.0, num_products=P)
    sc = eval_fn(P, num_sessions_organic, num_sessions, num_users_to_score, seed, latent_factor, num_flips, log_eps, sig_omega, PyTorchMLRAgent, pytorch_mlr_args, str(pytorch_mlr_args), True)
    res.append(first_element(sc, 'log reg'))
    random_args['P'] = P
    parameters = {**organic_user_count_args, **env_1_args, 'select_randomly': True, 'modelargs': random_args}
    sc = eval_fn(P, num_sessions_organic, num_sessions, num_users_to_score, seed, latent_factor, num_flips, log_eps, sig_omega, RandomAgent, parameters, 'randomagent', True)
    res.append(first_element(sc, 'random'))
    results = pd.concat(res)
    results['seed'], results['flips'], results['P'] = seed, num_flips, P
    r.append(results)
    pd.concat(r).to_csv('results/repro_table3_P%d%s.csv' % (P, '_quick' if os.environ.get('QUICK') else ''))
print(pd.concat(r))
