"""Regression checks for the working Phi-only Colab formulation."""
import contextlib
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ['WS_PINN_DEVICE'] = 'cpu'
os.environ['MPLBACKEND'] = 'Agg'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from ws_pinn import losses, wavefunctions as wf
from ws_pinn.config import load_config, training_settings
from ws_pinn.models import WaveNet3D
from ws_pinn.normalization import configure_normalization, restore_checkpoint_normalization
from ws_pinn.parameters import WSParams
from ws_pinn.runtime import configure_runtime

torch.set_num_threads(2)


class PhiNormalizationTests(unittest.TestCase):
    def setUp(self):
        configure_runtime('cpu', 0)
        self.sample = dict(A=48, Z=20, is_proton=True,
                           states=[dict(nr=0,l=2,j=2.5,energy=-9.0)])
        self.st = self.sample['states'][0]
        self.E = torch.tensor([-9.0])
        self.wave = WaveNet3D(1, hidden=12, depth=2)
        self.grids = (96, 128, 64)
        self.params = WSParams(**{k:torch.tensor(v) for k,v in dict(
            V0=51.,kappa=.67,r0=1.27,a=.67,lam_so=32.,r0_so=1.27).items()})

    def measure(self):
        torch.manual_seed(7)
        return (
            losses.loss_pde_radial_full3d(self.wave,self.E,self.sample,self.st,
                                          self.params,*self.grids,N=32),
            losses.loss_bc_full3d(self.wave,self.E,self.sample,self.st,*self.grids,Nb=4),
            wf.make_param_input_from_radial_psi(self.wave,self.sample,self.E,12,*self.grids),
        )

    def test_phi_scaling_cancels_but_original_theta_coupling_remains(self):
        state = configure_normalization(self.wave,[self.sample],self.grids)
        baseline = [x.detach() for x in self.measure()]
        for component in ('Phi','Theta'):
            method = getattr(self.wave,component)
            for gain in (.1,2.):
                if component == 'Phi':
                    def scaled(*a,**k):
                        x,y=method(*a,**k)
                        return gain*x,gain*y
                else:
                    def scaled(*a,**k): return gain*method(*a,**k)
                with patch.object(self.wave,component,scaled):
                    actual = self.measure()
                factor = 1. if component=='Phi' else 1/gain**2
                for x,y in zip(actual[:2],baseline[:2]):
                    torch.testing.assert_close(x,y*factor,rtol=4e-4,atol=2e-6)
                factor = 1. if component=='Phi' else 1/gain
                torch.testing.assert_close(actual[2][:12],baseline[2][:12]*factor,
                                           rtol=4e-4,atol=2e-6)
                torch.testing.assert_close(actual[2][12:],baseline[2][12:],rtol=0,atol=0)
        self.assertEqual(state, self.wave.normalization_state)

    def test_initial_values_and_theta_gradients_are_preserved(self):
        old = self.measure()
        theta = list(self.wave.Th_net.parameters())
        old_gradient = torch.autograd.grad(old[0],theta)
        old_probe_gradient = torch.autograd.grad(old[2].sum(),theta)
        rng = torch.get_rng_state().clone()
        configure_normalization(self.wave,[self.sample],self.grids)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        new = self.measure()
        for a,b in zip(old,new): torch.testing.assert_close(a,b,rtol=0,atol=0)
        for a,b in zip(old_gradient,torch.autograd.grad(new[0],theta)):
            torch.testing.assert_close(a,b,rtol=3e-5,atol=2e-6)
        for a,b in zip(old_probe_gradient,torch.autograd.grad(new[2].sum(),theta)):
            torch.testing.assert_close(a,b,rtol=3e-5,atol=2e-6)

    def test_references_distinguish_repeated_noisy_observations(self):
        noisy = deepcopy(self.sample); noisy['states'][0]['energy'] += .1
        state=configure_normalization(self.wave,[self.sample,noisy],self.grids)
        self.assertEqual(len(state['phi_norms']),2)
        clone=WaveNet3D(1,hidden=12,depth=2)
        clone.load_state_dict(self.wave.state_dict())
        configure_normalization(clone,[self.sample,noisy],self.grids,state=state)
        self.assertEqual(clone.normalization_state,state)
        with self.assertRaisesRegex(ValueError,'contexts differ'):
            configure_normalization(clone,[self.sample],self.grids,state=state)
        bad=deepcopy(state);bad['phi_norms'][next(iter(bad['phi_norms']))]=0.
        with self.assertRaisesRegex(ValueError,'Invalid'):
            configure_normalization(clone,[self.sample,noisy],self.grids,state=bad)

    def test_polar_quadrature_and_original_angular_losses(self):
        _,theta,_=wf.make_grids(12,512,12)
        theta=theta.squeeze()
        self.assertAlmostEqual(float(wf.polar_integral(torch.sin(theta),theta)),2.,places=5)
        # Jacobian cancels the difficult 1/sin(theta) term.
        actual=wf.polar_integral(1/torch.sin(theta),theta)
        expected=-2*np.log(np.tan(1e-5/2))
        self.assertAlmostEqual(float(actual),expected,places=4)
        configure_normalization(self.wave,[self.sample],self.grids)
        # The implemented ODEs remain the same for m=0 and nonzero m.
        for m in (0,1):
            torch.manual_seed(13)
            a=losses.loss_pde_theta_full3d(self.wave,self.E,self.sample,self.st,emm=m,N=32)
            torch.manual_seed(13)
            th=((torch.rand(32,1)*(1-2e-3)+1e-3)*np.pi).requires_grad_(True)
            T=self.wave.Theta(th,self.E,self.sample,self.st)
            Tp=wf.d(T,th);Tpp=wf.d(Tp,th)
            expected=(torch.sin(th)**2*Tpp+torch.sin(th)*torch.cos(th)*Tp
                      +(6*torch.sin(th)**2-m*m)*T).square().mean()
            torch.testing.assert_close(a,expected,rtol=0,atol=0)
            torch.manual_seed(13)
            b=losses.loss_pde_phi_full3d(self.wave,self.E,self.sample,self.st,emm=m,N=32,Np_bc=4)
            self.assertTrue(torch.isfinite(b))

    def test_legacy_loading_does_not_invent_initial_references(self):
        restore_checkpoint_normalization(self.wave,[self.sample],{}, {}, 'checkpoints/old.pt')
        self.assertIsNone(self.wave.phi_reference_norms)
        self.assertEqual(self.wave.polar_quadrature,'uniform_trapezoid_v1')
        with self.assertRaises(ValueError):
            restore_checkpoint_normalization(self.wave,[self.sample],{},
                {'run_manifest':{'normalization_correction':{'version':'component_normalization_v1'}}},
                'checkpoints/old.pt')


class JointTrainingTests(unittest.TestCase):
    @staticmethod
    def config(potential, mode, out, epochs):
        config=load_config(ROOT/f'configs/{potential}.yaml')
        # load_config resolves data.mode; switch using the original YAML when needed.
        if mode=='experimental':
            import yaml
            raw=yaml.safe_load((ROOT/f'configs/{potential}.yaml').read_text())
            config['data']=dict(raw['data']['datasets']['experimental'],mode=mode)
        config['data'].update(states_per_system=2,epochs=epochs,output_directory=str(out),
                              cases=config['data']['cases'][:1])
        config['model'].update(wave_width=12,param_width=12,wave_depth=2,param_depth=2)
        config['quadrature'].update(radial=96,polar=128,azimuthal=64,orthogonality_points=32)
        config['paramnet_input']['radial_probe_points']=12
        config['inference']['monte_carlo_samples']=8
        config['logging'].update(plot_every=None,log_every=1,print_every=1)
        config['output'].update(wavefunction_cases=0,heatmap_cases=0,overlap_cases=0,checkpoint_every=1)
        config['training']['kl_schedule']['after_epoch']=1
        return config

    def train(self,config):
        from ws_pinn import training
        configure_runtime('cpu',0)
        names=['plot_loss_history','plot_learning_rates','plot_parameter_history',
               'plot_parameter_summary','plot_radial_wavefunctions','plot_theta_phi_heatmaps',
               'plot_full_psi2_slice_vs_theta','plot_overlap_matrices']
        with contextlib.ExitStack() as stack:
            for name in names: stack.enter_context(patch.object(training,name,return_value={}))
            stack.enter_context(patch.object(training,'plot_global_energy_diagnostics',
                                              return_value={'MAE':0.,'RMSE':0.,'Bias':0.}))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            return training.train_global_multinucleus(training_settings(config))

    def test_all_four_experiment_modes_update_all_network_groups(self):
        from ws_pinn.checkpoints import load_models_and_samples
        with tempfile.TemporaryDirectory() as temporary:
            for potential in ('seminole','wahlborn'):
                for mode in ('synthetic','experimental'):
                    with self.subTest(potential=potential,mode=mode):
                        out=Path(temporary)/f'{potential}_{mode}'
                        cfg=self.config(potential,mode,out,1)
                        configure_runtime('cpu',0)
                        initial=WaveNet3D(2,hidden=12,depth=2).state_dict()
                        result=self.train(cfg)
                        for name in ('R_net','Th_net','Ph_net'):
                            pairs=[(v,initial[k]) for k,v in result['wave_net'].state_dict().items()
                                   if k.startswith(name)]
                            self.assertTrue(any(not torch.equal(a,b) for a,b in pairs))
                        for model in (result['wave_net'],result['param_net']):
                            self.assertTrue(all(p.requires_grad and p.grad is not None
                                                and torch.isfinite(p.grad).all() for p in model.parameters()))
                        ckpath=out/'checkpoints/latest_checkpoint.pt'
                        ck=torch.load(ckpath,weights_only=False)
                        self.assertEqual(ck['epoch'],1)
                        self.assertEqual(ck['history'][0]['wKL'],.001)
                        self.assertEqual(ck['normalization_state'],result['wave_net'].normalization_state)
                        with contextlib.redirect_stdout(io.StringIO()):
                            wave,param,samples,_,_=load_models_and_samples(cfg,ckpath)
                        self.assertEqual(wave.normalization_state,ck['normalization_state'])
                        args=[samples,12,96,128,64]
                        x=wf.make_global_param_context(wave,*args)
                        expected=wf.make_global_param_context(result['wave_net'],*args)
                        torch.testing.assert_close(x,expected,rtol=0,atol=0)
                        torch.testing.assert_close(param(x)[0],result['param_net'](expected)[0],rtol=0,atol=0)
                        best=torch.load(out/'checkpoints/best_training_loss.pt',weights_only=False)
                        self.assertEqual(best['normalization_state'],ck['normalization_state'])

    def test_resume_matches_uninterrupted_joint_training_and_kl_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            cfg=self.config('seminole','experimental',root/'split',1)
            self.train(cfg)
            refs=(root/'split/normalization_reference.json').read_bytes()
            cfg['data']['epochs']=2
            self.train(cfg)
            self.assertEqual(refs,(root/'split/normalization_reference.json').read_bytes())
            full=self.config('seminole','experimental',root/'whole',2)
            self.train(full)
            split=torch.load(root/'split/checkpoints/latest_checkpoint.pt',weights_only=False)
            whole=torch.load(root/'whole/checkpoints/latest_checkpoint.pt',weights_only=False)
            self.assertEqual(split['epoch'],2)
            self.assertEqual([row['wKL'] for row in split['history']],[.001,.0001])
            for model in ('wave_net','param_net'):
                for k,v in split[model].items(): torch.testing.assert_close(v,whole[model][k],rtol=0,atol=0)
            self.assertTrue(torch.equal(split['torch_rng_state'],whole['torch_rng_state']))
            self.assertEqual(split['normalization_state'],whole['normalization_state'])
            cfg['normalization']['phi_only']=False
            with self.assertRaisesRegex(ValueError,'Normalization settings differ'):
                self.train(cfg)


if __name__=='__main__': unittest.main()
