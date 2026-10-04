import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from src.utils.checkpoint_selection import validation_crps
from src.utils.probabilistic_metrics import grid_crps
from scipy.special import betainc


class SelectionTests(unittest.TestCase):
    def test_point_crps_clips_before_scoring(self):
        self.assertAlmostEqual(validation_crps([0,20,5],[-4,30,3]),1.)

    def test_hurdle_chunking_matches_full_grid(self):
        y=np.array([0.,1.,10.,21.,0.]);p=np.array([.1,.5,.9,1.,0.])
        a=np.array([.1,1,5,2,1]);b=np.array([.1,1,2,8,1]);g=np.linspace(0,21,1001)
        expected=grid_crps(y,1-p[:,None]+p[:,None]*betainc(a[:,None],b[:,None],g[None,:]/21),g)
        for chunk in [1,3,256]:
            self.assertAlmostEqual(validation_crps(y,p*a/(a+b)*21,p_pos=p,alpha=a,beta=b,chunk_size=chunk),expected,places=12)

    def test_invalid_scores_fail(self):
        with self.assertRaises(ValueError):validation_crps([0],[np.nan])

    def test_torch_selects_crps_epoch_and_test_labels_cannot_change_weights(self):
        from src.models.torch_trainer import train_and_evaluate_torch_model
        arr=np.random.default_rng(1).uniform(.1,10,(50,1))
        states=[]
        with tempfile.TemporaryDirectory() as td:
            for change in [0,2]:
                torch.manual_seed(5)
                model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(4,1))
                path=Path(td)/f'{change}.pt'
                with patch('src.models.torch_trainer.validation_crps',side_effect=[3.,1.,2.]):
                    info,_,_=train_and_evaluate_torch_model(model,arr,arr,arr+change,
                        lookback_steps=4,epochs=3,batch_size=16,device='cpu',save_model_path=str(path))
                self.assertEqual(info['best_epoch'],2)
                self.assertEqual(info['best_val_crps'],1.)
                history=json.loads(Path(str(path)+'.history.json').read_text())
                self.assertEqual(history['selection_metric'],'crps')
                states.append(torch.load(path,weights_only=True))
            for key in states[0]:torch.testing.assert_close(states[0][key],states[1][key],rtol=0,atol=0)

    def test_emfn_selects_crps_epoch(self):
        from src.models.emfn_trainer import train_and_evaluate_emfn
        torch.manual_seed(6);torch.set_num_threads(2)
        arr=np.random.default_rng(2).uniform(.1,10,(55,7));arr[::5,0]=0
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'emfn.pt'
            with patch('src.models.emfn_trainer.validation_crps',side_effect=[3.,1.,2.]):
                info,_,_=train_and_evaluate_emfn(arr,arr,arr,epochs=3,batch_size=16,
                    d_model=16,dilations=(1,2),device='cpu',save_model_path=str(path))
            self.assertEqual(info['best_epoch'],2)
            self.assertEqual(info['best_val_crps'],1.)
            history=json.loads(Path(str(path)+'.history.json').read_text())
            self.assertEqual([x['selection_score'] for x in history['epochs']],[3,1,2])

    def test_configurable_torch_patience(self):
        from src.models.torch_trainer import train_and_evaluate_torch_model
        torch.manual_seed(5)
        arr=np.random.default_rng(1).uniform(.1,10,(50,1))
        model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(4,1))
        with patch('src.models.torch_trainer.validation_crps',side_effect=[1.,2.]):
            info,_,_=train_and_evaluate_torch_model(model,arr,arr,arr,lookback_steps=4,
                epochs=5,batch_size=16,device='cpu',patience=1)
        self.assertEqual(info['best_epoch'],1)
        self.assertEqual(info['stopped_epoch'],2)


if __name__=='__main__':unittest.main()
