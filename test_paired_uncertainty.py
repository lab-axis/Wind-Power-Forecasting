import unittest
import numpy as np
from src.utils.paired_uncertainty import paired_block_interval


class PairedTests(unittest.TestCase):
    def test_constant_difference_and_partial_last_block(self):
        r=paired_block_interval(np.full(101,-.3),24)
        for k in ['difference','lower_95','upper_95']:self.assertAlmostEqual(r[k],-.3,places=12)

    def test_full_length_block_has_exact_mean(self):
        r=paired_block_interval(np.arange(31.),31)
        self.assertAlmostEqual(r['lower_95'],15.)
        self.assertAlmostEqual(r['upper_95'],15.)

    def test_seed_and_pair_sign(self):
        d=np.sin(np.arange(100)/5)
        a=paired_block_interval(d,24);b=paired_block_interval(-d,24)
        self.assertEqual(a,paired_block_interval(d,24))
        self.assertAlmostEqual(a['lower_95'],-b['upper_95'])

    def test_matches_explicit_circular_resampling(self):
        d=np.arange(37.)**2;b=7;repetitions=200;seed=11
        starts=np.random.default_rng(seed).integers(0,len(d),size=(repetitions,6))
        draws=[]
        for row in starts:
            indices=np.concatenate([(start+np.arange(b))%len(d) for start in row])[:len(d)]
            draws.append(d[indices].mean())
        result=paired_block_interval(d,b,repetitions,seed)
        np.testing.assert_allclose([result['lower_95'],result['upper_95']],np.quantile(draws,[.025,.975]))


if __name__=='__main__':unittest.main()
