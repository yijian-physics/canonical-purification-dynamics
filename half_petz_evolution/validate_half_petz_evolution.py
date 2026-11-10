"""Check local contractions and physical-window caches with complex loss data."""
import half_petz_evolution as algorithm
import tdvp_core as tn
import cluster_dephasing as base
import numpy as np


def main():
    rng = np.random.default_rng(20261109)
    n = 4
    factor = rng.normal(size=(16,16))+1j*rng.normal(size=(16,16))
    density = factor@factor.conj().T
    density /= np.trace(density)
    root,_ ,_ = algorithm.spectral_factors(density,1e-14)
    initial = tn.from_dense(tn.to_interleaved(root,n),n)
    loss = [np.diag([1.,np.sqrt(.93)]), np.sqrt(.07)*np.array([[0,1],[0,0]])]
    unitary = np.cos(.13)*base.I-1j*np.sin(.13)*base.X
    operations = [(1,loss),(3,[unitary]),(0,[np.sqrt(.97)*base.I,np.sqrt(.03)*base.Z]),(2,loss)]
    errors = []
    for buffer in (0,1,3):
        solver = algorithm.HalfPetzEvolution(initial,buffer=buffer,chi=16,cutoff=1e-14)
        physical = density.copy()
        for site,ks in operations:
            start,width = solver.site_window[site]
            current = tn.to_matrix(solver.state.dense(),n)
            local = base.marginal(physical,n,range(start,start+width))
            local_root,_,_ = algorithm.spectral_factors(local,1e-14)
            embedded_ks = [algorithm.embed(k,site-start,width) for k in ks]
            after = sum(k@local@k.conj().T for k in embedded_ks)
            _,inverse,_ = algorithm.spectral_factors(after,1e-14)
            def global_op(op):
                return np.kron(np.eye(2**start),np.kron(op,np.eye(2**(n-start-width))))
            expected = sum(global_op(inverse@k@local_root)@current@global_op(k).conj().T for k in embedded_ks)
            solver.update(site,ks)
            actual = tn.to_matrix(solver.state.dense(),n)
            error = np.linalg.norm(actual-expected)
            assert error < 1e-11
            errors.append(error)
            physical = sum(algorithm.embed(k,site,n)@physical@algorithm.embed(k,site,n).conj().T for k in ks)
            for key,value in solver.windows.items():
                a,w = key
                assert np.linalg.norm(value-base.marginal(physical,n,range(a,a+w))) < 1e-12
            if buffer == 3:
                exact,_,_ = algorithm.spectral_factors(physical,1e-14)
                assert np.linalg.norm(actual-exact) < 1e-11
    print('Complex Kraus updates, physical caches and full-window roots passed.')
    print('Maximum dense/MPS discrepancy',max(errors))


if __name__=='__main__':
    main()
