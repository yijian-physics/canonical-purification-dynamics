"""Independent dense checks for the MPS TDVP core and branch correction."""
import tdvp_core as tn
import cp_tdvp_evolution as evo
import tdvp_markov_diagnostic as old
import numpy as np
import argparse
from pathlib import Path


def two_site_dense(psi,force,n):
    left=[np.ones((1,1),complex)];right=[None]*(n+1);right[n]=np.ones((1,1),complex)
    for cut in range(1,n):
        u,s,vh=np.linalg.svd(psi.reshape(4**cut,-1),full_matrices=False);r=int(np.sum(s>1e-12*s[0]))
        left.append(u[:,:r]);right[cut]=vh[:r]
    result=np.zeros_like(force)
    for start in range(n-1):
        l,r=left[start],right[start+2];f=force.reshape(4**start,16,4**(n-start-2))
        core=np.einsum('al,apb,rb->lpr',l.conj(),f,r.conj(),optimize=True)
        result+=np.einsum('al,lpr,rb->apb',l,core,r,optimize=True).reshape(-1)
    for start in range(1,n-1):
        l,r=left[start],right[start+1];f=force.reshape(4**start,4,4**(n-start-1))
        core=np.einsum('al,apb,rb->lpr',l.conj(),f,r.conj(),optimize=True)
        result-=np.einsum('al,lpr,rb->apb',l,core,r,optimize=True).reshape(-1)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=Path(__file__).parent/'results')
    out=parser.parse_args().outdir
    rng=np.random.default_rng(20261103);rows=[]
    for n,chi in ((3,2),(4,3),(5,4)):
        vector=rng.normal(size=4**n)+1j*rng.normal(size=4**n);vector/=np.linalg.norm(vector)
        state=tn.from_dense(vector,n,chi);state,_=tn.compress(state,chi,normalize=True)
        force=rng.normal(size=4**n)+1j*rng.normal(size=4**n);fm=tn.from_dense(force,n)
        for mode in ('one','two'):
            expected=old.tangent_project(state.dense(),force,n)[0] if mode=='one' else two_site_dense(state.dense(),force,n)
            projector=tn.tangent_project if mode=='one' else tn.tangent_project_two
            projected=projector(state,fm);error=np.linalg.norm(projected.dense()-expected)
            repeated=projector(state,projected);idempotence=tn.distance(projected,repeated)
            assert max(error,idempotence)<1e-10
            rows.append(dict(test='tangent projector',n=n,mode=mode,error=error,idempotence=idempotence))
    n=3;_,_,liouv,_=evo.spin_model(n,'ising_loss');a=rng.normal(size=(8,8))+1j*rng.normal(size=(8,8))
    rho=a@a.conj().T;rho/=np.trace(rho);root=evo.positive_root(rho)
    state=tn.from_dense(tn.to_interleaved(root,n),n);force,_=evo.DenseForce(n,liouv)(.4,state)
    derivative=tn.to_matrix(force.dense(),n)/.8
    g=(liouv@rho.reshape(-1,order='F')).reshape(rho.shape,order='F')
    error=np.linalg.norm(root@derivative+derivative@root-g);assert error<1e-11
    rows.append(dict(test='Sylvester identity',n=n,error=error))
    amplitude=a/np.linalg.norm(a);u,s,vh=np.linalg.svd(amplitude);positive=(u*s)@u.conj().T
    unitary=vh.conj().T@u.conj().T
    density_error=np.linalg.norm(amplitude@amplitude.conj().T-positive@positive)
    trace_gap=float(np.sum(s)-np.trace(amplitude).real)
    branch=np.linalg.norm(amplitude-positive)
    assert density_error<1e-12 and branch**2<=2*max(s)*trace_gap+1e-12
    assert np.linalg.norm(amplitude@unitary-positive)<1e-12
    rows.append(dict(test='polar trace correction',n=n,error=density_error,branch_error=branch,trace_gap_bound=np.sqrt(2*max(s)*trace_gap)))
    # Stable Kraus-factor startup agrees with the channel at density level,
    # while retaining root eigenvalues lost by a density spectral cutoff.
    n=4;ham,jumps,_,rho0=evo.spin_model(n,'ising_loss');seed_time=.0025
    stable=evo.positive_start_root(evo.positive_root(rho0),ham,jumps,seed_time)
    channel=evo.positive_start(rho0,ham,jumps,seed_time)
    seed_density_error=np.linalg.norm(stable@stable-channel)
    assert seed_density_error<1e-12
    smallest=float(np.linalg.eigvalsh(stable)[0])
    assert smallest>1e-10
    rows.append(dict(test='Kraus-factor startup',n=n,error=seed_density_error,smallest_root_eigenvalue=smallest))
    out.mkdir(parents=True,exist_ok=True);evo.save(out/'validation.csv',rows)
    print('TDVP projector, Sylvester, and polar-gauge checks passed.')


if __name__=='__main__':main()
