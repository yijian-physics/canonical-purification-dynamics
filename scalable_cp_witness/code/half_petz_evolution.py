"""Repeated onsite-channel evolution of a CP MPS by the explicit half-Petz lift.

Physical windows are initialized from the input MPS and then evolved with the
known onsite channels. The driver never receives an exact future CP.
"""
import tdvp_core as tn
from dataclasses import dataclass
import numpy as np


def physical_factor(state, start, width):
    state.move_center(start)
    block = state.tensors[start]
    for tensor in state.tensors[start+1:start+width]:
        block = np.tensordot(block, tensor, axes=(-1, 0))
    order = [0] + [1+2*j for j in range(width)] + [2+2*j for j in range(width)] + [2*width+1]
    block = block.reshape([block.shape[0]]+[2,2]*width+[block.shape[-1]])
    block = block.transpose(order).reshape(block.shape[0], 2**width, 2**width, block.shape[-1])
    return block.transpose(1,0,2,3).reshape(2**width,-1)


def physical_marginal(state, start, width):
    factor = physical_factor(state,start,width)
    return factor@factor.conj().T


def factor_spectrum(factor, cutoff):
    basis,singular,_ = np.linalg.svd(factor,full_matrices=False)
    keep = singular > cutoff*max(singular[0],1e-300)
    if not np.any(keep):
        raise ValueError('The physical window factor is zero.')
    return basis[:,keep],singular[keep],float(np.sum(singular[~keep]**2))


def embed(operator, position, width):
    return np.kron(np.eye(2**position),
                   np.kron(operator, np.eye(2**(width-position-1))))


def spectral_factors(density, cutoff):
    density = (density+density.conj().T)/2
    values, vectors = np.linalg.eigh(density)
    if values[0] < -1e-10:
        raise ValueError('The physical marginal has a negative eigenvalue.')
    retained = values > cutoff * max(values[-1], 1e-300)
    root = (vectors[:,retained]*np.sqrt(values[retained])) @ vectors[:,retained].conj().T
    inverse = (vectors[:,retained]/np.sqrt(values[retained])) @ vectors[:,retained].conj().T
    return root, inverse, dict(
        discarded_density_mass=float(np.maximum(values[~retained],0).sum()),
        inverse_norm=float(1/np.sqrt(values[retained][0])),
        retained_rank=int(retained.sum()))


def split_block(state, start, width, block, chi):
    discarded = []
    tensors = []
    left = block.shape[0]
    for offset in range(width-1):
        u, singular, vh = np.linalg.svd(block.reshape(left*4,-1), full_matrices=False)
        rank = max(1, min(chi, int(np.sum(singular > 1e-14*singular[0]))))
        discarded.append(float(np.sum(singular[rank:]**2)))
        tensors.append(u[:,:rank].reshape(left,4,rank))
        block = singular[:rank,None]*vh[:rank]
        left = rank
    tensors.append(block.reshape(left,4,-1))
    state.tensors[start:start+width] = tensors
    state.center = start+width-1
    state.max_bond = chi
    return float(sum(np.sqrt(discarded)))


@dataclass
class UpdateDiagnostics:
    site: int
    width: int
    raw_norm: float
    output_norm: float
    truncation_bound: float
    discarded_density_mass: float
    inverse_norm: float
    cache_discarded_mass: float


class HalfPetzEvolution:
    def __init__(self, initial, buffer=1, chi=4, cutoff=1e-13):
        if initial.n < 2 or buffer < 0 or chi < 1 or not 0 < cutoff < 1:
            raise ValueError('Require n>=2, buffer>=0, chi>=1 and 0<cutoff<1.')
        import copy
        self.state = copy.deepcopy(initial)
        length = self.state.norm()
        if length == 0:
            raise ValueError('The input amplitude is zero.')
        self.state.tensors[self.state.center] /= length
        self.buffer, self.chi, self.cutoff = buffer, chi, cutoff
        self.windows = {}
        self.factors = {}
        self.cache_discarded_mass = 0.
        self.site_window = []
        self.affected = [[] for _ in range(initial.n)]
        for site in range(initial.n):
            start = max(0,site-buffer)
            stop = min(initial.n,site+buffer+1)
            key = (start,stop-start)
            self.site_window.append(key)
            if key not in self.windows:
                factor = physical_factor(self.state,*key)
                basis,singular,discarded = factor_spectrum(factor,self.cutoff)
                self.cache_discarded_mass += discarded
                self.factors[key] = basis*singular
                self.windows[key] = self.factors[key]@self.factors[key].conj().T
                for included in range(start,stop):
                    self.affected[included].append(key)

    def update(self, site, operators):
        if not 0 <= site < self.state.n:
            raise ValueError('Site index is outside the chain.')
        operators = [np.asarray(k,dtype=complex) for k in operators]
        if not operators or any(k.shape != (2,2) or not np.isfinite(k).all() for k in operators):
            raise ValueError('Supply finite 2x2 Kraus operators.')
        completeness = sum(k.conj().T@k for k in operators)
        if np.linalg.norm(completeness-np.eye(2)) > 1e-10:
            raise ValueError('Kraus operators must define a trace-preserving channel.')
        start,width = self.site_window[site]
        local_kraus = [embed(k,site-start,width) for k in operators]
        basis,scale,old_mass = factor_spectrum(self.factors[(start,width)],self.cutoff)
        after_factor = np.column_stack([k@self.factors[(start,width)] for k in local_kraus])
        new_basis,new_scale,new_mass = factor_spectrum(after_factor,self.cutoff)
        left_operators = [
            new_basis@((new_basis.conj().T@k@basis)*scale[None,:]/new_scale[:,None])@basis.conj().T
            for k in local_kraus]
        state = self.state
        state.move_center(start)
        block = state.tensors[start]
        for tensor in state.tensors[start+1:start+width]:
            block = np.tensordot(block,tensor,axes=(-1,0))
        order = [0]+[1+2*j for j in range(width)]+[2+2*j for j in range(width)]+[2*width+1]
        paired = block.reshape([block.shape[0]]+[2,2]*width+[block.shape[-1]])
        operator = paired.transpose(order).reshape(block.shape[0],2**width,2**width,block.shape[-1])
        updated = np.zeros_like(operator,dtype=complex)
        for left,k in zip(left_operators,local_kraus):
            updated += np.einsum('pi,aiqb,qj->apjb',left,operator,k.conj().T,optimize=True)
        raw_norm = float(np.linalg.norm(updated))
        unpair = [0]+[v for j in range(width) for v in (1+j,1+width+j)]+[2*width+1]
        updated = updated.reshape([updated.shape[0]]+[2]*2*width+[updated.shape[-1]])
        updated = updated.transpose(unpair).reshape([updated.shape[0]]+[4]*width+[updated.shape[-1]])
        truncation = split_block(state,start,width,updated,self.chi)
        output_norm = float(np.linalg.norm(state.tensors[state.center]))
        # Onsite channels outside a window disappear under its partial trace.
        for key in self.affected[site]:
            left_edge,window_width = key
            ks = [embed(k,site-left_edge,window_width) for k in operators]
            factor = np.column_stack([k@self.factors[key] for k in ks])
            u,s,discarded = factor_spectrum(factor,self.cutoff)
            self.cache_discarded_mass += discarded
            self.factors[key] = u*s
            self.windows[key] = self.factors[key]@self.factors[key].conj().T
        return UpdateDiagnostics(site,width,raw_norm,output_norm,truncation,
                                 old_mass+new_mass,float(1/new_scale[-1]),self.cache_discarded_mass)
