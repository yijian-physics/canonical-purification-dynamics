"""Sylvester preconditioning and endpoint checks for canonical-root TDVP.

Sign iteration uses the coupled Newton--Schulz recurrence. The MPO backend
rounds its intermediate products. Acceptance refers to the supplied physical
density and its trace-error budget. Bounds are evaluated in floating point.
"""
from dataclasses import dataclass
import copy
import tdvp_core as tn
import mpo_cp_force as mpo
import numpy as np


class Rejected(RuntimeError):
    def __init__(self, reason, **measurements):
        self.reason = reason
        self.measurements = measurements
        self.candidate = None
        super().__init__(f'{reason}: {measurements}')


class DenseAlgebra:
    @staticmethod
    def product(a, b):
        return a @ b

    @staticmethod
    def combine(a, b, alpha, beta):
        return alpha * a + beta * b

    @staticmethod
    def scale(a, c):
        return c * a

    @staticmethod
    def norm(a):
        return np.linalg.norm(a)

    @staticmethod
    def adjoint(a):
        return a.conj().T


class MPOAlgebra:
    def __init__(self, bond, max_elements=6_000_000):
        self.bond = bond
        self.max_elements = max_elements

    def product(self, a, b):
        # Bound the size of each zip-up physical block before allocating it.
        carry = 1
        for x, y in zip(a.tensors, b.tensors):
            count = 4 * carry * x.shape[-1] * y.shape[-1]
            if count > self.max_elements:
                raise Rejected('product_workspace', elements=count)
            carry = min(carry * 4, 2 * self.bond, x.shape[-1] * y.shape[-1])
        # Put the weight at the beginning of the zip sweep. Left-isometric
        # inputs can make an early local SVD discard equally weighted columns
        # whose actual global Schmidt weights differ by orders of magnitude.
        left, right = copy.deepcopy(a), copy.deepcopy(b)
        for state in (left, right):
            state.center = state.n-1
            state.move_center(0)
        return mpo.multiply_zip(left, right, self.bond)

    def combine(self, a, b, alpha, beta):
        return tn.compress(tn.add(a, b, alpha, beta), self.bond)[0]

    norm = staticmethod(tn.norm)
    scale = staticmethod(tn.scale)
    adjoint = staticmethod(mpo.adjoint)


def exact_product(a, b, max_elements):
    count = sum(x.shape[0]*y.shape[0]*4*x.shape[-1]*y.shape[-1]
                for x, y in zip(a.tensors, b.tensors))
    if count > max_elements:
        raise Rejected('audit_workspace', elements=count, limit=max_elements)
    return mpo.multiply(a, b)


def sylvester_sign(root, rhs, algebra, lower, upper, audit, tolerance,
                   max_iterations=90, check_every=4):
    """Solve S_Q(F)=rhs, with a supplied positive spectral enclosure.

    audit(F) returns an upper bound on the absolute Frobenius force error,
    evaluated for the original unrounded equation. upper may be ||Q||_F.
    A resolved zero eigenvalue needs a support-aware startup outside this solve.
    """
    if not (0 < lower <= upper < np.inf):
        raise Rejected('positive_floor_unresolved', lower=lower, upper=upper)
    if not (np.isfinite(tolerance) and tolerance > 0):
        raise ValueError('Require a positive finite absolute force tolerance.')
    # Geometric upper/lower bounds can be dramatically tighter than ||Q||_F.
    x = algebra.scale(root, 1/upper)
    y = algebra.scale(rhs, 1/upper)
    if isinstance(algebra, MPOAlgebra):
        x = tn.compress(x, algebra.bond)[0]
        y = tn.compress(y, algebra.bond)[0]
    product, combine = algebra.product, algebra.combine
    best = np.inf
    best_force = None
    stale = 0
    minimum = lower/upper
    for iteration in range(max_iterations + 1):
        if iteration % check_every == 0 or minimum > .999999 or iteration == max_iterations:
            force = combine(y, algebra.adjoint(y), .25, .25)
            bound = float(audit(force))
            if not np.isfinite(bound):
                raise Rejected('nonfinite_audit', iteration=iteration)
            if bound <= tolerance:
                return force, dict(iterations=iteration, force_error_bound=bound)
            stale = stale + 1 if bound >= best * .999 else 0
            if bound<best:
                best, best_force=bound,force
            if stale >= 4 or (minimum > 1-1e-13 and iteration > 0):
                rejection=Rejected('force_tolerance', best_bound=best, iteration=iteration,
                                   requested=tolerance)
                rejection.candidate=best_force
                raise rejection
        if iteration == max_iterations:
            break
        xx = product(x, x)
        xy = product(x, y)
        xyx = product(xy, x)
        xxy = product(xx, y)
        symmetric = combine(xxy, algebra.adjoint(xxy), .5, .5)
        y = combine(combine(y, xyx, 1.5, .5), symmetric, 1, -1)
        x = combine(x, product(xx, x), 1.5, -.5)
        x = combine(x, algebra.adjoint(x), .5, .5)
        y = combine(y, algebra.adjoint(y), .5, .5)
        if isinstance(algebra,MPOAlgebra) and algebra.norm(x)>2*2.**(x.n/2):
            rejection=Rejected('unstable_preconditioner',iteration=iteration+1,best_bound=best)
            rejection.candidate=best_force
            raise rejection
        minimum = .5 * minimum * (3-minimum**2)
    rejection=Rejected('iteration_limit', best_bound=best, requested=tolerance)
    rejection.candidate=best_force
    raise rejection


def mpo_force_audit(root, rhs, lower, rhs_error=0., max_elements=6_000_000):
    """rhs_error bounds ||rhs - L(Q^2)||_F for the supplied current Q."""
    if lower <= 0 or rhs_error < 0:
        raise ValueError('A positive floor and nonnegative RHS error are required.')
    def audit(force):
        first = exact_product(root, force, max_elements)
        second = exact_product(force, root, max_elements)
        # QR before addition controls the transient bond and avoids cancellation
        # from subtracting three independently computed squared norms.
        first.center = first.n-1
        first.move_center(0)
        second.center = second.n-1
        second.move_center(0)
        residual = tn.add(tn.add(first, second), rhs, beta=-1)
        return (tn.norm(residual) + rhs_error)/(2*lower)
    return audit


def adaptive_mpo_force(frame, rhs, rhs_error, tolerance, bonds):
    """Retry a force-tolerance rejection at larger configured working bonds."""
    if not bonds or any(b<=0 for b in bonds) or list(bonds)!=sorted(set(bonds)):
        raise ValueError('Supply increasing positive integer force bonds.')
    attempts=[]
    guess=None
    best=np.inf
    for bond in bonds:
        algebra=MPOAlgebra(bond)
        for refinement in range(2):
            attempts.append(bond)
            source=rhs
            base=guess
            if base is not None:
                try:
                    left=exact_product(frame.root,base,frame.max_elements)
                    right=exact_product(base,frame.root,frame.max_elements)
                except Rejected as failure:
                    if failure.reason!='audit_workspace':
                        raise
                    # Only the proposed correction uses these rounded products.
                    # Acceptance still calls the independent original-equation audit.
                    work_root=tn.compress(frame.root,bond)[0]
                    left=algebra.product(work_root,base)
                    right=algebra.product(base,work_root)
                source=tn.compress(tn.add(rhs,tn.add(left,right),beta=-1),bond)[0]
            def updated(delta):
                return delta if base is None else algebra.combine(base,delta,1.,1.)
            try:
                correction,info=sylvester_sign(frame.root,source,algebra,frame.lower,
                    frame.upper,lambda f:frame.force_error(updated(f),rhs,rhs_error),tolerance)
                return updated(correction),dict(**info,force_bond=bond,attempted_bonds=tuple(attempts))
            except Rejected as failure:
                if failure.reason in ('nonfinite_audit','unstable_preconditioner'):
                    if failure.candidate is not None and failure.measurements['best_bound']<best:
                        best,guess=failure.measurements['best_bound'],updated(failure.candidate)
                    break
                if failure.reason not in ('force_tolerance','iteration_limit') or failure.candidate is None:
                    raise
                bound=failure.measurements['best_bound']
                if bound>=.98*best:
                    break
                best,guess=bound,updated(failure.candidate)
    failure=Rejected('force_tolerance',attempted_bonds=tuple(attempts),best_bound=best,requested=tolerance)
    failure.candidate=guess
    raise failure


def product_mpo(matrices):
    state = tn.MPS(len(matrices), max_bond=1)
    state.tensors = [matrix.reshape(1,4,1).copy() for matrix in matrices]
    return state


def whiten(operator, matrices, both=True):
    result = copy.deepcopy(operator)
    for j,(tensor,w) in enumerate(zip(operator.tensors, matrices)):
        block = tensor.reshape(tensor.shape[0],2,2,tensor.shape[-1])
        block = np.einsum('pi,aijb->apjb',w,block,optimize=True)
        if both:
            block = np.einsum('aijb,jq->aiqb',block,w.conj().T,optimize=True)
        result.tensors[j] = block.reshape(tensor.shape)
    result.center=result.n-1
    return result


def operator_bound(state):
    # A Pauli expansion has ||O||_op <= sum |c_P|. Triangle inequalities
    # on virtual paths bound that sum, even in a noncanonical MPO gauge.
    y = np.array([[0,-1j],[1j,0]])
    pauli = np.stack([mpo.I,mpo.X,y,mpo.Z]).conj().reshape(4,4)/2
    weights = np.ones(1)
    for tensor in state.tensors:
        coefficients=np.einsum('ks,asb->akb',pauli,tensor,optimize=True)
        weights = weights@np.abs(coefficients).sum(axis=1)
    return min(float(weights[0]),tn.norm(state))


def grouped_operator_bound(state,width=4):
    """Triangle bound on virtual paths, keeping cancellation within each block."""
    weights=np.ones(1)
    for start in range(0,state.n,width):
        tensors=state.tensors[start:start+width]
        block=tensors[0]
        for tensor in tensors[1:]:
            block=np.tensordot(block,tensor,axes=(-1,0))
        length=len(tensors)
        left,right=block.shape[0],block.shape[-1]
        order=[0,2*length+1]+list(range(1,2*length+1,2))+list(range(2,2*length+1,2))
        blocks=block.reshape([left]+[2,2]*length+[right]).transpose(order)
        matrices=blocks.reshape(left,right,2**length,2**length)
        weights=weights@np.linalg.svd(matrices,compute_uv=False)[...,0]
    return float(weights[0])


def power_bound(herm, target, bond, max_elements, rounds=3):
    """Upper-bound ||H||op via rounded products of H*H, with propagated error."""
    approximate = herm
    power = 1
    error = 0.
    best = operator_bound(herm)
    if best>=target:
        best=min(best,grouped_operator_bound(herm))
    for _ in range(rounds):
        if best < target:
            break
        size = operator_bound(approximate)
        try:
            square = exact_product(mpo.adjoint(approximate), approximate, max_elements)
        except Rejected:
            break
        approximate,info = tn.compress(square,bond)
        error = error*(2*size+error)+info['discarded_bound']
        power *= 2
        size=operator_bound(approximate)
        if (size+error)**(1/power)>=target:
            size=min(size,grouped_operator_bound(approximate))
        best = min(best,(size+error)**(1/power))
    return best


class ProductFrame:
    """Certify a correlated Hermitian root using a trial product congruence."""
    def __init__(self, herm, max_elements=6_000_000, anchor=None):
        matrices = []
        self.upper = self.lower = 1.
        for density in mpo.one_site_marginals(herm):
            values, basis = np.linalg.eigh(density)
            if values[0] <= 32*np.finfo(float).eps:
                raise Rejected('local_support_unresolved', population=float(values[0]))
            matrices.append((basis*values**(-.25))@basis.conj().T)
            self.upper *= np.sqrt(values[-1])
            self.lower *= np.sqrt(values[0])
        self.matrices = matrices
        difference = tn.add(whiten(herm, matrices),product_mpo([mpo.I]*herm.n),beta=-1)
        # Pay for rounding explicitly in the norm bound.
        rounded, info = tn.compress(difference, max(4,herm.bond_dimension()+1))
        self.defect = power_bound(rounded,.7,max(8,herm.bond_dimension()),max_elements)+info['discarded_bound']
        self.c = 1-self.defect
        anchor_lower = 0.
        anchor_upper = np.inf
        if self.c<=0 and anchor is not None:
            reference, floor, ceiling = anchor
            delta, info=tn.compress(tn.add(herm,reference,beta=-1),herm.bond_dimension()+reference.bond_dimension())
            bound=power_bound(delta,.5*floor, max(12,delta.bond_dimension()),max_elements)+info['discarded_bound']
            anchor_lower,anchor_upper = floor-bound,ceiling+bound
            self.c = max(self.c,anchor_lower/self.upper)
        if not np.isfinite(self.c) or self.c<=0:
            raise Rejected('positive_frame_unresolved', frame_defect=self.defect)
        self.w_norm = 1/np.sqrt(self.lower)
        self.upper = min(self.upper*(1+self.defect),anchor_upper)
        self.lower = max(self.lower*self.c,anchor_lower)
        self.root = herm
        self.max_elements=max_elements

    def force_error(self, force, rhs, rhs_error=0.):
        left = exact_product(self.root,force,self.max_elements)
        right = exact_product(force,self.root,self.max_elements)
        r = tn.add(tn.add(left,right),rhs,beta=-1)
        return (tn.norm(whiten(r,self.matrices))+rhs_error*self.w_norm**2)/(2*self.c)

    def endpoint(self, density, trace_error=0.):
        square = exact_product(self.root,self.root,self.max_elements)
        defect=tn.add(square,density,beta=-1)
        frobenius=tn.norm(defect)
        weighted=self.w_norm*(tn.norm(whiten(defect,self.matrices,both=False))+
                              self.w_norm*trace_error)/self.c
        bound=min(weighted,(frobenius+trace_error)/self.lower,
                  np.sqrt(2.**(density.n/2)*frobenius+trace_error))
        return dict(cp_bound=bound, frame_defect=self.defect, lower=self.lower)

    @classmethod
    def from_anchor(cls, herm, anchor, matrices, lower_relative, upper_relative,
                    max_elements=6_000_000):
        """Use c I <= W anchor W* <= C I and bound the change from that anchor."""
        if not 0<lower_relative<=upper_relative:
            raise ValueError('Invalid anchor enclosure.')
        result=cls.__new__(cls)
        delta=whiten(tn.add(herm,anchor,beta=-1),matrices)
        rounded,info=tn.compress(delta,herm.bond_dimension()+anchor.bond_dimension())
        drift=power_bound(rounded,.4*lower_relative,max(12,rounded.bond_dimension()),
                          max_elements)+info['discarded_bound']
        result.c=lower_relative-drift
        if result.c<=0 or not np.isfinite(result.c):
            raise Rejected('positive_frame_unresolved', frame_drift=drift)
        result.w_norm=float(np.prod([np.linalg.norm(w,2) for w in matrices]))
        inv_norm=float(np.prod([1/np.linalg.svd(w,compute_uv=False)[-1] for w in matrices]))
        result.upper=(upper_relative+drift)*inv_norm**2
        result.lower=result.c/result.w_norm**2
        result.defect=max(abs(1-result.c),abs(upper_relative+drift-1))
        result.matrices,result.root,result.max_elements=matrices,herm,max_elements
        return result


@dataclass
class SpectralAudit:
    """Small-chain diagnostic, using the spectrum of the *current* amplitude."""
    q: np.ndarray
    rhs: np.ndarray
    def __post_init__(self):
        scale = np.linalg.norm(self.q)
        herm_error = np.linalg.norm(self.q - self.q.conj().T)
        if not np.isfinite(scale) or herm_error > 1e-12*max(scale, 1e-300):
            raise Rejected('nonhermitian_branch', defect=herm_error)
        self.values, self.vectors = np.linalg.eigh((self.q+self.q.conj().T)/2)
        floor = 32*np.finfo(float).eps*len(self.q)*scale
        if self.values[0] < -floor:
            raise Rejected('negative_branch', minimum=self.values[0])
        kernel = self.values <= floor
        if kernel.any():
            basis = self.vectors[:,kernel]
            leakage = np.linalg.norm(basis.conj().T @ self.rhs @ basis)
            # A finite precision support test cannot distinguish an arbitrarily
            # small population from an exact zero without external support data.
            reason = 'rank_bridge_required' if leakage > 1e-12*max(np.linalg.norm(self.rhs),1e-15) else 'support_unresolved'
            raise Rejected(reason, leakage=leakage, minimum=self.values[0], resolution=floor)
        self.denominator = self.values[:,None]+self.values[None,:]
        self.transformed_rhs = self.vectors.conj().T @ self.rhs @ self.vectors

    def error(self, candidate):
        # Evaluate by coordinates to avoid cancellation in a tiny denominator.
        coordinates = self.vectors.conj().T @ candidate @ self.vectors
        return np.linalg.norm(coordinates-self.transformed_rhs/self.denominator)


def regularized_stage(amplitude,floor,budget):
    """Build a positive provisional stage. The endpoint requires an independent audit."""
    values,basis=np.linalg.eigh((amplitude+amplitude.conj().T)/2)
    clipped=np.maximum(values,floor)
    root=(basis*clipped)@basis.conj().T
    root/=np.linalg.norm(root)
    repair=np.linalg.norm(root-amplitude)
    if not np.isfinite(repair) or repair>budget:
        raise Rejected('stage_repair_budget',repair=repair,budget=budget)
    return root,dict(repair=repair,soft_dimensions=int(np.sum(values<floor)))


def endpoint_dense(candidate, density, density_trace_error=0.):
    """Endpoint bound using soft spectral sectors of the candidate only."""
    if not np.isfinite(density_trace_error) or density_trace_error < 0:
        raise ValueError('Invalid physical density error bound.')
    if np.linalg.norm(density-density.conj().T)>1e-11:
        raise ValueError('The physical reference must be Hermitian.')
    values, basis = np.linalg.eigh((candidate+candidate.conj().T)/2)
    s = np.maximum(values, 0)
    positive = (basis*s)@basis.conj().T
    branch = np.linalg.norm(candidate-positive)
    physical = basis.conj().T@density@basis
    defect = np.diag(s*s)-physical
    trace_error = np.sum(abs(np.linalg.eigvalsh((defect+defect.conj().T)/2)))+density_trace_error
    gap_free = branch+np.sqrt(trace_error)
    # In mixed eigenbases E_ij=(s_i+sqrt(rho)_j) * amplitude_error_ij.
    # For soft rows, positivity instead bounds the cross term in ||P(H-S)||².
    soft_mass = np.r_[0.,np.cumsum(s*s+np.real(np.diag(physical)))]
    row_norm = np.linalg.norm(defect,axis=1)
    eligible = int(np.count_nonzero(s<=32*np.finfo(float).eps*max(s[-1],1e-300)))
    weights = row_norm[eligible:]/s[eligible:]
    hard_norm = np.r_[np.sqrt(np.cumsum(weights[::-1]**2)[::-1]),0.]
    if density_trace_error:
        hard_norm[:-1] += density_trace_error/s[eligible:]
    soft = soft_mass[eligible:].copy()
    soft += density_trace_error*(np.arange(eligible,len(s)+1)>0)
    bounds = branch + np.sqrt(hard_norm**2 + np.maximum(soft,0.))
    cut = int(np.argmin(bounds))
    # Integrate the square-root derivative on the straight positive density
    # path from D=H²+shift*I to rho. S_path >= (1-t) S_D, and integrating
    # 1/sqrt(1-t) gives 2. This weights both indices of a soft-hard coherence.
    d=len(s)
    shifts=[0.] if s[0]>0 else []
    shifts.extend(max(s[-1]**2,1e-300)*10.**np.arange(-16.,-3.,2.))
    path_bound=np.inf
    for shift in shifts:
        diag=s*s+shift
        diff=defect.copy()
        diff.flat[::d+1]+=shift
        squared=np.sum(np.abs(diff)**2/(diag[:,None]+diag[None,:]))
        weighted=np.sqrt(squared)+density_trace_error/np.sqrt(2*diag[0])
        path_bound=min(path_bound,branch+np.linalg.norm(np.sqrt(diag)-s)+2*weighted)
    return dict(cp_bound=float(min(gap_free,bounds[cut],path_bound)), branch=branch,
                density_trace_error=float(trace_error), gap_free_bound=float(gap_free),
                soft_dimension=eligible+cut, spectral_path_bound=float(path_bound))


def integrate_guarded(initial, start, stop, step, chi, provider, retract, audit,
                      cp_tolerance, max_halvings=8, bridge=None, bridge_width=None):
    """Retain the last accepted state on failure. Times use the provider's clock."""
    if not(0 <= start < stop and step > 0 and cp_tolerance > 0 and max_halvings >= 0):
        raise ValueError('Invalid integration interval or budget.')
    if bridge_width is not None and (bridge_width<=0 or not np.isfinite(bridge_width)):
        raise ValueError('A finite positive bridge width is required.')
    current, coordinate = initial, start
    history = []
    last_bound=None
    def result(status, **details):
        return current, dict(status=status, coordinate=coordinate,
                             last_accepted_bound=last_bound,
                             accepted=sum(r['accepted'] for r in history),
                             history=history, **details)
    try:
        initial_check = audit(coordinate,current)
    except Rejected as failure:
        return None, result(failure.reason, initial_accepted=False, **failure.measurements)[1]
    except (np.linalg.LinAlgError,FloatingPointError,MemoryError) as failure:
        return None,result('numerical_failure',initial_accepted=False,detail=str(failure))[1]
    if not np.isfinite(initial_check['cp_bound']) or initial_check['cp_bound'] > cp_tolerance:
        return None, result('initial_cp_budget', initial_accepted=False, **initial_check)[1]
    last_bound=initial_check['cp_bound']
    nominal = step
    while coordinate < stop:
        h = min(nominal, stop-coordinate)
        for retry in range(max_halvings+1):
            stage = 'initial_force'
            provisional_stages=[]
            if coordinate+h==coordinate:
                return result('step_floor')
            def force_at(t, state):
                nonlocal stage
                stage = 'initial_force' if t == coordinate else 'midpoint_force'
                force,diagnostics=provider(t,state)
                if diagnostics.get('provisional',False):
                    provisional_stages.append(stage)
                return force,diagnostics
            try:
                candidate, _ = tn.midpoint_step(current,coordinate,h,force_at,chi,retract,mode='two')
                stage = 'endpoint'
                check = audit(coordinate+h,candidate)
            except Rejected as failure:
                history.append(dict(accepted=False, coordinate=coordinate, h=h,
                                    stage=stage, reason=failure.reason, **failure.measurements))
                if stage=='initial_force' and failure.reason=='rank_bridge_required' and bridge:
                    h=min(bridge_width or h,stop-coordinate)
                    for bridge_try in range(max_halvings+1):
                        try:
                            candidate=bridge(current,coordinate,coordinate+h)
                            check=audit(coordinate+h,candidate)
                        except Rejected as bridge_failure:
                            return result(bridge_failure.reason,**bridge_failure.measurements)
                        except (np.linalg.LinAlgError,FloatingPointError,MemoryError) as bridge_failure:
                            return result('numerical_failure',stage='rank_bridge',detail=str(bridge_failure))
                        accepted=np.isfinite(check['cp_bound']) and check['cp_bound']<=cp_tolerance
                        history.append(dict(accepted=bool(accepted),coordinate=coordinate,h=h,
                                            stage='rank_bridge',**check))
                        if accepted:
                            current,coordinate=candidate,coordinate+h
                            last_bound=check['cp_bound']
                            break
                        h*=.5
                    else:
                        return result('rank_bridge_budget',**check)
                    break
                if stage!='initial_force' and retry<max_halvings and failure.reason not in ('product_workspace','audit_workspace'):
                    h *= .5
                    continue
                return result(failure.reason, **failure.measurements)
            except (np.linalg.LinAlgError,FloatingPointError,MemoryError) as failure:
                history.append(dict(accepted=False,coordinate=coordinate,h=h,stage=stage,
                                    reason='numerical_failure',detail=str(failure)))
                return result('numerical_failure',stage=stage,detail=str(failure))
            accepted = np.isfinite(check['cp_bound']) and check['cp_bound'] <= cp_tolerance
            history.append(dict(accepted=bool(accepted), coordinate=coordinate, h=h,
                                provisional_forces=len(provisional_stages), **check))
            if accepted:
                current, coordinate = candidate, coordinate+h
                last_bound=check['cp_bound']
                nominal = min(step,h*1.25)
                break
            if retry == max_halvings:
                return result('endpoint_cp_budget', rejected_cp_bound=check['cp_bound'])
            h *= .5
        if coordinate < stop and nominal < step/2**max_halvings:
            return result('step_floor')
    return result('accepted', cp_bound=last_bound)


def endpoint_mpo(candidate, density, anchor, anchor_floor, density_trace_error=0.,
                 max_elements=6_000_000):
    """Global MPO check using a Hermitian anchor with a known spectral floor."""
    if not np.isfinite(anchor_floor) or not np.isfinite(density_trace_error) or density_trace_error < 0:
        raise ValueError('Invalid anchor or density-error input.')
    herm = tn.add(candidate, mpo.adjoint(candidate), .5, .5)
    skew = tn.distance(candidate, herm)
    lower = anchor_floor - tn.distance(herm, anchor)
    if lower <= 0:
        raise Rejected('positive_floor_unresolved', lower=lower)
    square = exact_product(herm, herm, max_elements)
    defect = tn.distance(square, density)
    # Powers-Størmer, or the Sylvester identity for herm - sqrt(rho).
    bound = skew+min(np.sqrt(2.**(candidate.n/2)*defect+density_trace_error),
                     (defect+density_trace_error)/lower)
    return dict(cp_bound=bound, lower=lower, hermiticity_correction=skew)
