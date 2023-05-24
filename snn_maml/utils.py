import typing
import torch
import numpy as np

from collections import OrderedDict

def compute_accuracy(logits, targets):
    """Compute the accuracy"""
    with torch.no_grad():
        _, predictions = torch.max(logits, dim=1)
        accuracy = torch.mean(predictions.eq(targets).float())
    return accuracy.item()


def compute_accuracy_lava(logits, targets):
    """Compute the accuracy of lava spike train using rate coding"""
    # Assuming that the spike train in its entirety is given
    with torch.no_grad():
        rate = torch.mean(logits,dim=-1)
        predictions = torch.max(rate.reshape(logits.shape[0], -1), dim=1)[1]
        accuracy = torch.mean(predictions.eq(targets).float())
    return accuracy.item()
        


def to_cuda(x):
    try:
        return x.cuda()
    except:
        return torch.from_numpy(x).float().cuda()


def to_tensor(x):
    if type(x) == np.ndarray:
        return torch.from_numpy(x).float()
    elif type(x) == torch.Tensor:
        return x
    else:
        print("Type error. Input should be either numpy array or torch tensor")
    

def to_device(x, GPU=False):
    if GPU:
        return to_cuda(x)
    else:
        return to_tensor(x)
    
    
def to_numpy(x):
    if type(x) == np.ndarray:
        return x
    else:
        try:
            return x.data.numpy()
        except:
            return x.cpu().data.numpy()


def tensors_to_device(tensors, device=torch.device('cpu')):
    """Place a collection of tensors in a specific device"""
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, (list, tuple)):
        return type(tensors)(tensors_to_device(tensor, device=device)
            for tensor in tensors)
    elif isinstance(tensors, (dict, OrderedDict)):
        return type(tensors)([(name, tensors_to_device(tensor, device=device))
            for (name, tensor) in tensors.items()])
    else:
        raise NotImplementedError()

class ToTensor1D(object):
    """Convert a `numpy.ndarray` to tensor. Unlike `ToTensor` from torchvision,
    this converts numpy arrays regardless of the number of dimensions.

    Converts automatically the array to `float32`.
    """
    def __call__(self, array):
        return torch.from_numpy(array.astype('float32'))

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
# def device_update_nonlin_symm(update_tensor, weight_tensor, threshold=.5):
#     return update_tensor*(1 - 1.66*weight_tensor)*(weight_tensor+threshold>=0).float()

class SoftSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        return  torch.sign(input_).type(input_.dtype)

    def backward(aux, grad_output):
        # grad_input = grad_output.clone()
        input, = aux.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < -.5] = 0
        grad_input[input > .5] = 0
        return grad_input
softsign = SoftSign.apply

# beta = -.14*1.25 *.2
# alpha = .1 *.2
# gamma = -1.25*1.25 *.2

@torch.no_grad()
def cg_solve(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10, x_init=None):
    """
    Goal: Solve Ax=b equivalent to minimizing f(x) = 1/2 x^T A x - x^T b
    Assumption: A is PSD, no damping term is used here (must be damped externally in f_Ax)
    Algorithm template from wikipedia
    Verbose mode works only with numpy
    """
       
    if type(b) == torch.Tensor:
        x = torch.zeros(b.shape[0]) if x_init is None else x_init
        x = x.to(b.device)
        if b.dtype == torch.float16:
            x = x.half()
        r = b - f_Ax(x)
        p = r.clone()
    elif type(b) == np.ndarray:
        x = np.zeros_like(b) if x_init is None else x_init
        r = b - f_Ax(x)
        p = r.copy()
    else:
        print("Type error in cg")

    fmtstr = "%10i %10.3g %10.3g %10.3g"
    titlestr = "%10s %10s %10s %10s"
    if verbose: print(titlestr % ("iter", "residual norm", "soln norm", "obj fn"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose:
            obj_fn = 0.5*x.dot(f_Ax(x)) - 0.5*b.dot(x)
            norm_x = torch.norm(x) if type(x) == torch.Tensor else np.linalg.norm(x)
            print(fmtstr % (i, r.dot(r), norm_x, obj_fn))

        rdotr = r.dot(r)
        Ap = f_Ax(p)
        alpha = rdotr/(p.dot(Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        newrdotr = r.dot(r)
        beta = newrdotr/rdotr
        p = r + beta * p
        
        if newrdotr < residual_tol:
            # print("Early CG termination because the residual was small")
            break

    if callback is not None:
        callback(x)
    if verbose: 
        obj_fn = 0.5*x.dot(f_Ax(x)) - 0.5*b.dot(x)
        norm_x = torch.norm(x) if type(x) == torch.Tensor else np.linalg.norm(x)
        print(fmtstr % (i, r.dot(r), norm_x, obj_fn))
    return x


@torch.enable_grad()
def hessian_vector_product(inner_loss, vector, params):
    """
    Performs hessian vector product on the train set in task with the provided vector
    """
    tloss = inner_loss
    grad_ft = torch.autograd.grad(tloss, params, create_graph=True, retain_graph=True)
    flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_ft])
    vec = vector.to(inner_loss.device)
    #h = torch.sum(flat_grad * vec)
    #hvp = torch.autograd.grad(h, params, create_graph=False, retain_graph=True)
    hvp2 = torch.autograd.grad(flat_grad, params, create_graph=False, retain_graph=True, grad_outputs=vec)

    #hvp_flat = torch.cat([g.contiguous().view(-1) for g in hvp])
    hvp2_flat = torch.cat([g.contiguous().view(-1) for g in hvp2])
    #print(hvp_flat.sum(),hvp2_flat.sum())
    return hvp2_flat

def matrix_evaluator(inner_loss, params, regu_coef=1.0, lamda=0.0, lam_damping=10.0):
    """
    Constructor function that can be given to CG optimizer
    Works for both type(lam) == float 
    """
    def evaluator(v):
        hvp = hessian_vector_product(inner_loss, v, params)
        Av = (1.0 + regu_coef) * v + hvp / (lamda + lam_damping)
        return Av
    return evaluator



def create_fixed_quantizers():
    import qtorch
    from qtorch import FixedPoint
    from qtorch.quant import quantizer
    from qtorch.optim import OptimLP

    fixed_quantizers = {}
    fixed_quantizers['16'] = quantizer(forward_number=FixedPoint(wl=16, fl=15, clamp=True),
                          forward_rounding="nearest",
                          backward_rounding="stochastic",
                          clamping_grad_zero=True)
    fixed_quantizers['8'] = quantizer(forward_number=FixedPoint(wl=8, fl=7, clamp=True),
                         forward_rounding="nearest",
                         backward_rounding="stochastic",
                         clamping_grad_zero=True)
    
    fixed_quantizers['4'] = quantizer(forward_number=FixedPoint(wl=4, fl=3, clamp=True, symmetric=True),
                         forward_rounding="nearest",
                         backward_rounding="stochastic",
                         clamping_grad_zero=True)
    fixed_quantizers['3'] = quantizer(forward_number=FixedPoint(wl=3,fl=1, clamp=True, symmetric=True),
                         forward_rounding="nearest",
                         backward_rounding="stochastic",
                         clamping_grad_zero=True)
    
    fixed_quantizers['2'] = quantizer(forward_number=FixedPoint(wl=2, fl=1, clamp=True, symmetric=True),
                         forward_rounding="nearest",
                         backward_rounding="stochastic",
                         clamping_grad_zero=True)
    
    fixed_quantizers['16s'] = quantizer(forward_number=FixedPoint(wl=16, fl=15, clamp=True),
                          forward_rounding="stochastic",
                          backward_rounding="stochastic",
                          clamping_grad_zero=True)
    fixed_quantizers['8s'] = quantizer(forward_number=FixedPoint(wl=8, fl=7, clamp=True),
                         forward_rounding="stochastic",
                         backward_rounding="stochastic",
                         clamping_grad_zero=True)
    
    fixed_quantizers['4s'] = quantizer(forward_number=FixedPoint(wl=4, fl=3, clamp=True, symmetric=True),
                         forward_rounding="stochastic",
                         backward_rounding="stochastic",
                         clamping_grad_zero=True)
    
    fixed_quantizers['3s'] = quantizer(forward_number=FixedPoint(wl=3,fl=1, clamp=True, symmetric=True),
                         forward_rounding="stochastic",
                         backward_rounding="stochastic",
                         clamping_grad_zero=True)
    
    fixed_quantizers['2s'] = quantizer(forward_number=FixedPoint(wl=2, fl=1, clamp=True, symmetric=True),
                         forward_rounding="stochastic",
                         backward_rounding="stochastic",
                         clamping_grad_zero=True)

    return fixed_quantizers

def quantize_parameters(params: OrderedDict, quantizer : typing.Callable):
    for (name, param) in params.items():
        params[name] = quantizer(param)
    return params
    
