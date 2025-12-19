import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

def get_activation_lipschitz(activation):
    """
    Infer Lipschitz constant from activation type.
    - None -> 1.0 (identity)
    - ReLU (F.relu, torch.relu, nn.ReLU) -> 1.0
    - SinusoidalActivation -> omega
    - FINER -> unbounded (math.inf)
    - GaussianActivation -> 1/(a*sqrt(e))
    """
    if activation is None:
        c = 1.0
    elif isinstance(activation, (nn.ReLU, torch.nn.modules.activation.ReLU)):
        c = 1.0
    elif activation in [F.relu, torch.relu]:
        c = 1.0
    elif isinstance(activation, GaussianActivation):
        c = 1 / (activation.a*math.sqrt(math.e))
    elif isinstance(activation, SinusoidalActivation):
        c = activation.omega.item()
    elif isinstance(activation, FinerActivation):
        # max of deriv. is global infinity, it is not bounded 
        # problem is the |x| in hidden layers as we cant assume [-1,1] range
        c = math.inf
    elif isinstance(activation, (GaborComplexActivation, GaborRealActivation)):
        omega = float(activation.omega.item())
        sigma = float(activation.sigma.item())
        if sigma == 0.0:
            c = abs(omega)
        elif omega**2 >= 2.0 * sigma**2:
            c = abs(omega)
        else:
            term = math.sqrt(2.0) * sigma * math.exp((omega**2) / (4.0 * sigma**2) - 0.5)
            c = max(abs(omega), term)
    else:
        raise ValueError(f"Unsupported activation type: {type(activation)}")
    return c

def matrix_info(weight, activation, eps=1e-12):
    """
    Compute spectral norm, Frobenius norm, stable rank, and 
    effective rank (Shannon entropy) for a weight matrix.
    """
    act_lip = get_activation_lipschitz(activation)
    
    # Detach from computational graph for analysis
    with torch.no_grad():
        weight_detached = weight.detach()
        
        # 1. Get singular values. 
        #    We add eps for numerical stability to prevent division by zero 
        #    or log(0), especially for singular matrices (min(sv)=0).
        sv = torch.linalg.svdvals(weight_detached) + eps
        
        # 2. Calculate norms from singular values
        spectral_norm = torch.max(sv)           # Spectral norm (L2 norm)
        fro_norm_squared = torch.sum(sv ** 2)
        fro_norm = torch.sqrt(fro_norm_squared) # Frobenius norm
        
        # 3. Calculate Stable Rank
        #    srank(W) = ||W||_F^2 / ||W||_2^2
        stable_rank = fro_norm_squared / (spectral_norm ** 2)
        
        # 4. Calculate Effective Rank (based on Shannon Entropy)
        #    p_i = sigma_i^2 / ||W||_F^2  (distribution of "energy")
        p_i = (sv ** 2) / fro_norm_squared
        #    Shannon Entropy: H = -sum(p_i * log(p_i))
        shannon_entropy = -torch.sum(p_i * torch.log(p_i))
        #    Effective Rank = exp(H)
        effective_rank = torch.exp(shannon_entropy)
        
        # 5. Calculate Spectral Condition Number
        #    cond(W) = max(sigma_i) / min(sigma_i)
        spectral_condition_no = spectral_norm / torch.min(sv)
    
    return {
        'linear_spectral_norm': spectral_norm.item(),
        'activation_spectral_norm': act_lip,
        'combined_spectral_norm': spectral_norm.item() * act_lip,
        'frobenius_norm': fro_norm.item(),
        'stable_rank': stable_rank.item(),
        'effective_rank': effective_rank.item(),  # <-- Here is the new metric
        'spectral_condition_no': spectral_condition_no.item(),
    }

class ActivatedLinear(nn.Module):
    """Linear layer with activation and spectral norm tracking."""
    def __init__(self, in_features, out_features, activation=None, layer_type="hidden", init_siren=False, omega_for_init=None, bias_scale=0.0, init_bias=False, init_mfn=False, weight_scale=1.0):
        super().__init__()

        self.activation = activation
        self.in_features = in_features
        self.out_features = out_features
        self.layer_type = layer_type
        self.init_mfn = init_mfn
        self.weight_scale = weight_scale       

        # Determine if the linear layer should be complex
        is_complex_layer = False
        if layer_type == "wire_last":
            # The final WIRE layer is always complex
            is_complex_layer = True
        elif isinstance(activation, GaborComplexActivation) and layer_type != "first":
            # This is a HIDDEN wire layer, so it should be complex.
            # The FIRST wire layer (layer_type == "first") remains real.
            is_complex_layer = True

        # Create the linear layer with the correct dtype
        if is_complex_layer:
            self.linear = nn.Linear(in_features, out_features, dtype=torch.cfloat)
        else:
            # This handles:
            # 1. All ReLU, Siren, Gaussian, etc. layers
            # 2. The FIRST GaborComplexActivation layer (layer_type == "first")
            self.linear = nn.Linear(in_features, out_features)

        # edge case SIREN:
        # we do this for all sinusoidal activations
        if isinstance(activation, (SinusoidalActivation, FinerActivation)):
            self.init_weights_siren(layer_type=self.layer_type,
                                    omega=self.activation.omega.item())
            
        # for regular linear layers only if and when we use SIREN init (!)
        if init_siren and self.layer_type in ["last"]:
            # Use the passed omega_for_init value here
            self.init_weights_siren(layer_type=self.layer_type, omega=omega_for_init)

        self.first_bias_scale = bias_scale
        if init_bias:
            self.init_bias()

        if init_mfn:
            self.init_weights_mfn(self.out_features, self.weight_scale)

    def init_weights_mfn(self, hidden_dim, weight_scale):
        """MFN-style initialization."""
        with torch.no_grad():
            limit = np.sqrt(weight_scale / hidden_dim)
            self.linear.weight.uniform_(-limit, limit)


    """ Weight init as in SIREN paper """
    def init_weights_siren(self, layer_type="first", omega=30):    
        with torch.no_grad():
            if layer_type == "first":
                print("Initializing first layer with SIREN init")
                # no division by omega here, as in the paper
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            elif layer_type == "hidden":
                print("Initializing hidden layer with SIREN init")
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / omega, 
                                             np.sqrt(6 / self.in_features) / omega)
            
            elif layer_type == "last":
                print(f"Initializing last layer with SIREN init using omega={omega}")
                # Check if omega was actually provided
                if omega is None:
                    raise ValueError("omega must be provided for the last layer's SIREN initialization.")
                
                # Corrected initialization with division by omega
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / omega, 
                                            np.sqrt(6 / self.in_features) / omega)
                
            else:
                raise ValueError(f"Unknown layer_type '{layer_type}' for SIREN initialization.")

    def init_bias(self):
        print(f"Initializing bias with uniform in [-{self.first_bias_scale}, {self.first_bias_scale}]")
        with torch.no_grad():
            self.linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)        
                    
    def forward(self, x):
        if self.activation is None:
            if self.layer_type == "wire_last":
                return self.linear(x).real
            else:
                return self.linear(x)    
        else:           
            return self.activation(self.linear(x))

    def get_info(self):
        return matrix_info(self.linear.weight, self.activation)

class SinusoidalActivation(nn.Module):
    """Sinusoidal activation function as described in SIREN, Sitzmann et al. NeurIPS'20.
    
    Implements the Gaussian activation: sin(omega * x)
    """
    def __init__(self, omega=1.0):
        super().__init__()
        self.register_buffer('omega', torch.tensor(omega))

    def forward(self, x):
        return torch.sin(self.omega * x)

class FinerActivation(nn.Module):
    """
    FINER activation function.

    Implements the activation: sin(omega * scale * x), where scale = |x| + 1.
    The scale factor introduces a data-dependent frequency modulation.
    """
    def __init__(self, omega=1.0, scale_req_grad=False):
        """
        Args:
            omega (float): The base frequency for the sinusoidal activation.
            scale_req_grad (bool): If True, gradients are backpropagated through the
                                   scale calculation. Defaults to False.
        """
        super().__init__()
        self.register_buffer('omega', torch.tensor(omega))
        self.scale_req_grad = scale_req_grad

    def forward(self, x):
        """Applies the FINER activation."""
        if self.scale_req_grad:
            scale = torch.abs(x) + 1
        else:
            with torch.no_grad():
                scale = torch.abs(x) + 1
        
        return torch.sin(self.omega * scale * x)

class GaussianActivation(nn.Module):
    """Gaussian activation function as described in Table 1, Beyond Periodicity: Towards a 
    Unifying Framework for Activations in Coordinate-MLPs, Ramasingh et al. ECCV'22.
    
    Implements the Gaussian activation: exp(-x²/(2a²))
    """
    def __init__(self, a=1.0):
        super().__init__()
        self.register_buffer('a', torch.tensor(a))

    def forward(self, x):
        return torch.exp(-0.5 * x**2 / (self.a**2))
    
class GaborComplexActivation(nn.Module):
    """Gabor activation as described in Saragadam WIRE'23.
    """
    def __init__(self, omega=10, sigma=10):
        super().__init__()
        self.register_buffer('omega', torch.tensor(omega))
        self.register_buffer('sigma', torch.tensor(sigma))

    def forward(self, x):
        if not x.is_complex():
            x = x.to(dtype=torch.cfloat)
        omega_lin = self.omega * x
        scale_lin = self.sigma * x
        # Complex Gabor: complex exponential with Gaussian envelope
        # 1:1 as in WIRE paper
        return torch.exp(1j * omega_lin - scale_lin.abs().square())   

class ReluMLP(nn.Module):
    """
    ReLU-activated MLP without FFs.
    """
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=3, num_layers=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # layers = [ActivatedLinear(input_dim, hidden_dim, torch.nn.ReLU())]
        layers = [ActivatedLinear(input_dim, hidden_dim, torch.nn.ReLU())]
        for i in range(num_layers - 2):
            layers.append(ActivatedLinear(hidden_dim, hidden_dim, torch.nn.ReLU()))

        # DO NOT CHANGE: we use ActivatedLinear to get the spectral norm info, even without activation
        # layers.append(ActivatedLinear(hidden_dim, 126, activation=None))
        layers.append(ActivatedLinear(hidden_dim, output_dim, activation=None))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    def get_layer_infos(self):
        return [m.get_info() for m in self.mlp if isinstance(m, ActivatedLinear)]

    def get_end_to_end_spectral_bound(self):
        total = 1.0
        for info in self.get_layer_infos():
            total *= info['combined_spectral_norm']
        return total

    def get_detailed_matrix_info(self):
        infos = self.get_layer_infos()
        return {
            'layer_infos': infos,
            'end_to_end_spectral_bound': self.get_end_to_end_spectral_bound(),
            'num_layers': len(infos)
        }

class ReluPosEncoding(nn.Module):
    """
    Positional Encoding Network using NeRF-style (Mildenhall et al.) encoding
    with ReLU activations.

    The mapping_size (L) defines the number of frequency bands used for encoding.
    """
    def __init__(self, input_dim=3, mapping_size=10, hidden_dim=256,
                 output_dim=3, num_layers=4):
        super().__init__()
        self.input_dim = input_dim
        # mapping_size (L) is now the number of frequency bands (L)
        self.L = mapping_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Calculate the required input dimension for the MLP
        # Each input dimension (e.g., x, y, z) is encoded into 2*L components (sin and cos).
        # Total MLP input dimension = input_dim * 2 * L
        num_freq_bands = self.L // 2
        mlp_input_dim = input_dim * 2 * num_freq_bands

        print("Using NeRF Positional Encoding with L =", self.L)
        print("MLP input dimension after encoding:", mlp_input_dim)

        # Build MLP
        layers = [ActivatedLinear(mlp_input_dim, hidden_dim, nn.ReLU())]
        for _ in range(num_layers - 2):
            layers.append(ActivatedLinear(hidden_dim, hidden_dim, nn.ReLU()))
        # DO NOT CHANGE: we use ActivatedLinear to get the spectral norm info
        layers.append(ActivatedLinear(hidden_dim, output_dim, activation=None))
        self.mlp = nn.Sequential(*layers)

    def positional_encoding(self, p):
        """
        Applies the NeRF positional encoding:
        γ(p) = [sin(2^0πp), cos(2^0πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp)]
        applied element-wise across the input features.

        Args:
            p (torch.Tensor): Input tensor (e.g., coordinates), shape (..., D).

        Returns:
            torch.Tensor: Encoded tensor, shape (..., D * 2 * L).
        """
        # 1. Expand input: (..., D) -> (..., D, 1)
        p_expanded = p.unsqueeze(-1)

        # 2. Frequencies: [2^0, 2^1, ..., 2^(L-1)]. Shape: (L,)
        frequencies = 2 ** torch.arange(
            self.L // 2,
            device=p.device,
            dtype=p.dtype
        )

        # 3. Apply frequencies and π: Shape (..., D, L)
        freq_p = frequencies * math.pi * p_expanded

        # 4. Compute sin and cos: Shape (..., D, L)
        sin_p = torch.sin(freq_p)
        cos_p = torch.cos(freq_p)

        # 5. Stack sin/cos and flatten the last two dimensions (L, 2) -> (2*L)
        # Shape: (..., D, 2*L)
        gamma_p = torch.stack([sin_p, cos_p], dim=-1).flatten(start_dim=-2)

        # 6. Flatten the remaining feature dimensions (D, 2*L) -> (D*2*L)
        # Final Shape: (..., D*2*L)
        return gamma_p.flatten(start_dim=-2)

    def forward(self, x):
        """
        Processes input x through positional encoding and the MLP.
        """
        encoded_x = self.positional_encoding(x)
        return self.mlp(encoded_x)

    def get_layer_infos(self):
        """
        Retrieves information from ActivatedLinear layers (assuming they provide a get_info method).
        """
        return [m.get_info() for m in self.mlp if isinstance(m, ActivatedLinear)]

    # RFF-specific methods (like get_fourier_features_lipschitz_constant, etc.)
    # are removed as they are not applicable to the NeRF positional encoding.

    def get_end_to_end_spectral_bound(self):
        """
        Provides a placeholder for the spectral bound calculation.
        Note: The Lipschitz constant for NeRF PE is L_max, which is 2*pi*2^(L-1).
        """
        # Lipschitz constant for NeRF PE is roughly 2*pi*2^(L-1) along each dimension.
        # This is a simplification; the exact end-to-end bound is context-dependent.
        pe_lipschitz_approx = 2 * math.pi * (2 ** (self.L - 1))
        total = pe_lipschitz_approx
        for info in self.get_layer_infos():
            total *= info['combined_spectral_norm']
        return total

    def get_detailed_matrix_info(self):
        """
        Provides detailed information about the network layers.
        """
        infos = self.get_layer_infos()
        return {
            'layer_infos': infos,
            'end_to_end_spectral_bound': self.get_end_to_end_spectral_bound(),
            'positional_encoding_bands': self.L,
            'num_layers': len(infos)
        }

class ReluFFN(nn.Module):
    """
    Fourier Feature Network (FFN) with ReLU activations.
    """
    def __init__(self, input_dim=2, mapping_size=128, hidden_dim=256,
                 output_dim=3, num_layers=4, sigma=5.0):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sigma = sigma
        
        # Random Fourier matrix (fixed, NOT learnable)
        B = torch.randn(mapping_size, input_dim) * sigma

        print(f"Fourier Feature mapping matrix B with shape {B.shape} created with sigma={sigma}")

        self.register_buffer('B', B)

        print("MLP input dimension after Fourier Feature mapping:", 2 * mapping_size)

        # Build MLP
        layers = [ActivatedLinear(2 * mapping_size, hidden_dim, nn.ReLU())]
        for _ in range(num_layers - 2): # 
            layers.append(ActivatedLinear(hidden_dim, hidden_dim, nn.ReLU()))
        # DO NOT CHANGE: we use ActivatedLinear to get the spectral norm info, even without activation
        layers.append(ActivatedLinear(hidden_dim, output_dim, activation=None))
        self.mlp = nn.Sequential(*layers)

    def fourier_feature_mapping(self, x):
        x_proj = torch.matmul(x, self.B.T)
        return torch.cat([torch.cos(2 * math.pi * x_proj), torch.sin(2 * math.pi * x_proj)], dim=-1)

    def forward(self, x):
        return self.mlp(self.fourier_feature_mapping(x))

    def get_fourier_features_lipschitz_constant(self):
        return 2 * math.pi * torch.linalg.matrix_norm(self.B, ord=2).item()
    
    def get_fourier_features_frobenius_constant(self):
        return 2 * math.pi * torch.linalg.matrix_norm(self.B, ord='fro').item()
    
    def get_fourier_features_vector_norm_constant(self):
        return 2 * math.pi * torch.linalg.matrix_norm(self.B, ord=1).item()
    
    def get_fourier_features_stable_rank(self):
        fro_norm = torch.linalg.matrix_norm(self.B, ord='fro').item()
        spec_norm = torch.linalg.matrix_norm(self.B, ord=2).item()
        return (fro_norm**2) / (spec_norm**2 + 1e-12)
    
    def get_fourier_features_spectral_condition_number(self):
        return torch.linalg.cond(self.B, p=2).item()

    def get_layer_infos(self):
        return [m.get_info() for m in self.mlp if isinstance(m, ActivatedLinear)]

    def get_end_to_end_spectral_bound(self):
        total = self.get_fourier_features_lipschitz_constant()
        for info in self.get_layer_infos():
            total *= info['combined_spectral_norm']
        return total

    def get_detailed_matrix_info(self):
        infos = self.get_layer_infos()
        return {
            'layer_infos': infos,
            'end_to_end_spectral_bound': self.get_end_to_end_spectral_bound(),
            'fourier_features_lipschitz': self.get_fourier_features_lipschitz_constant(),
            'fourier_matrix_spectral_norm': torch.linalg.matrix_norm(self.B, ord=2).item(),
            'fourier_matrix_frobenius_norm': torch.linalg.matrix_norm(self.B, ord='fro').item(),
            'fourier_matrix_stable_rank': self.get_fourier_features_stable_rank(),
            'fourier_matrix_spectral_condition_no': self.get_fourier_features_spectral_condition_number(),
            'fourier_sigma': self.sigma,
            'num_layers': len(infos)
        }

class SirenMLP(nn.Module):
    """
    Sinusoidal-activated MLP as in SIREN, Sitzmann et al. NeurIPS'20.
    """
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=3, num_layers=4, omega=30.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # omega could be a list for layer-specific values
        if isinstance(omega, (list, tuple)):
            if len(omega) != (num_layers-1):
                raise ValueError(f"Length of 'omega' list ({len(omega)}) must match num_layers -1 ({num_layers-1})")
            self.omega_values = omega
        else:
            self.omega_values = [omega] * (num_layers -1)

        # Build MLP with layer-specific 'a' values
        layers = [ActivatedLinear(input_dim, hidden_dim, SinusoidalActivation(omega=self.omega_values[0]), layer_type="first", init_siren=True)]
        for i in range(num_layers - 2):
            layers.append(ActivatedLinear(hidden_dim, hidden_dim, SinusoidalActivation(omega=self.omega_values[i+1]), layer_type="hidden", init_siren=True))
        # DO NOT CHANGE: we use ActivatedLinear to get the spectral norm info, even without activation
        # notice how we use init_siren=True here to trigger the special last-layer init, not the vanilla init for nn.Linear()
        #layers.append(ActivatedLinear(hidden_dim, output_dim, activation=None, layer_type="last", init_siren=True))
        #self.mlp = nn.Sequential(*layers)
        # For the final layer, pass the omega from the last hidden layer
        last_hidden_omega = self.omega_values[-1]
        layers.append(ActivatedLinear(hidden_dim, output_dim, activation=None, layer_type="last", 
                                      init_siren=True, omega_for_init=last_hidden_omega))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    def get_layer_infos(self):
        return [m.get_info() for m in self.mlp if isinstance(m, ActivatedLinear)]

    def get_end_to_end_spectral_bound(self):
        total = 1.0
        for info in self.get_layer_infos():
            total *= info['combined_spectral_norm']
        return total

    def get_detailed_matrix_info(self):
        infos = self.get_layer_infos()
        return {
            'layer_infos': infos,
            'end_to_end_spectral_bound': self.get_end_to_end_spectral_bound(),
            'omega_values': self.omega_values,
            'num_layers': len(infos)
        }

class FinerMLP(nn.Module):
    """
    Finer-activated MLP as in FINER, Liu et al. CVPR'24.
    """
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=3, num_layers=4, omega=30.0, scale_req_grad=False, init_bias=False, bias_scale=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # omega could be a list for layer-specific values
        if isinstance(omega, (list, tuple)):
            if len(omega) != (num_layers-1):
                raise ValueError(f"Length of 'omega' list ({len(omega)}) must match num_layers -1 ({num_layers-1})")
            self.omega_values = omega
        else:
            self.omega_values = [omega] * (num_layers -1)

        # Build MLP with layer-specific 'a' values
        layers = [ActivatedLinear(input_dim, hidden_dim, FinerActivation(omega=self.omega_values[0]), layer_type="first", init_siren=True, init_bias=init_bias, bias_scale=bias_scale)]
        for i in range(num_layers - 2):
            layers.append(ActivatedLinear(hidden_dim, hidden_dim, FinerActivation(omega=self.omega_values[i+1]), layer_type="hidden", init_siren=True))
        # DO NOT CHANGE: we use ActivatedLinear to get the spectral norm info, even without activation
        # notice how we use init_siren=True here to trigger the special last-layer init, not the vanilla init for nn.Linear()
        #layers.append(ActivatedLinear(hidden_dim, output_dim, activation=None, layer_type="last", init_siren=True))
        #self.mlp = nn.Sequential(*layers)
        # For the final layer, pass the omega from the last hidden layer
        last_hidden_omega = self.omega_values[-1]
        layers.append(ActivatedLinear(hidden_dim, output_dim, activation=None, layer_type="last", 
                                      init_siren=True, omega_for_init=last_hidden_omega))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    def get_layer_infos(self):
        return [m.get_info() for m in self.mlp if isinstance(m, ActivatedLinear)]

    def get_end_to_end_spectral_bound(self):
        total = 1.0
        for info in self.get_layer_infos():
            total *= info['combined_spectral_norm']
        return total

    def get_detailed_matrix_info(self):
        infos = self.get_layer_infos()
        return {
            'layer_infos': infos,
            'end_to_end_spectral_bound': self.get_end_to_end_spectral_bound(),
            'omega_values': self.omega_values,
            'num_layers': len(infos)
        }

class GaussMLP(nn.Module):
    """
    Gaussian-activated MLP without FFs.
    """
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=3, num_layers=4, a=5.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # a could be a list for layer-specific values
        if isinstance(a, (list, tuple)):
            if len(a) != (num_layers-1):
                raise ValueError(f"Length of 'a' list ({len(a)}) must match num_layers -1 ({num_layers-1})")
            self.a_values = a
        else:
            self.a_values = [a] * (num_layers -1)

        # Build MLP with layer-specific 'a' values
        layers = [ActivatedLinear(input_dim, hidden_dim, GaussianActivation(a=self.a_values[0]))]
        for i in range(num_layers - 2):
            layers.append(ActivatedLinear(hidden_dim, hidden_dim, GaussianActivation(a=self.a_values[i+1])))
        # DO NOT CHANGE: we use ActivatedLinear to get the spectral norm info, even without activation
        layers.append(ActivatedLinear(hidden_dim, output_dim, activation=None))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    def get_layer_infos(self):
        return [m.get_info() for m in self.mlp if isinstance(m, ActivatedLinear)]

    def get_end_to_end_spectral_bound(self):
        total = 1.0
        for info in self.get_layer_infos():
            total *= info['combined_spectral_norm']
        return total

    def get_detailed_matrix_info(self):
        infos = self.get_layer_infos()
        return {
            'layer_infos': infos,
            'end_to_end_spectral_bound': self.get_end_to_end_spectral_bound(),
            'gaussian_a_values': self.a_values,
            'num_layers': len(infos)
        }
    
class GaussFFN(nn.Module):
    """
    Gaussian Fourier Feature Network (FFN) with Gaussian activations.
    """
    def __init__(self, input_dim=2, mapping_size=128, hidden_dim=256,
                 output_dim=3, num_layers=4, sigma=5.0, a=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sigma = sigma

        # a could be a list for layer-specific values
        if isinstance(a, (list, tuple)):
            if len(a) != (num_layers -1):
                raise ValueError(f"Length of 'a' list ({len(a)}) must match num_layers -1 ({num_layers -1})")
            self.a_values = a
        else:
            self.a_values = [a] * (num_layers -1)
        
        # Random Fourier matrix (fixed, NOT learnable)
        B = torch.randn(mapping_size, input_dim) * sigma
        self.register_buffer('B', B)

        print(f"Fourier Feature mapping matrix B with shape {B.shape} created with sigma={sigma}")
        

        # Build MLP with layer-specific 'a' values
        layers = [ActivatedLinear(2 * mapping_size, hidden_dim, GaussianActivation(a=self.a_values[0]))]
        for i in range(num_layers - 2):
            layers.append(ActivatedLinear(hidden_dim, hidden_dim, GaussianActivation(a=self.a_values[i+1])))
        # DO NOT CHANGE: we use ActivatedLinear to get the spectral norm info, even without activation
        layers.append(ActivatedLinear(hidden_dim, output_dim, activation=None))
        self.mlp = nn.Sequential(*layers)

    def fourier_feature_mapping(self, x):
        x_proj = torch.matmul(x, self.B.T)
        return torch.cat([torch.cos(2 * math.pi * x_proj), torch.sin(2 * math.pi * x_proj)], dim=-1)

    def forward(self, x):
        return self.mlp(self.fourier_feature_mapping(x))

    def get_fourier_features_lipschitz_constant(self):
        return 2 * math.pi * torch.linalg.matrix_norm(self.B, ord=2).item()
    
    def get_fourier_features_frobenius_constant(self):
        return 2 * math.pi * torch.linalg.matrix_norm(self.B, ord='fro').item()
    
    def get_fourier_features_stable_rank(self):
        fro_norm = torch.linalg.matrix_norm(self.B, ord='fro').item()
        spec_norm = torch.linalg.matrix_norm(self.B, ord=2).item()
        return (fro_norm**2) / (spec_norm**2 + 1e-12)
    
    def get_fourier_features_spectral_condition_number(self):
        return torch.linalg.cond(self.B, p=2).item()

    def get_layer_infos(self):
        return [m.get_info() for m in self.mlp if isinstance(m, ActivatedLinear)]

    def get_end_to_end_spectral_bound(self):
        total = self.get_fourier_features_lipschitz_constant()
        for info in self.get_layer_infos():
            total *= info['combined_spectral_norm']
        return total

    def get_detailed_matrix_info(self):
        infos = self.get_layer_infos()
        return {
            'layer_infos': infos,
            'end_to_end_spectral_bound': self.get_end_to_end_spectral_bound(),
            'fourier_features_lipschitz': self.get_fourier_features_lipschitz_constant(),
            'fourier_matrix_spectral_norm': torch.linalg.matrix_norm(self.B, ord=2).item(),
            'fourier_matrix_frobenius_norm': torch.linalg.matrix_norm(self.B, ord='fro').item(),
            'fourier_matrix_stable_rank': self.get_fourier_features_stable_rank(),
            'fourier_matrix_spectral_condition_no': self.get_fourier_features_spectral_condition_number(),
            'fourier_sigma': self.sigma,
            'gaussian_a_values': self.a_values,
            'num_layers': len(infos)
        }

class WireMLP(nn.Module):
    """
    GaborWavelet-activated MLP as in WIRE, Saragadam et al. 
    """
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=3, num_layers=4, omega=10.0, sigma=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Reduce hidden features since complex numbers have 2x parameters, according to WIRE paper, to make it fair
        red_hidden_dim = int(hidden_dim / np.sqrt(2))
        self.hidden_dim = red_hidden_dim
        
        # omega could be a list for layer-specific values
        if isinstance(omega, (list, tuple)):
            if len(omega) != (num_layers-1):
                raise ValueError(f"Length of 'omega' list ({len(omega)}) must match num_layers -1 ({num_layers-1})")
            self.omega_values = omega
        else:
            self.omega_values = [omega] * (num_layers -1)
        
        # sigma could be a list for layer-specific values
        if isinstance(sigma, (list, tuple)):
            if len(sigma) != (num_layers-1):
                raise ValueError(f"Length of 'sigma' list ({len(sigma)}) must match num_layers -1 ({num_layers-1})")
            self.sigma_values = sigma
        else:
            self.sigma_values = [sigma] * (num_layers -1)

        # Build MLP with layer-specific 'omega' and 'sigma' values
        layers = [ActivatedLinear(input_dim, red_hidden_dim, GaborComplexActivation(omega=self.omega_values[0],sigma=self.sigma_values[0]), layer_type="first", init_siren=False)]
        for i in range(num_layers - 2):
            # these are complex layers
            layers.append(ActivatedLinear(red_hidden_dim, red_hidden_dim, GaborComplexActivation(omega=self.omega_values[i+1], sigma=self.sigma_values[i+1]), layer_type="hidden", init_siren=False))
        # DO NOT CHANGE: we use ActivatedLinear to get the spectral norm info, even without activation
        # notice how we use init_siren=True here to trigger the special last-layer init, not the vanilla init for nn.Linear()
        layers.append(ActivatedLinear(red_hidden_dim, output_dim, activation=None, layer_type="wire_last", init_siren=False))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        '''
        if not x.is_complex():
            x = x.to(dtype=torch.cfloat)
        return self.mlp(x)
        '''
        return self.mlp(x)

    def get_layer_infos(self):
        return [m.get_info() for m in self.mlp if isinstance(m, ActivatedLinear)]

    def get_end_to_end_spectral_bound(self):
        total = 1.0
        for info in self.get_layer_infos():
            total *= info['combined_spectral_norm']
        return total

    def get_detailed_matrix_info(self):
        infos = self.get_layer_infos()
        return {
            'layer_infos': infos,
            'end_to_end_spectral_bound': self.get_end_to_end_spectral_bound(),
            'omega_values': self.omega_values,
            'num_layers': len(infos)
        }

class GaborRealActivation(nn.Module):
    """
    Real-Valued Gabor activation as an alternate form of WIRE (Saragadam WIRE'23).
    Implements the activation: sin(omega * x) * exp(-(sigma * x)^2).
    """
    def __init__(self, omega=10, sigma=10):
        super().__init__()
        # omega controls the frequency of the sinusoid (sin term)
        self.register_buffer('omega', torch.tensor(omega))
        # sigma controls the spread/width of the Gaussian envelope (exp term)
        self.register_buffer('sigma', torch.tensor(sigma))

    def forward(self, x):
        # 1. Sinusoidal (periodic) component: sin(omega * x)
        sin_term = torch.sin(self.omega * x)
        
        # 2. Gaussian envelope (spatial compactness) component: exp(-(sigma * x)^2)
        # Note: (sigma * x)^2 is the same as (sigma * x).abs().square() for real x
        gaussian_term = torch.exp(-(self.sigma * x)**2)
        
        # 3. Element-wise multiplication to get the real Gabor wavelet
        return sin_term * gaussian_term

class WireRealMLP(nn.Module):
    """
    Real-Valued GaborWavelet-activated MLP (Alternate WIRE form).
    Uses GaborRealActivation and only standard real-valued linear layers.
    """
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=3, num_layers=4, omega=10.0, sigma=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # No need to reduce hidden_dim by sqrt(2) as all layers are now real-valued
        # and do not have doubled complex parameters.
        # (Assuming you would move that logic into this class)

        # omega could be a list for layer-specific values
        if isinstance(omega, (list, tuple)):
            if len(omega) != (num_layers-1):
                raise ValueError(f"Length of 'omega' list ({len(omega)}) must match num_layers -1 ({num_layers-1})")
            self.omega_values = omega
        else:
            self.omega_values = [omega] * (num_layers -1)
        
        # sigma could be a list for layer-specific values
        if isinstance(sigma, (list, tuple)):
            if len(sigma) != (num_layers-1):
                raise ValueError(f"Length of 'sigma' list ({len(sigma)}) must match num_layers -1 ({num_layers-1})")
            self.sigma_values = sigma
        else:
            self.sigma_values = [sigma] * (num_layers -1)

        # Build MLP with GaborRealActivation
        layers = []
        # First layer (real-valued linear layer)
        layers.append(ActivatedLinear(
            input_dim, hidden_dim, 
            GaborRealActivation(omega=self.omega_values[0], sigma=self.sigma_values[0]), 
            layer_type="first"
        ))
        
        # Hidden layers (real-valued linear layers)
        for i in range(num_layers - 2):
            layers.append(ActivatedLinear(
                hidden_dim, hidden_dim, 
                GaborRealActivation(omega=self.omega_values[i+1], sigma=self.sigma_values[i+1]), 
                layer_type="hidden"
            ))
            
        # Last layer (real-valued linear layer, no activation)
        layers.append(ActivatedLinear(hidden_dim, output_dim, activation=None, layer_type="last"))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    
    def get_layer_infos(self):
        return [m.get_info() for m in self.mlp if isinstance(m, ActivatedLinear)]

    def get_end_to_end_spectral_bound(self):
        total = 1.0
        for info in self.get_layer_infos():
            total *= info['combined_spectral_norm']
        return total

    def get_detailed_matrix_info(self):
        infos = self.get_layer_infos()
        return {
            'layer_infos': infos,
            'end_to_end_spectral_bound': self.get_end_to_end_spectral_bound(),
            'omega_values': self.omega_values,
            'num_layers': len(infos)
        }