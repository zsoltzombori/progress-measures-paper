
# %% auto 0
__all__ = ['Config', 'HookPoint', 'Embed', 'Unembed', 'PosEmbed', 'LayerNorm', 'Attention', 'MLP', 'TransformerBlock',
           'Transformer', 'make_fourier_basis', 'calculate_key_freqs', 'get_components_of_trig_loss',
           'calculate_excluded_loss', 'calculate_trig_loss', 'calculate_coefficients', 'gen_train_test', 'full_loss',
           'Trainer', 'train_model']

# %% ../transformer.ipynb 3
import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F
import einops
import random 
import helpers 
from dataclasses import dataclass
import os
import wandb
import copy
import argparse

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")



@dataclass(frozen = False)
class Config():
    # lr: float = 1e-3 #@param
    lr: float = 5*1e-4 #@param
    weight_decay: float = 1.0 #@param
    p: int = 113 #@param
    d_model: int = 128 #@param
    fn_name: str = 'add' #@param ['add', 'subtract', 'x2xyy2','rand']
    frac_train: float = 0.3 #@param
    # num_epochs: int = 50000 #@param
    num_epochs: int = 40000 #@param
    save_models: bool = True #@param
    save_every: int = 100 #@param

    # TODO for the first 1000 steps, save every 10 because 'interesting stuff happens at the start'
    # TODO add a helper function to generate indices here

    # Stop training when test loss is <stopping_thresh
    stopping_thresh: int = -1 #@param
    seed: int = 0 #@param

    num_layers: int = 1
    batch_style: str = 'full'
    d_vocab: int = p+1
    n_ctx: int = 3
    d_mlp: int = 4*d_model
    num_heads: int = 4

    act_type: str = 'ReLU' #@param ['ReLU', 'GeLU']


    device: t.device = t.device(DEVICE)

    # TODO ankify the privileged basis concept- a priori vs etc. ; consider writing up an explanation of privileged basis

    use_ln: bool = False

    take_metrics_every_n_epochs: int = 1000 #@param

    input_format=None
    embedding_matrix=None


    @property
    def d_head(self):
        return self.d_model // self.num_heads

    @property
    def random_answers(self):
        return np.random.randint(low=0, high=self.p, size=(self.p, self.p))

    @property 
    def fns_dict(self):
        return {
            'add': lambda x,y:(x+y) % self.p,
            'subtract': lambda x,y:(x-y) % self.p,
            'x2xyy2': lambda x,y:(x**2+x*y+y**2) % self.p,
            'rand': lambda x,y:self.random_answers[x][y]
            }

    @property
    def fn(self):
        return self.fns_dict[self.fn_name]

    def is_train_is_test(self, train, all_data):
        '''Creates an array of Boolean indices according to whether each data point is in train or test.
        Used to index into the big batch of all possible data'''

        batch1=len(all_data)
        batch2=len(train)
        is_train = np.array([(t.any(t.all((train==b).reshape(batch2, -1), dim=1))).item() for b in all_data])
        is_test = np.invert(is_train)
        return (is_train, is_test)

    def is_it_time_to_save(self, epoch):
        return (epoch % self.save_every == 0)

    def is_it_time_to_take_metrics(self, epoch):
        return epoch % self.take_metrics_every_n_epochs == 0

    def __init__(self,input_format):
        self.input_format=input_format
        if input_format == "word2vec":
            self.embedding_matrix = np.load('word2vec_abc_0830.npy')


# TODO make this an assert inside the consturctor
assert Config.d_model % Config.num_heads == 0

# %% ../transformer.ipynb 5
class HookPoint(nn.Module):
    '''A helper class to get access to intermediate activations (inspired by Garcon)
    It's a dummy module that is the identity function by default
    I can wrap any intermediate activation in a HookPoint and get a convenient way to add PyTorch hooks
    '''
    def __init__(self):
        super().__init__()
        self.fwd_hooks = []
        self.bwd_hooks = []
    
    def give_name(self, name):
        # Called by the model at initialisation
        self.name = name
    
    def add_hook(self, hook, dir='fwd'):
        # Hook format is fn(activation, hook_name)
        # Change it into PyTorch hook format (this includes input and output, 
        # which are the same for a HookPoint)
        def full_hook(module, module_input, module_output):
            return hook(module_output, name=self.name)
        if dir=='fwd':
            handle = self.register_forward_hook(full_hook)
            self.fwd_hooks.append(handle)
        elif dir=='bwd':
            handle = self.register_backward_hook(full_hook)
            self.bwd_hooks.append(handle)
        else:
            raise ValueError(f"Invalid direction {dir}")
    
    def remove_hooks(self, dir='fwd'):
        if (dir=='fwd') or (dir=='both'):
            for hook in self.fwd_hooks:
                hook.remove()
            self.fwd_hooks = []
        if (dir=='bwd') or (dir=='both'):
            for hook in self.bwd_hooks:
                hook.remove()
            self.bwd_hooks = []
        if dir not in ['fwd', 'bwd', 'both']:
            raise ValueError(f"Invalid direction {dir}")
    
    def forward(self, x):
        return x
    

# %% ../transformer.ipynb 6
class Embed(nn.Module):
    '''Define network architecture
    I defined my own transformer from scratch so I'd fully understand each component 
    - I expect this wasn't necessary or particularly important, and a bunch of this replicates existing Pyt functionality
    '''
    def __init__(self, d_vocab, d_model, input_format="onehot"):
        super().__init__()
        self.W_E = nn.Parameter(t.randn(d_model, d_vocab)/np.sqrt(d_model))
        self.input_format = input_format
        self.d_vocab = d_vocab
    
    def forward(self, x):
        if self.input_format == "orig":
            return t.einsum('dbp -> bpd', self.W_E[:, x])
        else:
            return t.einsum("dp,...ip -> ...id", self.W_E, x)


#| export
class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_U = nn.Parameter(t.randn(d_model, d_vocab)/np.sqrt(d_vocab))
    
    def forward(self, x):
        return (x @ self.W_U)

#| export
class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        self.W_pos = nn.Parameter(t.randn(max_ctx, d_model)/np.sqrt(d_model))
    
    def forward(self, x):
        return x+self.W_pos[:x.shape[-2]]

#| export
class LayerNorm(nn.Module):
    def __init__(self, d_model, epsilon = 1e-4, model=[None]):
        super().__init__()
        self.model = model
        self.w_ln = nn.Parameter(t.ones(d_model))
        self.b_ln = nn.Parameter(t.zeros(d_model))
        self.epsilon = epsilon
    
    def forward(self, x):
        if self.model[0].use_ln:
            x = x - x.mean(axis=-1)[..., None]
            x = x / (x.std(axis=-1)[..., None] + self.epsilon)
            x = x * self.w_ln
            x = x + self.b_ln
            return x
        else:
            return x

#| export
class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, n_ctx, model):
        super().__init__()
        self.model = model
        self.W_K = nn.Parameter(t.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_Q = nn.Parameter(t.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_V = nn.Parameter(t.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_O = nn.Parameter(t.randn(d_model, d_head * num_heads)/np.sqrt(d_model))
        self.register_buffer('mask', t.tril(t.ones((n_ctx, n_ctx))))
        self.d_head = d_head
        self.hook_k = HookPoint()
        self.hook_q = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn = HookPoint()
        self.hook_attn_pre = HookPoint()

    def forward(self, x):
        k = self.hook_k(t.einsum('ihd,bpd->biph', self.W_K, x))
        q = self.hook_q(t.einsum('ihd,bpd->biph', self.W_Q, x))
        v = self.hook_v(t.einsum('ihd,bpd->biph', self.W_V, x))
        attn_scores_pre = t.einsum('biph,biqh->biqp', k, q)
        attn_scores_masked = t.tril(attn_scores_pre) - 1e10 * (1 - self.mask[:x.shape[-2], :x.shape[-2]])
        attn_matrix = self.hook_attn(F.softmax(self.hook_attn_pre(attn_scores_masked/np.sqrt(self.d_head)), dim=-1))
        z = self.hook_z(t.einsum('biph,biqp->biqh', v, attn_matrix))
        z_flat = einops.rearrange(z, 'b i q h -> b q (i h)')
        out = t.einsum('df,bqf->bqd', self.W_O, z_flat)
        return out

#| export
class MLP(nn.Module):
    def __init__(self, d_model, d_mlp, act_type, model):
        super().__init__()
        self.model = model
        self.W_in = nn.Parameter(t.randn(d_mlp, d_model)/np.sqrt(d_model))
        self.b_in = nn.Parameter(t.zeros(d_mlp))
        self.W_out = nn.Parameter(t.randn(d_model, d_mlp)/np.sqrt(d_model))
        self.b_out = nn.Parameter(t.zeros(d_model))
        self.act_type = act_type
        # self.ln = LayerNorm(d_mlp, model=self.model)
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()
        assert act_type in ['ReLU', 'GeLU']
        
    def forward(self, x):
        x = self.hook_pre(t.einsum('md,bpd->bpm', self.W_in, x) + self.b_in)
        if self.act_type=='ReLU':
            x = F.relu(x)
        elif self.act_type=='GeLU':
            x = F.gelu(x)
        x = self.hook_post(x)
        x = t.einsum('dm,bpm->bpd', self.W_out, x) + self.b_out
        return x

# export
class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model):
        super().__init__()
        self.model = model
        # self.ln1 = LayerNorm(d_model, model=self.model)
        self.attn = Attention(d_model, num_heads, d_head, n_ctx, model=self.model)
        # self.ln2 = LayerNorm(d_model, model=self.model)
        self.mlp = MLP(d_model, d_mlp, act_type, model=self.model)
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()
    
    def forward(self, x):
        x = self.hook_resid_mid(x + self.hook_attn_out(self.attn((self.hook_resid_pre(x)))))
        x = self.hook_resid_post(x + self.hook_mlp_out(self.mlp((x))))
        return x

#| export
class Transformer(nn.Module):
    def __init__(self, config: Config, use_cache=False, use_ln=True):
        '''this function could be augmented to contain more options for creating different architectures'''
        super().__init__()
        self.cache = {}
        self.use_cache = use_cache
        self.embed = Embed(d_vocab = config.d_vocab, d_model = config.d_model, input_format=config.input_format)
        self.pos_embed = PosEmbed(max_ctx = config.n_ctx, d_model = config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model = config.d_model,
            d_mlp = config.d_mlp,
            d_head = config.d_head,
            num_heads = config.num_heads,
            n_ctx = config.n_ctx,
            act_type = config.act_type,
            model=[self]) for i in range(config.num_layers)])
        self.unembed = Unembed(d_vocab = config.d_vocab, d_model = config.d_model)
        self.use_ln = use_ln

        for name, module in self.named_modules():
            if type(module)==HookPoint:
                module.give_name(name)
    
    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        # x = self.ln(x)
        x = self.unembed(x)
        return x

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache
    
    def hook_points(self):
        return [module for name, module in self.named_modules() if 'hook' in name]

    def remove_all_hooks(self):
        for hp in self.hook_points():
            hp.remove_hooks('fwd')
            hp.remove_hooks('bwd')
    
    def cache_all(self, cache, incl_bwd=False):
        # Caches all activations wrapped in a HookPoint
        def save_hook(tensor, name):
            cache[name] = tensor.detach()
        def save_hook_back(tensor, name):
            cache[name+'_grad'] = tensor[0].detach()
        for hp in self.hook_points():
            hp.add_hook(save_hook, 'fwd')
            if incl_bwd:
                hp.add_hook(save_hook_back, 'bwd')

# %% ../transformer.ipynb 7
def make_fourier_basis(config: Config):
    fourier_basis = []
    fourier_basis.append(t.ones(config.p)/np.sqrt(config.p))
    fourier_basis_names = ['Const']
    # Note that if p is even, we need to explicitly add a term for cos(kpi), ie 
    # alternating +1 and -1
    for i in range(1, config.p//2 +1):
        fourier_basis.append(t.cos(2*t.pi*t.arange(config.p)*i/config.p))
        fourier_basis.append(t.sin(2*t.pi*t.arange(config.p)*i/config.p))
        fourier_basis[-2]/=fourier_basis[-2].norm()
        fourier_basis[-1]/=fourier_basis[-1].norm()
        fourier_basis_names.append(f'cos {i}')
        fourier_basis_names.append(f'sin {i}')
    return t.stack(fourier_basis, dim=0).to(config.device)


def calculate_key_freqs(config: Config, model: Transformer, all_data):
    cache = {}
    model.remove_all_hooks() # TODO is this line fucky??
    model.cache_all(cache)
    model(all_data)
    neuron_acts = cache['blocks.0.mlp.hook_post'][:, -1]
    # Center the neurons to remove the constant term
    neuron_acts_centered = neuron_acts - einops.reduce(neuron_acts, 'batch neuron -> 1 neuron', 'mean')
    # Note that fourier_neuron_acts[(0, 0), i]==0 for all i, because we centered the activations
    fourier_basis = make_fourier_basis(config = config)
    fourier_neuron_acts = helpers.fft2d(neuron_acts_centered, p = config.p, fourier_basis=fourier_basis)

    fourier_neuron_acts_square = fourier_neuron_acts.reshape(config.p, config.p, config.d_mlp)
    neuron_freqs = []
    neuron_frac_explained = []
    for ni in range(config.d_mlp):
        best_frac_explained = -1e6
        best_freq = -1
        for freq in range(1, config.p//2):
            # We extract the linear and quadratic fourier terms of frequency freq, 
            # and look at how much of the variance of the full vector this explains
            # If neurons specialise into specific frequencies, one frequency should 
            # have a large value
            numerator = helpers.extract_freq_2d(fourier_neuron_acts_square[:, :, ni], freq, p = config.p).pow(2).sum()
            denominator = fourier_neuron_acts_square[:, :, ni].pow(2).sum().item()
            frac_explained = numerator / denominator
            if frac_explained > best_frac_explained:
                best_freq = freq
                best_frac_explained = frac_explained
        neuron_freqs.append(best_freq)
        neuron_frac_explained.append(best_frac_explained)
    neuron_freqs = np.array(neuron_freqs)
    neuron_frac_explained = helpers.to_numpy(neuron_frac_explained)
    key_freqs, neuron_freq_counts = np.unique(neuron_freqs, return_counts=True)
    return key_freqs

def get_components_of_trig_loss(logits, freq, fourier_basis):
    cos = helpers.get_component_cos_xpy(logits, freq, fourier_basis=fourier_basis)
    sin = helpers.get_component_sin_xpy(logits, freq, fourier_basis=fourier_basis)
    return cos + sin


def calculate_excluded_loss(config: Config, fourier_basis, key_freqs, is_train, is_test, labels, logits):
    row = []
    for freq in key_freqs:
        cos = helpers.get_component_cos_xpy(logits, freq, fourier_basis=fourier_basis) 
        sin = helpers.get_component_sin_xpy(logits, freq, fourier_basis=fourier_basis) 
        value = helpers.test_logits(logits - cos - sin, bias_correction=False, mode='train', p = config.p,
           is_train = is_train, is_test = is_test, labels = labels)
        row.append(value.item())
    return row

def calculate_trig_loss(config: Config, model, train, logits, key_freqs, fourier_basis, is_train, is_test, labels, mode='all'):
    trig_logits = sum([get_components_of_trig_loss(logits, freq, fourier_basis) for freq in key_freqs])
    return helpers.test_logits(trig_logits, 
                        p = config.p,
                        is_train = is_train, 
                        is_test = is_test,
                        labels = labels,
                        bias_correction=True, 
                        original_logits=logits, 
                        mode=mode)


def calculate_coefficients(logits, fourier_basis, key_freqs, p, device):
    '''updated version from https://colab.research.google.com/drive/1ScVRL8OCtTFpOHpgfz0PLTFvX4g_YbuN?usp=sharing#scrollTo=WY4nPUDwl9UN
    '''
    x = t.arange(p)[None, :, None, None]
    y = t.arange(p)[None, None, :, None]
    z = t.arange(p)[None, None, None, :]
    w = t.arange(1, (p//2+1))[:, None, None, None]
    coses = t.cos(w*t.pi*2/p * (x + y - z)).to(device)
    coses = coses.reshape(p//2, p*p, p)
    coses/= coses.pow(2).sum([-2, -1], keepdim=True).sqrt()
    cos_coefficients = (coses * logits).sum([-2, -1])
    return cos_coefficients

# %% ../transformer.ipynb 8
# TODO move this into the config?
import dataclasses
from collections import defaultdict

def int_to_vec(n, input_format, d_vocab, embedding_matrix):
    result = t.zeros(d_vocab)
    if input_format == "onehot":
        result[n] = 1.0
    elif input_format == "binary":
        for i in range(d_vocab):
            if n == 0:
                break
            if (n % 2 != 0):
                result[i] = 1.0
            n = n // 2
    elif input_format == "binary_norm":
        ones = 0
        for i in range(d_vocab):
            if n == 0:
                break
            if (n % 2 != 0):
                result[i] = 1.0
                ones += 1
            n = n // 2
        if ones > 0:
            result /= np.sqrt(ones)
    elif input_format == "unary":
        for i in range(n):
            result[i] = 1.0
    elif input_format == "unary_norm":
        for i in range(n):
            result[i] = 1.0
        if n > 0:
            result /= np.sqrt(n)
    elif input_format == "onecold":
        result[n] = 1.0
        result = 1.0 - result
    elif input_format == "onecold_norm":
        result[n] = 1.0
        result = 1.0 - result
        result /= np.sqrt(d_vocab-1)
    elif input_format == "word2vec":
        embedding = embedding_matrix[n]
        for i in range(len(embedding)):
            result[i] = embedding[i]
    return result

def gen_train_test(config: Config):
    '''Generate train and test split'''
    num_to_generate = config.p
    pairs = [(i, j, num_to_generate) for i in range(num_to_generate) for j in range(num_to_generate)]
    labels = [config.fn(i, j) for i, j, _ in pairs]
    random.seed(config.seed)

    if config.input_format != "orig":
        pairs = [t.stack([int_to_vec(element, config.input_format, config.d_vocab, config.embedding_matrix) for element in item]) for item in pairs]

    pairs0 = pairs
    labels0 = labels
    
    temp = list(zip(pairs, labels))
    random.shuffle(temp)
    pairs, labels = zip(*temp)
    
    div = int(config.frac_train*len(pairs))
    if config.input_format == "orig":
        return pairs[:div], labels[:div], pairs[div:], labels[div:], pairs, labels
    else:
        train_data = t.stack(pairs[:div]).to(DEVICE)
        test_data = t.stack(pairs[div:]).to(DEVICE)
        all_data = t.stack(pairs0).to(DEVICE)
        train_labels = t.tensor(labels[:div]).to(DEVICE)
        test_labels = t.tensor(labels[div:]).to(DEVICE)
        all_labels = t.tensor(labels0).to(DEVICE)
        return train_data, train_labels, test_data, test_labels, all_data, all_labels

# TODO what type for model?
def full_loss(model: Transformer, data, labels):
    '''Takes the cross entropy loss of the model on the data'''
    # Take the final position only
    logits = model(data)[:, -1]
    return helpers.cross_entropy_high_precision(logits, labels)


class Trainer:
    '''TODO
    ways this stinks:
    - callbacks every k epochs 
    - training on infinite data
    - general abstract class w/o assumption and subclasses w/ more assumptions
    - check out hugging face trainer
    - disentangle optimization step and taking gradients
    - forward compatibility, e.g. batches per step
    '''

    def __init__(self, config : Config, model = None) -> None:
        # wandb.init(project = "grokking", config = dataclasses.asdict(config))
        self.model = model if model is not None else Transformer(config, use_cache=False)
        self.model.to(config.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr = config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.98))
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: min(step/10, 1)) # TODO make this a config option
        self.run_name = f"grok_{int(time.time())}"
        self.train_data, self.train_labels, self.test_data, self.test_labels, self.all_data, self.all_labels = gen_train_test(config = config)
        self.metrics_dictionary = defaultdict(dict) # so we can safely call 'update' on keys
        print('training length = ', len(self.train_data))
        print('testing length = ', len(self.test_data))
        self.train_losses = []
        self.test_losses = []
        self.config = config
        self.state_dicts = []
        self.epochs = []

    def save_epoch(self, epoch, save_to_wandb = True):
        ''' precondition! train loss and test losses have been appended to '''
        save_dict = {
            'model': self.model.state_dict(),
            'train_loss': self.train_losses[-1],
            'test_loss': self.test_losses[-1],
            'epoch': epoch,
        }

        tmp_model = copy.deepcopy(self.model)
        self.state_dicts.append(tmp_model.state_dict())
        self.epochs.append(epoch)
        
        # if save_to_wandb:
            # wandb.log(save_dict)
            # print("Saved epoch to wandb")
        if self.config.save_models: 
            t.save(save_dict, helpers.root/self.run_name/f"{epoch}.pth")
            print(f"Saved model to {helpers.root/self.run_name/f'{epoch}.pth'}")
        self.metrics_dictionary[epoch].update(save_dict)

    def do_a_training_step(self, epoch: int):
        '''returns train_loss, test_loss'''
        train_loss = full_loss(model = self.model, data = self.train_data, labels=self.train_labels)
        test_loss = full_loss(model = self.model, data = self.test_data, labels=self.test_labels)
        self.train_losses.append(train_loss.item())
        self.test_losses.append(test_loss.item())
        if epoch % 100 == 0:
            # TODO is this ok? this was np.log, and it was barking at me ; i think np.log was being interpreted as a logging module
            print(f'Epoch {epoch}, train loss {t.log(train_loss).item():.4f}, test loss {t.log(test_loss).item():.4f}')
            # print(f'Epoch {epoch}, train loss {train_loss.item():.4f}, test loss {test_loss.item():.4f}')
        train_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        return train_loss, test_loss

    def initial_save_if_appropriate(self):
        if self.config.save_models:
            os.mkdir(helpers.root/self.run_name)
            save_dict = {
                'model': self.model.state_dict(),
                'train_data' : self.train_data,
                'test_data' : self.test_data}
            t.save(save_dict, helpers.root/self.run_name/'init.pth')


    def post_training_save(self, save_optimizer_and_scheduler = True, log_to_wandb = True):
        if not self.config.save_models:
            os.makedirs(helpers.root/self.run_name, exist_ok=True)
        save_dict = {
            'config': self.config,
            'model': self.model.state_dict(),
            'train_loss': self.train_losses[-1],
            'test_loss': self.test_losses[-1],
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'epoch': self.config.num_epochs,
            'state_dicts': self.state_dicts,
            'epochs': self.epochs,
        }
        if save_optimizer_and_scheduler:
            save_dict['optimizer'] = self.optimizer.state_dict()
            save_dict['scheduler'] = self.scheduler.state_dict()
        # if log_to_wandb:
            # wandb.log(save_dict)
        outfile =f'{helpers.outdir}/{self.config.input_format}.pth'
        # t.save(save_dict, helpers.root/self.run_name/f"final.pth")
        # print(f"Saved model to {helpers.root/self.run_name/f'final.pth'}")
        t.save(save_dict, outfile)
        print(f"Saved model to {outfile}")
        self.metrics_dictionary[save_dict['epoch']].update(save_dict)


    def take_metrics(self, train, epoch, all_data, all_labels):
        with t.inference_mode():
            def sum_sq_weights():
                # TODO refactor- taken from app code
                row = []
                for name, param in self.model.named_parameters():
                    row.append(param.pow(2).sum().item())
                return row

            print('taking metrics')

            # TODO calculate key freqs is the most expensive part of this
            key_freqs = calculate_key_freqs(config = self.config, model = self.model, all_data = all_data)
            logits = self.model(all_data)[:, -1, :-1] # TODO i think this is equivalent to what's in the new paper?
            fourier_basis = make_fourier_basis(config = self.config)
            is_train, is_test = self.config.is_train_is_test(train = train, all_data=all_data)

            metrics = {
                'epoch': epoch, 
                'trig_loss': calculate_trig_loss(config = self.config,
                    model = self.model,
                    train = train,
                    key_freqs = key_freqs,
                    is_test=is_test,
                    is_train=is_train,
                    labels=all_labels,
                    logits = logits,
                    fourier_basis=fourier_basis),
                'sum_of_squared_weights': sum_sq_weights(),
                'excluded_loss': calculate_excluded_loss(
                    logits = logits,
                    key_freqs = key_freqs,
                    fourier_basis = fourier_basis,
                    is_train=is_train,
                    config = self.config,
                    is_test = is_test,
                    labels=all_labels),
                'coefficients': calculate_coefficients(p = self.config.p, logits = logits, fourier_basis = fourier_basis, key_freqs = key_freqs, device = self.config.device),
            }
            # wandb.log(metrics)
            # print("Logged metrics to wandb")
            self.metrics_dictionary[epoch].update(metrics)

def train_model(config: Config):
    world = Trainer(config = config)
    print(f'Run name {world.run_name}')
    world.initial_save_if_appropriate()

    for epoch in range(config.num_epochs):
        train_loss, test_loss = world.do_a_training_step(epoch)
        if test_loss.item() < config.stopping_thresh:
            break
        if config.is_it_time_to_save(epoch = epoch):
            # TODO this also used to do a check about test loss- pretty sure not necessary
            world.save_epoch(epoch = epoch, save_to_wandb=False)
        if config.is_it_time_to_take_metrics(epoch = epoch):
            world.take_metrics(epoch = epoch, train = world.train_data, all_data=world.all_data, all_labels=world.all_labels)

    world.post_training_save(save_optimizer_and_scheduler=True)
    helpers.lines([world.train_losses, world.test_losses], labels=['train', 'test'], log_y=True)
    return world # to export the dictionary with the training metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_format', dest="input_format", default="onehot", help="onehot,unary,binary,onecold,onecold_norm")
    args = parser.parse_args()
    
    config = Config(args.input_format)
    train_model(config)
