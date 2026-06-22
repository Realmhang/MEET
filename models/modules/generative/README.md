## Independent Framework of Iterative Generative Models

This folder contains a self-implemented framework for iterative generative models commonly used in AI4Science (typically Diffusion and Flow Matching). If you would like to contribute or apply this framework to a specific task, please read the following tutorials. **Feedback on theoretical inaccuracies or engineering inefficiencies is highly encouraged.**

### Theoretical Concepts

The basic idea is that the forward corruption process in both Diffusion and Flow Matching can be formulated as an interpolation between a data point $x_0\sim p_0$ and a prior point $x_1\sim p_1$, i.e.

$$x_t=\eta(t)x_0 + \sigma(t)x_1$$

From the Flow Matching perspective, the corresponding vector field is given by the derivative as

$$v(x_t, t)=\eta'(t)x_0 + \sigma'(t)x_1$$

In Euclidean space, the prior distribution $p_1$ is usually selected as the standard normal distribution $N(0,I)$. According to the linearity of the Gaussian distribution, $x_t$ also follows an gaussian distribution as $q(x_t|x_0)\sim N(\eta(t)x_0, \sigma^2(t)I)$, where the corresponding score is defined as

$$s(x_t, t) = -\frac{x_1}{\sigma(t)}$$

For Diffusion models, the interpolation also has a Markovian SDE form as

$$dx = f(x,t)dt + g(t)dW$$

In particular, DDPM defines a family of Variance Preserving (VP) paths, where

$$f(x,t) = -1/2\beta(t)x, g(t)=\sqrt(\beta(t))$$

Such definitation also leads to closed-form interpolation coefficients:

$$\eta(t)=\exp(-0.5\int_0^t beta(s) ds), \sigma_t=\sqrt(1-\eta^2(t))$$

For generation, we can either solve the reversed ODE

$$x_0 = x_1 + \int_1^0 v(x_t, t) dt$$

or the reversed SDE

$$x_0 = x_1 + \int_1^0 f(x_t, t) - g^2(t) s(x_t, t) dt + \int_1^0 g(t) dW$$

### Implementation Details

#### Class Hierarchy

This folder models the training and generation process of DM/FM via the following 5 core classes:

- `XtInfo` in `path.py`: Determine the interface of x_t state. Specifically, it associates `xt` with three optional tensors `x0`, `eps` and `vector_field`. Once the model predict one of them, the other two can be solved linearly.
- `ProbabilityPath` in `path.py`: Define the interpolation paths introduced above.
- `BaseSampler` in `sampler.py`: Define the one-step update of (first-order) solvers.
- `IterativeGenerativeModel` in `base_model.py`: An abstract class integrate the backbone model and the multi-field paths. This serves as the main interface for applying the framework and is explained in detail below.
- `Solver` in `solver.py`: The final SDE/ODE solver wraps the trained model and one-step samplers.

#### Quick Start

If you want to apply this folder as a black-box for your own task, the only component you need to implement by yourself is to instantiate the base `IterativeGenerativeModel` for your own data.

Its `__init__` function takes 5 arguments, which are detailed as

```
backbone: the neural network for prediction. Note that there is no explicit constraints on the input/output interfaces of the backbone model itself.
path: a dict of ProbabilityPath for each field. The key is the name of the each field and the value is the corresponding ProbabilityPath.
model_targets: a dict to define the model prediction of each field, including "x0", "eps", "minus_eps", "vector_field".
time_scheduler: the scheduler for time sampling. Default sampler yield uniform time from [0,1].
loss_weights: a dict to define loss weights for different fields. If None (default), all weights are set to 1.0.
```

You should also instantiate the following abstract methods:

- `preprocess_data`

```
@abstractmethod
def preprocess_data(self, batch):
    '''
        preprocess the data for training
        :param batch: the input batch data

        :return: 
            a tuple containing:
            - x0: dict[str, torch.Tensor] the original data for each path
            - batch_size: batch size
            - batch_ids: optional batch ids for variable sized data
            - context_info: other information if needed. A simplest way is to directly return the input batch itself
    '''
    pass
```

This function is actually the input interface. Its main idea is to decompose the arbitrary raw batch data into diffusable fields and the other contextual information. Some examples for the latter could be pocket graphs, pre-defined edge connections, e.t.c.. Note that if the diffusable data is not available in the given batch (e.g. inference from a new pocket without known ligands), you should still define the placeholder (e.g. torch.zeros) for x0 to enable prior sampling.

- `postprocess_data`

```
@abstractmethod
def postprocess_data(self, xt_infos: dict[str, XtInfo], t: torch.Tensor, batch_ids: Optional[torch.Tensor] = None, context_info = None):

    '''
        convert xt_infos back to whatever you like
    '''
    pass
```

This function is the output interface to combine the current x_t and the separated contextual information together into user defined outputs, e.g. PyG data or even CIF format files.

- `model_prediction`

```
@abstractmethod
def model_prediction(self, xt_infos: dict[str, XtInfo], t: torch.Tensor, batch_ids: Optional[torch.Tensor] = None, context_info = None) -> dict[str, torch.Tensor]:
    '''
        Given x_t and t, predict the target for the model
        :param xt_infos: the current state information at time t
        :param t: current time step
        :param batch_ids: optional batch ids for variable sized data
        :param context_info: other information if needed
        :return: the model prediction for each path
    '''
    pass
```

This function wraps the backbone model and yields the tensor-wise model prediction. That's why there's no requirements of the interfaces for the backbone model itself. Note that in the last two functions, the `context_info` directly use that decomposed from `preprocess_data`, hence you should maintain its completeness from the beginning.

This folder is self-contained and can be copied out as a standalone Python package.

#### Gallery of Paths and Samplers

Users can acquire the paths and samplers by two api functions:

```
from generative import make_path, make_sampler_partial
```

**Implemented paths:**

`make_path` takes a string input, which can be selected from:

- `LinearFlowMatching`: Standard linear interpolation
- `LinearDDPM`: DDPM path with Linear $\beta(t)$
- `CosineDDPM`: DDPM path with cosine-style $\eta(t)$



**Example samplers with hyperparameters:**

```
diffusion_sde = {
    'default': make_sampler_partial('DiffusionSDE')
}

diffusion_ode = {
    'default': make_sampler_partial('DiffusionODE')
}

diffusion_sde_conc = {
    'default': make_sampler_partial('DiffusionSDE',marginal_concentration_factor = 2.0)
}

diffusion_anc = {
    'default': make_sampler_partial('AncestralSampler')
}

ddim_sde = {
    'default': make_sampler_partial('DDIMSDE')
}

ddim_ode = {
    'default': make_sampler_partial('DDIMODE')
}

flow_matching = {
    'default': make_sampler_partial('FlowMatchingODE')
}

partials = {
    'Diffusion SDE': diffusion_sde,
    'Diffusion ODE': diffusion_ode,
    'Diffusion SDE (marginal concentration = 2.0)': diffusion_sde_conc,
    'Diffusion Ancestral Sampling': diffusion_anc,
    'Flow Matching ODE': flow_matching
}

diffusion_keys = ['Diffusion ODE', 'Diffusion SDE', 'Diffusion SDE (marginal concentration = 2.0)', 'Diffusion Ancestral Sampling', 'DDIMODE', 'DDIMSDE']
flow_keys = ['Flow Matching ODE']
```

One visualization on two moons example is shown as follows for first impressions.

![Two-Moon-20250926](./gym/generation_two_moons.png)
