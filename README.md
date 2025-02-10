## Casadi Warp
This repository shows how to combine the popular optimization framework casadi with NVIDIA Warp.

### Installation
First clone the repository:
```bash
git clone git@github.com:mgomezandreu/casadi_warp.git
```

To install the required packages, run the following command:
```bash
cd casadi_warp
pip install -r requirements.txt
```


# Related Blog Article
### Why Casadi Warp?
Casadi is a powerful tool for optimization and automatic differentiation. It is widely used in the optimization community. Unfortunatly there is currently no native GPU support for casadi.
Initial explorations to provide an OPENCL backend have been made by the casadi team, but currently there is no official version available.

This is very unfortunate, as models that have been trained on GPUs cannot be used in casadi without a significant performance hit.

[L4Casadi](https://github.com/Tim-Salzmann/l4casadi) is upcoming projects that allows the usage of pytorch models in casadi. This is a great step forward, but only works for models that can be expressed in pytorch.
Since it used TorchScript to convert the model, it is not possible to really customize the model, because TorchScript is limited in its capabilities.

Because my current project requires the usage of a custom model, I decided to make the first steps to accelerate casadi with [NVIDIA Warp](https://github.com/NVIDIA/warp).

### How it works
Casadi allows the user to define custom callback functions. These functions are then called by the optimization algorithm. The callback function is called for every iteration of the optimization algorithm. This is the perfect place to insert the GPU code.
Since there is limited documentation on how to define such a callback function, I will provide a small example here.

```python
from casadi import *

"""
f(y) = (y - 4) ** 4
f'(y) = 4 * (y - 4) ** 3
f''(y) = 3 * 4 * (y- 4) ** 2
"""

class F_cas(Callback):
    def __init__(self, name, xs,opts={}):
        Callback.__init__(self)
        self.construct(name, opts)

    def get_n_in(self): return 1
    def get_n_out(self): return 1

    def get_sparsity_in(self,i):
        return Sparsity.dense(1,1)

    def get_sparsity_out(self,i):
        return Sparsity.dense(1,1)

    # Evaluate numerically
    def eval(self, arg):
        y = arg[0].__float__()

        out = (y -4) ** 4 
        return [out]


    def has_jacobian(self): 
        return True
    def get_jacobian(self,name,inames,onames,opts):
        class JacFun(Callback):
            def __init__(self,opts={}):
                Callback.__init__(self)
                self.construct(name, opts)

            def get_n_in(self): return 2
            def get_n_out(self): return 1

            def get_sparsity_in(self,i):
                if i == 0:
                    return Sparsity.dense(1,1)
                else:
                    return Sparsity.dense(1,1)

            def get_sparsity_out(self,i):
                return Sparsity.dense(1,1)

            def eval(self, arg):
                print("input_jac",arg)
                y = arg[0].__float__()

                out = 4 * (y - 4) ** 3                    
                return [out]

            def has_jacobian(self): 
                    return True
            def get_jacobian(self,name,inames,onames,opts):

                class HessFun(Callback):
                    def __init__(self,opts={}):
                        Callback.__init__(self)
                        self.construct(name, opts)

                    def get_n_in(self): return 3
                    def get_n_out(self): return 2

                    def get_sparsity_in(self,i):
                        return Sparsity.dense(1,1)

                    def get_sparsity_out(self,i):
                        return Sparsity.dense(1,1)

                    def eval(self, arg):
                        y = arg[0].__float__()
                        hess = 3 * 4 * (y- 4) ** 2
                        return [hess, 0] # I have zero idea why this has to be the the output format but it is.


                self.hess_callback = HessFun()
                print("Returning HessFun")
                return self.hess_callback


        self.jac_callback = JacFun()
        print("Returning JacFun")
        return self.jac_callback



x = MX.sym('x')
func_custom = F_cas("f", x)

x = MX.sym('x')
cost = func_custom(x)

nlp = {'x': x, 'f': cost}
solver = nlpsol('solver', 'ipopt', nlp)
res = solver(x0=1.0)
print("Solution:", res['x'])
```

Using this structure we can then insert calls to GPU kernels, that evalute our cost function, it's jacobian and hessian, respectively.
This allows for fast computution of the quantities while still relying on the powerful optimization algorithms provided by casadi.
The following code shows a simple example of how to use the callback function to evaluate the cost function on the GPU.

```python
"""
f(y) = \sum_{i=1}^{n} (x_i - y)^2
f'(y) = -2 \sum_{i=1}^{n} (x_i - y)
f''(y) = -2 * n
"""


wp.init()
wp.set_device("cuda:0")
class F_cas(Callback):
    def __init__(self, name, xs,opts={}):
        Callback.__init__(self)
        self.construct(name, opts)

        self.xs = xs
        self.dim = len(xs)
        self.x_gpu = wp.array(xs, dtype=float)
        self.y_gpu = wp.array([0], dtype=float)
        self.intermediate = wp.zeros(self.dim, dtype=float)
        self.out = wp.array([0], dtype=float)

        
        @wp.kernel
        def f(x: wp.array(dtype=float),
            y: wp.array(dtype=float),
            intermediate: wp.array(dtype=float)):

            tid = wp.tid()
            diff = x[tid] - y[0]
            intermediate[tid] = diff * diff

        self.f = f

    def get_n_in(self): return 1
    def get_n_out(self): return 1

    def get_sparsity_in(self,i):
        return Sparsity.dense(1,1)

    def get_sparsity_out(self,i):
        return Sparsity.dense(1,1)

    # Evaluate numerically
    def eval(self, arg):
        y = arg[0].__float__()
        self.y_gpu = wp.array([y], dtype=float) 
        self.intermediate.zero_()

        wp.launch(kernel = self.f,
                dim = self.dim,
                inputs = [self.x_gpu, self.y_gpu,self.intermediate])
        wp.utils.array_sum(self.intermediate, out = self.out, value_count = self.dim)
        
        out = self.out.numpy()[0]
        return [out]


    def has_jacobian(self): 
        return True
    def get_jacobian(self,name,inames,onames,opts):
        class JacFun(Callback):
            def __init__(self, x_gpu,opts={}):
                Callback.__init__(self)
                self.construct(name, opts)

                self.x_gpu = x_gpu
                self.dim = len(x_gpu)
                self.y_gpu = wp.array([0], dtype=float)
                self.intermediate = wp.zeros(self.dim, dtype=float)
                self.out = wp.array([0], dtype=float)

                @wp.kernel        
                def f_jac(x: wp.array(dtype=float),
                        y: wp.array(dtype=float),
                        intermediate: wp.array(dtype=float)):

                    tid = wp.tid()
                    diff = x[tid] - y[0]
                    intermediate[tid] = -2.0 * diff

                self.f_jac = f_jac



            def get_n_in(self): return 2
            def get_n_out(self): return 1

            def get_sparsity_in(self,i):
                if i == 0:
                    return Sparsity.dense(1,1)
                else:
                    return Sparsity.dense(1,1)

            def get_sparsity_out(self,i):
                return Sparsity.dense(1,1)

            def eval(self, arg):
                y = arg[0].__float__()
                self.y_gpu = wp.array([y], dtype=float) 
                self.intermediate.zero_()


                wp.launch(kernel = self.f_jac,
                        dim = self.dim,
                        inputs = [self.x_gpu, self.y_gpu,self.intermediate])

                wp.utils.array_sum(self.intermediate, out = self.out, value_count = self.dim)                    
                out = self.out.numpy()[0]

                return [out]

            def has_jacobian(self): 
                    return True
            def get_jacobian(self,name,inames,onames,opts):
                class HessFun(Callback):
                    def __init__(self, x_gpu,opts={}):
                        Callback.__init__(self)
                        self.construct(name, opts)

                        self.x_gpu = x_gpu
                        self.dim = len(x_gpu)
                        self.y_gpu = wp.array([0], dtype=float)
                        self.out = wp.array([0], dtype=float)


                    def get_n_in(self): return 3
                    def get_n_out(self): return 2

                    def get_sparsity_in(self,i):
                        return Sparsity.dense(1,1)

                    def get_sparsity_out(self,i):
                        return Sparsity.dense(1,1)

                    def eval(self, arg):
                        out = self.dim * 2.0
                        return [out,0]
                self.jac_callback = HessFun(self.x_gpu)
                return self.jac_callback


        self.jac_callback = JacFun(self.x_gpu)
        return self.jac_callback


dim = 100_000_000
x =  np.random.rand(dim).astype(float)


opts = {}
opts["print_time"] = False
opts["ipopt"] = {}
opts["ipopt"]["print_level"] = 3
opts["ipopt"]["sb"] = "yes"
opts["ipopt"]["max_iter"] = 100
opts["ipopt"]["tol"] = 1e-2
opts["ipopt"]["acceptable_tol"] = 1e-6

# custom solve
f = F_cas("f", x)
y = MX.sym("y")

cost = f(y)
solver= nlpsol("solver", "ipopt", {"x":y, "f":cost}, opts)
res = solver(x0=0.0)


print("Solution:", res["x"])
```

### Conclusion
The presented code is a first step to accelerate casadi with NVIDIA Warp. The obvious draw back is the you have to calcualate the jacobian and hessian expressions yourself and make them run on the GPU.

NVIDIA Warp does support [automatic differentiation on its kernels](https://nvidia.github.io/warp/index.html), so this is a route that I will explore in the future.

### Repository
The code is available on [github]()

