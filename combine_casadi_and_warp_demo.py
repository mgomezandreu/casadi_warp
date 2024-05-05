"""
@Author: Mario Gomez Andreu
@Description: This script shows how to combine casadi and warp.
@Date: 2024-05-05
"""


import warp as wp
from casadi import *
import numpy as np

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







