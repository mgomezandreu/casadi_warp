"""
@Author: Mario Gomez Andreu
@Description: This is a test file to see if we can implement a custom jacobian and hessian in casadi.
@Date: 2024-05-05
"""
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