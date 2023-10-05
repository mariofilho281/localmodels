import numpy as np
import scipy.optimize as opt


class TrilocalModel:
    def __init__(self, *args):
        """
        Possible signatures:
        TrilocalModel(p_alpha, p_beta, p_gamma, p_a, p_b, p_c)
        TrilocalModel(c_alpha, c_beta, c_gamma, ma, mb,mc, x)

        Constructs a trilocal model either:
        i) from the hidden variable distributions ``p_alpha``, ``p_beta``,
        ``p_gamma`` and response functions ``p_a``, ``p_b``, ``p_c``, or;
        ii) from the hidden variable cardinalities ``c_alpha``, ``c_beta``,
        ``c_gamma``, output cardinalities ``m_a``, ``m_b``, ``m_c`` and array
        of free parameters ``x``.
        """
        if len(args) == 6:
            p_alpha, p_beta, p_gamma, p_a, p_b, p_c = args
            self.p_alpha = np.array(p_alpha).flatten()
            self.p_beta = np.array(p_beta).flatten()
            self.p_gamma = np.array(p_gamma).flatten()
            self.p_a = np.array(p_a)
            self.p_b = np.array(p_b)
            self.p_c = np.array(p_c)
        elif len(args) == 7:
            c_alpha, c_beta, c_gamma, ma, mb, mc, x = args
            x = np.array(x).flatten()
            end_alpha = c_alpha - 1
            end_beta = end_alpha + c_beta - 1
            end_gamma = end_beta + c_gamma - 1
            end_a = end_gamma + (ma - 1) * c_beta * c_gamma
            end_b = end_a + (mb - 1) * c_gamma * c_alpha
            # end_c = end_b + (mc - 1) * c_alpha * c_beta
            self.p_alpha, self.p_beta, self.p_gamma, self.p_a, self.p_b, \
                self.p_c = np.split(x, [end_alpha, end_beta, end_gamma,
                                        end_a, end_b])
            self.p_alpha = np.concatenate((self.p_alpha,
                                           [1 - np.sum(self.p_alpha)]))
            self.p_beta = np.concatenate((self.p_beta,
                                          [1 - np.sum(self.p_beta)]))
            self.p_gamma = np.concatenate((self.p_gamma,
                                           [1 - np.sum(self.p_gamma)]))
            self.p_a = self.p_a.reshape((ma - 1, c_beta, c_gamma))
            self.p_a = np.concatenate((self.p_a,
                                       1 - self.p_a.sum(axis=0, keepdims=True)),
                                      axis=0)
            self.p_b = self.p_b.reshape((mb - 1, c_gamma, c_alpha))
            self.p_b = np.concatenate((self.p_b,
                                       1 - self.p_b.sum(axis=0, keepdims=True)),
                                      axis=0)
            self.p_c = self.p_c.reshape((mc - 1, c_alpha, c_beta))
            self.p_c = np.concatenate((self.p_c,
                                       1 - self.p_c.sum(axis=0, keepdims=True)),
                                      axis=0)
        else:
            raise ValueError(f'Either 6 or 7 arguments expected. '
                             f'Got {len(args)} argument(s) instead.')
        self.c_alpha = len(self.p_alpha)
        self.c_beta = len(self.p_beta)
        self.c_gamma = len(self.p_gamma)
        self.ma = self.p_a.shape[0]
        self.mb = self.p_b.shape[0]
        self.mc = self.p_c.shape[0]

    def __str__(self):
        """
        Returns string representation of hidden variable distributions and
        response functions.
        """
        return f'p_alpha = {str(self.p_alpha)}\n' \
               f'p_beta  = {str(self.p_beta)}\n' \
               f'p_gamma = {str(self.p_gamma)}\n' \
               f'p_a =\n{str(self.p_a[0:-1, :, :])}\n' \
               f'p_b =\n{str(self.p_b[0:-1, :, :])}\n' \
               f'p_c =\n{str(self.p_c[0:-1, :, :])}'

    def show_hidden_variable_distributions(self):
        """
        Prints hidden variable distributions.
        """
        print(f'p_alpha = {str(self.p_alpha)}')
        print(f'p_beta  = {str(self.p_beta)}')
        print(f'p_gamma = {str(self.p_gamma)}')

    def show_response_functions(self, decimal_places=8):
        """
        Prints response functions.
        """
        print(f'p_a =\n{str(self.p_a[0:-1, :, :].round(decimal_places))}')
        print(f'p_b =\n{str(self.p_b[0:-1, :, :].round(decimal_places))}')
        print(f'p_c =\n{str(self.p_c[0:-1, :, :].round(decimal_places))}')

    def cardinalities(self):
        """
        Return tuple with hidden variable cardinalities ``c_alpha``,
        ``c_beta``, ``c_gamma`` and output cardinalities ``ma``, ``mb``, ``mc``.
        """
        return (self.c_alpha, self.c_beta, self.c_gamma,
                self.ma, self.mb, self.mc)

    def degrees_of_freedom(self):
        """
        Calculates the number of free parameters in the trilocal model.
        """
        return self.c_alpha + self.c_beta + self.c_gamma - 3 \
            + self.c_alpha * self.c_beta * (self.mc - 1) \
            + self.c_beta * self.c_gamma * (self.ma - 1) \
            + self.c_alpha * self.c_gamma * (self.mb - 1)

    def behavior(self):
        """
        Calculates the statistical behavior p(a,b,c) for the trilocal model.
        """
        # Array indices for np.einsum:
        #   p_alpha: alpha -> i
        #   p_beta: beta -> j
        #   p_gamma: gamma -> k
        #   p_a: a, beta, gamma -> ljk
        #   p_b: b, gamma, alpha -> mki
        #   p_c: c, alpha, beta -> nij
        #   px: a, b, c -> lmn
        return np.einsum('i,j,k,ljk,mki,nij->lmn',
                         self.p_alpha, self.p_beta, self.p_gamma,
                         self.p_a, self.p_b, self.p_c)

    def cost(self, p):
        """
        Calculates the sum of squared errors between the model behavior and a
        given target behavior ``p``.
        """
        return np.sum((self.behavior() - p) ** 2)

    @staticmethod
    def cost_for_optimizer(x, p, c_alpha, c_beta, c_gamma, ma, mb, mc):
        """
        Cost function for the optimizer.
        """
        return TrilocalModel(c_alpha, c_beta, c_gamma, ma, mb, mc, x).cost(p)

    @staticmethod
    def uniform(c_alpha, c_beta, c_gamma, ma, mb, mc):
        """
        Creates trilocal model with uniform probability distributions with
        cardinalities ``c_alpha``, ``c_beta``, ``c_gamma``, ``ma``, ``mb``,
        ``mc``.
        """
        p_alpha = 1 / c_alpha * np.ones((c_alpha,))
        p_beta = 1 / c_beta * np.ones((c_beta,))
        p_gamma = 1 / c_gamma * np.ones((c_gamma,))
        p_a = 1 / ma * np.ones((ma, c_beta, c_gamma))
        p_b = 1 / mb * np.ones((mb, c_gamma, c_alpha))
        p_c = 1 / mc * np.ones((mc, c_alpha, c_beta))
        return TrilocalModel(p_alpha, p_beta, p_gamma, p_a, p_b, p_c)

    @staticmethod
    def random(c_alpha, c_beta, c_gamma, ma, mb, mc):
        """
        Creates random trilocal model with cardinalities ``c_alpha``,
        ``c_beta``, ``c_gamma``, ``ma``, ``mb``, ``mc``.
        """
        p_alpha = np.random.dirichlet(np.ones(c_alpha))
        p_beta = np.random.dirichlet(np.ones(c_beta))
        p_gamma = np.random.dirichlet(np.ones(c_gamma))
        p_a = np.moveaxis(np.random.dirichlet(np.ones(ma), (c_beta, c_gamma)),
                          -1, 0)
        p_b = np.moveaxis(np.random.dirichlet(np.ones(mb), (c_gamma, c_alpha)),
                          -1, 0)
        p_c = np.moveaxis(np.random.dirichlet(np.ones(mc), (c_alpha, c_beta)),
                          -1, 0)
        return TrilocalModel(p_alpha, p_beta, p_gamma, p_a, p_b, p_c)

    def optimizer_representation(self):
        """
        Returns representation consisting only of free parameters ``x``.
        Useful for the optimizer.
        """
        end_alpha = self.c_alpha - 1
        end_beta = end_alpha + self.c_beta - 1
        end_gamma = end_beta + self.c_gamma - 1
        end_a = end_gamma + (self.ma - 1) * self.c_beta * self.c_gamma
        end_b = end_a + (self.mb - 1) * self.c_gamma * self.c_alpha
        end_c = end_b + (self.mc - 1) * self.c_alpha * self.c_beta
        x = np.zeros(end_c)
        x[0:end_alpha] = self.p_alpha[0:end_alpha]
        x[end_alpha:end_beta] = self.p_beta[0:end_beta - end_alpha]
        x[end_beta:end_gamma] = self.p_gamma[0:end_gamma - end_beta]
        x[end_gamma:end_a] = self.p_a[0:self.ma - 1, :, :].flatten()
        x[end_a:end_b] = self.p_b[0:self.mb - 1, :, :].flatten()
        x[end_b:end_c] = self.p_c[0:self.mc - 1, :, :].flatten()
        return x

    def optimize(self, p, initial_guess=None, number_of_trials=1, tol=1e-4):
        """
        Returns model optimized to replicate given behavior ``p``.
        """
        dof = self.degrees_of_freedom()
        bounds = opt.Bounds(np.zeros(dof), np.ones(dof))
        # The code below assembles the hidden variable positivity constraints
        n_constraints = (3 + self.c_beta * self.c_gamma
                         + self.c_gamma * self.c_alpha
                         + self.c_alpha * self.c_beta)
        end_alpha = self.c_alpha - 1
        end_beta = end_alpha + self.c_beta - 1
        end_gamma = end_beta + self.c_gamma - 1
        coeffs = np.zeros(shape=(n_constraints, dof))
        coeffs[0, 0:end_alpha] = 1
        coeffs[1, end_alpha:end_beta] = 1
        coeffs[2, end_beta:end_gamma] = 1
        # The code below assembles the response function positivity constraints
        row_a = 3 + self.c_beta * self.c_gamma
        row_b = row_a + self.c_gamma * self.c_alpha
        row_c = row_b + self.c_alpha * self.c_beta
        end_a = end_gamma + (self.ma - 1) * self.c_beta * self.c_gamma
        end_b = end_a + (self.mb - 1) * self.c_gamma * self.c_alpha
        end_c = end_b + (self.mc - 1) * self.c_alpha * self.c_beta
        coeffs[3:row_a, end_gamma:end_a] = np.tile(np.eye(self.c_beta
                                                          * self.c_gamma),
                                                   self.ma - 1)
        coeffs[row_a:row_b, end_a:end_b] = np.tile(np.eye(self.c_gamma
                                                          * self.c_alpha),
                                                   self.mb - 1)
        coeffs[row_b:row_c, end_b:end_c] = np.tile(np.eye(self.c_alpha
                                                          * self.c_beta),
                                                   self.mc - 1)
        linear_constraints = opt.LinearConstraint(coeffs,
                                                  -np.inf * np.ones(row_c),
                                                  np.ones(row_c))
        if initial_guess is None:
            initial_guess = TrilocalModel\
                .random(self.c_alpha, self.c_beta, self.c_gamma, self.ma,
                        self.mb, self.mc).optimizer_representation()
        # ----------------------------------------------------------------------
        optimized_model = self
        error = self.cost(p)
        for i in range(number_of_trials):
            solution = opt.minimize(TrilocalModel.cost_for_optimizer,
                                    initial_guess,
                                    args=(p, self.c_alpha, self.c_beta,
                                          self.c_gamma, self.ma, self.mb,
                                          self.mc),
                                    method='trust-constr',
                                    constraints=linear_constraints,
                                    options={'verbose': 1}, bounds=bounds)
            initial_guess = TrilocalModel\
                .random(self.c_alpha, self.c_beta, self.c_gamma,
                        self.ma, self.mb, self.mc).optimizer_representation()
            partial_model = TrilocalModel(self.c_alpha, self.c_beta,
                                          self.c_gamma, self.ma, self.mb,
                                          self.mc, solution.x)
            partial_error = np.sqrt(partial_model.cost(p) / (self.ma * self.mb
                                                             * self.mc))
            if partial_error < error:
                optimized_model = partial_model
                error = partial_error
            if partial_error < tol:
                break
        return optimized_model

    def relabel_hidden_variable(self, variable, new_labels):
        """
        Relabels the hidden variable indicated by the integer parameter
        ``variable``, 0 being alpha, 1 being beta and 2 being gamma. The new
        labels are indicated by the array ``new_labels``, which must be an array
        containing the integers 0, 1, ..., c-1, where c is the cardinality of
        the hidden variable being relabelled.
        """
        if variable == 0:
            self.p_alpha = self.p_alpha.take(new_labels)
            self.p_b = self.p_b.take(new_labels, axis=2)
            self.p_c = self.p_c.take(new_labels, axis=1)
        elif variable == 1:
            self.p_beta = self.p_beta.take(new_labels)
            self.p_a = self.p_a.take(new_labels, axis=1)
            self.p_c = self.p_c.take(new_labels, axis=2)
        elif variable == 2:
            self.p_gamma = self.p_gamma.take(new_labels)
            self.p_a = self.p_a.take(new_labels, axis=2)
            self.p_b = self.p_b.take(new_labels, axis=1)

    def remove_hidden_variable_labels(self, variable, labels):
        """
        Removes labels for the hidden variable indicated by the integer
        parameter ``variable``, 0 being alpha, 1 being beta and 2 being gamma.
        The labels to be removed are indicated by the array ``labels``, which
        must be an array of integers in the set {0, 1, ..., c-1}, where c is
        the cardinality of the hidden variable being altered.
        """
        if variable == 0:
            self.p_alpha = np.delete(self.p_alpha, labels)
            self.p_alpha = self.p_alpha / self.p_alpha.sum()
            self.p_b = np.delete(self.p_b, labels, axis=2)
            self.p_c = np.delete(self.p_c, labels, axis=1)
        elif variable == 1:
            self.p_beta = np.delete(self.p_beta, labels)
            self.p_beta = self.p_beta / self.p_beta.sum()
            self.p_a = np.delete(self.p_a, labels, axis=1)
            self.p_c = np.delete(self.p_c, labels, axis=2)
        elif variable == 2:
            self.p_gamma = np.delete(self.p_gamma, labels)
            self.p_gamma = self.p_gamma / self.p_gamma.sum()
            self.p_a = np.delete(self.p_a, labels, axis=2)
            self.p_b = np.delete(self.p_b, labels, axis=1)

    def relabel_output(self, party, new_labels):
        """
        Relabels the output indicated by the integer parameter ``variable``,
        0 being a, 1 being b and 2 being c. The new labels are indicated by the
        array ``new_labels``, which must be an array containing the integers
        0, 1, ..., m-1, where m is the cardinality of the output being
        relabelled.
        """
        if party == 0:
            self.p_a = self.p_a.take(new_labels, axis=0)
        elif party == 1:
            self.p_b = self.p_b.take(new_labels, axis=0)
        elif party == 2:
            self.p_c = self.p_c.take(new_labels, axis=0)

    def exchange_hidden_variables(self, variable1, variable2):
        """
        Exchange the hidden variables indicated by the integer parameters
        ``variable1`` and ``variable2``, 0 being alpha, 1 being beta and
        2 being gamma.
        """
        if {variable1, variable2} == {0, 1}:
            self.p_alpha, self.p_beta = self.p_beta, self.p_alpha
            self.p_a, self.p_b = (self.p_b.swapaxes(1, 2),
                                  self.p_a.swapaxes(1, 2))
            self.p_c = self.p_c.swapaxes(1, 2)
        elif {variable1, variable2} == {1, 2}:
            self.p_beta, self.p_gamma = self.p_gamma, self.p_beta
            self.p_b, self.p_c = (self.p_c.swapaxes(1, 2),
                                  self.p_b.swapaxes(1, 2))
            self.p_a = self.p_a.swapaxes(1, 2)
        elif {variable1, variable2} == {0, 2}:
            self.p_alpha, self.p_gamma = self.p_gamma, self.p_alpha
            self.p_a, self.p_c = (self.p_c.swapaxes(1, 2),
                                  self.p_a.swapaxes(1, 2))
            self.p_b = self.p_b.swapaxes(1, 2)

    def exchange_parties(self, party1, party2):
        """
        Exchange the parties indicated by the integer parameters ``party1``
        and ``party2``, 0 being a, 1 being b and 2 being c.
        """
        self.exchange_hidden_variables(party1, party2)

    def standardize(self, exchange_parties_allowed=True):
        """
        Relabels hidden variables so that probabilities are listed in
        descending order. Also, if the exchange of parties is allowed, reorder
        hidden variables so that c_alpha >= c_beta >= c_gamma.
        """
        c_alpha, c_beta, c_gamma, ma, mb, mc = self.cardinalities()
        # Rearrange hidden variables in descending cardinalities
        if exchange_parties_allowed:
            order_alpha, order_beta, order_gamma = np.argsort((c_alpha, c_beta,
                                                               c_gamma))
            if (order_alpha, order_beta, order_gamma) == (0, 1, 2):  # a b c
                self.exchange_hidden_variables(0, 2)
            elif (order_alpha, order_beta, order_gamma) == (0, 2, 1):  # a c b
                self.exchange_hidden_variables(0, 1)
                self.exchange_hidden_variables(1, 2)
            elif (order_alpha, order_beta, order_gamma) == (1, 0, 2):  # b a c
                self.exchange_hidden_variables(0, 2)
                self.exchange_hidden_variables(1, 2)
            elif (order_alpha, order_beta, order_gamma) == (1, 2, 0):  # b c a
                self.exchange_hidden_variables(1, 2)
            elif (order_alpha, order_beta, order_gamma) == (2, 0, 1):  # c a b
                self.exchange_hidden_variables(0, 1)
        # Rearrange hidden variable labels in descending probabilities
        self.relabel_hidden_variable(0, np.argsort(-self.p_alpha))
        self.relabel_hidden_variable(1, np.argsort(-self.p_beta))
        self.relabel_hidden_variable(2, np.argsort(-self.p_gamma))

    def round(self, tol=1e-4):
        """
        Removes hidden variable labels with probabilities less than ``tol``.
        """
        self.remove_hidden_variable_labels(0, self.p_alpha < tol)
        self.remove_hidden_variable_labels(1, self.p_beta < tol)
        self.remove_hidden_variable_labels(2, self.p_gamma < tol)
