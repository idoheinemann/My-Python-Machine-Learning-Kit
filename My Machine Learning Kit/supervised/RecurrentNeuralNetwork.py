import numpy as np


class RecurrentNeuralNetwork:
    @staticmethod
    def create_layers(*sizes):
        a = ((sizes[0], sizes[1]),)
        for i in xrange(1, len(sizes)-1):
            a += ((sizes[i], sizes[i+1]),)
        return a

    def __init__(self, layers_size, non_linear):
        layers = list(layers_size)
        self.W = []
        self.U = []
        self.B = []
        self.w_derivatives = None
        self.u_derivatives = None
        self.b_derivatives = None
        if hasattr(non_linear, '__iter__') or hasattr(non_linear, '__getitem__'):
            assert len(layers) == len(non_linear)
        else:
            non_linear = (non_linear,)*len(layers)
        self.functions = non_linear
        for i in xrange(len(layers)):
            self.U.append(np.random.rand(layers[i][0], layers[i][1])-0.5)
            self.W.append(np.random.rand(layers[i][1], layers[i][1])-0.5)
            self.B.append(np.random.rand(layers[i][1])-0.5)

    def __call__(self, inp):
        states = [[np.array([0])]]

        for i in self.W:
            states[0].append(np.array([np.zeros(i.shape[0])]))

        for i in inp:
            prev_state = states[-1]
            state = []
            d_inp = i
            state.append(d_inp)

            for j in xrange(len(self.U)):
                d_inp = self.functions[j](d_inp.dot(self.U[j])+prev_state[j+1].dot(self.W[j])+self.B[j])
                state.append(d_inp)

            states.append(state)
        return states

    def hypot(self, inp):
        return map(lambda l: l[-1], self(inp))[1:]

    def unfold(self, layer, delta, states, step):
        if step <= 0:
            return
        state = states[-1]
        prev_state = states[-2]
        for j in xrange(layer, -1, -1):
            # print j
            # print "delta: " + str(delta)
            self.u_derivatives[j] += state[j].T.dot(delta)
            self.w_derivatives[j] += prev_state[j + 1].T.dot(delta)
            self.b_derivatives[j] += delta.reshape(self.b_derivatives[j].shape)
            delta_t = delta.dot(self.W[j].T) * self.functions[j](prev_state[j+1], True)
            self.unfold(layer=j, delta=delta_t, states=states[:-1], step=min(step-1, len(states)-1))
            if j > 0:
                delta = delta.dot(self.U[j].T) * self.functions[j - 1](state[j], True)

    def get_gradient(self, inp, labels, step=3):
        states = self(inp)
        self.u_derivatives = []
        self.w_derivatives = []
        self.b_derivatives = []
        for i in xrange(len(self.U)):
            self.u_derivatives.append(np.zeros_like(self.U[i]))
            self.w_derivatives.append(np.zeros_like(self.W[i]))
            self.b_derivatives.append(np.zeros_like(self.B[i]))

        for i in xrange(len(labels)):
            output = states[i+1][len(states[i+1])-1]
            state = states[i+1]
            prev_state = states[i]
            error = output - labels[i]
            delta = error*self.functions[len(self.functions)-1](output, True)
            # print "state: "+str(state)
            # print "prev state: "+str(prev_state)
            # print "delta: "+str(delta)
            # print "U: "+str(self.U)
            # print "W: "+str(self.W)
            # print "B: " + str(self.B)
            for j in xrange(len(self.U)-1, -1, -1):
                # print j
                # print "delta: " + str(delta)
                self.u_derivatives[j] += state[j].T.dot(delta)
                self.w_derivatives[j] += prev_state[j+1].T.dot(delta)
                self.b_derivatives[j] += delta.reshape(self.b_derivatives[j].shape)
                delta_t = delta.dot(self.W[j].T)*self.functions[j](prev_state[j+1], True)
                self.unfold(layer=j, delta=delta_t, states=states[:i+1], step=min(step, i))
                if j > 0:
                    delta = delta.dot(self.U[j].T)*self.functions[j-1](state[j], True)

    def cost(self, inp, labels):
        j = 0
        for i in xrange(len(inp)):
            outputs = np.array(self.hypot(inp[i]))-np.array(labels[i])
            j += np.sum(outputs*outputs)
        return j

    def iteration(self, inp, label, step=3, alpha=0.001):
        def __addition(x, y):
            for i in xrange(len(x)):
                x[i] += y[i]
        gu = []
        for j in self.U:
            gu.append(np.zeros_like(j))
        gw = []
        for j in self.W:
            gw.append(np.zeros_like(j))
        gb = []
        for j in self.B:
            gb.append(np.zeros_like(j))
        for j in xrange(len(inp)):
            self.get_gradient(inp[j], label[j], step)
            __addition(gu, self.u_derivatives)
            __addition(gb, self.b_derivatives)
            __addition(gw, self.w_derivatives)

        for j in xrange(len(self.U)):
            self.U[j] -= alpha * gu[j]
            self.W[j] -= alpha * gw[j]
            self.B[j] -= alpha * gb[j]
