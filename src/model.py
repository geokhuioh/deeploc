import mxnet as mx
import numpy as np
import time
import mxnet.ndarray as nd
import mxnet.initializer as init
from mxnet import npx, autograd, optimizer, gluon
from mxnet.gluon import nn, rnn
from mxboard import SummaryWriter

class ConvLayer(nn.Block):

    def __init__(self, n_filt_1, n_filt_2, **kwargs):
        super(ConvLayer, self).__init__(**kwargs)
        self._n_filt_1 = n_filt_1
        self._n_filt_2 = n_filt_2

        with self.name_scope():
            self.l_conv_01 = nn.Conv1D(prefix='01_', channels=n_filt_1, kernel_size=1,  padding=0,  layout='NCW', activation='relu')
            self.l_conv_03 = nn.Conv1D(prefix='03_', channels=n_filt_1, kernel_size=3,  padding=1,  layout='NCW', activation='relu')
            self.l_conv_05 = nn.Conv1D(prefix='05_', channels=n_filt_1, kernel_size=5,  padding=2,  layout='NCW', activation='relu')
            self.l_conv_09 = nn.Conv1D(prefix='09_', channels=n_filt_1, kernel_size=9,  padding=4,  layout='NCW', activation='relu')
            self.l_conv_15 = nn.Conv1D(prefix='15_', channels=n_filt_1, kernel_size=15, padding=7,  layout='NCW', activation='relu')
            self.l_conv_21 = nn.Conv1D(prefix='21_', channels=n_filt_1, kernel_size=21, padding=10, layout='NCW', activation='relu')

            self.l_conv_final = nn.Conv1D(prefix='conc_', channels=n_filt_2, kernel_size=3, padding=1, layout='NCW', activation='relu')

    def forward(self, x):
        a = self.l_conv_01(x)
        b = self.l_conv_03(x)
        c = self.l_conv_05(x)
        d = self.l_conv_09(x)
        e = self.l_conv_15(x)
        f = self.l_conv_21(x)

        conc = nd.concat(a, b, c, d, e, f, dim=1)

        return self.l_conv_final(conc)

class BidirectionalLSTM(nn.Block):

    def __init__(self, ctx, n_hid, n_filt_2, **kwargs):
        super(BidirectionalLSTM, self).__init__(**kwargs)
        self._n_hid = n_hid
        self._n_filt_2 = n_filt_2
        self._ctx = ctx

        with self.name_scope():
            self.l_fwd = rnn.LSTM(hidden_size=n_hid, layout='TNC', prefix='Fwd_')
            self.l_bck = rnn.LSTM(hidden_size=n_hid, layout='TNC', prefix='Bck_')

    def _reverse(self, x, mask):
        x_1 = nd.empty(x.shape, ctx=self._ctx)
        nd.reset_arrays(x_1, num_arrays=1)
        for i in range(x.shape[0]):
            size = nd.sum(mask[i]).astype('int32').asscalar()
            seq = x[i]
            seq_1 = nd.reverse(nd.slice(seq, begin=(0,0), end=(size, self._n_filt_2)), axis=1)
            seq_1.copyto(x_1[i, :size])
        return x_1

    def forward(self, x, mask):
        dimension = (1, 0, 2)
        x_fwd = nd.transpose(x, dimension)
        x_bck = nd.transpose(self._reverse(x, mask), dimension)
        fwd = self.l_fwd(x_fwd)
        bck = self.l_bck(x_bck)
        conc = nd.concat(fwd, bck, dim=2)
        return nd.transpose(conc, dimension)

class LSTMAttentionDecodeFeedback(nn.Block):
    def __init__(self,
                 ctx,
                 num_units,
                 aln_num_units,
                 n_decodesteps=10,
                 **kwargs):

        super(LSTMAttentionDecodeFeedback, self).__init__(**kwargs)

        self.num_units = num_units
        self.aln_num_units = aln_num_units
        self.n_decodesteps = n_decodesteps
        self.attention_softmax_function = nd.softmax
        self.peepholes = True
        self.ctx = ctx

        self.num_inputs = 512

        self.nonlinearity_align=nd.tanh

        self.nonlinearity_ingate = nd.sigmoid
        self.nonlinearity_forgetgate = nd.sigmoid
        self.nonlinearity_cell = nd.tanh
        self.nonlinearity_outgate = nd.sigmoid

        self.nonlinearity_out = nd.tanh

        self.W_hid_to_ingate = self.params.get('W_hid_to_ingate', shape=(num_units, num_units),
                                               init=init.Normal(0.1),
                                               allow_deferred_init=True)

        self.W_hid_to_forgetgate = self.params.get('W_hid_to_forgetgate', shape=(num_units, num_units),
                                                   init=init.Normal(0.1),
                                                   allow_deferred_init=True)

        self.W_hid_to_cell = self.params.get('W_hid_to_cell', shape=(num_units, num_units),
                                             init=init.Normal(0.1),
                                             allow_deferred_init=True)

        self.W_hid_to_outgate = self.params.get('W_hid_to_outgate', shape=(num_units, num_units),
                                                init=init.Normal(0.1),
                                                allow_deferred_init=True)

        self.b_ingate = self.params.get('b_ingate', shape=(num_units),
                                        init=init.Constant(0),
                                        allow_deferred_init=True)

        self.b_forgetgate = self.params.get('b_forgetgate', shape=(num_units),
                                            init=init.Constant(0),
                                            allow_deferred_init=True)

        self.b_cell = self.params.get('b_cell', shape=(num_units),
                                      init=init.Constant(0),
                                      allow_deferred_init=True)

        self.b_outgate = self.params.get('b_outgate', shape=(num_units),
                                         init=init.Constant(0),
                                         allow_deferred_init=True)

        self.W_weightedhid_to_ingate = self.params.get('W_weightedhid_to_ingate',
                                                       shape=(self.num_inputs, num_units),
                                                       init=init.Normal(0.1),
                                                       allow_deferred_init=True)

        self.W_weightedhid_to_forgetgate = self.params.get('W_weightedhid_to_forgetgate',
                                                           shape=(self.num_inputs, num_units),
                                                           init=init.Normal(0.1),
                                                           allow_deferred_init=True)

        self.W_weightedhid_to_cell = self.params.get('W_weightedhid_to_cell',
                                                     shape=(self.num_inputs, num_units),
                                                     init=init.Normal(0.1),
                                                     allow_deferred_init=True)

        self.W_weightedhid_to_outgate = self.params.get('W_weightedhid_to_outgate',
                                                        shape=(self.num_inputs, num_units),
                                                        init=init.Normal(0.1),
                                                        allow_deferred_init=True)

        self.W_cell_to_ingate = self.params.get('W_cell_to_ingate',
                                                shape=(num_units),
                                                init=init.Normal(0.1),
                                                allow_deferred_init=True)

        self.W_cell_to_forgetgate = self.params.get('W_cell_to_forgetgate',
                                                    shape=(num_units),
                                                    init=init.Normal(0.1),
                                                    allow_deferred_init=True)

        self.W_cell_to_outgate = self.params.get('W_cell_to_outgate',
                                                 shape=(num_units),
                                                 init=init.Normal(0.1),
                                                 allow_deferred_init=True)

        self.W_align = self.params.get('W_align',
                                       shape=(num_units, self.aln_num_units),
                                       init=init.Normal(0.1))

        self.U_align = self.params.get('U_align', shape=(self.num_inputs,self.aln_num_units),
                                       init=init.Normal(0.1),
                                       allow_deferred_init=True)

        self.v_align = self.params.get('v_align', shape=(self.aln_num_units, 1),
                                       init=init.Normal(0.1))

        with self.name_scope():
            pass

    def slice_w(self, x, n):
        return x[:, n*self.num_units:(n+1)*self.num_units]

    def step(self, cell_previous, hid_previous, alpha_prev, weighted_hidden_prev,
             input, mask, hUa, W_align, v_align,
             W_hid_stacked, W_weightedhid_stacked, W_cell_to_ingate,
             W_cell_to_forgetgate, W_cell_to_outgate,
             b_stacked, *args):

        sWa = nd.dot(hid_previous, W_align)  # (BS, aln_num_units)
        sWa = nd.expand_dims(sWa, axis=1)    # (BS, 1 aln_num_units)
        align_act = sWa + hUa
        tanh_sWahUa = nd.tanh(align_act)     # (BS, seqlen, num_units_aln)

        # CALCULATE WEIGHT FOR EACH HIDDEN STATE VECTOR
        a = nd.dot(tanh_sWahUa, v_align)  # (BS, Seqlen, 1)
        a = nd.reshape(a, (a.shape[0], a.shape[1]))
        #                                # (BS, Seqlen)
        # # ->(BS, seq_len)

        a = a*mask - (1-mask)*10000

        alpha = self.attention_softmax_function(a)

        # input: (BS, Seqlen, num_units)
        weighted_hidden = input * nd.expand_dims(alpha, axis=2)
        weighted_hidden = nd.sum(weighted_hidden, axis=1)  #sum seqlen out

        # (BS, dec_hid) x (dec_hid, dec_hid)
        gates = nd.dot(hid_previous, W_hid_stacked) + b_stacked
        # (BS, enc_hid) x (enc_hid, dec_hid)
        gates = gates + nd.dot(weighted_hidden, W_weightedhid_stacked)


        # Clip gradients
        # if self.grad_clipping is not False:
        #    gates = theano.gradient.grad_clip(
        #        gates, -self.grad_clipping, self.grad_clipping)

        # Extract the pre-activation gate values
        ingate = self.slice_w(gates, 0)
        forgetgate = self.slice_w(gates, 1)
        cell_input = self.slice_w(gates, 2)
        outgate = self.slice_w(gates, 3)

        if self.peepholes:
            # Compute peephole connections
            ingate = ingate + cell_previous*W_cell_to_ingate
            forgetgate = forgetgate + (cell_previous*W_cell_to_forgetgate)

        # Apply nonlinearities
        ingate = self.nonlinearity_ingate(ingate)
        forgetgate = self.nonlinearity_forgetgate(forgetgate)
        cell_input = self.nonlinearity_cell(cell_input)
        outgate = self.nonlinearity_outgate(outgate)

        # Compute new cell value
        cell = forgetgate*cell_previous + ingate*cell_input

        if self.peepholes:
            outgate = outgate + cell*W_cell_to_outgate

        # W_align:  (num_units, aln_num_units)
        # U_align:  (num_feats, aln_num_units)
        # v_align:  (aln_num_units, 1)
        # hUa:      (BS, Seqlen, aln_num_units)
        # hid:      (BS, num_units_dec)
        # input:    (BS, Seqlen, num_inputs)

        # Compute new hidden unit activation
        hid = outgate*self.nonlinearity_out(cell)

        return [cell, hid, alpha, weighted_hidden]


    def forward(self, input, mask):

        num_batch = input.shape[0]
        encode_seqlen = input.shape[1]

        W_hid_stacked = nd.concat(
            self.W_hid_to_ingate.data(),
            self.W_hid_to_forgetgate.data(),
            self.W_hid_to_cell.data(),
            self.W_hid_to_outgate.data(),
            dim=1)

        W_weightedhid_stacked = nd.concat(
            self.W_weightedhid_to_ingate.data(),
            self.W_weightedhid_to_forgetgate.data(),
            self.W_weightedhid_to_cell.data(),
            self.W_weightedhid_to_outgate.data(),
            dim=1)

        b_stacked = nd.concat(
            self.b_ingate.data(),
            self.b_forgetgate.data(),
            self.b_cell.data(),
            self.b_outgate.data(),
            dim=0)

        cell = nd.zeros((num_batch, self.num_units), ctx=self.ctx)
        hid = nd.zeros((num_batch, self.num_units), ctx=self.ctx)
        alpha = nd.zeros((num_batch, encode_seqlen), ctx=self.ctx)
        weighted_hidden = nd.zeros((num_batch, self.num_units), ctx=self.ctx)

        hUa = nd.dot(input, self.U_align.data())
        W_align = self.W_align.data()
        v_align = self.v_align.data()

        W_cell_to_ingate = self.W_cell_to_ingate.data()
        W_cell_to_forgetgate = self.W_cell_to_forgetgate.data()
        W_cell_to_outgate = self.W_cell_to_outgate.data()

        for i in range(self.n_decodesteps):
            cell, hid, alpha, weighted_hidden = self.step(cell, hid, alpha, weighted_hidden,
                                                          input, mask, hUa, W_align, v_align,
                                                          W_hid_stacked, W_weightedhid_stacked, W_cell_to_ingate,
                                                          W_cell_to_forgetgate, W_cell_to_outgate,
                                                          b_stacked)

        return weighted_hidden

class Model(nn.Block):
    def __init__(self, ctx, drop_per, n_class, n_hid, n_filt_1, n_filt_2, **kwargs):
        super(Model, self).__init__(**kwargs)
        self._drop_per = drop_per
        self._n_class = n_class
        self._n_hid = n_hid
        self._n_filt_1 = n_filt_1
        self._n_filt_2 = n_filt_2

        with self.name_scope():
            self.l_dropout_1 = nn.Dropout(rate=self._drop_per)
            self.l_dropout_2 = nn.Dropout(rate=self._drop_per)
            self.l_dropout_3 = nn.Dropout(rate=self._drop_per)
            self.l_dropout_4 = nn.Dropout(rate=self._drop_per)
            self.l_conv = ConvLayer(n_filt_1, n_filt_2, prefix='Conv_')
            self.l_lstm = BidirectionalLSTM(ctx, n_hid, n_filt_2, prefix='BLSTM_')
            self.l_dense = nn.Dense(units=self._n_class, activation='relu')
            self.l_decoder = LSTMAttentionDecodeFeedback(
                ctx,
                prefix='Decoder_',
                num_units=2*self._n_hid, aln_num_units=self._n_hid, n_decodesteps=10)

    def forward(self, input, mask):
        x = self.l_dropout_1.forward(input)
        x = nd.transpose(x, (0, 2, 1))
        x = self.l_conv.forward(x)
        x = nd.transpose(x, (0, 2, 1))
        x = self.l_dropout_2.forward(x)
        x = self.l_lstm.forward(x, mask)
        x = self.l_decoder(x, mask)
        x = self.l_dropout_3.forward(x)
        x = self.l_dense.forward(x)
        x = self.l_dropout_4.forward(x)

        return x
