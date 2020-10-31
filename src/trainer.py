import mxnet as mx
import numpy as np
import time
import mxnet.ndarray as nd
from mxnet import npx, autograd, optimizer, gluon
from mxboard import SummaryWriter
from model import Model
from util import ConfusionMatrix

epoch = 200                #-- integer, epoch
batch_size = 32            #-- integer, minibatches size
max_seq_size = 1000        #-- integer, maximum sequence size
n_hid = 256                #-- integer, number of hidden neurons
n_feat = 20                #-- integer, number of features encoded  X_test.shape[2]
n_class = 10               #-- integer, number of classes to output
lr = 0.0005                #-- float, learning rate
drop_per  = 0.2            #-- float, input dropout
drop_hid = 0.5             #-- float, hidden neurons dropout
n_filt_1 = 20              #-- integer, number of filter in the first convolutional layer
n_filt_2 = 128             #-- integer, number of filter in the second convolutional layer
seed     = 123456          #-- seed
loss_fn  = 'cosine'        #-- 'cross_entropy' or 'cosine' loss function
encoding = 'blomap'        #-- 'onehot', 'blomap', 'blosum62' or 'profile'

#
# Initialization
#
ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()
mx.random.seed(seed)
npx.random.seed(seed)
np.random.seed(seed)


def generate_run_id():
    t = time.gmtime()
    return "{0}{1:0>2d}{2:0>2d}-{3:0>2d}{4:0>2d}".format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)


#
# Training Set and Test Set
#
if encoding == 'profile':
    train_file = 'subcellular_localization/data/train.npz'
    test_file  = 'subcellular_localization/data/test.npz'
elif encoding == 'blomap':
    train_file = 'data/deeploc_blomap.npz'
    test_file  = 'data/deeploc_blomap.npz'
elif encoding == 'blosum62':
    train_file = 'data/deeploc_blosum62.npz'
    test_file  = 'data/deeploc_blosum62.npz'
elif encoding == 'balanced_blosum62':
    train_file = 'data/deeploc_balanced_blosum62.npz'
    test_file  = 'data/deeploc_balanced_blosum62.npz'
elif encoding == 'balanced_blomap':
    train_file = 'data/deeploc_balanced_blomap.npz'
    test_file  = 'data/deeploc_balanced_blomap.npz'
elif encoding == 'balanced_blomap':
    train_file = 'data/deeploc_balanced_onehot.npz'
    test_file  = 'data/deeploc_balanced_onehot.npz'
elif encoding == 'onehot':
    train_file = 'data/deeploc_onehot.npz'
    test_file  = 'data/deeploc_onehot.npz'

train_npz = np.load(train_file)
test_npz = np.load(test_file)

mask_train = train_npz['mask_train']
partition = train_npz['partition']
X_train = train_npz['X_train']
y_train = train_npz['y_train']
X_test = test_npz['X_test']
mask_test = test_npz['mask_test']
y_test = test_npz['y_test']

train_npz.close()
test_npz.close()

#
# Training loop
#

run_id = generate_run_id()

hparams = {
    'run_id' : run_id,
    'epoch' : epoch,
    'batch_size' : batch_size,
    'n_hid' : n_hid,
    'n_feat' : n_feat,
    'n_class' : n_class,
    'lr' : lr,
    'drop_per' : drop_per,
    'drop_hid' : drop_hid,
    'n_filt_1' : n_filt_1,
    'n_filt_2' : n_filt_2,
    'seed' : seed,
    'loss_fn' : loss_fn,
    'encoding' : encoding}

ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()
mx.random.seed(seed)
npx.random.seed(seed)
np.random.seed(seed)


def save_hparams(hparams, hparams_path):
    import json
    json = json.dumps(hparams)
    f = open(hparams_path,"w")
    f.write(json)
    f.close()

hparams_path='./models/{}_hparams.json'.format(run_id)
net_params_path='./models/{}.params'.format(run_id)
logdir='./logs/{}-{}/{}'.format(encoding, loss_fn, run_id)

save_hparams(hparams, hparams_path)

if (loss_fn == 'cosine'):
    loss_function = gluon.loss.CosineEmbeddingLoss()
elif (loss_fn == 'cross_entropy'):
    loss_function =  gluon.loss.SoftmaxCrossEntropyLoss()

net = Model(ctx, drop_per, n_class, n_hid, n_filt_1, n_filt_2, prefix='net_')
net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

def train():

    sw = SummaryWriter(logdir=logdir)

    params = net.collect_params()
    params.reset_ctx([ctx])

    trainer = gluon.Trainer(params=params,
                            optimizer='adam', optimizer_params={'learning_rate':lr})

    for p in range(1, 5):

        # Train and validation sets
        train_index = np.where(partition != p)
        val_index = np.where(partition == p)
        X_tr = nd.from_numpy(X_train[train_index].astype(np.float32)).as_in_context(ctx)
        X_val = nd.from_numpy(X_train[val_index].astype(np.float32)).as_in_context(ctx)
        y_tr = nd.from_numpy(y_train[train_index].astype(np.int32)).as_in_context(ctx)
        y_val = nd.from_numpy(y_train[val_index].astype(np.int32)).as_in_context(ctx)
        mask_tr = nd.from_numpy(mask_train[train_index].astype(np.float32)).as_in_context(ctx)
        mask_val = nd.from_numpy(mask_train[val_index].astype(np.float32)).as_in_context(ctx)

        train_iter = mx.io.NDArrayIter([X_tr, mask_tr], y_tr, batch_size, shuffle=True)
        val_iter = mx.io.NDArrayIter([X_val, mask_val], y_val, batch_size, shuffle=False)

        eps = []
        best_val_acc = 0

        for e in range(1, epoch + 1):
            step = ((p - 1) * epoch) + e
            begin_time = time.perf_counter()
            train_loss = 0.
            train_acc = mx.metric.Accuracy()
            train_iter.reset()
            val_iter.reset()

            # Full pass training set
            train_err = 0
            train_batches = 0
            confusion_train = ConfusionMatrix(n_class)

            for batch in train_iter:
                input = batch.data[0]
                mask = batch.data[1]
                label = batch.label[0]

                with mx.autograd.record():
                    output = net(input, mask)

                    if (loss_fn == 'cosine'):
                        # Cosine loss
                        l = loss_function(output, nd.one_hot(label, n_class), nd.array([1.0], ctx=ctx))
                    else:
                        # Softmax cross entropy
                        l = loss_function(output, label)

                l.backward()
                trainer.step(batch_size)

                train_err += l.mean().asscalar()
                preds = output.argmax(axis=1)
                train_acc.update(label, preds)
                train_batches += 1
                np_label = label.astype('int32').asnumpy()
                np_preds = preds.astype('int32').asnumpy()
                confusion_train.batch_add(np_label, np_preds)

            stop_time = time.perf_counter()
            train_time = stop_time - begin_time

            train_loss = train_err / train_batches
            train_accuracy = confusion_train.accuracy()
            cf_train = confusion_train.ret_mat()

            sw.add_scalar(tag='train_time', value=train_time, global_step=step)
            sw.add_scalar(tag='train_loss', value=train_loss, global_step=step)
            sw.add_scalar(tag='train_accuracy', value=train_accuracy, global_step=step)

            param_names = net.collect_params().keys()
            grads = [i.grad() for i in net.collect_params().values()]
            assert len(grads) == len(param_names)
            # logging the gradients of parameters for checking convergence
            for i, name in enumerate(param_names):
                sw.add_histogram(tag=name, values=grads[i], global_step=step, bins=1000)

            print("%d,%.5f,%.5f,%.5f" % (e, train_time, train_accuracy, train_loss))

            # Full pass validation set
            val_err = 0
            val_batches = 0
            val_acc = mx.metric.Accuracy()
            confusion_valid = ConfusionMatrix(n_class)

            for batch in val_iter:
                input = batch.data[0]
                mask = batch.data[1]
                label = batch.label[0]

                with mx.autograd.predict_mode():
                    output = net(input, mask)

                    if (loss_fn == 'cosine'):
                        # Cosine loss
                        l = loss_function(output, nd.one_hot(label, n_class), nd.array([1.0], ctx=ctx))
                    else:
                        # Softmax cross entropy
                        l = loss_function(output, label)

                preds = output.argmax(axis=1)
                np_label = label.asnumpy()
                val_acc.update(label, preds)
                np_preds = preds.astype('int32').asnumpy()
                confusion_valid.batch_add(np_label, np_preds)
                val_batches += 1
                val_err += l.mean().asscalar()

            val_loss = val_err / val_batches
            val_accuracy = confusion_valid.accuracy()
            cf_val = confusion_valid.ret_mat()

            sw.add_scalar(tag='val_loss', value=val_loss, global_step=step)
            sw.add_scalar(tag='val_accuracy', value=val_accuracy, global_step=step)

            f_val_acc = val_accuracy

            # Full pass test set if validation accuracy is higher
            if f_val_acc >= best_val_acc:
                test_batches = 0

                confusion_test = ConfusionMatrix(n_class)

                mask_nd = nd.from_numpy(mask_test.astype(np.float32)).as_in_context(ctx)
                X_nd = nd.from_numpy(X_test.astype(np.float32)).as_in_context(ctx)
                y_nd = nd.from_numpy(y_test.astype(np.float32)).as_in_context(ctx)

                test_iter = mx.io.NDArrayIter([X_nd, mask_nd], y_nd, batch_size, shuffle=False)

                for batch in test_iter:
                    input = batch.data[0]
                    mask = batch.data[1]
                    label = batch.label[0]

                    with mx.autograd.predict_mode():
                        output = net(input, mask)

                    preds = output.argmax(axis=1)
                    np_label = label.astype('int32').asnumpy()
                    np_preds = preds.astype('int32').asnumpy()
                    confusion_test.batch_add(np_label, np_preds)

                print(confusion_test.accuracy())
                print(confusion_test.ret_mat())

                best_val_acc = f_val_acc

                net.save_parameters(net_params_path)

    sw.close()

if __name__ == 'main':
    train()
