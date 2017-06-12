
import numpy as np
import tensorflow as tf
from multiprocessing import Process


from keras.layers import Input, Dense
from keras.models import Model


def build_model():
    inp = Input(shape=(10,))
    out = Dense(2, activation='relu')(inp)
    model = Model(inputs=inp, outputs=out)
    model_input, model_output = model.input, model.output
    adam = tf.train.AdamOptimizer(1e-4)
    grad_var = adam.compute_gradients(model_output, var_list=model.weights)
    model_var = [var for grad, var in grad_var]
    return model, model_var

def build_submodel(model, model_var):
    inp = Input(shape=(10,))
    out = Dense(2, activation='relu')(inp)
    submodel = Model(inputs=inp, outputs=out)
    input, output = submodel.input, submodel.output
    target = tf.placeholder(tf.float32, shape=(None, 2))
    sub_loss = tf.nn.l2_loss(submodel.output - target)
    adam = tf.train.AdamOptimizer(1e-4)
    grad_var_sub = adam.compute_gradients(sub_loss, var_list=submodel.weights)
    submodel_grad_model_var = [(grad, var) for (grad, _), var in zip(grad_var_sub, model_var)]
    opt = adam.apply_gradients(submodel_grad_model_var)
    return submodel, opt, input, output, target



def agent(model, submodel, input_batch, output_batch):
    pass
    sess = tf.Session()
    submodel, opt, input, output, target = submodel
    #~ sess.run(tf.variables_initializer(submodel.weights))
    #~ print sess.run(model.weights)
    #~ print sess.run(submodel.weights)
    submodel.set_weights(model.get_weights())


input_batch = np.random.rand(3200, 784)
output_batch = np.random.rand(3200, 10)

model, model_var = build_model()
sess = tf.Session()
sess.run(tf.variables_initializer(model.weights))

submodel_list = [build_submodel(model, model_var) for _ in range(4)]

proc1 = Process(target=agent, args=(model, submodel_list[0], input_batch, output_batch))
proc1.start()
proc1.join()

#~ sess = tf.Session()
#~ sess.run(opt, feed_dict={input_main: input_batch, output_main: output_batch})


