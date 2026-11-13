"""DE-GAN's official Keras models; released MSE GAN + 100 BCE training rule."""
import os
import random
os.environ.setdefault('TF_USE_LEGACY_KERAS','1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')

import numpy as np
import tensorflow as tf

from ..util import SEED, require, write_json
from .restoration import module_file


def train(e):
    tf.keras.utils.set_random_seed(SEED)
    # tf-keras 2.16 passes float 1e9 to Random.randint; Python 3.12 rejects
    # integral floats. Preserve the seeded RNG sequence with integer bounds.
    from tf_keras.src import backend
    class IntegerBoundsRandom(random.Random):
        def randint(self,a,b):return super().randint(int(a),int(b))
    backend._SEED_GENERATOR.generator=IntegerBoundsRandom(SEED)
    devices=tf.config.list_physical_devices('GPU');require(len(devices)==1,'exactly one assigned GPU required')
    for device in devices:tf.config.experimental.set_memory_growth(device,True)
    tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(2)
    m=module_file('paper2_degan_official',e.args.upstream/'DE-GAN/models/models.py')
    # Compatibility alias only: modern Keras renamed lr to learning_rate.
    m.get_optimizer=lambda:tf.keras.optimizers.Adam(learning_rate=1e-4*e.args.learning_rate_scale)
    g=m.generator_model(biggest_layer=1024);d=m.discriminator_model()
    if e.args.init:g.load_weights(str(e.args.init/'generator.h5'));d.load_weights(str(e.args.init/'discriminator.h5'))
    go=m.get_optimizer();do=m.get_optimizer();go.build(g.trainable_variables);do.build(d.trainable_variables)
    config={'framework':tf.__version__,'generator_parameters':g.count_params(),'discriminator_parameters':d.count_params(),
        'architecture':'unchanged official generator biggest_layer1024 and discriminator',
        'loss':'released code MSE adversarial + 100 binary cross entropy reconstruction',
        'update':'D(real), D(detached fake from eval G), G through eval D; two D optimizer steps',
        'adaptation':'modern optimizer alias and Python3.12 integer RNG bounds; same paired 256 inputs and shortened budget',
        'initialization':str(e.args.init) if e.args.init else 'official initialization','learning_rate_scale':e.args.learning_rate_scale}
    write_json(e.out/'model_config.json',config);e.log(config)

    @tf.function(reduce_retracing=True)
    def update(x,y):
        fake=tf.stop_gradient(g(x,training=False))
        with tf.GradientTape() as tape:
            real=d([y,x],training=True);dlr=tf.reduce_mean(tf.square(real-1))
        do.apply_gradients(zip(tape.gradient(dlr,d.trainable_variables),d.trainable_variables))
        with tf.GradientTape() as tape:
            rejected=d([fake,x],training=True);dlf=tf.reduce_mean(tf.square(rejected))
        do.apply_gradients(zip(tape.gradient(dlf,d.trainable_variables),d.trainable_variables))
        with tf.GradientTape() as tape:
            reconstructed=g(x,training=True);valid=d([reconstructed,x],training=False)
            gl=tf.reduce_mean(tf.square(valid-1))+100*tf.reduce_mean(tf.keras.backend.binary_crossentropy(y,reconstructed))
        go.apply_gradients(zip(tape.gradient(gl,g.trainable_variables),g.trainable_variables))
        return gl,(dlr+dlf)/2

    def predict(x):return np.clip(g(x[...,None],training=False).numpy()[...,0],0,1)

    def save(path):
        path.mkdir();g.save_weights(str(path/'generator.h5'));d.save_weights(str(path/'discriminator.h5'))
        tf.train.Checkpoint(generator=g,discriminator=d,g_optimizer=go,d_optimizer=do).write(str(path/'state'))
        # Verify that the standalone inference weights really reload.
        probe=np.asarray(e.x[(e.val if len(e.val) else e.train)[:1]],np.float32)/255;before=predict(probe)   # data without a synthetic validation: a training tile
        g.load_weights(str(path/'generator.h5'));require(np.array_equal(before,predict(probe)),'DE-GAN reload mismatch')

    for step in range(1,e.args.steps+1):
        if step==1:before_g=g.trainable_variables[0].numpy().copy();before_d=d.trainable_variables[0].numpy().copy()
        x,y=e.batch();gl,dl=update(x[...,None],y[...,None]);gl=float(gl);dl=float(dl)
        require(np.isfinite(gl) and np.isfinite(dl),'nonfinite DE-GAN loss')
        if step==1:
            require(not np.array_equal(before_g,g.trainable_variables[0].numpy()),'generator did not update')
            require(not np.array_equal(before_d,d.trainable_variables[0].numpy()),'discriminator did not update')
        if step==1 or step%100==0:e.log({'step':step,'generator_loss':gl,'discriminator_loss':dl})
        if step%e.args.eval_every==0 or step==e.args.steps:
            e.evaluate(predict,step,save)
            if e.stop:break
    e.finish()
