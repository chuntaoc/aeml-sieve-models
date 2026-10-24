"""Convert the StarDist 2D_versatile_fluo pretrained model to ONNX (AEML, NTUST).
Requires: pip install stardist tensorflow-cpu tf2onnx onnxruntime
"""
import os
import numpy as np
import tensorflow as tf
import tf2onnx
import onnxruntime as ort
from stardist.models import StarDist2D

m = StarDist2D.from_pretrained('2D_versatile_fluo')
km = m.keras_model


@tf.function(input_signature=[tf.TensorSpec([1, None, None, 1], tf.float32, name='input')])
def f(x):
    p, d = km(x, training=False)
    return {'prob': p, 'dist': d}


tf2onnx.convert.from_function(f, input_signature=f.input_signature, opset=17,
                              output_path='stardist_2d_versatile_fluo.onnx')

# parity check
x = np.random.rand(1, 256, 320, 1).astype(np.float32)
p, d = km.predict(x, verbose=0)
s = ort.InferenceSession('stardist_2d_versatile_fluo.onnx')
out = dict(zip([o.name for o in s.get_outputs()], s.run(None, {'input': x})))
print('max |dprob| =', np.abs(out['prob'] - p).max(), ' max |ddist| =', np.abs(out['dist'] - d).max())
