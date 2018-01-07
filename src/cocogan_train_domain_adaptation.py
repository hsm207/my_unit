import tensorflow as tf

from utils.pipelines import train_input_fn_builder, test_input_fn_builder

sess = tf.InteractiveSession()

train_input_fn = train_input_fn_builder((64 * 3 + 2, 64 * 3 + 1))
test_input_fn = test_input_fn_builder(100)

