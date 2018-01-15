import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.keras.backend import set_learning_phase

from utils.models import UNIT_DA_SHVN_TO_MNIST
from utils.pipelines import train_input_fn_builder, test_input_fn_builder

hooks = [tf_debug.LocalCLIDebugHook()]


def model_fn(features, labels, mode, params):
    model = UNIT_DA_SHVN_TO_MNIST(data_format='channels_last')

    if mode == tf.estimator.ModeKeys.PREDICT:
        set_learning_phase(False)
        image_b = features['image']
        predictions = model.get_predictions(image_b)

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions)

    if mode == tf.estimator.ModeKeys.TRAIN:
        set_learning_phase(True)

        image_a, image_b = features['image1'], features['image2']
        labels_a, labels_b = labels['label1'], labels['label2']

        discriminator_loss = model.update_discriminator(image_a, image_b, labels_a)
        generator_loss, _, _, _, _ = model.update_generator(image_a, image_b)

        predictions_b = model.get_predictions(image_b)
        metrics = model.get_metrics(labels_b, predictions_b['class'])

        total_loss = discriminator_loss + generator_loss
        train_op = model.get_train_op(loss_fn_discriminator=discriminator_loss,
                                      loss_fn_generator=generator_loss)

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=total_loss,
            eval_metric_ops=metrics,
            train_op=train_op
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        set_learning_phase(False)

        img, lab = features['image'], labels['label']

        pred_logits = model.classify_image_b(img)
        loss = model.batch_cross_entropy(lab, pred_logits)

        pred_class = model.get_predictions(img)['class']
        metrics = model.get_metrics(lab, pred_class)

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops=metrics
        )


train_input_fn = train_input_fn_builder((65, 66), data_format='channels_last')
test_input_fn = test_input_fn_builder(100, data_format='channels_last')

da_model = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir='./model_dir/'
)

# da_model.train(train_input_fn, steps=2, hooks=hooks)
da_model.train(train_input_fn, steps=2)
# da_model.evaluate(test_input_fn)
# sess = tf.InteractiveSession()
#
# model = UNIT_DA_SHVN_TO_MNIST(data_format='channels_last')
# images, labels = train_input_fn()
# img_a, img_b = images['image1'], images['image2']
# labels_a, labels_b = labels['label1'], labels['label2']
#
# op1 = model.dis(img_a, img_b)
# op1
#
# init = tf.global_variables_initializer()
# sess.run(init)
# sess.run(tf.shape(op1[1]), feed_dict={learning_phase(): 1})
