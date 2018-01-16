import tensorflow as tf
from tensorflow.contrib.learn import RunConfig
from tensorflow.python.keras.backend import set_learning_phase

from utils.models import UNIT_DA_SHVN_TO_MNIST
from utils.pipelines import train_input_fn_builder, test_input_fn_builder


def model_fn(features, labels, mode, params):
    model = UNIT_DA_SHVN_TO_MNIST(data_format=params['data_format'], batch_size=params['batch_size'])

    if mode == tf.estimator.ModeKeys.PREDICT:
        set_learning_phase(False)
        image_b = features['image']
        predictions = model.get_predictions(image_b)

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions)
            })

    if mode == tf.estimator.ModeKeys.TRAIN:
        set_learning_phase(True)

        image_a, image_b = features['image1'], features['image2']
        labels_a, labels_b = labels['label1'], labels['label2']

        discriminator_loss = model.update_discriminator(image_a, image_b, labels_a)
        generator_loss, _, _, _, _ = model.update_generator(image_a, image_b)

        predictions_b = model.get_predictions(image_b)['class']
        metrics = model.get_metrics(labels_b, predictions_b)

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


# Define a dictionary of hyper parameters
hyper_params = {
    'data_format': 'channels_first',
    'batch_size': 64,
    'subset_train': (-1, -1),
    'subset_test': -1,
    'n_epochs': 100000
}

# Input pipelines
train_input_fn = train_input_fn_builder(subset=hyper_params['subset_train'], data_format=hyper_params['data_format'],
                                        batch_size=hyper_params['batch_size'], num_epochs=1)

test_input_fn = test_input_fn_builder(subset=hyper_params['subset_test'], data_format=hyper_params['data_format'])

# Configuration for the Estimator
config = RunConfig(save_summary_steps=187500,
                   save_checkpoints_steps=9375,
                   keep_checkpoint_max=2,
                   model_dir='../model_dir')

# Create the Estimator
da_model = tf.estimator.Estimator(
    model_fn=model_fn,
    config=config,
    params=hyper_params
)

# Placeholder to build a Serving input function
if hyper_params['data_format'] == 'channels_last':
    img_serv = tf.placeholder(tf.float32, (64, 32, 32, 1))
else:
    img_serv = tf.placeholder(tf.float32, (64, 1, 32, 32))

# Train the model and save the model that gives the best accuracy
best_accuracy = 0

for i in range(hyper_params['n_epochs']):
    print('Epoch: {}'.format(i + 1))
    da_model.train(train_input_fn)
    res = da_model.evaluate(test_input_fn)

    if res['classification_accuracy'] > best_accuracy:
        best_accuracy = res['classification_accuracy']
        print('best accuracy is {} and achieved on epoch {}'.format(best_accuracy, i+1))

        serv_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
            'image': img_serv
        }, default_batch_size=64)

        da_model.export_savedmodel('../best_model', serv_input_fn)
