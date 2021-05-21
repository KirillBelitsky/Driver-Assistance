import os
import shutil
import numpy as np
import tensorflow as tf
from train.yoloTrainModel import YoloTrainModel
from train.dataset import Dataset
from train.trainConfig import cfg
from util.util import Util
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

class TrainNeuralNetwork:

    def __init__(self):
        self.trainset = None
        self.testset = None

        self.first_stage_epochs = None
        self.second_stage_epochs = None

        self.global_steps = None
        self.warmup_steps = None
        self.total_steps = None

        self.freeze_layers = None

        self.model = None
        self.yolo_model = None
        self.optimizer = None
        self.writer = None

    def train(self, _argv):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        self.trainset = Dataset(_argv, is_training=True)
        self.testset = Dataset(_argv, is_training=False)

        logdir = "logs/log"
        isfreeze = False

        steps_per_epoch = len(self.trainset)
        self.first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS

        self.global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
        self.warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
        self.total_steps = (self.first_stage_epochs + self.second_stage_epochs) * steps_per_epoch
        # train_steps = (first_stage_epochs + second_stage_epochs) * steps_per_period

        input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])

        # STRIDES, ANCHORS, NUM_CLASS, XYSCALE, IOU_LOSS_THRESH
        _, _, NUM_CLASS, _ = Util.load_config(_argv)

        self.freeze_layers = Util.load_freeze_layer(_argv.tiny)

        self.yolo_model = YoloTrainModel(input_layer, NUM_CLASS, _argv)
        conv_layers = self.yolo_model.create_tiny_model() if _argv.tiny else self.yolo_model.create_full_model()

        if _argv.tiny:
            bbox_tensors = []
            for i, fm in enumerate(conv_layers):
                if i == 0:
                    bbox_tensor = self.yolo_model.decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, i)
                else:
                    bbox_tensor = self.yolo_model.decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, i)
                bbox_tensors.append(fm)
                bbox_tensors.append(bbox_tensor)
        else:
            bbox_tensors = []
            for i, fm in enumerate(conv_layers):
                if i == 0:
                    bbox_tensor = self.yolo_model.decode_train(fm, cfg.TRAIN.INPUT_SIZE // 8, i)
                elif i == 1:
                    bbox_tensor = self.yolo_model.decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, i)
                else:
                    bbox_tensor = self.yolo_model.decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, i)
                bbox_tensors.append(fm)
                bbox_tensors.append(bbox_tensor)

        self.model = tf.keras.Model(input_layer, bbox_tensors)
        self.model.summary()

        if _argv.weights == None:
            print("Training from scratch")
        else:
            if _argv.weights.split(".")[len(_argv.weights.split(".")) - 1] == "weights":
                Util.load_weights(self.model, _argv.weights, _argv.tiny)
            else:
                self.model.load_weights(_argv.weights)
            print('Restoring weights from: %s ... ' % _argv.weights)


        self.optimizer = tf.keras.optimizers.Adam()
        if os.path.exists(logdir):
            shutil.rmtree(logdir)
        self.writer = tf.summary.create_file_writer(logdir)

        for epoch in range(self.first_stage_epochs + self.second_stage_epochs):
            if epoch < self.first_stage_epochs:
                if not isfreeze:
                    isfreeze = True
                    for name in self.freeze_layers:
                        freeze = self.model.get_layer(name)
                        Util.freeze_all(freeze)
            elif epoch >= self.first_stage_epochs:
                if isfreeze:
                    isfreeze = False
                    for name in self.freeze_layers:
                        freeze = self.model.get_layer(name)
                        Util.unfreeze_all(freeze)
            for image_data, target in self.trainset:
                self.train_step(image_data, target)
            # for image_data, target in self.testset:
            #     self.test_step(image_data, target)

        self.model.save("config/trained.h5")
        print('.h5 saved')

        full_model = tf.function(lambda x: self.model(x))
        full_model = full_model.get_concrete_function(
            tf.TensorSpec(self.model.inputs[0].shape, self.model.inputs[0].dtype))
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()

        tf.io.write_graph(frozen_func.graph, 'config', 'trained.pb',
                             as_text=False)
        tf.io.write_graph(frozen_func.graph, 'config', 'trained.pbtxt',
                          as_text=True)

    # define training step function
    # @tf.function
    def train_step(self, image_data, target):
        with tf.GradientTape() as tape:
            pred_result = self.model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(self.freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = self.yolo_model.compute_loss(pred, conv, target[i][0], target[i][1], i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            tf.print("=> STEP %4d/%4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (self.global_steps, self.total_steps, self.optimizer.lr.numpy(),
                                                               giou_loss, conf_loss,
                                                               prob_loss, total_loss))

            # update learning rate
            self.global_steps.assign_add(1)
            if self.global_steps < self.warmup_steps:
                lr = self.global_steps / self.warmup_steps * cfg.TRAIN.LR_INIT
            else:
                lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                    (1 + tf.cos((self.global_steps - self.warmup_steps) / (self.total_steps - self.warmup_steps) * np.pi))
                )
            self.optimizer.lr.assign(lr.numpy())

            # writing summary data
            with self.writer.as_default():
                tf.summary.scalar("lr", self.optimizer.lr, step=self.global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=self.global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=self.global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=self.global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=self.global_steps)
            self.writer.flush()

    def test_step(self, image_data, target):
        with tf.GradientTape() as tape:
            pred_result = self.model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(self.freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = self.yolo_model.compute_loss(pred, conv, target[i][0], target[i][1], i=i)

                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss
            tf.print("=> TEST STEP %4d   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (self.global_steps, giou_loss, conf_loss,
                                                                   prob_loss, total_loss))


