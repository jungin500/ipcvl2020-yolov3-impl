import tensorflow as tf
import random

class VisualizeYolo(tf.keras.callbacks.Callback):
    def __init__(self, test_dataset, visualizer_fn, model, display_count=1, display_on_begin=False, display_freq_epoches=200, **kwargs):
        self.display_on_begin = display_on_begin
        self.display_freq_epoches = display_freq_epoches
        self.display_count = display_count
        self.test_dataset = test_dataset
        self.visualize_fn = visualizer_fn
        super(VisualizeYolo, self).__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.display_freq_epoches == 0:
            if epoch == 0 and not self.display_on_begin:
                return

            batch_range = random.randrange(0, self.test_dataset.__len__())
            image, gt_label = self.test_dataset.__getitem__(batch_range)
            net_out = self.model.predict(image)

            batch_image_index = [random.randrange(0, self.test_dataset.batch_size) for _ in range(min(self.display_count, self.test_dataset.batch_size))]
            print("\n[Visualizer] Pulling out batch index [%d/%d]" % (batch_range + 1, self.test_dataset.__len__()))
            print("[Visualizer] Displaying one image out of batch " + str(batch_image_index) + " of %d" % self.test_dataset.batch_size)
            self.visualize_fn(image, gt_label, net_out, batch_range=batch_image_index, scale_range=[0, 1, 2],
                      anchor_range='merge', pred_threshold=0.5, verbose=True)