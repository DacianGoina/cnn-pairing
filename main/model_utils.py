"""
Model utils
"""

import time
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.metrics import Precision, Recall, AUC, F1Score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Auxiliary class used for storing the model performance results (obtained values)
class CustomMetricsCallback(tf.keras.callbacks.Callback):
  def __init__(self):
    super().__init__()
    self.epoch_times = []
    self.metrics_log = {
      "loss": [],
      "accuracy": [],
      "auc": [],
      "precision": [],
      "recall": [],
      "f1_score": [],
      "val_loss": [],
      "val_accuracy": [],
      "val_auc": [],
      "val_precision": [],
      "val_recall": [],
      "val_f1_score": []
    }

  def on_epoch_begin(self, epoch, logs=None):
    self.start_time = time.time()  # Start time for the epoch

  def on_epoch_end(self, epoch, logs=None):
    # Record epoch time
    epoch_time = time.time() - self.start_time
    self.epoch_times.append(epoch_time)

    # Capture metrics from logs
    if logs:
      for metric in self.metrics_log.keys():
        self.metrics_log[metric].append(logs.get(metric, None))


# Compute confusion matrix from multiclass matrix
def get_confusion_matrix(model, x_test, y_test, no_of_classes_val):
  y_pred = model.predict(x_test)  # Outputs probabilities for each class

  # Convert predictions to class labels
  y_pred_labels = tf.argmax(y_pred, axis=1).numpy()  # For multiclass
  y_true_labels = tf.argmax(y_test, axis=1).numpy()  # True labels

  # Compute the confusion matrix
  conf_matrix = tf.math.confusion_matrix(
      labels =y_true_labels,
      predictions=y_pred_labels,
      num_classes=no_of_classes_val  # Replace with the number of classes in your dataset
  )

  return conf_matrix.numpy()


# Create the model, feed it with data and start training and validation
def create_and_train_model( x_train, x_test, y_train, y_test, rows_num, cols_num):

  # Obs:
  # accuracy	Training data	Measures the model's performance on the training set.
  # val_accuracy	Validation data	Indicates how well the model generalizes to unseen data.

  tf.keras.backend.clear_session()

  callback = CustomMetricsCallback()

  model = Sequential([
    Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=(rows_num, cols_num, 1)),
    MaxPooling2D(pool_size=(2, 2), padding='same'),

    Conv2D(16, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2), padding='same'),

    Flatten(),
    #Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
  ])

  # Step 6: Compile the model
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', AUC(), Precision(), Recall(), F1Score(name='f1_score', average='macro') ])

  # Step 7: Train the model
  model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test), callbacks=[callback])

  # Step 8: Evaluate the model on the test set
  test_results = model.evaluate(x_test, y_test)
  # print(f'Test accuracy: {test_accuracy:.4f}')
  print("test results: ", test_results)
  test_results_float = [float(metric_val) for metric_val in  test_results]
  results_on_testing_at_end = list(zip(["loss", "accuracy", "auc", "precision", "recall", "f1"] , test_results_float))

  print("-" * 15)

  epoch_times = callback.epoch_times
  metrics_log = callback.metrics_log
  metrics_log['results_on_testing_at_end'] = results_on_testing_at_end

  return model, epoch_times, metrics_log