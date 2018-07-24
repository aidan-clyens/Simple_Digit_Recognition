# Simple Digit Recognition

## Dependencies
- Numpy
- OpenCV

## Example Usage
This will create a training dataset given a set of images and save to data files. Then, the data files will be loaded into Numpy arrays, and trained using a K-Nearest Neighbours algorithm. Then, using a test image, the digits will be predicted and displayed with the results.

```python
  contour_area_threshhold = 60
  crop_width = 50
  crop_height = 50
  crop_margin = 5
  trainer = Trainer(contour_area_threshhold, crop_width, crop_height, crop_margin)

  trainer.create_training_data()
  train_data, train_labels = trainer.load_training_data()
  test_data = trainer.create_test_image('numbers_test.jpg')

  results = trainer.knn_train(train_data, train_labels, test_data)

  trainer.show_result_image(results)
```
