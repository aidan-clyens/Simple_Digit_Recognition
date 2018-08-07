# Simple Digit Recognition

## Getting Started
### Prerequisites
1. Install the following:
- Python 2.7
- Numpy
- OpenCV

### Installing
1. Clone the directory with `git clone git@github.com:aidanclyens88/Simple_Digit_Recognition.git`.
2. Create and run your own Python 2.7 script, making sure to import trainer.py.
3. The first time the script is run, training data files will be created in the [datasets](./datasets) folder, named train_data.data and train_labels.data.

### Adding training data
1. The training images are stored in the [datasets/digits/training](./datasets/digits/training) folder. Each digit is stored in it's respective folder.
2. You may add you own training images to their respective digits folder following the naming scheme (e.g. digits2a.jpg).
3. Each image must be a clear scan with white text on a black background, roughly 1500x1500 px in size.
4. Delete the train_data.data and train_labels.data files to have them be recreated.

## Example Usage
This will create a training dataset given a set of images and save to data files. Then, the data files will be loaded into Numpy arrays, and trained using a K-Nearest Neighbours algorithm. Then, using a test image, the digits will be predicted and displayed with the results.

```python
    # Create a Trainer object
    contour_area_threshhold = 60
    crop_width = 50
    crop_height = 50
    crop_margin = 5
    trainer = Trainer(contour_area_threshhold, crop_width, crop_height, crop_margin)

    # Create training data files if they have not been already created
    if not os.path.exists(trainer.train_data_file) or not os.path.exists(trainer.train_labels_file):
        print "Creating new data files."
        trainer.create_training_data()

    # Load training data files into Numpy arrays
    train_data, train_labels = trainer.load_training_data()
    # Load a test image and convert it into a Numpy array
    test_data = trainer.create_test_image('example/example.jpg')
    # Use the training data and test image to train the algorithm and predict the drawn digits
    results = trainer.knn_train(train_data, train_labels, test_data)

    # Generate test labels for the test image provided in this example
    test_labels = []
    for x in range(1,10):
        for y in range(0,9):
            test_labels.append(x)
    for x in range(0,9):
        test_labels.append(0)

    # Show the resulting test image with the results printed on top
    trainer.show_result_image(results)
    # Get the accuracy of the results
    accuracy = trainer.get_result_accuracy(results, test_labels)
    print "accuracy =", accuracy, "%"
```

#### Example Test Image Before:
![](./example/example.jpg)

#### Example Test Image After:
![](./example/example_result.jpg)

This example is 94.44% accurate.

## Author
Copyright 2018 Aidan Clyens
