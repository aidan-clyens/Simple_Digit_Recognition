from trainer import Trainer
import os

if __name__ == '__main__':
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
    test_data = trainer.create_test_image('docs/images/example.jpg')
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
