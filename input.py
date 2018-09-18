from trainer import Trainer
import os
import sys

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Enter file name: python", sys.argv[0], "filename"
        print "Enter python input.py --help for more commands"
        exit()

    if sys.argv[1] == "--help":
        print "Usage: python input.py filename --command"
        exit()

    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print input_file, "is not a valid file name."
        exit()

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

    # Show the resulting test image with the results printed on top
    trainer.show_result_image(results)
