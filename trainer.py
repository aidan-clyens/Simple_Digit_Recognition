# Imports
import numpy as np
import cv2
import os
import re

"""Trainer
"""
class Trainer():
    datasets_path = './datasets'
    training_path = os.path.join(datasets_path, './digits/training')
    train_data_file = os.path.join(datasets_path, 'train_data.data')
    train_labels_file = os.path.join(datasets_path, 'train_labels.data')

    """Constructor
    """
    def __init__(self, threshold_area, crop_width, crop_height, crop_margin):
        self.threshold_area = threshold_area
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.crop_margin = crop_margin

    """show_image
    """
    def show_image(self, title, image):
        cv2.imshow(title, image)
        cv2.waitKey()
        cv2.destroyAllWindows()

    """preprocess_image
    """
    def preprocess_image(self, image_link, invert=False):
        # Read the image using the path given and convert it to grayscale
        image = cv2.imread(image_link)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply a threshold and invert the image colour if specified
        if invert:
            thresh = cv2.threshold(gray, 200, 300, cv2.THRESH_BINARY_INV)[1]
        else:
            thresh = cv2.threshold(gray, 200, 300, cv2.THRESH_BINARY)[1]

        return thresh

    """find_contours
    """
    def find_contours(self, image):
        # Find every contour in the image using the OpenCV function
        contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        new_contours = []
        # Filter the contours by area, to eliminate invalid contours
        for c in contours:
            [x, y, w, h] = cv2.boundingRect(c)
            area = cv2.contourArea(c)

            if area > self.threshold_area:
                # Draw a rectangle at each contour on the image for viewing purposes
                cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,255),1)
                new_contours.append(c)

        return new_contours

    """convert_image
    """
    def convert_image(self, image, contour):
        # Crop the image at the region of interest, specified by the contour
        [x, y, w, h] = cv2.boundingRect(contour)
        roi = image[y:y+h, x:x+w]
        # Only resize the cropped image if the region of interest has an area greater than 0
        if not roi.size == 0:
            # Resize the region of interest to the specified width and height reshape the pixels to be in 1 row
            roi = cv2.resize(roi, (self.crop_width, self.crop_height))
            sample = roi.reshape((1, self.crop_width*self.crop_height))

        return sample

    """get_digit_data
    """
    def get_digit_data(self, data, image, contours):
        for c in contours:
            sample = self.convert_image(image, c)
            data = np.append(data, sample, 0)

        return data

    """create_training_data
    """
    def create_training_data(self):
        # Create an array for images of each number
        imgs = [[],[],[],[],[],[],[],[],[],[]]
        # Read through every file in the training directory
        for root, dirs, files in os.walk(self.training_path):
            for f in files:
                img_path = os.path.join(root, f)
                # Preprocess the training image, extract the digit number from the filename and add to it's respective position in the array
                image = self.preprocess_image(img_path)
                num = int(re.findall(r'\d+', f)[0])
                imgs[num].append(image)

        train_data = np.empty((0, self.crop_width*self.crop_height), np.float32)
        train_labels = []

        # Iterate through each image in the array
        for [n, num] in enumerate(imgs):
            for img in num:
                # Find the contours of the image and then iterate through each contour, converting into a numpy array
                contours = self.find_contours(img)
                train_data = self.get_digit_data(train_data, img, contours)

                print n
                for c in contours:
                    train_labels.append(n)

        # Save the labels as type int64
        train_labels = np.array(train_labels, np.int64)
        train_labels.reshape((train_labels.size, 1))
        # Save the numpy arrays for the training data and training labels to data files
        self.save_training_data(train_data, train_labels)

    """save_training_data
    """
    def save_training_data(self, train_data, train_labels):
        np.savetxt(self.train_data_file, train_data, fmt='%s')
        np.savetxt(self.train_labels_file, train_labels, fmt='%s')

    """load_training_data
    """
    def load_training_data(self):
        train_data = np.loadtxt(self.train_data_file, dtype=np.float32)
        train_labels = np.loadtxt(self.train_labels_file, dtype=np.int64)

        return train_data, train_labels

    """create_test_image
    """
    def create_test_image(self, image_link):
        # Preprocess the test image, inverting the colour and then find the contours
        test_image = self.preprocess_image(image_link, invert=True)
        test_contours = self.find_contours(test_image)
        self.test_image = test_image
        self.test_contours = test_contours
        # Get a numpy array of each cropped digit in the test image
        test_data = np.empty((0, self.crop_width*self.crop_height), np.float32)
        test_data = self.get_digit_data(test_data, test_image, test_contours)

        return test_data

    """knn_train
    """
    def knn_train(self, train_data, train_labels, test_data):
        # Using the training data and training labels, train the K-Nearest Neighbours algorithm
        knn = cv2.KNearest()
        knn.train(train_data, train_labels)
        _, results, _, _ = knn.find_nearest(test_data, k=5)

        return results

    """show_result_image
    """
    def show_result_image(self, results):
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (x-self.crop_margin,y-self.crop_margin)
        font_scale = 0.7
        font_colour = (255,0,255)
        font_type = 2
        for [i,c] in enumerate(self.test_contours):
            [x, y, w, h] = cv2.boundingRect(c)
            cv2.putText(self.test_image, str(results[i][0]), position, font, font_scale, font_colour, font_type)
        self.show_image('result', self.test_image)

if __name__ == '__main__':
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
