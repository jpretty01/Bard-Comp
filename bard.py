import cv2
import numpy as np
import os
import pytesseract

russia_1 = os.path.join(os.path.dirname(__file__), 'russia1.jpeg')
russia_2 = os.path.join(os.path.dirname(__file__), 'russia2.jpeg')
texas = os.path.join(os.path.dirname(__file__), 'texas.jpeg')
license_plate_stuff = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

def detect_license_plates(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the cascade classifier to detect the license plates
    license_plate_locations = license_plate_stuff.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    return license_plate_locations

def extract_license_plate(image, license_plate_location):
    # Crop the image to the region of the license plate
    x, y, w, h = license_plate_location
    crop = image[y:y+h, x:x+w]

    # Binarize the image
    binary = cv2.threshold(crop, 127, 255, cv2.THRESH_BINARY)[1]

    # Apply a Gaussian blur to the image
    blurred = cv2.GaussianBlur(binary, (5, 5), 0)

    return blurred

def recognize_license_plate(image):
    # Apply the character recognition algorithm to the license plate
    characters = pytesseract.image_to_string(image, lang='rus', config='--psm 6')

    return characters


def main():
    # Load the first image
    image = cv2.imread(russia_1)

    # Detect the license plates
    license_plate_locations = detect_license_plates(image)

    # Extract the license plates
    license_plates = list(map(extract_license_plate, [image]*len(license_plate_locations), license_plate_locations))

    # Recognize the license plates
    license_plate_characters = list(map(recognize_license_plate, license_plates))

    # Print the detected license plate locations and characters
    print("License plate locations:")
    print(license_plate_locations)

    print("License plate characters:")
    print(license_plate_characters)


# Find two more Russian license plates

# Image 2
# image2 = cv2.imread(russia_2)

# Image 3
# image3 = cv2.imread(texas)


if __name__ == '__main__':
    main()
