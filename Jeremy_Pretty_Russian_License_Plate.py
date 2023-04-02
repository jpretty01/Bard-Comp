import cv2
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
 # Load the images
    images = [(russia_1, "russia1"), (russia_2, "russia2"), (texas, "texas")]

    for image_path, image_name in images:
        image = cv2.imread(image_path)

        # Detect the license plates
        license_plate_locations = detect_license_plates(image)

        # Extract the license plates
        license_plates = list(map(extract_license_plate, [image]*len(license_plate_locations), license_plate_locations))

        # Recognize the license plates
        license_plate_characters = list(map(recognize_license_plate, license_plates))

        # Print the detected license plate locations and characters
        print(f"License plate locations ({image_name}):")
        print(license_plate_locations)

        print(f"License plate characters ({image_name}):")
        print(license_plate_characters)


if __name__ == '__main__':
    main()
