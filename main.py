import cv2
import numpy as np
import webcolors


class Options:
    window_name = "options"

    def __init__(self):
        # Define Window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # Define Trackbars
        cv2.createTrackbar('thresh_value', self.window_name, 77, 255, lambda x: None)
        cv2.createTrackbar('blur_value', self.window_name, 5, 255, lambda x: None)
        cv2.createTrackbar('scale_factor', self.window_name, 11, 100, lambda x: None)
        cv2.createTrackbar('min_neighbors', self.window_name, 1, 100, lambda x: None)
        cv2.createTrackbar('dilate_value', self.window_name, 1, 10, lambda x: None)
        cv2.createTrackbar('structure_element', self.window_name, 1, 10, lambda x: None)
        cv2.createTrackbar('green_value', self.window_name, 0, 255, lambda x: None)

        self.switch = '0 : OFF \n1 : ON'
        cv2.createTrackbar(self.switch, self.window_name, 0, 1, lambda x: None)

    def find_color(self, rect, image):
        # get the center of the rectangle
        x, y, w, h = rect
        center_x = x + w // 2
        center_y = y + h // 2

        # get the average color of the rectangle by getting the average of the pixels in the rectangle
        average_color = np.average(image[center_y - 5:center_y + 5, center_x - 5:center_x + 5], axis=(0, 1))

        # convert the average color to HSV
        # average_color = cv2.cvtColor(np.array([[average_color]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0][0]

        # get closes color string
        name = self.closest_colour(average_color)
        return name

    def closest_colour(self, requested_colour):
        min_colours = {}
        for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_colour[0]) ** 2
            gd = (g_c - requested_colour[1]) ** 2
            bd = (b_c - requested_colour[2]) ** 2
            min_colours[(rd + gd + bd)] = name
        return min_colours[min(min_colours.keys())]


class Detector:
    window_name = 'video'
    video = 'video/video1.mp4'
    haar_cascade = 'cars.xml'

    def __init__(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.cap = cv2.VideoCapture(self.video)
        self.car_cascade = cv2.CascadeClassifier(self.haar_cascade)
        self.options = Options()
        self.run()

    def run(self):
        while True:

            # Get Trackbar Values
            thresh_value = cv2.getTrackbarPos('thresh_value', self.options.window_name)
            scale_factor = max(cv2.getTrackbarPos('scale_factor', self.options.window_name), 11) / 10
            min_neighbors = cv2.getTrackbarPos('min_neighbors', self.options.window_name)
            blur_value = 2 * cv2.getTrackbarPos('blur_value', self.options.window_name) + 1
            dilate_value = 2 * cv2.getTrackbarPos('dilate_value', self.options.window_name) + 1
            structure_element = cv2.getTrackbarPos('structure_element', self.options.window_name) + 1
            green_value = cv2.getTrackbarPos('green_value', self.options.window_name)
            s = cv2.getTrackbarPos(self.options.switch, self.options.window_name)

            # Read frames from a video
            ret, frames = self.cap.read()

            # filter green color using green_value
            green_mask = cv2.inRange(frames, (0, green_value, 0), (100, 255, 100))
            inv_mask = cv2.bitwise_not(green_mask)
            no_green = cv2.bitwise_and(frames, frames, mask=inv_mask)

            # convert to gray scale of each frames
            processed = cv2.cvtColor(no_green, cv2.COLOR_BGR2GRAY)

            # process threshold
            ret, processed = cv2.threshold(processed, thresh_value, 255, cv2.THRESH_BINARY)

            # process gaussian blur
            processed = cv2.GaussianBlur(processed, (blur_value, blur_value), 0)

            # process dilation
            processed = cv2.dilate(processed, np.ones((dilate_value, dilate_value)))

            # get kernel from structuring element
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (structure_element, structure_element))

            # use morphology for better detection
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

            # Detects cars of different sizes in the input image
            cars = self.car_cascade.detectMultiScale(processed, scale_factor, min_neighbors)

            # Display switch
            display_on = processed if s == 0 else frames

            for (x, y, w, h) in cars:
                # Find color of car
                color = self.options.find_color((x, y, w, h), frames)

                # Find car bounding box
                cv2.rectangle(display_on, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # Display car color (approx)
                cv2.putText(display_on, str(color), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # display frames in a window
            cv2.imshow(self.window_name, display_on)

            # wait for Esc key to stop
            if cv2.waitKey(33) == 27:
                break

        # De-allocate any associated memory usage
        cv2.destroyAllWindows()


if __name__ == '__main__':
    detector = Detector()


