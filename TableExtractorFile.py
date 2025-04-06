import cv2
import numpy as np

class TableExtractor:
    def __init__(self, all_images, no_of_images):
        self.all_images = all_images
        self.no_of_images = no_of_images
        
    def execute(self):
        self.perspective_corrected_images = []
        for i in range(self.no_of_images):
            self.read_Image(i)
            self.store_Process_Image("0_raw_image", self.raw_image)
            # self.show_Image(self.raw_image)
            
            self.threshold_Image_Invert_Image()
            self.store_Process_Image("1_thresholded_inv_image", self.thresholded_image)
            # self.show_Image(self.thresholded_image)
            
            self.dilate_Image()
            self.store_Process_Image("2_dilated_image", self.dilated_image)
            # self.show_Image(self.dilated_image)
            
            self.find_All_Contours()
            self.image_with_all_contours = self.draw_Countours(self.all_contours, self.thresholded_image)
            self.store_Process_Image("3_image_with_all_contours", self.image_with_all_contours)
            # self.show_Image(self.image_with_all_contours)
            
            self.filter_contours_and_leave_only_rectangles()
            self.image_with_rectangular_contours = self.draw_Countours(self.rectangular_contours, self.thresholded_image)
            self.store_Process_Image("4_image_with_rectangular_contours", self.image_with_rectangular_contours)
            # self.show_Image(self.image_with_rectangular_contours)
            
            self.find_Largest_Contour_By_Area()
            self.image_with_max_area_contours = self.draw_Countours([self.contour_with_max_area], self.thresholded_image)
            self.store_Process_Image("5_image_with_max_area_contours", self.image_with_max_area_contours)
            # self.show_Image(self.image_with_max_area_contours)
            
            self.order_Points_In_The_Contour_With_Max_Area(self.raw_image)
            self.store_Process_Image("6_image_with_edge_points", self.four_point_image)
            # self.show_Image(self.four_point_image)
            
            self.calculate_New_Width_And_Height_Of_Image(self.raw_image)
            
            self.apply_Perspective_Transform()
            self.store_Process_Image("7_perspective_corrected_image", self.perspective_corrected_image)
            # self.show_Image(self.perspective_corrected_image)
            self.perspective_corrected_images.append(self.perspective_corrected_image)
        return self.perspective_corrected_images
        
        
    def read_Image(self, i):
        self.raw_image = self.all_images[i]
        
    def threshold_Image_Invert_Image(self):
        self.thresholded_image = cv2.adaptiveThreshold(
            self.raw_image.copy(),       # Input image
            255,                         # Maximum value to use with THRESH_BINARY_INV
            cv2.ADAPTIVE_THRESH_MEAN_C,  # Adaptive method (Mean)
            cv2.THRESH_BINARY_INV,       # Thresholding type
            blockSize=11,                # Size of the local neighborhood (odd number)
            C=30                         # Constant subtracted from the mean
        )
        # _, self.thresholded_image = cv2.threshold(self.raw_image.copy(), 75, 255, cv2.THRESH_BINARY_INV)

    def dilate_Image(self):
        self.dilated_image = cv2.dilate(self.thresholded_image.copy(), None, iterations=4)

    def find_All_Contours(self):
        self.all_contours, _ = cv2.findContours(self.dilated_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    def filter_contours_and_leave_only_rectangles(self):
        self.rectangular_contours = []
        for contour in self.all_contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.1 * peri, True)
            if len(approx) == 4:
                self.rectangular_contours.append(approx)

    def find_Largest_Contour_By_Area(self):
        max_area = 0
        self.contour_with_max_area = None
        for contour in self.rectangular_contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                self.contour_with_max_area = contour

    def order_Points_In_The_Contour_With_Max_Area(self, image):
        self.contour_with_max_area_ordered = self.order_Points()
        for point in self.contour_with_max_area_ordered:
            point_coordinates = (int(point[0]), int(point[1]))
            self.four_point_image = cv2.circle(image, point_coordinates, 10, (0, 0, 255), -1)

    def order_Points(self):
        self.contour_with_max_area = self.contour_with_max_area.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = self.contour_with_max_area.sum(axis=1)
        rect[0] = self.contour_with_max_area[np.argmin(s)]
        rect[2] = self.contour_with_max_area[np.argmax(s)]
        diff = np.diff(self.contour_with_max_area, axis=1)
        rect[1] = self.contour_with_max_area[np.argmin(diff)]
        rect[3] = self.contour_with_max_area[np.argmax(diff)]
        return rect
    
    def calculate_New_Width_And_Height_Of_Image(self, image):
        existing_image_width = image.shape[1]
        existing_image_width_reduced_by_10_percent = int(existing_image_width * 0.9)
        
        distance_between_top_left_and_top_right = self.calculateDistanceBetween2Points(self.contour_with_max_area_ordered[0], self.contour_with_max_area_ordered[1])
        distance_between_top_left_and_bottom_left = self.calculateDistanceBetween2Points(self.contour_with_max_area_ordered[0], self.contour_with_max_area_ordered[3])

        aspect_ratio = distance_between_top_left_and_bottom_left / distance_between_top_left_and_top_right

        self.new_image_width = existing_image_width_reduced_by_10_percent
        self.new_image_height = int(self.new_image_width * aspect_ratio)
        return self.new_image_width, self.new_image_height,self.contour_with_max_area_ordered
    
    def calculateDistanceBetween2Points(self, p1, p2):
        dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        return dis
        
    def apply_Perspective_Transform(self):
        pts1 = np.float32(self.contour_with_max_area_ordered)
        pts2 = np.float32([[0, 0], [self.new_image_width, 0], [self.new_image_width, self.new_image_height], [0, self.new_image_height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        self.perspective_corrected_image = cv2.warpPerspective(self.raw_image, matrix, (self.new_image_width, self.new_image_height))

    def draw_Countours(self, contour, image):
        image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(image, contour, -1, (0, 255, 0), 3)
        return image

    def store_Process_Image(self, file_name, image):
        path = f"./all_processed_images/2_ProcessImages/1_TableExtractedImages/{file_name}.jpg"
        cv2.imwrite(path, image)
        
    def show_Image(self, image):
        cv2.imshow("a", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
