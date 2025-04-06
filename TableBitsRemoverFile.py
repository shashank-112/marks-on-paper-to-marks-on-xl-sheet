import numpy as np
import cv2
class TableLinesRemover:

    def __init__(self, raw_perspective_corrected_images, no_of_images):
        self.raw_perspective_corrected_images = raw_perspective_corrected_images
        self.no_of_images = no_of_images
        
    def execute(self):
        self.all_bits_images = []
        for self.i in range(self.no_of_images):
            self.store_Process_Image("0_raw_image", self.raw_perspective_corrected_images[self.i])
            # self.show_Image(self.raw_perspective_corrected_images)
            
            self.important_Bits_trimmer()
            
            self.store_Process_Image("1_only_marks_row_image", self.only_marks_row_image)
            # self.show_Image(self.only_marks_row_image)
            
            self.resize_The_Image()
            self.store_Process_Image("2_only_marks_row_image_resized", self.only_marks_row_image)
            # self.show_Image(self.only_marks_row_image)
            
            self.threshold_Image_Invert_Image()
            self.store_Process_Image("3_thresholded_inv_image", self.thresholded_image)
            # self.show_Image(self.thresholded_image)
            
            self.dilate_Image()
            self.store_Process_Image("4_dilated_image", self.dilated_image)
            # self.show_Image(self.dilated_image)
            
            self.find_All_Contours()
            self.image_with_all_contours = self.draw_Countours(self.all_contours, self.dilated_image)
            self.store_Process_Image("5_image_with_bits_contours", self.image_with_all_contours)
            # self.show_Image(self.image_with_all_contours)
            
            self.filter_Contours_And_Leave_Only_Rectangles()
            self.image_with_rectangular_contours = self.draw_Countours(self.rectangular_contours, self.dilated_image)
            self.store_Process_Image("6_image_with_rectangular_bits_contours", self.image_with_rectangular_contours)
            # self.show_Image(self.image_with_rectangular_contours)
            
            self.image_Nois_Remover()
            self.nois_less_reduced_image = self.draw_Countours(self.perfect_contour, self.dilated_image)
            self.store_Process_Image("7_image_with_no_nois_rectangular_bits_contours", self.nois_less_reduced_image)
            # self.show_Image(self.nois_less_reduced_image)
            
            self.sorting_Perfect_Contour()
            self.individual_Sorted_Bits_Storer()
            # self.display_All_Bits()
            self.all_bits_images.append(self.all_bits_image)
        return self.all_bits_images
    # morphology
    
    
    def resize_The_Image(self):
        self.only_marks_row_image = cv2.resize(self.only_marks_row_image, (830, 109))
            
    def threshold_Image_Invert_Image(self):
        _, self.thresholded_image = cv2.threshold(self.only_marks_row_image.copy(), 75, 255, cv2.THRESH_BINARY_INV)    
    
    def dilate_Image(self):
        self.dilated_image = cv2.dilate(self.thresholded_image.copy(), None, iterations=0)
    
    def important_Bits_trimmer(self):
        original_height, original_width = self.raw_perspective_corrected_images[self.i].shape[:2]
        start_width = int(original_width * (11 / 100))
        start_height = int(original_height * (53 / 100))
        end_width = int(original_width * (83 / 100))
        end_height = int(original_height * (72 / 100))
        self.only_marks_row_image = self.raw_perspective_corrected_images[self.i][start_height:end_height,start_width:end_width]
    
    def find_All_Contours(self):
        self.all_contours, _ = cv2.findContours(self.dilated_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    def filter_Contours_And_Leave_Only_Rectangles(self):
        self.rectangular_contours = []
        for contour in self.all_contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.1 * peri, True)
            if len(approx) == 4:
                self.rectangular_contours.append(approx)
    
    def image_Nois_Remover(self):
        self.perfect_contour = []
        for con in self.rectangular_contours:
            if cv2.contourArea(con) > 1000:
                self.perfect_contour.append(con)
    
    def sorting_Perfect_Contour(self):
        self.sorted_bits_contour_list = sorted(self.perfect_contour, key=lambda x: x[0][0][0])
           
    def individual_Sorted_Bits_Storer(self):
        self.all_bits_image = []
        for i in range(len(self.sorted_bits_contour_list)):
            points = self.sorted_bits_contour_list[i]
            points = np.array(points).reshape(-1, 2)
            x, y, w, h = cv2.boundingRect(points)
            cropped_image = self.only_marks_row_image[y+2:y+h-2, x+2:x+w-2]
            # cropped_image = self.only_marks_row_image[y-7:y+h+7, x-7:x+w+7]
            self.all_bits_image.append(cropped_image)    
            
    def store_All_List_Images(self):
        
        for i in range(len(self.sorted_bits_contour_list)):
            path = "./all_processed_images/2_ProcessImages/3_AllBitsFromTableStore/"+str(i)+'.jpg'
            cv2.imwrite(path, self.sorted_bits_contour_list[i])
            
    def display_All_Bits(self):
        for i in range(len(self.all_bits_image)):
            image = self.all_bits_image[i]
            cv2.imshow("a", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
             
           
         
    def draw_Countours(self, contour, image):
        image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(image, contour, -1, (0, 255, 0), 3)
        return image
    
    def store_Process_Image(self, file_name, image):
        # path = "./2_ProcessImages/2_TableBitsExtractedImages/" + file_name + ".jpg"
        path = "./all_processed_images/2_ProcessImages/2_TableBitsExtractedImages/" + file_name + ".jpg"
        cv2.imwrite(path, image)
        
    def show_Image(self, image):
        cv2.imshow("a", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()