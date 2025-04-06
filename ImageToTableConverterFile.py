import numpy as np
import matplotlib.pyplot as plt
import csv
import openpyxl
import cv2

class ImageToTableConverter:
    def __init__(self,list_of_images, no_of_images, model):
        self.list_of_images = list_of_images
        self.no_of_images = no_of_images
        self.model = model
        
    def execute(self):
        plt.gray()
        self.all_digital_digits_marks = []
        for self.i in range(self.no_of_images):
            self.store_All_List_Images()
            self.digital_digits_marks = []
            for i in range(17):
                self.re_Obtain_Image(i)
                # self.show_Images(self.image, i)
                
                self.re_Size_The_Image()
                # self.show_Images(self.resized_image, i)
                
                self.re_Size_The_Image_Thresholding()
                # self.show_Images(self.thresholded_resized_image, i)
                
                self.find_Countour_On_resize_Image()
                
                self.filter_Countours()

                self.sorting_Perfect_Countour()
                self.result = [0]
                
                for self.contour in self.perfectly_sorted_contours:
                    self.add_Padding_To_Individual_Contour()
                
                    self.normalizing_The_Image()
                
                    self.model_Predict_Digit()
                # print(self.result,end="\n")
                if len(self.result) == 3:
                    self.result[0] = int(str(self.result[1]) + str(self.result[2]))
                    
                elif len(self.result) == 2:
                    self.result[0] = self.result[1]
                self.digital_digits_marks.append(self.result[0])
                    
            self.all_digital_digits_marks.append(self.digital_digits_marks)
        self.save_File_As_Csv()
        self.save_File_As_XLSX()
        return self.all_digital_digits_marks
            
    def store_All_List_Images(self):
        for i in range(17):
            path = "./all_processed_images/2_ProcessImages/3_AllBitsFromTableStore/"+str(i)+'.jpg'
            cv2.imwrite(path, self.list_of_images[self.i][i])
            
    def re_Obtain_Image(self,i):
        self.image = cv2.imread(f"./all_processed_images/2_ProcessImages/3_AllBitsFromTableStore/{i}.jpg", 0)
        
    def re_Size_The_Image(self):
        self.resized_image = cv2.resize(self.image, (28, 28))
        
    def re_Size_The_Image_Thresholding(self):
        _, self.thresholded_resized_image = cv2.threshold(self.resized_image, 110, 255, cv2.THRESH_BINARY_INV)

    def find_Countour_On_resize_Image(self):
        self.contours, _ = cv2.findContours(self.thresholded_resized_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    def filter_Countours(self):
        self.perfect_contours = [contour for contour in self.contours if cv2.contourArea(contour) >= 10]    
        
    def sorting_Perfect_Countour(self):
        self.perfectly_sorted_contours = sorted(self.perfect_contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
        
    def add_Padding_To_Individual_Contour(self):
        x, y, w, h = cv2.boundingRect(self.contour)
        cropped_image = self.thresholded_resized_image[y:y+h, x:x+w]
        h1 = h // 2 + 4
        padded_image = cv2.copyMakeBorder(cropped_image, 5, 5, h1 - (w // 2) + 2, h1 - (w // 2) + 2, cv2.BORDER_CONSTANT, value=0)
        self.padded_image = cv2.resize(padded_image, (28, 28))    
        
    def normalizing_The_Image(self):
        padded_image = self.padded_image.astype(np.float32) / 255
        self.normalized_padded_image = np.expand_dims(padded_image, axis=0)
        
    def model_Predict_Digit(self):
        prediction = self.model.predict(self.normalized_padded_image)
        self.result.append(np.argmax(prediction))
        
    def save_File_As_Csv(self):
        with open('./all_processed_images/3_output_files/output.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.all_digital_digits_marks)
        
    def save_File_As_XLSX(self):
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        
        for row_index, row in enumerate(self.all_digital_digits_marks, start=1):
            for col_index, value in enumerate(row, start=1):
                sheet.cell(row=row_index, column=col_index, value=value)

        workbook.save("./all_processed_images/3_output_files/output.xlsx")
        
    def show_Images(self, image, i):
        plt.matshow(image, fignum=False) 
        plt.show() 
        # if self.i == 12: 
        #     plt.matshow(image, fignum=False) 
        #     plt.show() 
        # else:
        #     pass 
            