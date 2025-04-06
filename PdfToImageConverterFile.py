from PIL import Image, ImageEnhance
from pdf2image import convert_from_path
import cv2
poppler_path = r'C:\Program Files\poppler-24.08.0\Library\bin'

class PdfToImageConverter:
    def __init__(self,path):
        self.pdf_path = path
        
    def execute(self):
        self.obtain_Image()
        self.dimn_All_Images()
        self.reobtain_All_Images()
        return self.all_Perfect_Images, len(self.all_Imperfect_Images)

    def obtain_Image(self):
        self.all_Imperfect_Images = convert_from_path(self.pdf_path, poppler_path=poppler_path)

    def dimn_All_Images(self):
        for i, image in enumerate(self.all_Imperfect_Images):
            en = ImageEnhance.Brightness(image)
            image = en.enhance(0.9)
            image.save(f"./all_processed_images/1_storeAllImages/{i}.jpg")
            
    def reobtain_All_Images(self):
        self.all_Perfect_Images = []
        for i in range(len(self.all_Imperfect_Images)):
            path = f"./all_processed_images/1_storeAllImages/{i}.jpg"
            image = cv2.imread(path, 0)
            self.all_Perfect_Images.append(image)