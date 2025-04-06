import PdfToImageConverterFile as pic
import TableExtractorFile as te 
import TableBitsRemoverFile as tbr 
import ImageToTableConverterFile as itc
import cv2
import tensorflow as tf


# path = './all_processed_images/0_assets/final1.pdf'
path = './uploads/final.pdf'
image_extractor = pic.PdfToImageConverter(path)
all_images, no_of_images = image_extractor.execute()

table_extractor = te.TableExtractor(all_images, no_of_images)
perspective_corrected_images = table_extractor.execute()

table_bits_remover = tbr.TableLinesRemover(perspective_corrected_images, no_of_images)
list_of_marks_bits_images = table_bits_remover.execute()

model = tf.keras.models.load_model("./uploads/cnn_mnist_model.h5")
bits_to_table = itc.ImageToTableConverter(list_of_marks_bits_images, no_of_images, model)
bits_to_table.execute()

