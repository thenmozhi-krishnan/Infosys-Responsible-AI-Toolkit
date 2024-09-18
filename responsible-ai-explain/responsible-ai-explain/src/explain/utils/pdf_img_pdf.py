'''
Copyright 2024 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies 
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE 
AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

# -*- coding: utf-8 -*-
"""
    Function: pdf_img_pdf
    Objective: 1. Converts pages of a pdf file into multiple images into a folder,
               2. Applies a processing function on each of the image 
               3. Converts back all processed images into a pdf
    Arguments: 
        ip_pdf,tmp_op_fldr,op_pdf
        ip_pdf : Specify path of the pdf file (input file)
        tmp_op_fldr : Specify path of the output folder where converted images will be stored
        op_pdf : Name of the output pdf file
        
    Required Library: PyMuPDF 
"""
from explain.config.logger import CustomLogger
import fitz
import os

log=CustomLogger()

# Function to convert pdf pages into individual images
def pdf_to_img(pdf_path, output_folder, format='png', resolution=200):
  fn = os.path.basename(pdf_path)[:-4]

  doc = fitz.open(pdf_path)
  for page_num in range(doc.page_count):
    page = doc.load_page(page_num)
    pix = page.get_pixmap(matrix=fitz.Matrix(resolution/72, resolution/72))  # Adjust resolution as needed
    output_file = f"{output_folder}/{fn}_{page_num+1}.{format}"
    pix.save(output_file)

# Function to process the image
def proc_img(img,imgfl):
  log.info("Processed Image {} Width:{} Height:{}".format(imgfl,img.width,img.height))

# Function to convert multiple images into a single pdf
def images_to_pdf(image_folder, output_pdf):
  doc = fitz.open()
  image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
  image_files.sort()  # Sort images for consistent order
  for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    img = fitz.Pixmap(image_path)
    #---- Add Image processing code here call function proc_img------
    proc_img(img,image_file)
    # -----------------------------------------
    page = doc.new_page(width=img.width, height=img.height)
    # page.insert_pixmap(0, img)
    page.insert_image(rect=(0, 0, img.width, img.height), pixmap=img)
    os.remove(image_path) #to delete image files in the temp folder
  doc.save(output_pdf)
