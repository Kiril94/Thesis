#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 17:38:43 2021

@author: neus
"""

from multilingual_pdf2text.pdf2text import PDF2Text
from multilingual_pdf2text.models.document_model.document import Document as pdfDocument
import logging
from glob import iglob
import re 
import os
logging.basicConfig(level=logging.INFO)


path = '/home/neus/Documents/09.UCPH/MasterThesis/ucph_sif_data/2019_02'
target_words = ['microbleed', 'infarkt','ingen microbleeds']

def get_docs_path_list(scan_dir):
    reports = iglob(f"{scan_dir}/*/*/DOC/*/*.pdf")
    reports_list = [x for x in reports] 
    return reports_list

docs_path_list = get_docs_path_list(path)
class Document():

    def __init__(self, doc_dir,language='dan'):
        self.language = language
        self.path = doc_dir
        
        split_path = doc_dir.split('/')
        self.patient_id = split_path[-5]
        self.study_id = split_path[-4]
        
        self.document = pdfDocument(
        document_path=doc_dir,
        language=language
        )
        
        pdf2text = PDF2Text(document=self.document)
        self.content = pdf2text.extract()
    
    def get_path(self):
        return self.path 
    
    def get_patientId(self):
        return self.patient_id
    
    def get_content(self):
        return self.content
    

    def find_word(self,word):
        """
        

        Parameters
        ----------
        word : str
            word to look for.

        Returns
        -------
        list
            list of the indexes where the word is.

        """
        
        indexes = []
        for page in self.content:
            index = [m.start() for m in re.finditer(word, page['text'].lower())]
            indexes.append({'page_number':page['page_number'],'indexes':index})
        
        return indexes
    
    

doc = Document('/home/neus/Documents/09.UCPH/MasterThesis/ucph_sif_data/2019_02/00b91959df63a6fdc7d5e692637acb0a/0ea1e64cb419ee3473b152fb184388b5/DOC/c43c9091a30993384335b057dd2ffca4/69a85927d80da5fc3e4154a290cce785.pdf')
print(doc.find_word('microbleed'))
print(doc.find_word('ingen microbleed'))