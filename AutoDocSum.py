import subprocess
import matplotlib
import mglearn
import numpy
import pandas
import pdfminer
import sklearn

import sys
import os
from io import StringIO
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def convert_pdf_to_text(pdf_path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    with open(pdf_path, 'rb') as fp:
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.get_pages(fp, check_extractable=True):
            interpreter.process_page(page)
    text = retstr.getvalue()
    device.close()
    retstr.close()
    return text


def lda_analysis_on_folder(folder_path, n_topics=5):
    # Get all PDF files in the folder
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("No PDF files found in the folder.")
        return
    print(f"Found {len(pdf_files)} PDF files. Extracting text...")
    documents = []
    for pdf in pdf_files:
        try:
            text = convert_pdf_to_text(pdf)
            documents.append(text)
        except Exception as e:
            print(f"Failed to extract {pdf}: {e}")
    if not documents:
        print("No text extracted from PDFs.")
        return
    print("Performing LDA topic modeling...")
    vect = CountVectorizer(ngram_range=(1,1), stop_words='english')
    dtm = vect.fit_transform(documents)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_dtf = lda.fit_transform(dtm)
    features = np.array(vect.get_feature_names_out())
    sorting = np.argsort(lda.components_)[:, ::-1]
    for topic_idx in range(n_topics):
        top_words = features[sorting[topic_idx][:10]]
        print(f"\nTopic {topic_idx}: {', '.join(top_words)}")
    print("\nLDA analysis complete.")


if __name__ == "__main__":
    folder = input("Enter path to folder containing PDF files: ")
    lda_analysis_on_folder(folder)


