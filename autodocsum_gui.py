import tkinter as tk
from tkinter import filedialog, messagebox
import pyLDAvis
import numpy as np
import pandas as pd
from io import StringIO
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import webbrowser
import os

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
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    if not pdf_files:
        raise Exception('No PDF files found in the folder.')
    documents = []
    for pdf in pdf_files:
        print(f"Processing: {os.path.basename(pdf)}")
        try:
            text = convert_pdf_to_text(pdf)
            documents.append(text)
        except Exception as e:
            print(f"Failed to process {os.path.basename(pdf)}: {e}")
            documents.append('')
    vect = CountVectorizer(ngram_range=(1, 1), stop_words='english')
    dtm = vect.fit_transform(documents)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_dtf = lda.fit_transform(dtm)
    features = np.array(vect.get_feature_names_out())
    sorting = np.argsort(lda.components_)[:, ::-1]
    topics = []
    for topic_idx in range(n_topics):
        top_words = features[sorting[topic_idx][:10]]
        topics.append(f"Topic {topic_idx}: {', '.join(top_words)}")
    # Map each file to its dominant topic
    file_topic_map = []
    for i, pdf in enumerate(pdf_files):
        dominant_topic = np.argmax(lda_dtf[i])
        file_topic_map.append((os.path.basename(pdf), dominant_topic, lda_dtf[i][dominant_topic]))
    # For latest pyLDAvis, need to provide vocab and term_frequency explicitly
    vocab = vect.get_feature_names_out()
    term_frequency = np.asarray(dtm.sum(axis=0)).flatten()
    vis_data = pyLDAvis.prepare(
        model=lda,
        dtm=dtm,
        vocab=vocab,
        term_frequency=term_frequency
    )
    html_path = os.path.abspath('lda_gui_vis.html')
    pyLDAvis.save_html(vis_data, html_path)
    return topics, file_topic_map, html_path

def open_html(html_path):
    webbrowser.open(f'file://{html_path}')

def run_lda():
    folder_path = filedialog.askdirectory()
    if not folder_path:
        return
    try:
        topics, file_topic_map, html_path = lda_analysis_on_folder(folder_path)
        topics_text.delete('1.0', tk.END)
        topics_text.insert(tk.END, '\n'.join(topics))
        mapping_text.delete('1.0', tk.END)
        for fname, topic, prob in file_topic_map:
            mapping_text.insert(tk.END, f"{fname} -> Topic {topic} (score: {prob:.2f})\n")
        if messagebox.askyesno('LDA Visualization', 'LDA analysis complete. Open interactive visualization in browser?'):
            open_html(html_path)
    except Exception as e:
        messagebox.showerror('Error', str(e))

root = tk.Tk()
root.title('AutoDocSum LDA Topic Visualizer')
root.geometry('800x600')

frame = tk.Frame(root)
frame.pack(pady=20)

select_btn = tk.Button(frame, text='Select Folder and Run LDA', command=run_lda)
select_btn.pack()

topics_label = tk.Label(root, text='Top Words per Topic:')
topics_label.pack(pady=(20, 0))

topics_text = tk.Text(root, height=8, width=90)
topics_text.pack(pady=10)

mapping_label = tk.Label(root, text='File to Dominant Topic Mapping:')
mapping_label.pack(pady=(10, 0))

mapping_text = tk.Text(root, height=15, width=90)
mapping_text.pack(pady=10)

root.mainloop()
