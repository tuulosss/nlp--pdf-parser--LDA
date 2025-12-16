import tkinter as tk
from tkinter import filedialog, messagebox
import pyLDAvis
import pyLDAvis.sklearn
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


def lda_analysis(pdf_path, n_topics=5):
    text = convert_pdf_to_text(pdf_path)
    vect = CountVectorizer(ngram_range=(1, 1), stop_words='english')
    dtm = vect.fit_transform([text])
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_dtf = lda.fit_transform(dtm)
    features = np.array(vect.get_feature_names_out())
    sorting = np.argsort(lda.components_)[:, ::-1]
    topics = []
    for topic_idx in range(n_topics):
        top_words = features[sorting[topic_idx][:10]]
        topics.append(f"Topic {topic_idx}: {', '.join(top_words)}")
    # Visualization
    vis_data = pyLDAvis.sklearn.prepare(lda, dtm, vect)
    html_path = os.path.abspath('lda_gui_vis.html')
    pyLDAvis.save_html(vis_data, html_path)
    return topics, html_path


def open_html(html_path):
    webbrowser.open(f'file://{html_path}')


def run_lda():
    pdf_path = filedialog.askopenfilename(filetypes=[('PDF Files', '*.pdf')])
    if not pdf_path:
        return
    try:
        topics, html_path = lda_analysis(pdf_path)
        topics_text.delete('1.0', tk.END)
        topics_text.insert(tk.END, '\n'.join(topics))
        if messagebox.askyesno('LDA Visualization', 'LDA analysis complete. Open interactive visualization in browser?'):
            open_html(html_path)
    except Exception as e:
        messagebox.showerror('Error', str(e))


root = tk.Tk()
root.title('LDA PDF Topic Visualizer')
root.geometry('600x400')

frame = tk.Frame(root)
frame.pack(pady=20)

select_btn = tk.Button(frame, text='Select PDF and Run LDA', command=run_lda)
select_btn.pack()

topics_label = tk.Label(root, text='Top Words per Topic:')
topics_label.pack(pady=(20, 0))

topics_text = tk.Text(root, height=10, width=70)
topics_text.pack(pady=10)

root.mainloop()
