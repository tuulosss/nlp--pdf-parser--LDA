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
import webbrowser
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt


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
        raise Exception("No PDF files found in the folder.")
    documents = []
    for pdf in pdf_files:
        try:
            text = convert_pdf_to_text(pdf)
            documents.append(text)
        except Exception as e:
            documents.append('')
    vect = CountVectorizer(ngram_range=(1,1), stop_words='english')
    dtm = vect.fit_transform(documents)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_dtf = lda.fit_transform(dtm)
    features = np.array(vect.get_feature_names_out())
    sorting = np.argsort(lda.components_)[:, ::-1]
    topics = []
    for topic_idx in range(n_topics):
        top_words = features[sorting[topic_idx][:10]]
        topics.append(f"Topic {topic_idx}: {', '.join(top_words)}")
    # Group files by dominant topic
    topic_to_files = {i: [] for i in range(n_topics)}
    for i, pdf in enumerate(pdf_files):
        dominant_topic = int(np.argmax(lda_dtf[i]))
        topic_to_files[dominant_topic].append(pdf)
    global last_topic_to_files
    last_topic_to_files = topic_to_files
    return topics, topic_to_files


def open_file(path):
    webbrowser.open(f'file://{os.path.abspath(path)}')


def run_lda():
    folder_path = filedialog.askdirectory()
    if not folder_path:
        return
    try:
        global last_topic_to_files
        topics, last_topic_to_files = lda_analysis_on_folder(folder_path)
        output_text.config(state='normal')
        output_text.delete('1.0', tk.END)
        for idx, topic_desc in enumerate(topics):
            output_text.insert(tk.END, f"{topic_desc}\n", f"topic{idx}")
            output_text.insert(tk.END, "Articles:\n")
            files = last_topic_to_files.get(idx, [])
            if files:
                for fidx, fname in enumerate(files):
                    tag = f"file_{idx}_{fidx}"
                    output_text.insert(tk.END, f"  {os.path.basename(fname)}\n", tag)
                    output_text.tag_bind(tag, "<Button-1>", lambda e, f=fname: open_file(f))
                    output_text.tag_config(tag, foreground="blue", underline=1)
            else:
                output_text.insert(tk.END, "  (None)\n")
            output_text.insert(tk.END, "\n")
        output_text.config(state='disabled')
        show_topic_distribution(last_topic_to_files)
    except Exception as e:
        messagebox.showerror('Error', str(e))


def save_output():
    file_path = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    if file_path:
        text = output_text.get("1.0", tk.END)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)


def show_topic_distribution(topic_to_files):
    topic_nums = list(topic_to_files.keys())
    counts = [len(topic_to_files[k]) for k in topic_nums]
    plt.figure(figsize=(8,4))
    plt.bar([f"Topic {k}" for k in topic_nums], counts, color='skyblue')
    plt.xlabel("Topic")
    plt.ylabel("Number of Articles")
    plt.title("Number of Articles per Topic")
    plt.tight_layout()
    plt.show()


root = tk.Tk()
root.title('AutoDocSum LDA Topic Visualizer')
root.geometry('800x600')

frame = tk.Frame(root)
frame.pack(pady=20)

select_btn = tk.Button(frame, text='Select Folder and Run LDA', command=run_lda)
select_btn.pack(side=tk.LEFT, padx=5)

save_btn = tk.Button(frame, text='Save Output to File', command=save_output)
save_btn.pack(side=tk.LEFT, padx=5)

visualize_btn = tk.Button(frame, text='Show Topic Distribution', command=lambda: show_topic_distribution(last_topic_to_files))
visualize_btn.pack(side=tk.LEFT, padx=5)

output_text = tk.Text(root, height=30, width=100)
output_text.pack(pady=10)
output_text.config(state='disabled')

last_topic_to_files = {}

root.mainloop()


