import os
import time
import requests
from io import BytesIO
from PIL import Image
import pytesseract
import pandas as pd
from fpdf import FPDF
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from openai import OpenAI
import logging
from urllib.parse import urljoin
import tkinter as tk
from tkinter import ttk
import threading
import html
from concurrent.futures import ThreadPoolExecutor

# ------------------ SETTINGS ------------------
OUTPUT_DIR = os.path.join(os.getcwd(), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(OUTPUT_DIR, "web_scraper.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

MAX_THREADS = 5  # Paralel işleme için
TIMEOUT = 15     # Request timeout

# ------------------ SCRAPER FUNCTIONS ------------------

def summarize_text(client, text, max_chunk=2000):
    try:
        summary = ""
        for i in range(0, len(text), max_chunk):
            chunk = text[i:i+max_chunk]
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Summarize this text in bullet points:\n{chunk}"}],
                temperature=0.3
            )
            summary += response.choices[0].message.content + "\n"
        return summary.strip()
    except Exception as e:
        logging.error(f"Summarization error: {e}")
        return text

def fetch_page(url, retries=3, proxy=None):
    attempt = 0
    while attempt < retries:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            proxies = {"http": proxy, "https": proxy} if proxy else None
            response = requests.get(url, timeout=TIMEOUT, headers=headers, verify=False, proxies=proxies)
            response.encoding = response.apparent_encoding
            soup = BeautifulSoup(response.text, 'html.parser')
            for s in soup(["script", "style"]):
                s.extract()
            if len(soup.get_text(strip=True)) > 50 and "Loading" not in soup.get_text():
                return soup
            # Dinamik sayfa
            chrome_options = Options()
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--window-size=1920,1080")
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(url)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            html_content = driver.page_source
            driver.quit()
            soup = BeautifulSoup(html_content, 'html.parser')
            for s in soup(["script", "style"]):
                s.extract()
            return soup
        except Exception as e:
            logging.warning(f"Page fetch failed (attempt {attempt+1}): {url} -> {e}")
            attempt += 1
            time.sleep(2)
    logging.error(f"Failed to fetch page after {retries} attempts: {url}")
    return None

def extract_text(soup):
    return [p.get_text().strip() for p in soup.find_all(['p','h1','h2','h3','h4','h5','h6']) if p.get_text().strip()]

def extract_links(soup):
    return [a['href'] for a in soup.find_all('a', href=True) if a['href'].startswith(('http', 'https'))]

def extract_tables(soup):
    tables = []
    for table_tag in soup.find_all('table'):
        rows = table_tag.find_all('tr')
        table_data = []
        for row in rows:
            cells = [c.get_text().strip() for c in row.find_all(['td', 'th'])]
            if cells:
                table_data.append(cells)
        if table_data:
            tables.append(table_data)
    return tables

def extract_image_text(img_url):
    try:
        if not img_url.startswith('http'):
            img_url = f"https:{img_url}"
        img_data = requests.get(img_url, timeout=10, verify=False).content
        img = Image.open(BytesIO(img_data))
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        logging.warning(f"Image could not be processed: {img_url} -> {e}")
        return ""

def save_pdf(text_list, filename="summary.pdf"):
    if not text_list:
        logging.info("No text data to save in PDF.")
        return
    clean_text_list = [html.unescape(t) for t in text_list]
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, "\n\n".join(clean_text_list))
    file_path = os.path.join(OUTPUT_DIR, filename)
    pdf.output(file_path)
    return file_path

def save_csv(data_list, filename="output.csv"):
    if not data_list:
        logging.info(f"No data found to save in {filename}.")
        return
    try:
        df = pd.DataFrame(data_list)
        file_path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        return file_path
    except Exception as e:
        logging.error(f"CSV save error: {e}")
        return None

def get_next_page_url(soup, base_url):
    next_link = soup.find('a', string=lambda t: t and ('Next' in t or '>' in t))
    if next_link and next_link.get('href'):
        return urljoin(base_url, next_link['href'])
    return None

# ------------------ SCRAPE ALL ------------------

def scrape_all_pages(start_url, max_pages=5):
    all_texts, all_links, all_tables, all_image_texts = [], [], [], []
    url = start_url
    page_count = 0

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        while url and page_count < max_pages:
            soup = fetch_page(url)
            if not soup:
                print(f"⚠ Could not fetch page: {url}")
                break
            all_texts.extend(extract_text(soup))
            all_links.extend(extract_links(soup))
            all_tables.extend(extract_tables(soup))
            image_urls = [img['src'] for img in soup.find_all('img', src=True)]
            all_image_texts.extend(executor.map(extract_image_text, image_urls))
            url = get_next_page_url(soup, url)
            page_count += 1
    return all_texts, all_links, all_tables, all_image_texts

# ------------------ GUI ------------------

pdf_filename = "summary.pdf"
csv_filename = "texts.csv"
client = None

def set_api_key():
    global client
    key = api_key_entry.get().strip()
    if key:
        client = OpenAI(api_key=key)
        status_label.config(text="API Key set successfully.", fg="green")
        root.after(3000, lambda: status_label.config(text=""))

def set_pdf_name():
    global pdf_filename
    name = file_name_entry.get().strip()
    if name:
        pdf_filename = name + ".pdf"
        status_label.config(text=f"PDF file will be saved as: {pdf_filename}", fg="green")
        root.after(3000, lambda: status_label.config(text=""))

def set_csv_name():
    global csv_filename
    name = file_name_entry.get().strip()
    if name:
        csv_filename = name + ".csv"
        status_label.config(text=f"CSV file will be saved as: {csv_filename}", fg="green")
        root.after(3000, lambda: status_label.config(text=""))

def start_scraping_threaded():
    threading.Thread(target=start_scraping, daemon=True).start()

def start_scraping():
    if not client:
        status_label.config(text="Please set your OpenAI API Key first.", fg="red")
        return
    url = url_entry.get().strip()
    if not url:
        status_label.config(text="Please enter a valid URL.", fg="red")
        return

    status_label.config(text="Scraping in progress...", fg="blue")
    progress_bar['value'] = 0
    root.update()

    try:
        texts, links, tables, image_texts = scrape_all_pages(url)
        status_label.config(text="Summarization in progress...", fg="orange")
        root.update()

        summarized_texts = [summarize_text(client, t) for t in texts] if texts else []
        summarized_image_texts = [summarize_text(client, t) for t in image_texts if t] if image_texts else []

        pdf_path = save_pdf(summarized_texts + summarized_image_texts, pdf_filename)
        table_paths = [save_csv(table, f"{os.path.splitext(csv_filename)[0]}_table{i}.csv") for i, table in enumerate(tables, start=1)]
        link_path = save_csv([{"Link": l} for l in links], f"{os.path.splitext(csv_filename)[0]}_links.csv") if links else None

        msg = "Scraping & summarization complete!\n"
        msg += f"PDF saved at: {pdf_path}\n" if pdf_path else ""
        for p in table_paths:
            if p:
                msg += f"Table CSV saved at: {p}\n"
        if link_path:
            msg += f"Links CSV saved at: {link_path}\n"
        status_label.config(text=msg, fg="green")
    except Exception as e:
        status_label.config(text=f"Error: {e}", fg="red")
        logging.error(f"Scraping error: {e}")

# ------------------ MAIN GUI ------------------

root = tk.Tk()
root.title("Web Scraper & Summarizer")
root.geometry("750x500")

tk.Label(root, text="Enter OpenAI API Key:").pack(pady=5)
api_key_entry = tk.Entry(root, width=70, show="*")
api_key_entry.pack(pady=5)
tk.Button(root, text="Set API Key", command=set_api_key, width=20, bg="green", fg="white").pack(pady=5)

tk.Label(root, text="Enter starting URL:").pack(pady=5)
url_entry = tk.Entry(root, width=70)
url_entry.pack(pady=5)

tk.Label(root, text="File name (without extension, e.g., 'summary')").pack(pady=5)
file_name_entry = tk.Entry(root, width=40)
file_name_entry.pack(pady=5)

btn_frame = tk.Frame(root)
btn_frame.pack(pady=5)
tk.Button(btn_frame, text="Set PDF Name", command=set_pdf_name, width=15).grid(row=0, column=0, padx=5)
tk.Button(btn_frame, text="Set CSV Name", command=set_csv_name, width=15).grid(row=0, column=1, padx=5)

tk.Button(root, text="Start Scraping", command=start_scraping_threaded, width=25, bg="blue", fg="white").pack(pady=10)

progress_bar = ttk.Progressbar(root, length=700, mode='determinate')
progress_bar.pack(pady=10)

status_label = tk.Label(root, text="", fg="blue", justify="left")
status_label.pack(pady=5)

root.mainloop()
