import fitz  # PyMuPDF

def load_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""

    for page in doc:
        full_text += page.get_text()
    
    return full_text

def extract_images(pdf_path):
    doc = fitz.open(pdf_path)
    image_captions = []

    for page_num, page in enumerate(doc):
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            # You can extract actual image bytes if needed:
            # xref = img[0]
            # base_image = doc.extract_image(xref)
            # image_bytes = base_image["image"]
            image_captions.append(f"[Image on Page {page_num + 1} - placeholder caption]")
    
    return image_captions

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
