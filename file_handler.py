from PyPDF2 import PdfReader


def load_txt(file):
    try:
        return file.read().decode("utf-8")
    except Exception as e:
        raise Exception(f"TXT Read Error: {e}")


def load_pdf(file):
    try:
        reader = PdfReader(file)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text

    except Exception as e:
        raise Exception(f"PDF Read Error: {e}")