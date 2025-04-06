from flask import Flask, render_template, request, jsonify, send_file
import os
import subprocess
import threading
from threading import Event
import fitz  # PyMuPDF
from urllib.parse import unquote

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
PREVIEW_FOLDER = 'static/previews'
OUTPUT_FOLDER = 'Flask/all_processed_images/3_output_files'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREVIEW_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables
conversion_status = "processing"
conversion_done = Event()

# Helper functions
def generate_pdf_preview(pdf_path, image_path):
    """Generates an image for the first page of the PDF."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap()
    pix.save(image_path)
    doc.close()

def run_conversion_script():
    """Runs the conversion script in a separate thread."""
    global conversion_status
    try:
        print("Starting script execution...")
        result = subprocess.run(
            [r'C:\Users\shashank\.conda\envs\tabels_project_env\python.exe', r'D:\flask_prep\Flask\main.py'],
            capture_output=True, text=True, encoding='utf-8'
        )
        print(f"Script completed with return code: {result.returncode}")
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)

        conversion_status = "completed" if result.returncode == 0 else "failed"
    except Exception as e:
        print("Error running script:", str(e))
        conversion_status = "failed"
    finally:
        conversion_done.set()

# Routes
@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return "No file selected", 400

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'final.pdf')
        file.save(file_path)

        preview_path = os.path.join(PREVIEW_FOLDER, "preview.jpg")
        generate_pdf_preview(file_path, preview_path)

        return render_template('previewpage.html', preview_image=preview_path, pdf_name=file.filename)

    return render_template('index.html')

@app.route("/convert_to_xlsx", methods=['POST'])
def convert_to_xlsx():
    global conversion_done
    conversion_done.clear()

    thread = threading.Thread(target=run_conversion_script)
    thread.start()

    return render_template('loading.html')

@app.route("/status")
def check_status():
    return jsonify(status=conversion_status if conversion_done.is_set() else "processing")

@app.route("/conversion_complete")
def conversion_complete():
    relative_path = "./all_processed_images/3_output_files/output.csv".replace("\\", "/")
    return render_template("final.html", xlsx_file=relative_path)

@app.route("/download")
def download_file():
    relative_path = request.args.get("xlsx_file")
    if relative_path:
        decoded_path = unquote(relative_path)
        absolute_path = os.path.abspath(os.path.join(decoded_path))

        if os.path.exists(absolute_path):
            return send_file(absolute_path, as_attachment=True)

        print("File not found:", absolute_path)

    return "File not found", 404

if __name__ == "__main__":
    app.run(debug=True)
