from flask import Flask,render_template,url_for,request,redirect
import pickle
from werkzeug.utils import secure_filename
import os
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import HTMLConverter,TextConverter,XMLConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import io

import sys, getopt

UPLOAD_FOLDER ='C:/Users/hind/Desktop/start_up/cv_app/uploads/'
ALLOWED_EXTENSIONS = set(['pdf','txt'])
app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS
app.config['UPLOAD_EXTENSIONS'] = ['.pdf','.txt','.csv']
app.config['UPLOAD_PATH'] = 'uploads'

app.config['UPLOAD_FOLDER'] =UPLOAD_FOLDER


def convert_pdf_to_txt(data):
    fp = open(data, 'rb')
    manager = PDFResourceManager()
    output = io.StringIO()
    #codec = 'utf-8'
    converter = TextConverter(manager, output, laparams=LAParams())
    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(manager, converter)
    # Process each page contained in the document.

    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
        
    convertedPDF =  output.getvalue()
    fp.close(); converter.close(); output.close()
    return convertedPDF

@app.route('/')
def home():
	#return render_template('home.html')

    return render_template('upload.html')


@app.route('/uploads/<filename>')
def predire(filename):

	#Alternative Usage of Saved Model
	model = pickle.load(open("model.pkl","rb"))
	transformer = pickle.load(open("transformer.pkl","rb"))
	file_path = UPLOAD_FOLDER + filename
	with open(file, 'r') as file:
		data = file.read().replace('\n', '')
	vect=model.predict(transformer.transform([data]))
		
	#return render_template('results.html',prediction = vect)
	return render_template('index.html',prediction = vect)	

@app.route('/upload_file', methods=['POST'])
def upload_files():

    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            abort(400)
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))

    model = pickle.load(open("model.pkl","rb"))
    transformer = pickle.load(open("transformer.pkl","rb"))
    file_path = UPLOAD_FOLDER + filename
    data=convert_pdf_to_txt(file_path)
    with open(file_path, 'r') as file:
    	data = data.replace('\n', '')
    vect=model.predict(transformer.transform([data]))

    return render_template('index.html',prediction = vect)
		
	#return render_template('results.html',prediction = vect)
		




if __name__=='__main__':
	app.run(debug=True)
