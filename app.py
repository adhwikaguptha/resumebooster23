import os
import io
import uuid
import tempfile
import logging
from flask import Flask, request, render_template, flash, redirect, url_for, session, send_file
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from document_processor import extract_text, calculate_ats_score, create_pdf, create_docx
from resume_optimizer import generate_resume_feedback

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
fh = logging.FileHandler('app.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(fh)

# Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key-for-development")
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
TEMP_FOLDER = tempfile.gettempdir()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    try:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id

        if 'resume' not in request.files:
            flash('No resume file uploaded')
            return redirect(url_for('index'))

        resume_file = request.files['resume']
        job_description = request.form.get('job_description', '')

        if resume_file.filename == '':
            flash('No resume file selected')
            return redirect(url_for('index'))

        if not job_description:
            flash('Job description is required')
            return redirect(url_for('index'))

        if resume_file and allowed_file(resume_file.filename):
            filename = secure_filename(resume_file.filename)
            file_path = os.path.join(TEMP_FOLDER, f"{session_id}_{filename}")
            resume_file.save(file_path)
            session['resume_path'] = file_path
            session['original_filename'] = filename

            try:
                resume_text = extract_text(file_path)

                # Provider/model logic (can be dynamic later)
                provider = "groq"  # or "together", "huggingface", "openrouter"
                model = "mixtral"  # or "gemma", "llama2-70b", etc.

                ai_output = generate_resume_feedback(resume_text, job_description, provider=provider, model=model)

                match_analysis = f"MATCH SCORE: {int(ai_output['ats_score'] * 100)}%"
                suggestions = "\n".join(ai_output['suggestions'])
                rewritten_resume = ai_output['optimized_resume']

                initial_score = ai_output['ats_score']
                initial_score_normalized = int(initial_score * 100)

                new_score = initial_score  # Could be recalculated from rewritten_resume
                new_score_normalized = initial_score_normalized

                # Store rewritten resume in temp file
                rewritten_resume_path = os.path.join(TEMP_FOLDER, f"{session_id}_rewritten_resume.txt")
                with open(rewritten_resume_path, 'w') as f:
                    f.write(rewritten_resume)

                session['rewritten_resume_path'] = rewritten_resume_path
                session['initial_score'] = initial_score_normalized
                session['new_score'] = new_score_normalized

                return render_template('index.html',
                                       initial_score=initial_score_normalized,
                                       new_score=new_score_normalized,
                                       suggestions=suggestions,
                                       rewritten_resume=rewritten_resume,
                                       resume_text=resume_text,
                                       job_description=job_description,
                                       match_analysis=match_analysis,
                                       has_detailed_analysis=True,
                                       analysis_complete=True)

            except Exception as e:
                logger.error(f"Error processing resume: {str(e)}")
                flash(f"Error processing resume: {str(e)}")
                return redirect(url_for('index'))

        else:
            flash('File type not allowed. Please upload a PDF or DOCX file.')
            return redirect(url_for('index'))

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        flash(f'An unexpected error occurred: {str(e)}')
        return redirect(url_for('index'))

@app.route('/download/<format>', methods=['GET'])
def download_resume(format):
    try:
        if 'rewritten_resume_path' not in session:
            flash('No resume data available. Please analyze a resume first.')
            return redirect(url_for('index'))

        rewritten_resume_path = session.get('rewritten_resume_path')
        if not rewritten_resume_path or not os.path.exists(rewritten_resume_path):
            flash('Resume content is missing or expired. Please try analyzing your resume again.')
            return redirect(url_for('index'))

        with open(rewritten_resume_path, 'r') as f:
            rewritten_resume = f.read()

        original_filename = session.get('original_filename', 'resume')
        base_filename = original_filename.rsplit('.', 1)[0]

        if format == 'docx':
            output = create_docx(rewritten_resume)
            mimetype = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            filename = f"{base_filename}_rewritten.docx"
        elif format == 'pdf':
            output = create_pdf(rewritten_resume)
            mimetype = 'application/pdf'
            filename = f"{base_filename}_rewritten.pdf"
        else:
            flash('Invalid format specified')
            return redirect(url_for('index'))

        file_obj = io.BytesIO(output.getvalue())
        file_obj.seek(0)

        return send_file(file_obj, mimetype=mimetype, as_attachment=True, download_name=filename)

    except Exception as e:
        logger.error(f"Error generating downloadable file: {str(e)}")
        flash('An error occurred while generating the downloadable file. Please try again.')
        return redirect(url_for('index'))

@app.errorhandler(413)
def request_entity_too_large(error):
    flash('The file is too large')
    return redirect(url_for('index')), 413

@app.errorhandler(500)
def internal_server_error(error):
    flash('An internal server error occurred')
    return redirect(url_for('index')), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}")
    flash('An unexpected error occurred')
    return redirect(url_for('index')), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
