from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from detector_module import process_video
import sqlite3
import os
import json
import uuid

# Add custom Jinja filter for extracting filename from path (cross-platform)
def basename_filter(path):
    """Extract filename from path, works on both Windows and Unix"""
    return os.path.basename(path)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Change to the script's directory to ensure relative paths work
os.chdir(BASE_DIR)

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'  # Change this to a random secret key

# Register custom Jinja filter
app.jinja_env.filters['basename'] = basename_filter

# Configuration
DB_PATH = os.path.join(BASE_DIR, 'users.db')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_db():
    """Initialize the database with users and processing history tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    
    # Processing history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processing_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            original_filename TEXT NOT NULL,
            input_path TEXT NOT NULL,
            output_path TEXT NOT NULL,
            stats TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    """Home page - always shows login page first"""
    # Always redirect to signin to ensure users see login page first
    # Session will be checked in dashboard route if user is already logged in
    return redirect(url_for('signin'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """User registration page"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if not username or not email or not password:
            flash('All fields are required!', 'error')
            return render_template('signup.html')
        
        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return render_template('signup.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long!', 'error')
            return render_template('signup.html')
        
        # Check if user already exists
        conn = get_db_connection()
        existing_user = conn.execute(
            'SELECT * FROM users WHERE username = ? OR email = ?',
            (username, email)
        ).fetchone()
        
        if existing_user:
            flash('Username or email already exists!', 'error')
            conn.close()
            return render_template('signup.html')
        
        # Create new user
        hashed_password = generate_password_hash(password)
        conn.execute(
            'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
            (username, email, hashed_password)
        )
        conn.commit()
        conn.close()
        
        flash('Account created successfully! Please sign in.', 'success')
        return redirect(url_for('signin'))
    
    return render_template('signup.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    """User login page"""
    # If already logged in, redirect to dashboard
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Please enter both username and password!', 'error')
            return render_template('signin.html')
        
        # Check user credentials
        conn = get_db_connection()
        user = conn.execute(
            'SELECT * FROM users WHERE username = ?',
            (username,)
        ).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash(f'Welcome back, {username}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!', 'error')
            return render_template('signin.html')
    
    return render_template('signin.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page - shown after successful login"""
    if 'user_id' not in session:
        flash('Please sign in to access the dashboard.', 'error')
        return redirect(url_for('signin'))
    
    # Get processing history for the user
    conn = get_db_connection()
    history = conn.execute(
        'SELECT * FROM processing_history WHERE user_id = ? ORDER BY created_at DESC LIMIT 10',
        (session['user_id'],)
    ).fetchall()
    conn.close()
    
    return render_template('dashboard.html', 
                         username=session.get('username'),
                         history=history)

@app.route('/detector', methods=['GET', 'POST'])
def detector():
    """Video upload and processing page"""
    if 'user_id' not in session:
        flash('Please sign in to use the detector.', 'error')
        return redirect(url_for('signin'))
    
    if request.method == 'POST':
        # Check if file was uploaded
        if 'video' not in request.files:
            flash('No file selected! Please choose a video file.', 'error')
            return redirect(request.url)
        
        file = request.files['video']
        
        # Check if file was actually selected (not just empty filename)
        if file.filename == '' or file.filename is None:
            flash('No file selected! Please choose a video file before submitting.', 'error')
            return redirect(request.url)
        
        # Additional check: verify file object is valid
        if not hasattr(file, 'filename') or not file.filename:
            flash('Invalid file upload. Please try selecting the file again.', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Generate unique filename
            filename = secure_filename(file.filename)
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{unique_id}_{filename}"
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)
            
            # Generate output filename
            output_filename = f"output_{unique_id}.avi"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            try:
                # Process video
                flash('Processing video... This may take a few minutes.', 'success')
                stats = process_video(input_path, output_path, 
                                    user_id=session['user_id'], 
                                    display=False)
                
                # Save to database
                conn = get_db_connection()
                conn.execute(
                    'INSERT INTO processing_history (user_id, original_filename, input_path, output_path, stats) VALUES (?, ?, ?, ?, ?)',
                    (session['user_id'], file.filename, input_path, output_path, json.dumps(stats))
                )
                conn.commit()
                conn.close()
                
                # Redirect to results page
                return redirect(url_for('results', output_file=output_filename))
                
            except Exception as e:
                flash(f'Error processing video: {str(e)}', 'error')
                # Clean up on error - handle Windows file locking issue
                if os.path.exists(input_path):
                    try:
                        os.remove(input_path)
                    except (PermissionError, OSError):
                        # File might still be in use, skip deletion
                        # It will be cleaned up later or can be manually deleted
                        pass
                return redirect(request.url)
        else:
            flash('Invalid file type! Please upload a video file (mp4, avi, mov, etc.)', 'error')
            return redirect(request.url)
    
    return render_template('detector.html', username=session.get('username'))

@app.route('/results')
def results():
    """Display processing results and download link"""
    if 'user_id' not in session:
        flash('Please sign in to view results.', 'error')
        return redirect(url_for('signin'))
    
    output_file = request.args.get('output_file')
    if not output_file:
        flash('No output file specified.', 'error')
        return redirect(url_for('dashboard'))
    
    output_path = os.path.join(OUTPUT_FOLDER, output_file)
    
    # Get processing details from database
    conn = get_db_connection()
    record = conn.execute(
        'SELECT * FROM processing_history WHERE user_id = ? AND output_path LIKE ?',
        (session['user_id'], f'%{output_file}')
    ).fetchone()
    conn.close()
    
    if not record or not os.path.exists(output_path):
        flash('Results not found.', 'error')
        return redirect(url_for('dashboard'))
    
    # Parse stats
    stats = json.loads(record['stats']) if record['stats'] else {}
    
    return render_template('results.html', 
                         username=session.get('username'),
                         output_file=output_file,
                         stats=stats,
                         record=record)

@app.route('/download/<filename>')
def download_file(filename):
    """Download processed video file"""
    if 'user_id' not in session:
        flash('Please sign in to download files.', 'error')
        return redirect(url_for('signin'))
    
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    
    # Verify file belongs to user
    conn = get_db_connection()
    record = conn.execute(
        'SELECT * FROM processing_history WHERE user_id = ? AND output_path LIKE ?',
        (session['user_id'], f'%{filename}')
    ).fetchone()
    conn.close()
    
    if record and os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        flash('File not found or access denied.', 'error')
        return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('signin'))

if __name__ == '__main__':
    # Initialize database on first run
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)

