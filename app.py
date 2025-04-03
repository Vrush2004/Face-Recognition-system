from flask import Flask, render_template, request, Response, jsonify, redirect, url_for, flash, session, send_file, send_from_directory
from werkzeug.utils import secure_filename
from flask import session
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import threading
import mysql.connector
from datetime import datetime
from response import ChatbotResponse
from face_recognition_module import load_existing_model, train
from tensorflow.keras.preprocessing import image  
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import time
import smtplib
import secrets
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import csv

chatbot_res = ChatbotResponse()

video_cap = cv2.VideoCapture(0)
last_recognition_status = "not_recognized"

app = Flask(__name__)
app.secret_key = 'vrush_chaube_734'
app.debug = True
UPLOAD_FOLDER = "./data"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="vrush@123",
        database="face_recg_attendance",
        unix_socket="MySQL" 
    )

@app.route('/')
def index():
    if 'isLoggedIn' in session and session['isLoggedIn']:
        return render_template('index.html') 
    else:
        return redirect(url_for('login'))  

######################## developer page
@app.route("/developer")
def developer():
    return render_template('developer.html')

######################## about page
@app.route("/about")
def about():
    return render_template('about.html')

######################## dataset page
@app.route("/dataset")
def dataset():
    return render_template('dataset.html')

###################### Chatbot page
@app.route("/chatbot")
def chatbot():
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()  
        user_input = data.get("msg") 

        if not user_input:
            return jsonify({"error": "No input provided"}), 400  

        response = chatbot_res.get_response(user_input)
        return jsonify({"response": response})  

    except Exception as e:
        print("Error:", e)  
        return jsonify({"error": "An error occurred on the server"}), 500

# ################ Train_data page
@app.route("/train_data")
def train_data():
    return render_template('train_data.html')

@app.route("/train", methods=['POST'])
def train_model():
    return train()

load_existing_model()

######################### dataset page
@app.route("/upload", methods=["POST"])
def upload():
    if "dataset_folder" not in request.files:
        return jsonify({"message": "No file part"}), 400
    
    files = request.files.getlist("dataset_folder")
    for file in files:
        folder_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(folder_path)

    return jsonify({"message": "Folder uploaded successfully!"})

@app.route("/get-datasets")
def get_datasets():
    folders = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isdir(os.path.join(UPLOAD_FOLDER, f))]
    dataset_info = {}

    for folder in folders:
        folder_path = os.path.join(UPLOAD_FOLDER, folder)
        images = [img for img in os.listdir(folder_path) if img.endswith((".png", ".jpg", ".jpeg"))][:10] 
        dataset_info[folder] = images  

    return jsonify({"datasets": dataset_info})

@app.route("/dataset-images/<folder>/<filename>")
def dataset_images(folder, filename):
    return send_from_directory(os.path.join(UPLOAD_FOLDER, folder), filename)

@app.route("/get-images", methods=["GET"])
def get_images():
    folder_name = request.args.get("folder")
    folder_path = os.path.join(UPLOAD_FOLDER, folder_name)
    
    if not os.path.exists(folder_path):
        return jsonify({"images": []})

    images = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".png", ".jpeg"))]
    return jsonify({"images": images})

############################### Students page
@app.route("/students")
def students():
    return render_template('students.html')

import csv
import os

STUDENT_CSV_FILE = os.path.join(os.getcwd(), 'students_data.csv')  
ATTENDANCE_FILE = os.path.join(os.getcwd(), 'attendance_data.csv') 

def save_all_students_to_csv():
    """Fetch all student data from the database and write to students_data.csv (DOES NOT MODIFY attendance.csv)."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT std_id, roll, std_name, email, phone, address, gender, dob, dep, course, year, semester, photo FROM student")
        students = cursor.fetchall()

        with open(STUDENT_CSV_FILE, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["StudentID", "RollNo", "Name", "Email", "Phone", "Address", "Gender", "DOB", "Department", "Course", "Year", "Semester", "PhotoSamples"])
            writer.writerows(students)  

        print(f"Student data successfully saved to {STUDENT_CSV_FILE}")

    except Exception as e:
        print(f"Error saving student data to CSV: {str(e)}")

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@app.route('/add_data', methods=['POST'])
def add_data():
    """Add a student record to the database and update students_data.csv (DOES NOT AFFECT attendance.csv)."""
    try:
        data = request.get_json()
        std_id = data.get('std_id')
        roll = data.get('roll')
        std_name = data.get('std_name')
        email = data.get('email')
        phone = data.get('phone')
        address = data.get('address')
        gender = data.get('gender')
        dob = data.get('dob')
        dep = data.get('dep')
        course = data.get('course')
        year = data.get('year')
        semester = data.get('semester')
        photo = data.get('photo', 'no')

        if not std_id or not roll:
            return jsonify({"error": "Student ID and Roll are required!"}), 400

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO student (std_id, roll, std_name, email, phone, address, gender, dob, dep, course, year, semester, photo) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (std_id, roll, std_name, email, phone, address, gender, dob, dep, course, year, semester, photo)
        )
        conn.commit()

        save_all_students_to_csv()

        return jsonify({
            'success': True,
            'message': f"Student {std_name} added successfully!",
            'student': {
                'std_id': std_id,
                'roll': roll,
                'std_name': std_name,
                'email': email,
                'phone': phone,
                'address': address,
                'gender': gender,
                'dob': dob,
                'semester': semester,
                'dep': dep,
                'course': course,
                'year': year,
                'photo': photo,
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

    finally:
        if conn:
            conn.close()

@app.route('/get_students', methods=['GET'])
def get_students():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM student")  
        students = cursor.fetchall()
        conn.close()

        student_data = [
            {
                "std_id": student[0],        
                "roll": student[1],
                "std_name": student[2],
                "email": student[3],
                "phone": student[4],
                "address": student[5],
                "gender": student[6],
                "dob": student[7],
                "dep": student[8],
                "course": student[9],
                "year": student[10],
                "semester": student[11],
                "photo": student[12],
            }
            for student in students
        ]
        return jsonify({"success": True, "data": student_data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
save_all_students_to_csv()
        
@app.route('/generate_dataset', methods=['POST'])
def generate_dataset():
    if request.method == 'POST':
        std_id = request.form['std_id']
        if std_id == "":
            flash("Student ID is required!", "error")
            return redirect(url_for('index'))

        face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        cap = cv2.VideoCapture(0)
        img_id = 0
        try:
            while True:
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
                    for (x, y, w, h) in faces:
                        img_id += 1
                        face = cv2.resize(frame[y:y+h, x:x+w], (450, 450))
                        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                        file_name_path = f"static/images/user.{std_id}.{img_id}.jpg"
                        cv2.imwrite(file_name_path, face_gray)
                        cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
                        cv2.imshow("Cropped Face", face)

                    if img_id == 50:
                        break

            cap.release()
            cv2.destroyAllWindows()

            flash("Generating datasets completed successfully", "success")
            return redirect(url_for('index'))
        except Exception as e:
            flash(f"Error: {str(e)}", "error")
            cap.release()
            cv2.destroyAllWindows()
            return redirect(url_for('index'))
        

@app.route('/fetch_data')
def fetch_data():
    conn = get_db_connection()
    my_cursor = conn.cursor()
    my_cursor.execute("SELECT * FROM student")
    data = my_cursor.fetchall()
    conn.close()

    return render_template('index.html', students=data)


@app.route('/search_data', methods=['GET', 'POST'])
def search_data():
    search_by = request.args.get('searchBy')  
    search_value = request.args.get('searchTerm')

    try:
        conn = get_db_connection()
        my_cursor = conn.cursor()

        if not search_by and not search_value:
            my_cursor.execute("SELECT * FROM student")
            data = my_cursor.fetchall()
            conn.close()
            return jsonify({"success": True, "data": data})

        if search_by == "Select":
            return jsonify({"error": "Please select a search criterion."})

        if not search_value:
            return jsonify({"error": "Please enter a value to search."})

        if search_by == "RollNo":
            my_cursor.execute("SELECT * FROM student WHERE Roll=%s", (search_value,))
        elif search_by == "Phone No":
            my_cursor.execute("SELECT * FROM student WHERE Phone=%s", (search_value,))
        else:
            return jsonify({"error": "Invalid search criterion."})

        data = my_cursor.fetchall()
        conn.close()

        if len(data) != 0:
            return jsonify({"success": True, "data": data})
        else:
            return jsonify({"error": "No records found."})

    except Exception as es:
        return jsonify({"error": f"Due to: {str(es)}"})

@app.route('/get_cursor', methods=['POST'])
def get_cursor():
    try:
        data = request.get_json()
        selected_row_id = data.get('selected_row_id')

        if not selected_row_id:
            return jsonify({"error": "No row selected."})

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM student WHERE std_id = %s", (selected_row_id,))
        row_data = cursor.fetchone()
        conn.close()

        if row_data:
            return jsonify({"success": True, "data": row_data}), 200
        else:
            return jsonify({"error": "No data found for the selected row."}), 404

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    
    
@app.route('/update_data', methods=['POST'])
def update_data():
    try:
        data = request.get_json()

        std_id = data.get('std_id')
        roll = data.get('roll')
        std_name = data.get('std_name')
        email = data.get('email')
        phone = data.get('phone')
        address = data.get('address')
        gender = data.get('gender')
        dob = data.get('dob')
        dep = data.get('dep')
        course = data.get('course')
        year = data.get('year')
        semester = data.get('semester')
        photo = data.get('photo', 'no')

        if dep == "Select Department" or not std_name or not std_id or not email or not roll:
            return jsonify({"error": "All Fields are required."})

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(""" 
            UPDATE student 
            SET roll=%s, std_name=%s, email=%s, phone=%s, address=%s, gender=%s, dob=%s, dep=%s, course=%s, year=%s, semester=%s, photo=%s
            WHERE std_id=%s
        """, (roll, std_name, email, phone, address, gender, dob, dep, course, year, semester, photo, std_id))

        conn.commit()

        if cursor.rowcount == 0:
            return jsonify({"error": "No student found with the provided ID or no changes were made."})

        cursor.execute("SELECT * FROM student WHERE std_id=%s", (std_id,))
        updated_student = cursor.fetchone()

        conn.close()

        if updated_student:
            updated_student_data = {
                "std_id": updated_student[0],
                "roll": updated_student[1],
                "std_name": updated_student[2],
                "email": updated_student[3],
                "phone": updated_student[4],
                "address": updated_student[5],
                "gender": updated_student[6],
                "dob": updated_student[7],
                "dep": updated_student[8],
                "course": updated_student[9],
                "year": updated_student[10],
                "semester": updated_student[11],
                "photo": updated_student[12]
            }

            return jsonify({"success": "Student details successfully updated", "student": updated_student_data})

        else:
            return jsonify({"error": "Updated student data not found!"})

    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"})


@app.route('/delete_data/<int:std_id>', methods=['DELETE'])
def delete_data(std_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM student WHERE std_id = %s', (std_id,))
        conn.commit()
        conn.close()

        return jsonify({"success": "Student deleted successfully"})

    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"})
    

@app.route('/reset_data', methods=['POST'])
def reset_data():
    try:
        reset_values = {
            'std_name': "",
            'std_id': "",
            'roll': "",
            'email': "",
            'phone': "",
            'address': "",
            'gender': "Male",
            'dob': "",
            'dep': "Select Department",
            'course': "Select Course",
            'year': "Select Year",
            'semester': "Select Semester",
            'radio1': ""
        }

        return jsonify(reset_values) 
    except Exception as e:
        return jsonify({"error": f"Due to: {str(e)}"})


########################### Attendance Page
@app.route("/attendance")
def attendance():
    return render_template('attendance.html')

# ######################## Register page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        contact = request.form.get('contact')
        securityQ = request.form.get('securityQ')
        securityA = request.form.get('securityA')
        password = request.form.get('password')
        conf_password = request.form.get('c_password')

        if not name or not email or securityQ == "Select":
            flash("All fields are required", "danger")
            return redirect(url_for('register'))
        
        if password != conf_password:
            flash("Password & Confirm Password must be the same", "danger")
            return redirect(url_for('register'))
        
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("SELECT * FROM register WHERE email=%s", (email,))
        user = cur.fetchone()
        if user:
            flash("User already exists!", "danger")
            return redirect(url_for('register'))

        cur.execute(
            "INSERT INTO register (Name, Email, Contact, SecurityQ, SecurityA, Password) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (name, email, contact, securityQ, securityA, password)
        )
        conn.commit()
        cur.close()
        conn.close()

        flash("Registered Successfully!", "success")
        return redirect(url_for('login'))

    return render_template('register.html')


############################# Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'isLoggedIn' in session and session['isLoggedIn']:
        return jsonify({"success": True, "redirect": url_for('index')})

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if not email or not password:
            return jsonify({"success": False, "message": "All fields are required"})

        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT password FROM register WHERE email=%s", (email,))
            row = cur.fetchone()
            cur.close()

            if row and row[0] == password: 
                session['isLoggedIn'] = True
                session['user_email'] = email
                session.modified = True
                return jsonify({"success": True, "redirect": url_for('index')})
            else:
                return jsonify({"success": False, "message": "Invalid Username & Password"})
        except Exception as e:
            print("Error in login:", str(e))  
            return jsonify({"success": False, "message": "Something went wrong"})

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('isLoggedIn', None)
    session.pop('user_email', None)
    flash("You have been logged out", "info")
    return redirect(url_for('login'))

################################# forgot password page
reset_tokens = {}

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        question = request.form.get('SecurityQ')
        answer = request.form.get('SecurityA')

        if not email or not question or not answer:
            return jsonify({"success": False, "message": "All fields are required."})

        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT * FROM register WHERE LOWER(Email)=%s AND LOWER(SecurityQ)=%s AND LOWER(SecurityA)=%s", 
            (email.lower(), question.lower(), answer.lower()))

            row = cur.fetchone()
            cur.close()
            conn.close()
        except Exception as e:
            return jsonify({"success": False, "message": f"Database error: {e}"})

        if row:
            reset_token = secrets.token_urlsafe(16)  
            reset_tokens[email] = reset_token 

            reset_link = f"http://127.0.0.1:5000/reset_password?token={reset_token}&email={email}"

            send_email(email, reset_link)

            return jsonify({"success": True, "message": "Password reset link sent to your email!"})
        else:
            return jsonify({"success": False, "message": "Invalid email or security answer."})

    return render_template('forgot_password.html')


def send_email(to_email, reset_link):
    from_email = ""  
    from_password = ''

    subject = "Password Reset Request"
    body = f"Click the following link to reset your password: {reset_link}"

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)  
        server.starttls()
        server.login(from_email, from_password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        print(f"Password reset email sent to {to_email}")
    except Exception as e:
        print(f"Error sending email: {e}")

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    email = request.args.get('email')
    token = request.args.get('token')

    if request.method == 'POST':
        new_password = request.form.get('new_password')

        if email in reset_tokens and reset_tokens[email] == token:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("UPDATE register SET password=%s WHERE email=%s", (new_password, email))
            conn.commit()
            cur.close()
            conn.close()

            del reset_tokens[email] 
            flash("Password reset successfully! You can now login.", "success")
            return redirect(url_for('login'))
        else:
            flash("Invalid or expired token.", "danger")
            return redirect(url_for('forgot_password'))

    return render_template("reset_password.html", email=email, token=token)


############################ Face Detector Page
@app.route("/face_detector")
def face_detector():
    return render_template('face_detector.html')

UPLOAD_FOLDER = 'data'
MODEL_PATH = 'face_recognition_googlenet_model.h5'
camera_active = False  

model = load_model(MODEL_PATH)

db = mysql.connector.connect(host="localhost", user="root", password="vrush@123", database="face_recg_attendance")
cursor = db.cursor()

cursor.execute("SELECT std_id FROM student")
students = [row[0] for row in cursor.fetchall()]
label_encoder = LabelEncoder()
label_encoder.fit(students)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = None  

def get_student_details(student_id):
    """Fetch student details from the database."""
    student_id = int(student_id)
    query = "SELECT roll, std_name, dep FROM student WHERE std_id = %s"
    cursor.execute(query, (student_id,))
    result = cursor.fetchone()
    return result if result else ("Unknown", "Unknown", "Unknown")

def mark_attendance(student_id):
    """Mark student attendance in the CSV file and update recognition status."""
    global last_recognition_status 

    roll, name, department = get_student_details(student_id)

    if roll != "Unknown":
        now = datetime.now()
        date = now.strftime("%d/%m/%Y")
        time = now.strftime("%H:%M:%S")

        if not os.path.exists(ATTENDANCE_FILE) or os.stat(ATTENDANCE_FILE).st_size == 0:
            df = pd.DataFrame(columns=["Student ID", "Roll", "Name", "Department", "Time", "Date", "Status"])
            df.to_csv(ATTENDANCE_FILE, index=False)

        try:
            if os.stat(ATTENDANCE_FILE).st_size > 0:
                df = pd.read_csv(ATTENDANCE_FILE)
            else:
                df = pd.DataFrame(columns=["Student ID", "Roll", "Name", "Department", "Time", "Date", "Status"])

            if not ((df["Student ID"] == student_id) & (df["Date"] == date)).any():
                with open(ATTENDANCE_FILE, "a") as f:
                    f.write(f"{student_id},{roll},{name},{department},{time},{date},Present\n")

                last_recognition_status = {
                    "status": "recognized",
                    "id": student_id
                }

        except pd.errors.EmptyDataError:
            print("Warning: Attendance file is empty. Initializing with headers.")
            df = pd.DataFrame(columns=["Student ID", "Roll", "Name", "Department", "Time", "Date", "Status"])
            df.to_csv(ATTENDANCE_FILE, index=False)
    
    else:
        last_recognition_status = {"status": "unrecognized"} 

@app.route('/video_feed')
def video_feed():
    """Handle real-time video feed with face recognition."""
    global cap, camera_active
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return Response("Error: Could not open camera.", status=500)

    def generate():
        global camera_active
        while camera_active:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face = frame[y:y + h, x:x + w]
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)

                predictions = model.predict(face)
                predicted_class = np.argmax(predictions)
                confidence = predictions[0][predicted_class]   

                if confidence > 0.90:
                    student_id = label_encoder.inverse_transform([predicted_class])[0]
                    mark_attendance(student_id)
                    
                    global last_recognition_status
                    last_recognition_status = "recognized" 

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {student_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                break
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/recognition_status")
def recognition_status():
    global last_recognition_status
    return jsonify({"status": last_recognition_status})

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Start the camera feed."""
    global cap, camera_active
    camera_active = True
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({"status": "error", "message": "Failed to start camera"}), 500
    return jsonify({"status": "success", "message": "Camera started"})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop the camera feed and release resources properly."""
    global cap, camera_active
    camera_active = False  

    if cap is not None and cap.isOpened():
        cap.release()
        cap = None
        cv2.destroyAllWindows()

    return jsonify({"status": "success", "message": "Camera stopped"})

def allowed_file(filename, allowed_extensions={'jpg', 'jpeg', 'png'}):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/save-photo', methods=['POST'])
def save_photo():
    try:
        if 'photo' not in request.files:
            print("No photo found in request.")
            return jsonify({"error": "No photo part"}), 400
        
        photo = request.files['photo']
        
        if photo.filename == '':
            print("No selected file.")
            return jsonify({"error": "No selected file"}), 400

        photoCount = int(request.form.get('photoCount', 0))  
        
        if photo and allowed_file(photo.filename):
            student_id = request.form['std_id']
            student_folder = os.path.join(UPLOAD_FOLDER, f"Student-{student_id}")
            os.makedirs(student_folder, exist_ok=True)  
            
            filename = secure_filename(f"{student_id}-{photoCount + 1}.jpg")
            filepath = os.path.join(student_folder, filename)

            photo.save(filepath)

            return jsonify({"message": f"Photo {filename} saved successfully!"}), 200
        else:
            print("Invalid file format.")
            return jsonify({"error": "Invalid file format"}), 400

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/delete-photos/<int:std_id>', methods=['DELETE'])
def delete_photos(std_id):
    student_folder = os.path.join(UPLOAD_FOLDER, std_id)

    if os.path.exists(student_folder):
        for file in os.listdir(student_folder):
            file_path = os.path.join(student_folder, file)
            try:
                os.remove(file_path)
            except Exception as e:
                return jsonify({"success": False, "error": str(e)})

        try:
            os.rmdir(student_folder)
        except OSError as e:
            pass

        return jsonify({"success": True, "message": "Photos deleted successfully."})
    else:
        return jsonify({"success": False, "error": "Student folder does not exist."})


# ####################### Attendance page

attendance_data = []

@app.route("/dashboard")
def dashboard():
    return render_template('dashboard.html')

CSV_FILE = "attendance_data.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Attendance ID", "Roll", "Name", "Department", "Time", "Date", "Attendance"])


# Import CSV Route
@app.route('/import_csv', methods=['POST'])
def import_csv():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'Error', 'message': 'No file part'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'status': 'Error', 'message': 'No selected file'})

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            df = pd.read_csv(file_path)
            for index, row in df.iterrows():
                new_attendance = {
                    'attendanceId': str(row['Attendance ID']),
                    'rollNo': str(row['Roll']),
                    'name': row['Name'],
                    'date': row['Date'],
                    'department': row['Department'],
                    'time': row['Time'],
                    'attendanceStatus': row['Attendance']
                }
                attendance_data.append(new_attendance)
            
            return jsonify({'status': 'CSV imported successfully', 'attendance': attendance_data})

        return jsonify({'status': 'Error', 'message': 'Invalid file format. Only CSV is allowed.'})

    except Exception as e:
        return jsonify({'status': 'Error', 'message': str(e)})

@app.route('/get_attendance')
def get_attendance():
    try:
        df = pd.read_csv('attendance_data.csv')
        data = df.iloc[1:].to_dict(orient='records')  
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_attendance_data')
def fetch_attendance_csv(): 
    csv_path = os.path.join(os.getcwd(), "attendance_data.csv")  
    if os.path.exists(csv_path):
        return send_file(csv_path, mimetype='text/csv')
    else:
        return Response("Attendance file not found", status=404)

@app.route('/export_csv')
def export_csv():
    return send_file('attendance_data.csv', as_attachment=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)