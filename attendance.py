import cv2
import face_recognition
import numpy as np
import pickle
from datetime import datetime
import pandas as pd
import os
from tabulate import tabulate

# 1. 📂 CSV Database එක Load කරමු
def get_student_db():
    try:
        # CSV එක කියවලා 'Name' එක key එක විදිහට dictionary එකක් හදනවා
        df = pd.read_csv('students.csv')
        # නමේ තියෙන හිස්තැන් අයින් කරලා (Strip) Capital කරලා ගන්නවා
        return df.set_index(df['Name'].str.upper().str.strip()).to_dict('index')
    except Exception as e:
        print(f"❌ Database Error: {e}")
        return {}

student_db = get_student_db()

# 2. Model එක Load කරමු
if not os.path.exists("trained_face_model.pkl"):
    print("❌ Model file එක නැහැ! කලින් 'train_model.py' run කරලා ඉන්න.")
    exit()

with open("trained_face_model.pkl", "rb") as f:
    data = pickle.loads(f.read())
knownEncodings = data["encodings"]
knownNames = data["names"]

# 3. ප්‍රධාන ලොජික් එක
cap = cv2.VideoCapture(0)
approved = False

print("\n" + "="*50)
print("🚀 ATTENDANCE SYSTEM STARTED (SMART MATCH ACTIVE)")
print("="*50 + "\n")

while True:
    success, img = cap.read()
    if not success: break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        faceDis = face_recognition.face_distance(knownEncodings, encodeFace)
        matchIndex = np.argmin(faceDis)
        
        if faceDis[matchIndex] < 0.45:
            # AI එක අඳුරගත්ත folder නම (උදා: 'E2340067 WKD WIJEWICRAMA')
            folder_full_name = knownNames[matchIndex].upper().strip() 
            now = datetime.now().strftime('%Y-%m-%d | %H:%M:%S')

            # --- Smart Matching Logic ---
            found_match = False
            for student_name_in_csv in student_db:
                # CSV එකේ නම Folder එකේ නම ඇතුළේ තියෙනවද කියලා බලනවා
                if student_name_in_csv in folder_full_name:
                    details = student_db[student_name_in_csv]
                    
                    # Terminal Table Output
                    display_data = [
                        ["NIC", details['NIC']],
                        ["Name", details['Name']],
                        ["Grade", details['Grade']],
                        ["Address", details['Address']],
                        ["Time", now],
                        ["Status", "✅ APPROVED"]
                    ]
                    
                    print("\n" + tabulate(display_data, tablefmt="fancy_grid"))
                    
                    # Attendance Log එකට ලියමු
                    with open('attendance_log.csv', 'a') as f:
                        f.writelines(f"\n{details['NIC']},{details['Name']},{now}")

                    # Webcam Screen එකේ feedback එක
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, details['Name'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    cv2.imshow('Attendance System', img)
                    cv2.waitKey(2000)
                    approved = True
                    found_match = True
                    break
            
            if not found_match:
                print(f"⚠️ Warning: Folder '{folder_full_name}' matching record එකක් CSV එකේ නැහැ!")

    if approved: break
    cv2.imshow('Attendance System', img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()