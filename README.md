# Face Attendance System - SAP (Student Attendance Portal)

A sophisticated facial recognition-based attendance management system with a modern monochrome UI, real-time dashboard, and voice feedback capabilities.

## 🚀 Features

### Core Functionality
- **Facial Recognition Attendance**: Automatically identifies students/staff using facial recognition
- **Real-time Dashboard**: Live attendance statistics with hourly breakdown charts
- **Voice Feedback**: Text-to-speech announcements for attendance confirmation
- **Smart Duplicate Detection**: Prevents multiple attendance marks for the same person on the same day
- **Image Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization) for better recognition in varying lighting conditions
- **Student Management**: Admin panel for managing student records with real-time search
- **Attendance Logs**: CSV-based logging with timestamps
- **Photo Management**: Organized student photo storage with folder-based structure

### Advanced Features
- **Animated UI Elements**: Smooth transitions, fade effects, and pulse animations
- **Toast Notifications**: Non-intrusive feedback for system events
- **Status Bar**: Visual progress indicators for system operations
- **Student Detail Panel**: Comprehensive student information display
- **Recent Activity Sidebar**: Quick access to latest attendance records
- **Dark Mode UI**: Professional monochrome design with high contrast
- **Hash-based Model Training**: Smart incremental training that only processes new/modified images

## 📋 System Requirements

### Operating System
- Windows 10/11 (tested and optimized)
- Linux (may require additional configuration)
- macOS (may require additional configuration)

### Hardware Requirements
- **Webcam**: Required for face recognition
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: 500MB free space for dependencies and model files
- **Display**: Minimum 1280x720 resolution

### Software Requirements
- **Python**: 3.8 or higher (3.9 recommended)
- **C++ Redistributable**: Visual C++ Redistributable for Visual Studio 2015-2019
- **CMake**: Required for dlib compilation (Windows)

## 🔧 Installation

### Step 1: Install Python
Download and install Python 3.9 or higher from [python.org](https://www.python.org/downloads/)

**Important**: During installation, check "Add Python to PATH"

### Step 2: Install C++ Redistributable (Windows Only)
Download and install from [Microsoft](https://aka.ms/vs/17/release/vc_redist.x64.exe)

### Step 3: Install CMake (Windows Only)
Download from [cmake.org](https://cmake.org/download/) or use pip:
```bash
pip install cmake
```

### Step 4: Install Project Dependencies

Open Command Prompt or Terminal in the project directory and run:

```bash
pip install opencv-python
```

```bash
pip install face-recognition
```

```bash
pip install numpy
```

```bash
pip install pandas
```

```bash
pip install pillow
```

```bash
pip install matplotlib
```

```bash
pip install tkcalendar
```

```bash
pip install pyttsx3
```

```bash
pip install tabulate
```

### Alternative: Install All Dependencies at Once

```bash
pip install opencv-python face-recognition numpy pandas pillow matplotlib tkcalendar pyttsx3 tabulate
```

## 📁 Project Structure

```
Face identify/
├── main_app.py              # Main GUI application (Tkinter-based)
├── train_model.py           # Face encoding training script
├── attendance.py            # Console-based attendance system
├── students.csv             # Student database
├── attendance_log.csv       # Generated attendance records
├── trained_face_model.pkl   # Generated face encoding model
├── Face_Attendance_System.spec  # PyInstaller specification
├── images/                  # Student photos directory
│   ├── E2340065 AIN Hatharasinghe/
│   ├── E2340066 Sathira Heesara/
│   ├── E2340067 WKD Wijewicrama/
│   └── E2340068 Hansa Ranawaka/
├── alerts/                  # Alert/snapshot storage
│   └── unknown_people/      # Unknown face captures
└── build/                   # PyInstaller build directory
```

## 🎯 Usage Guide

### 1. Prepare Student Database

Create a `students.csv` file with the following structure:

```csv
NIC,Name,Grade,Address,Personal_Number,Home_Number
E2340065,AIN Hatharasinghe,12-A,Galle,712345678,94703052181
E2340066,Sathira Heesara,12-B,Colombo,778889990,94771234567
```

### 2. Add Student Photos

Organize student photos in the `images/` directory:
```
images/
├── E2340065 AIN Hatharasinghe/
│   ├── photo1.jpg
│   └── photo2.jpg
├── E2340066 Sathira Heesara/
│   └── photo1.jpg
```

**Photo Requirements**:
- Format: JPG, JPEG, or PNG
- Minimum resolution: 640x480 pixels
- Recommended: Multiple angles per student (3-5 photos)
- Clear, well-lit frontal face photos work best

### 3. Train the Face Recognition Model

Run the training script to create face encodings:

```bash
python train_model.py
```

**Output**:
- Creates `trained_face_model.pkl` with face encodings
- Shows training progress and statistics
- Supports incremental training (only processes new/modified images)

### 4. Launch the Main Application

```bash
python main_app.py
```

**Main Application Features**:
- **Mark Attendance**: Click "Mark Attendance" button to scan faces
- **Admin Panel**: Manage student records, add/edit/delete students
- **Dashboard**: View real-time attendance statistics
- **Recent Activity**: Click on recent entries for detailed view

### 5. Console-based Attendance (Alternative)

For command-line usage:

```bash
python attendance.py
```

## 🎨 User Interface

### Main Dashboard
- **Dark Monochrome Theme**: Professional black and white design
- **Real-time Charts**: Hourly attendance visualization
- **Activity Feed**: Latest attendance records with photos
- **Student Details**: Comprehensive information panel

### Admin Panel
- **Search Functionality**: Real-time filtering by name or NIC
- **Data Grid**: Sortable, scrollable student list
- **CRUD Operations**: Add, edit, and delete student records
- **Bulk Management**: Handle multiple students efficiently

### Attendance Scanner
- **Live Camera Feed**: Real-time face detection
- **Visual Feedback**: Bounding boxes and status indicators
- **Voice Announcements**: Audio confirmation of attendance
- **Duplicate Prevention**: Smart detection of already-marked entries

## 🔒 Security & Privacy

- **Local Processing**: All face recognition happens locally (no cloud uploads)
- **Data Storage**: CSV files stored locally on your machine
- **Photo Management**: Images stored in organized folder structure
- **Access Control**: Admin panel for authorized personnel only

## 🛠️ Troubleshooting

### Common Issues

#### 1. "No module named 'face_recognition'"
```bash
pip install face-recognition
```

#### 2. CMake Error during Installation
```bash
pip install cmake
```
Then reinstall face-recognition:
```bash
pip install --no-cache-dir face-recognition
```

#### 3. "Microsoft Visual C++ 14.0 or greater is required"
Download and install: [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

#### 4. Webcam Not Working
- Check webcam permissions in Windows Settings
- Ensure no other application is using the webcam
- Try restarting the application

#### 5. Poor Recognition Accuracy
- Add more photos of each student (different angles)
- Ensure good lighting conditions
- Retrain the model after adding new photos
- Adjust face detection distance threshold in code (default: 0.45)

#### 6. Application Crashes on Startup
- Verify all dependencies are installed
- Check Python version (must be 3.8+)
- Run as Administrator if permission issues occur

### Performance Optimization

#### For Slow Performance:
1. Reduce camera resolution in code (currently 0.25 scale)
2. Process faces on every other frame
3. Use smaller images for training
4. Limit number of known faces in database

#### For Better Accuracy:
1. Use high-quality, well-lit photos
2. Include multiple angles per person
3. Ensure photos are recent and match current appearance
4. Retrain model regularly with new photos

## 📊 Data Management

### Backup Your Data
Regularly backup these critical files:
- `students.csv` - Student database
- `attendance_log.csv` - Attendance records
- `trained_face_model.pkl` - Trained model
- `images/` - Student photos directory

### Export Attendance Data
Attendance data is stored in `attendance_log.csv` with format:
```
NIC,Name,DateTime
E2340065,AIN Hatharasinghe,2024-01-15 | 08:30:45
```

## 🚀 Building Executable (.exe)

To create a standalone Windows executable:

### 1. Install PyInstaller
```bash
pip install pyinstaller
```

### 2. Build Executable
```bash
pyinstaller --noconfirm --onefile --windowed --name "Face_Attendance_System" --add-data "students.csv;." --add-data "trained_face_model.pkl;." --add-data "images;images" main_app.py
```

Or use the provided spec file:
```bash
pyinstaller Face_Attendance_System.spec
```

### 3. Distribution
The executable will be in the `dist/` folder. Distribute along with:
- `students.csv`
- `trained_face_model.pkl`
- `images/` folder

## 📝 Configuration

### Adjusting Recognition Sensitivity
Edit `main_app.py` line 693:
```python
if dis[idx] < 0.45:  # Lower = stricter matching (0.4-0.6 recommended)
```

### Customizing UI Theme
Edit color constants in `main_app.py` (lines 66-97):
```python
BG = "#080808"        # Background color
ACCENT = "#FFFFFF"    # Accent/highlight color
TEXT = "#F0F0F0"      # Text color
```

### Camera Settings
Adjust camera resolution in `main_app.py` line 684:
```python
imgS = cv2.cvtColor(cv2.resize(img, (0, 0), None, 0.25, 0.25), cv2.COLOR_BGR2RGB)
# Change 0.25 to adjust scale (lower = faster, higher = more accurate)
```

## 🤝 Contributing

This is a complete attendance management system. If you'd like to contribute:

1. **Bug Reports**: Document the issue with steps to reproduce
2. **Feature Requests**: Describe the feature and its use case
3. **Code Improvements**: Submit clean, documented code with tests
4. **Documentation**: Help improve this README or add tutorials

## 📄 License

This project is provided as-is for educational and commercial use.

## 🙏 Acknowledgments

### Technologies Used:
- **OpenCV**: Computer vision and image processing
- **face_recognition**: State-of-the-art facial recognition library
- **Tkinter**: Python's standard GUI framework
- **Pandas**: Data manipulation and CSV handling
- **Matplotlib**: Chart generation for dashboard
- **Pillow**: Image processing and manipulation
- **pyttsx3**: Text-to-speech for voice feedback
- **tkcalendar**: Date picker widget

### Libraries & Dependencies:
```
opencv-python>=4.5.0
face-recognition>=1.3.0
numpy>=1.21.0
pandas>=1.3.0
Pillow>=8.0.0
matplotlib>=3.4.0
tkcalendar>=1.6.1
pyttsx3>=2.90
tabulate>=0.8.9
```

## 📞 Support

### Getting Help:
1. Check this README's troubleshooting section
2. Review the code comments (includes Sinhala documentation)
3. Test with sample data first
4. Ensure all dependencies are properly installed

### Known Limitations:
- Requires good lighting for accurate recognition
- Works best with frontal face photos
- Performance depends on hardware capabilities
- Webcam quality affects recognition accuracy

## 🔄 Updates & Changelog

### Current Version: 1.0.0
- ✅ Complete GUI with modern dark theme
- ✅ Real-time attendance marking with voice feedback
- ✅ Student management admin panel
- ✅ Dashboard with live statistics
- ✅ Hash-based incremental model training
- ✅ Duplicate attendance prevention
- ✅ Image enhancement with CLAHE
- ✅ Comprehensive logging system

### Planned Features:
- [ ] Database integration (SQLite/MySQL)
- [ ] Report generation (PDF/Excel)
- [ ] Multi-camera support
- [ ] Mobile app integration
- [ ] Cloud backup option
- [ ] Advanced analytics dashboard

---

**Developed with ❤️ for educational institutions**

*For any questions or support, please refer to the inline code documentation or raise an issue in the project repository.*#   F a c e - a t t e n d a n c e - s y s t e m  
 