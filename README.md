# Automatic Car Number Plate Recognition (ACNPR) System - Enhanced Version

An **advanced, high-accuracy** image processing system that uses computer vision and OCR (Optical Character Recognition) to automatically detect and recognize vehicle license plates from images with improved accuracy and robustness.

## 🚀 Latest Improvements (v2.0)

### Enhanced Detection Accuracy
- **Multi-method Detection**: Combines Haar cascades, edge detection, and MSER for better plate localization
- **Advanced Preprocessing**: 6 different image enhancement techniques including CLAHE and denoising
- **Smart OCR**: 8 different Tesseract configurations with optimal parameters
- **Multiple Attempts**: Tries enhanced image, original image, and different regions automatically
- **Confidence Scoring**: Advanced scoring system considering method, preprocessing, and text patterns

### Improved Text Recognition
- **Optimal Sizing**: Automatically resizes images to optimal OCR dimensions (32-48px height)
- **Pattern Validation**: Enhanced validation for various international license plate formats
- **OCR Error Correction**: Smart correction of common OCR errors (O/0, I/1, S/5, etc.)
- **Context-aware Processing**: Adapts correction based on expected character distribution

### Better Robustness
- **Graceful Degradation**: Falls back to alternative methods when primary detection fails
- **Regional Processing**: Tries center and bottom regions when full detection fails
- **Error Handling**: Comprehensive error handling with detailed logging
- **Performance Optimization**: Intelligent processing with time/accuracy balance

## Features

- **Intelligent Plate Detection**: Uses multiple detection methods with confidence scoring
- **Advanced Image Preprocessing**: 6 different enhancement techniques for optimal OCR
- **Robust OCR**: Multiple OCR configurations for maximum text extraction accuracy
- **Smart Pattern Validation**: Validates against common license plate formats worldwide
- **Confidence Scoring**: Provides detailed confidence levels (high, medium, low)
- **Multi-attempt Processing**: Tries multiple approaches automatically for best results
- **Web Interface**: User-friendly web interface with detailed processing feedback
- **Real-time Processing**: Fast processing with intelligent method selection

## System Architecture

The system consists of several key components:

1. **Image Preprocessing Module**: Enhances image quality for better OCR results
2. **License Plate Detection**: Locates license plate regions in vehicle images
3. **OCR Engine**: Extracts text from detected license plates using Tesseract
4. **Text Validation**: Validates and cleans extracted text using pattern matching
5. **Web Interface**: Flask-based web application for user interaction

## Installation

### Prerequisites

1. **Python 3.7+** installed on your system
2. **Tesseract OCR** installed:
   - **Windows**: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)
   - **Linux**: `sudo apt-get install tesseract-ocr`
   - **macOS**: `brew install tesseract`

### Setup

1. **Clone or download the project**:
   ```bash
   cd path/to/your/project
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Tesseract installation**:
   ```bash
   tesseract --version
   ```

## Usage

### Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Upload an image** containing a vehicle with visible license plate

4. **View results** including:
   - Detected license plate number
   - Confidence level
   - Processing steps information

### Supported Image Formats

- PNG, JPG, JPEG, GIF, BMP, TIFF
- Maximum file size: 16MB
- Recommended: Clear images with visible license plates

## Technical Details

### Image Processing Pipeline

1. **Image Resizing**: Large images are resized for optimal processing speed
2. **Grayscale Conversion**: Converts color images to grayscale
3. **Noise Reduction**: Applies Gaussian blur to reduce image noise
4. **Adaptive Thresholding**: Creates binary images for better text detection
5. **Morphological Operations**: Cleans up the binary image

### License Plate Detection Methods

1. **Haar Cascade Classification**: Uses pre-trained cascades when available
2. **Edge Detection**: Canny edge detection followed by contour analysis
3. **Aspect Ratio Filtering**: Filters contours based on typical license plate dimensions

### OCR Configuration

- Uses Tesseract OCR with multiple PSM (Page Segmentation Mode) configurations
- Character whitelist for alphanumeric characters only
- Multiple attempts with different configurations for better accuracy

### Pattern Validation

Supports various license plate formats:
- XX00XX0000 (2 letters, 2 numbers, 2 letters, 4 numbers)
- XXX000 (3 letters, 3 numbers)
- XXX0000 (3 letters, 4 numbers)
- 000XXX (3 numbers, 3 letters)
- And many more regional variations

## Applications

This ACNPR system can be used in various scenarios:

- **Traffic Management**: Monitor and track vehicles in traffic systems
- **Toll Collection**: Automated toll collection based on license plate recognition
- **Parking Automation**: Automatic entry/exit control in parking facilities
- **Law Enforcement**: Vehicle tracking and monitoring for security purposes
- **Access Control**: Restricted area access based on vehicle identification
- **Fleet Management**: Track company vehicles and logistics

## Performance Optimization

- **Image Resizing**: Automatically resizes large images for faster processing
- **Multi-method Detection**: Uses multiple detection algorithms for better accuracy
- **Efficient OCR**: Optimized OCR configurations for license plate text
- **Error Handling**: Robust error handling for various image types and qualities

## Troubleshooting

### Common Issues

1. **Tesseract not found error**:
   - Ensure Tesseract OCR is properly installed
   - Add Tesseract to your system PATH

2. **Poor recognition accuracy**:
   - Use high-quality, well-lit images
   - Ensure license plate is clearly visible and not obscured
   - Try images with minimal skew or rotation

3. **No license plate detected**:
   - Check if the license plate is clearly visible in the image
   - Try images with better contrast between plate and background
   - Ensure the license plate takes up a reasonable portion of the image

### Improving Accuracy

- Use images with good lighting conditions
- Ensure license plates are not tilted or skewed
- Clean license plates work better than dirty or damaged ones
- Higher resolution images generally provide better results

## Future Enhancements

- Support for multiple license plates in a single image
- Real-time video processing capabilities
- Integration with databases for vehicle tracking
- Support for more international license plate formats
- Machine learning models for improved detection accuracy

## Dependencies

- **Flask**: Web framework for the user interface
- **OpenCV**: Computer vision library for image processing
- **Pytesseract**: Python wrapper for Tesseract OCR
- **NumPy**: Numerical computing library
- **Pillow**: Python Imaging Library for image handling

## License

This project is for educational and research purposes. Please ensure compliance with local laws and regulations when using for commercial applications involving vehicle monitoring or data collection.
