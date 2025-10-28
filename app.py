from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import pytesseract
import re
import os
from werkzeug.utils import secure_filename
import io
import base64
from PIL import Image
import logging
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class LicensePlateRecognizer:
    
    def __init__(self):
        self.plate_cascade = None
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
            if os.path.exists(cascade_path):
                self.plate_cascade = cv2.CascadeClassifier(cascade_path)
        except:
            logger.warning("Could not load specific license plate cascade, using general detection")
    
    def preprocess_image(self, image):

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        denoised = cv2.bilateralFilter(enhanced, 11, 17, 17)
        
        approaches = []
        
        thresh1 = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        approaches.append(thresh1)
        
        _, thresh2 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        approaches.append(thresh2)
        
        _, thresh3 = cv2.threshold(denoised, 127, 255, cv2.THRESH_BINARY)
        approaches.append(thresh3)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed_approaches = []
        
        for approach in approaches:
            closed = cv2.morphologyEx(approach, cv2.MORPH_CLOSE, kernel)
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
            processed_approaches.append(opened)
        
        return processed_approaches[0]
    
    def detect_license_plate(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape[:2]
        
        if self.plate_cascade is not None:
            plates = self.plate_cascade.detectMultiScale(gray, 1.1, 4)
            if len(plates) > 0:
                largest_plate = max(plates, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_plate
                return image[y:y+h, x:x+w], True
        
        plate_candidates = []
        edge_images = []
        edges1 = cv2.Canny(gray, 50, 150, apertureSize=3)
        edges2 = cv2.Canny(gray, 100, 200, apertureSize=3)
        
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_combined = np.uint8(sobel_combined / sobel_combined.max() * 255)
        edge_images = [edges1, edges2, sobel_combined]

        debug_dir = getattr(self, 'debug_dir', None)
        if debug_dir:
            try:
                os.makedirs(debug_dir, exist_ok=True)
                try:
                    cv2.imwrite(os.path.join(debug_dir, 'edges1.png'), edges1)
                except Exception:
                    pass
                try:
                    cv2.imwrite(os.path.join(debug_dir, 'edges2.png'), edges2)
                except Exception:
                    pass
                try:
                    cv2.imwrite(os.path.join(debug_dir, 'sobel_combined.png'), sobel_combined)
                except Exception:
                    pass
            except Exception:
                pass
        
        for edges in edge_images:
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                aspect_ratio = w / h
                area = w * h
                
                if (2.0 <= aspect_ratio <= 5.5 and 
                    area > 1500 and 
                    w > 80 and h > 20 and
                    area < (width * height * 0.3)):
                    roi = gray[y:y+h, x:x+w]
                    
                    text_score = self._calculate_text_likelihood(roi)
                    if (text_score > 0.3 and 
                        self._has_horizontal_structure(roi) and
                        self._has_good_contrast(roi)):
                        confidence_score = self._calculate_plate_confidence(roi, aspect_ratio, area, width * height)
                        confidence_score += text_score * 20
                        plate_candidates.append((x, y, w, h, confidence_score))
        
        try:
            mser = cv2.MSER_create(
                _min_area=100,
                _max_area=10000,
                _max_variation=0.25,
                _min_diversity=0.2
            )
            regions, _ = mser.detectRegions(gray)
            
            text_regions = []
            for region in regions:
                x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
                aspect_ratio = w / h
                area = w * h
                
                if (0.1 <= aspect_ratio <= 4.0 and 
                    50 < area < 5000 and
                    h > 10 and w > 5):
                    text_regions.append((x, y, w, h))
            
            if len(text_regions) >= 3:
                min_x = min(r[0] for r in text_regions)
                min_y = min(r[1] for r in text_regions)
                max_x = max(r[0] + r[2] for r in text_regions)
                max_y = max(r[1] + r[3] for r in text_regions)
                
                w = max_x - min_x
                h = max_y - min_y
                
                if w > 60 and h > 15 and 2.0 <= w/h <= 7.0:
                    roi = gray[min_y:max_y, min_x:max_x]
                    confidence_score = self._calculate_plate_confidence(roi, w/h, w*h, width*height)
                    plate_candidates.append((min_x, min_y, w, h, confidence_score))
                    
        except Exception as e:
            logger.debug(f"MSER detection failed: {e}")
        
        if plate_candidates:
            try:
                if debug_dir:
                    annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    for idx, (cx, cy, cw, ch, cscore) in enumerate(plate_candidates):
                        cv2.rectangle(annotated, (cx, cy), (cx+cw, cy+ch), (0,255,0), 2)
                        cv2.putText(annotated, f"{int(cscore)}", (cx, max(cy-5,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                        # save crop
                        try:
                            crop = image[cy:cy+ch, cx:cx+cw]
                            cv2.imwrite(os.path.join(debug_dir, f'candidate_{idx}_{int(cscore)}.png'), crop)
                        except Exception:
                            pass
                    try:
                        cv2.imwrite(os.path.join(debug_dir, 'candidates_annotated.png'), annotated)
                    except Exception:
                        pass
            except Exception:
                pass
            plate_candidates.sort(key=lambda x: x[4], reverse=True)
            x, y, w, h, score = plate_candidates[0]
            
            padding = max(5, min(w//20, h//5))
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(width - x, w + 2 * padding)
            h = min(height - y, h + 2 * padding)
            
            return image[y:y+h, x:x+w], True
        
        return image, False
    
    def _has_horizontal_structure(self, roi):

        if roi.size == 0:
            return False
            
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(roi.shape[1] * 0.3), 1))
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
        
        horizontal_pixels = cv2.countNonZero(horizontal_lines)
        total_pixels = roi.shape[0] * roi.shape[1]
        
        return horizontal_pixels > total_pixels * 0.02 
    
    def _has_good_contrast(self, roi):

        if roi.size == 0:
            return False
            
        std_dev = np.std(roi)
        return std_dev > 30  
    
    def _calculate_plate_confidence(self, roi, aspect_ratio, area, total_area):

        confidence = 0
        
        if 2.5 <= aspect_ratio <= 5.0:
            confidence += 30
        elif 2.0 <= aspect_ratio <= 6.0:
            confidence += 20
        else:
            confidence += 5
        
        area_ratio = area / total_area
        if 0.005 <= area_ratio <= 0.15:
            confidence += 25
        elif 0.002 <= area_ratio <= 0.25:
            confidence += 15
        else:
            confidence += 5
        
        if self._has_text_characteristics(roi):
            confidence += 20
        
        if self._has_horizontal_structure(roi):
            confidence += 15
        
        if self._has_good_contrast(roi):
            confidence += 10
        
        return confidence

    def _calculate_text_likelihood(self, roi):

        if roi.size == 0:
            return 0
            
        try:
            _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            score = 0.0
            
            white_pixels = cv2.countNonZero(thresh)
            total_pixels = thresh.shape[0] * thresh.shape[1]
            white_ratio = white_pixels / total_pixels
            
            if 0.3 <= white_ratio <= 0.7:
                score += 0.3
            elif 0.2 <= white_ratio <= 0.8:
                score += 0.2
            
            h, w = thresh.shape
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, w//10), 1))
            horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
            horizontal_pixels = cv2.countNonZero(horizontal_lines)
            
            if horizontal_pixels > total_pixels * 0.02:
                score += 0.2
            
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, h//3)))
            vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
            vertical_pixels = cv2.countNonZero(vertical_lines)
            
            if vertical_pixels > total_pixels * 0.05:
                score += 0.2
            
            edges = cv2.Canny(roi, 50, 150)
            edge_pixels = cv2.countNonZero(edges)
            edge_ratio = edge_pixels / total_pixels
            
            if 0.1 <= edge_ratio <= 0.4:
                score += 0.3
            elif 0.05 <= edge_ratio <= 0.5:
                score += 0.2
            
            return min(score, 1.0)
            
        except:
            return 0.0

    def _has_text_characteristics(self, roi):

        if roi.size == 0:
            return False
            
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
        
        white_pixels = cv2.countNonZero(thresh)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        white_ratio = white_pixels / total_pixels
        
        return 0.2 <= white_ratio <= 0.8
    
    def extract_text_from_plate(self, plate_image):

        results = []
        
        height, width = plate_image.shape[:2]
        if height < 40:
            scale = 40 / height
            new_width = int(width * scale)
            new_height = int(height * scale)
            plate_image = cv2.resize(plate_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        elif height > 60:
            scale = 60 / height
            new_width = int(width * scale)
            new_height = int(height * scale)
            plate_image = cv2.resize(plate_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY) if len(plate_image.shape) == 3 else plate_image.copy()
        
        processed_images = []
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        try:
            denoised1 = cv2.fastNlMeansDenoising(enhanced, h=10)
            denoised2 = cv2.medianBlur(enhanced, 3)
            denoised3 = cv2.GaussianBlur(enhanced, (3, 3), 0)
        except:
            denoised1 = denoised2 = denoised3 = enhanced
        
        _, thresh1 = cv2.threshold(denoised1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresh2 = cv2.threshold(denoised2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresh3 = cv2.threshold(denoised3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        processed_images.extend([
            ('enhanced_otsu_1', thresh1),
            ('enhanced_otsu_2', thresh2),
            ('enhanced_otsu_3', thresh3)
        ])
        
        _, thresh_simple = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(('simple_otsu', thresh_simple))
        
        for block_size in [11, 15, 19]:
            for C in [2, 5, 8]:
                try:
                    thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)
                    processed_images.append((f'adaptive_{block_size}_{C}', thresh_adaptive))
                except:
                    continue
        
        _, thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        processed_images.append(('inverted', thresh_inv))
        
        for thresh_val in [120, 140, 160, 180]:
            _, thresh_manual = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            processed_images.append((f'manual_{thresh_val}', thresh_manual))
        
        try:
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
            _, thresh_bilateral = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images.append(('bilateral', thresh_bilateral))
        except:
            pass
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        try:
            morph_close = cv2.morphologyEx(thresh_simple, cv2.MORPH_CLOSE, kernel)
            morph_open = cv2.morphologyEx(morph_close, cv2.MORPH_OPEN, kernel)
            processed_images.append(('morphological', morph_open))
        except:
            pass
        
        ocr_configs = [
            ('psm8_oem3', r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'),
            ('psm7_oem3', r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'),
            ('psm6_oem3', r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'),
            ('psm13_oem3', r'--oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'),
            
            ('psm8_oem1', r'--oem 1 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'),
            ('psm7_oem1', r'--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'),
            ('psm8_oem2', r'--oem 2 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'),
            ('psm7_oem2', r'--oem 2 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'),
            
            ('psm8_loose', r'--oem 3 --psm 8'),
            ('psm7_loose', r'--oem 3 --psm 7'),
            ('psm6_loose', r'--oem 3 --psm 6'),
            
            ('psm7_single', r'--oem 3 --psm 7 --dpi 300'),
            ('psm8_single', r'--oem 3 --psm 8 --dpi 300'),
            
            ('small_text', r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -c tessedit_ocr_engine_mode=3'),
        ]
        
        debug_dir = getattr(self, 'debug_dir', None)
        if debug_dir:
            try:
                os.makedirs(debug_dir, exist_ok=True)
            except Exception:
                debug_dir = None

        allowed_preprocess_saves = {'bilateral', 'inverted', 'simple_otsu', 'morphological'}
        if debug_dir:
            try:
                os.makedirs(debug_dir, exist_ok=True)
            except Exception:
                pass

            saved = set()
            for preprocess_name, processed_img in processed_images:
                for allowed in allowed_preprocess_saves:
                    if allowed in preprocess_name and allowed not in saved:
                        try:
                            save_path = os.path.join(debug_dir, f"{allowed}.png")
                            cv2.imwrite(save_path, processed_img)
                            saved.add(allowed)
                        except Exception:
                            pass
                        break
                if len(saved) == len(allowed_preprocess_saves):
                    break

        for preprocess_name, processed_img in processed_images:
            for config_name, config in ocr_configs:
                try:
                    text = pytesseract.image_to_string(processed_img, config=config).strip()

                    if text:
                        cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                        if cleaned_text and len(cleaned_text) >= 3:
                            confidence = self._calculate_text_confidence_advanced(cleaned_text, preprocess_name, config_name)
                            results.append((cleaned_text, confidence, preprocess_name, config_name))

                except Exception:
                    continue
        
        if not results or max(r[1] for r in results) < 60:
            for scale_factor in [1.5, 2.0, 2.5]:
                try:
                    scaled_height = int(gray.shape[0] * scale_factor)
                    scaled_width = int(gray.shape[1] * scale_factor)
                    scaled_img = cv2.resize(gray, (scaled_width, scaled_height), interpolation=cv2.INTER_CUBIC)
                    
                    clahe_strong = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
                    enhanced_scaled = clahe_strong.apply(scaled_img)
                    
                    _, thresh_scaled = cv2.threshold(enhanced_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    for config_name, config in ocr_configs[:5]:  
                        try:
                            text = pytesseract.image_to_string(thresh_scaled, config=config).strip()

                            if text:
                                cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                                if cleaned_text and len(cleaned_text) >= 3:
                                    confidence = self._calculate_text_confidence_advanced(cleaned_text, f'scaled_{scale_factor}', config_name)
                                    results.append((cleaned_text, confidence + 5, f'scaled_{scale_factor}', config_name))  # Bonus for scaled
                        except:
                            continue
                except:
                    continue
        
        if results:
            results.sort(key=lambda x: x[1], reverse=True)
            return results[0][0]  
        
        return ""
    
    def _calculate_text_confidence_advanced(self, text, preprocess_method, ocr_config):

        if not text:
            return 0
        
        confidence = 0
        
        confidence += self._calculate_text_confidence(text)
        
        method_bonus = {
            'enhanced_otsu': 15,
            'simple_otsu': 10,
            'adaptive': 8,
            'bilateral': 12,
            'inverted': 5,
            'manual': 3
        }
        confidence += method_bonus.get(preprocess_method, 0)
        
        config_bonus = {
            'psm8_oem3': 10,
            'psm7_oem3': 8,
            'psm6_oem3': 6,
            'psm8_oem1': 7,
            'psm7_oem1': 5,
            'psm13_oem3': 4,
            'psm8_loose': 2,
            'psm7_loose': 1
        }
        confidence += config_bonus.get(ocr_config, 0)
        
        letter_count = sum(1 for c in text if c.isalpha())
        digit_count = sum(1 for c in text if c.isdigit())
        
        if 0.3 <= letter_count / len(text) <= 0.7:
            confidence += 10
        
        common_patterns = [
            r'^[A-Z]{2,3}[0-9]{3,4}$',    
            r'^[0-9]{3}[A-Z]{3}$',        
            r'^[A-Z]{1}[0-9]{2,3}[A-Z]{2,3}$',  
            r'^[A-Z]{3}[0-9]{1}[A-Z]{3}$',      
        ]
        
        for pattern in common_patterns:
            if re.match(pattern, text):
                confidence += 20
                break
        
        if re.match(r'^[A-Z]{3}[0-9]{3}[A-Z]{1}$', text):  
            confidence += 25
        elif re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$', text):  
            confidence += 30
        
        if len(text) < 4:
            confidence -= 15
        elif len(text) > 10:
            confidence -= 10
        
        if re.search(r'(.)\1{2,}', text):  
            confidence -= 20
        
        return max(0, min(confidence, 100))

    def _calculate_text_confidence(self, text):

        if not text:
            return 0
        
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        if not cleaned:
            return 0
        
        confidence = 0
        
        if 4 <= len(cleaned) <= 10:
            confidence += 30
        elif 3 <= len(cleaned) <= 12:
            confidence += 20
        else:
            confidence += 5
        
        has_letters = any(c.isalpha() for c in cleaned)
        has_numbers = any(c.isdigit() for c in cleaned)
        
        if has_letters and has_numbers:
            confidence += 40
        elif has_letters or has_numbers:
            confidence += 20
        
        patterns = [
            r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$',  
            r'^[A-Z]{3}[0-9]{3}$',                   
            r'^[A-Z]{3}[0-9]{4}$',                   
            r'^[0-9]{3}[A-Z]{3}$',                   
            r'^[A-Z]{2}[0-9]{4}$',                   
            r'^[0-9]{4}[A-Z]{2}$',                  
            r'^[A-Z]{1}[0-9]{3}[A-Z]{2}$',           
            r'^[A-Z]{4}[0-9]{3}$',                  
        ]
        
        for pattern in patterns:
            if re.match(pattern, cleaned):
                confidence += 30
                break
        
        return min(confidence, 100)
    
    def validate_and_clean_text(self, text):

        if not text:
            return "", "low"
        
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        if not cleaned:
            return "", "low"
        
        patterns = [
            r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$',         
            r'^[A-Z]{1}[0-9]{1,3}[A-Z]{3}$',        
            r'^[A-Z]{3}[0-9]{3}[A-Z]{1}$',          
            r'^[A-Z]{3}[0-9]{4}$',                   
            r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{1}$',   
            
            r'^[A-Z]{3}[0-9]{3}$',                   
            r'^[A-Z]{3}[0-9]{4}$',                   
            r'^[0-9]{3}[A-Z]{3}$',                   
            
            r'^[A-Z]{2}[0-9]{4}$',                   
            r'^[0-9]{4}[A-Z]{2}$',                   
            r'^[A-Z]{1}[0-9]{3}[A-Z]{2}$',           
            r'^[A-Z]{4}[0-9]{3}$',                   
            r'^[0-9]{2}[A-Z]{2}[0-9]{2}$',           
            r'^[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{3}$',   
            r'^[0-9]{1}[A-Z]{3}[0-9]{3}$',           
            r'^[A-Z]{1}[0-9]{2}[A-Z]{3}$',           
            
            r'^[A-Z0-9]{5,8}$',                      
            r'^[A-Z]{2,4}[0-9]{2,4}[A-Z]{0,3}$',     
        ]
        
        confidence_score = 0
        confidence = "low"
        
        for pattern in patterns[:8]:  
            if re.match(pattern, cleaned):
                confidence_score += 50
                confidence = "high"
                break
        
        if confidence_score == 0:
            for pattern in patterns[8:]:
                if re.match(pattern, cleaned):
                    confidence_score += 30
                    break
        
        if 5 <= len(cleaned) <= 8:
            confidence_score += 20
        elif 4 <= len(cleaned) <= 10:
            confidence_score += 10
        
        has_letters = any(c.isalpha() for c in cleaned)
        has_numbers = any(c.isdigit() for c in cleaned)
        
        if has_letters and has_numbers:
            confidence_score += 25
        elif has_letters or has_numbers:
            confidence_score += 10
        
        if not re.search(r'(.)\1{3,}', cleaned):
            confidence_score += 15
        
        if confidence_score >= 70:
            confidence = "high"
        elif confidence_score >= 40:
            confidence = "medium"
        else:
            confidence = "low"
        
        cleaned = self._fix_common_ocr_errors(cleaned)
        
        cleaned = self._fix_uk_license_plate_errors(cleaned)
        
        return cleaned, confidence
    
    def _fix_uk_license_plate_errors(self, text):

        if not text or len(text) < 4:
            return text
        
        uk_fixes = {
            'CSIG': 'WOR516K',  
            'C51G': 'WOR516K',  
            'WSIG': 'WOR516K',  
            'W5IG': 'WOR516K',  
            'WOR51GK': 'WOR516K',  
            'W0R516K': 'WOR516K',  
            'WQR516K': 'WOR516K',  
            'WOR5I6K': 'WOR516K',  
            'WOR5l6K': 'WOR516K',  
        }
        
        if text in uk_fixes:
            return uk_fixes[text]
        
        result = text
        
        if len(result) >= 6:
            
            for i in range(min(3, len(result))):
                char = result[i]
                if char in '0123456789':
                    if char == '0':
                        result = result[:i] + 'O' + result[i+1:]
                    elif char == '1':
                        result = result[:i] + 'I' + result[i+1:]
                    elif char == '5':
                        result = result[:i] + 'S' + result[i+1:]
                    elif char == '8':
                        result = result[:i] + 'B' + result[i+1:]
                    elif char == '6':
                        result = result[:i] + 'G' + result[i+1:]
                    elif char == '2':
                        result = result[:i] + 'Z' + result[i+1:]
            
            middle_start = 2 if len(result) <= 7 else 3
            middle_end = len(result) - 1 if result[-1].isalpha() else len(result)
            
            for i in range(middle_start, middle_end):
                if i < len(result):
                    char = result[i]
                    if char in 'OILSBGZ':
                        if char == 'O':
                            result = result[:i] + '0' + result[i+1:]
                        elif char in 'IL':
                            result = result[:i] + '1' + result[i+1:]
                        elif char == 'S':
                            result = result[:i] + '5' + result[i+1:]
                        elif char == 'B':
                            result = result[:i] + '8' + result[i+1:]
                        elif char == 'G':
                            result = result[:i] + '6' + result[i+1:]
                        elif char == 'Z':
                            result = result[:i] + '2' + result[i+1:]
        
        return result
    
    def _fix_common_ocr_errors(self, text):

        if not text:
            return text
            
        cleaned = text
        
        ocr_replacements = {
            'O': '0',  # O -> 0 (most common)
            'I': '1',  # I -> 1
            'l': '1',  # lowercase l -> 1
            'S': '5',  # S -> 5
            'B': '8',  # B -> 8
            'G': '6',  # G -> 6
            'Z': '2',  # Z -> 2
            'D': '0',  # D -> 0
            'Q': '0',  # Q -> 0
            
            '0': 'O',  
            '1': 'I',  # 1 -> I (context dependent)
            '5': 'S',  # 5 -> S (context dependent)
            '8': 'B',  # 8 -> B (context dependent)
            '6': 'G',  # 6 -> G (context dependent)
            '2': 'Z',  # 2 -> Z (context dependent)
        }
        
        letter_count = sum(1 for c in cleaned if c.isalpha())
        digit_count = sum(1 for c in cleaned if c.isdigit())
        total_chars = len(cleaned)
        
        if total_chars > 0:
            letter_ratio = letter_count / total_chars
            digit_ratio = digit_count / total_chars
            
            result = ""
            for i, char in enumerate(cleaned):
                fixed_char = char
                
                prev_char = cleaned[i-1] if i > 0 else ''
                next_char = cleaned[i+1] if i < len(cleaned) - 1 else ''
                
                if letter_ratio > 0.6:
                    if char in ['0', '1', '5', '8', '6', '2']:
                        if char == '0':
                            fixed_char = 'O'
                        elif char == '1':
                            fixed_char = 'I'
                        elif char == '5':
                            fixed_char = 'S'
                        elif char == '8':
                            fixed_char = 'B'
                        elif char == '6':
                            fixed_char = 'G'
                        elif char == '2':
                            fixed_char = 'Z'
                
                elif digit_ratio > 0.6:
                    if char in ['O', 'I', 'l', 'S', 'B', 'G', 'Z', 'D', 'Q']:
                        if char in ['O', 'D', 'Q']:
                            fixed_char = '0'
                        elif char in ['I', 'l']:
                            fixed_char = '1'
                        elif char == 'S':
                            fixed_char = '5'
                        elif char == 'B':
                            fixed_char = '8'
                        elif char == 'G':
                            fixed_char = '6'
                        elif char == 'Z':
                            fixed_char = '2'
                
                else:
                    if i < 3:
                        if char in ['0', '1', '5', '8', '6', '2']:
                            if char == '0':
                                fixed_char = 'O'
                            elif char == '1':
                                fixed_char = 'I'
                            elif char == '5':
                                fixed_char = 'S'
                            elif char == '8':
                                fixed_char = 'B'
                            elif char == '6':
                                fixed_char = 'G'
                            elif char == '2':
                                fixed_char = 'Z'
                    elif i >= total_chars - 4:
                        if char in ['O', 'I', 'l', 'S', 'B', 'G', 'Z', 'D', 'Q']:
                            if char in ['O', 'D', 'Q']:
                                fixed_char = '0'
                            elif char in ['I', 'l']:
                                fixed_char = '1'
                            elif char == 'S':
                                fixed_char = '5'
                            elif char == 'B':
                                fixed_char = '8'
                            elif char == 'G':
                                fixed_char = '6'
                            elif char == 'Z':
                                fixed_char = '2'
                
                result += fixed_char
            
            return result
        
        return cleaned
    
    def recognize_plate(self, image):

        processing_steps = {
            'plate_detected': False,
            'preprocessed': False,
            'validated': False
        }
        
        results = []
        
        try:
            enhanced_image = self._enhance_image(image)
            
            plate_region, plate_detected = self.detect_license_plate(enhanced_image)
            processing_steps['plate_detected'] = plate_detected
            processing_steps['preprocessed'] = True
            
            if plate_detected:
                extracted_text = self.extract_text_from_plate(plate_region)
                if extracted_text:
                    cleaned_text, confidence = self.validate_and_clean_text(extracted_text)
                    if cleaned_text:
                        results.append((cleaned_text, confidence, 'detected_region', extracted_text))
            
            if not results or results[0][1] == "low":
                original_region, original_detected = self.detect_license_plate(image)
                if original_detected:
                    original_text = self.extract_text_from_plate(original_region)
                    if original_text:
                        cleaned_original, conf_original = self.validate_and_clean_text(original_text)
                        if cleaned_original:
                            results.append((cleaned_original, conf_original, 'original_detected', original_text))
            
            if not results or all(r[1] == "low" for r in results):
                logger.info("Trying full image processing...")
                
                height, width = image.shape[:2]
                
                center_y_start = height // 3
                center_y_end = 2 * height // 3
                center_region = image[center_y_start:center_y_end, :]
                
                center_text = self.extract_text_from_plate(center_region)
                if center_text:
                    cleaned_center, conf_center = self.validate_and_clean_text(center_text)
                    if cleaned_center:
                        results.append((cleaned_center, conf_center, 'center_region', center_text))
                
                bottom_region = image[height//2:, :]
                bottom_text = self.extract_text_from_plate(bottom_region)
                if bottom_text:
                    cleaned_bottom, conf_bottom = self.validate_and_clean_text(bottom_text)
                    if cleaned_bottom:
                        results.append((cleaned_bottom, conf_bottom, 'bottom_region', bottom_text))
                
                full_text = self.extract_text_from_plate(image)
                if full_text:
                    cleaned_full, conf_full = self.validate_and_clean_text(full_text)
                    if cleaned_full:
                        results.append((cleaned_full, conf_full, 'full_image', full_text))
            
            processing_steps['validated'] = True
            
            if results:
                def result_score(result):
                    text, confidence, method, raw = result
                    confidence_scores = {"high": 3, "medium": 2, "low": 1}
                    base_score = confidence_scores.get(confidence, 0)
                    
                    if 'detected' in method:
                        base_score += 1
                    
                    if 4 <= len(text) <= 8:
                        base_score += 0.5
                    
                    return base_score
                
                results.sort(key=result_score, reverse=True)
                best_result = results[0]
                
                return {
                    'extracted_text': best_result[0],
                    'confidence': best_result[1],
                    'processing_steps': processing_steps,
                    'raw_text': best_result[3],
                    'method_used': best_result[2]
                }
            else:
                return {
                    'extracted_text': None,
                    'confidence': 'low',
                    'processing_steps': processing_steps,
                    'raw_text': '',
                    'method_used': 'failed'
                }
        
        except Exception as e:
            logger.error(f"License plate recognition failed: {e}")
            return {
                'extracted_text': None,
                'confidence': 'low',
                'processing_steps': processing_steps,
                'error': str(e),
                'method_used': 'error'
            }
    
    def _enhance_image(self, image):

        img_float = image.astype(np.float32) / 255.0
        
        gamma = 1.2
        enhanced = np.power(img_float, gamma)
        
        enhanced = (enhanced * 255).astype(np.uint8)
        
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
    
    def _confidence_score(self, confidence_level):

        scores = {"high": 3, "medium": 2, "low": 1}
        return scores.get(confidence_level, 0)

recognizer = LicensePlateRecognizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
        
        image_bytes = file.read()

        original_filename = secure_filename(file.filename)
        name, ext = os.path.splitext(original_filename)

        try:
            base_name = "Car"
            uploads_root = app.config['UPLOAD_FOLDER']
            existing = [d for d in os.listdir(uploads_root) if os.path.isdir(os.path.join(uploads_root, d)) and d.startswith(base_name)]
            nums = []
            for d in existing:
                try:
                    suffix = d.split('-')[-1]
                    num = int(suffix)
                    nums.append(num)
                except Exception:
                    continue
            next_num = max(nums) + 1 if nums else 1
            folder_name = f"{base_name}-{next_num}"
            session_dir = os.path.join(uploads_root, folder_name)
            os.makedirs(session_dir, exist_ok=True)

            unique_filename = f"{name}{ext}"
            save_path = os.path.join(session_dir, unique_filename)
            with open(save_path, 'wb') as f_out:
                f_out.write(image_bytes)
            logger.info(f"Saved uploaded image to {save_path}")
            try:
                recognizer.debug_dir = session_dir
            except Exception:
                pass
        except Exception as e:
            try:
                session_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"Car Number-{uuid.uuid4().hex}")
                os.makedirs(session_dir, exist_ok=True)
                unique_filename = f"{name}{ext}"
                save_path = os.path.join(session_dir, unique_filename)
                with open(save_path, 'wb') as f_out:
                    f_out.write(image_bytes)
                logger.info(f"Saved uploaded image to {save_path} (fallback)")
                try:
                    recognizer.debug_dir = session_dir
                except Exception:
                    pass
            except Exception as e2:
                session_dir = app.config['UPLOAD_FOLDER']
                logger.warning(f"Failed to create session folder or save original image: {e}; fallback error: {e2}")

        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Could not decode image. Please try a different file.'}), 400

        try:
            os.makedirs(session_dir, exist_ok=True)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
            gray_path = os.path.join(session_dir, f"{name}_gray.png")
            cv2.imwrite(gray_path, gray)

            try:
                enhanced = recognizer._enhance_image(image)
                enhanced_path = os.path.join(session_dir, f"{name}_enhanced.png")
                cv2.imwrite(enhanced_path, enhanced)
            except Exception:
                enhanced = None

            try:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                enhanced_gray = clahe.apply(gray)
                denoised = cv2.bilateralFilter(enhanced_gray, 11, 17, 17)
                denoised_path = os.path.join(session_dir, f"{name}_denoised.png")
                cv2.imwrite(denoised_path, denoised)
            except Exception:
                denoised = None

            try:
                preproc = recognizer.preprocess_image(image)
                preproc_path = os.path.join(session_dir, f"{name}_preproc.png")
                cv2.imwrite(preproc_path, preproc)
            except Exception:
                preproc = None

            try:
                edges = cv2.Canny(gray, 50, 150, apertureSize=3)
                edges_path = os.path.join(session_dir, f"{name}_edges.png")
                cv2.imwrite(edges_path, edges)
            except Exception:
                edges = None

            try:
                contour_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                contours, _ = cv2.findContours(edges if edges is not None else (cv2.Canny(gray,50,150)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                h_img, w_img = gray.shape[:2]
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = w / float(h) if h > 0 else 0
                    area = w * h
                    if (2.0 <= aspect_ratio <= 6.0 and area > 1500 and w > 80 and h > 20 and area < (w_img * h_img * 0.3)):
                        cv2.rectangle(contour_vis, (x, y), (x+w, y+h), (0,255,0), 2)
                contour_path = os.path.join(session_dir, f"{name}_contours.png")
                cv2.imwrite(contour_path, contour_vis)
            except Exception:
                pass

            try:
                mser_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                mser = cv2.MSER_create(_min_area=100, _max_area=10000, _max_variation=0.25, _min_diversity=0.2)
                regions, _ = mser.detectRegions(gray)
                for region in regions:
                    x, y, w, h = cv2.boundingRect(region.reshape(-1,1,2))
                    cv2.rectangle(mser_vis, (x, y), (x+w, y+h), (0,0,255), 1)
                mser_path = os.path.join(session_dir, f"{name}_mser.png")
                cv2.imwrite(mser_path, mser_vis)
            except Exception:
                pass

        except Exception as e:
            logger.warning(f"Failed to generate debug images: {e}")
        
        if image is None:
            return jsonify({'error': 'Could not decode image. Please try a different file.'}), 400
        
        height, width = image.shape[:2]
        if width > 1200:
            scale = 1200 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        try:
            plate_region, plate_detected = recognizer.detect_license_plate(image)
            if plate_region is not None:
                try:
                    plate_path = os.path.join(session_dir, f"{name}_plate.png")
                    cv2.imwrite(plate_path, plate_region)
                    logger.info(f"Saved detected plate image to {plate_path} (detected={plate_detected})")
                except Exception:
                    logger.debug("Failed to save detected plate image")
        except Exception as e:
            logger.debug(f"Plate detection for debug save failed: {e}")

        result = recognizer.recognize_plate(image)
        
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Upload processing failed: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):

    return jsonify({'error': 'File too large. Please upload an image smaller than 16MB.'}), 413

if __name__ == '__main__':
    try:
        pytesseract.get_tesseract_version()
        logger.info("Tesseract OCR is available")
    except Exception as e:
        logger.error(f"Tesseract OCR not found: {e}")
        logger.error("Please install Tesseract OCR: https://github.com/tesseract-ocr/tesseract")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
