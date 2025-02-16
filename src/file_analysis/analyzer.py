from typing import Dict, Any, List
import torch
from PIL import Image
import cv2
import pytesseract
from transformers import pipeline
import fitz  # PyMuPDF
from pathlib import Path
import magic
import numpy as np

class FileAnalyzer:
    def __init__(self):
        self.image_classifier = pipeline("image-classification")
        self.object_detector = pipeline("object-detection")
        self.text_classifier = pipeline("text-classification")
        
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Анализирует файл и возвращает результаты"""
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(file_path)
        
        if file_type.startswith('image'):
            return self.analyze_image(file_path)
        elif file_type.startswith('video'):
            return self.analyze_video(file_path)
        elif file_type == 'application/pdf':
            return self.analyze_pdf(file_path)
        elif file_type.startswith('text'):
            return self.analyze_text(file_path)
        else:
            return {"error": "Unsupported file type"}
            
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Анализирует изображение"""
        image = Image.open(image_path)
        
        # Классификация изображения
        classification = self.image_classifier(image)
        
        # Обнаружение объектов
        objects = self.object_detector(image)
        
        # OCR
        text = pytesseract.image_to_string(image)
        
        # Анализ цветов
        img_array = np.array(image)
        color_analysis = {
            'mean_rgb': img_array.mean(axis=(0, 1)).tolist(),
            'std_rgb': img_array.std(axis=(0, 1)).tolist()
        }
        
        return {
            'classification': classification,
            'objects': objects,
            'text': text,
            'color_analysis': color_analysis,
            'size': image.size,
            'format': image.format
        }
        
    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """Анализирует видео"""
        cap = cv2.VideoCapture(video_path)
        
        # Получаем информацию о видео
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        # Анализируем ключевые кадры
        frames_analysis = []
        for i in range(0, frame_count, int(fps)):  # Анализируем один кадр в секунду
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Конвертируем BGR в RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_image = Image.fromarray(frame_rgb)
                
                # Анализируем кадр
                frame_analysis = {
                    'timestamp': i / fps,
                    'objects': self.object_detector(frame_image)
                }
                frames_analysis.append(frame_analysis)
                
        cap.release()
        
        return {
            'duration': duration,
            'fps': fps,
            'frame_count': frame_count,
            'frames_analysis': frames_analysis
        }
        
    def analyze_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Анализирует PDF документ"""
        doc = fitz.open(pdf_path)
        
        pages_analysis = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Извлекаем текст
            text = page.get_text()
            
            # Извлекаем изображения
            images = []
            for img_index, img in enumerate(page.get_images()):
                xref = img[0]
                base_image = doc.extract_image(xref)
                images.append({
                    'size': base_image["size"],
                    'extension': base_image["ext"]
                })
                
            # Анализируем содержимое страницы
            if text:
                text_analysis = self.text_classifier(text[:512])  # Ограничиваем длину текста
            else:
                text_analysis = None
                
            pages_analysis.append({
                'page_number': page_num + 1,
                'text': text,
                'text_analysis': text_analysis,
                'images_count': len(images),
                'images_info': images
            })
            
        return {
            'pages_count': len(doc),
            'pages_analysis': pages_analysis
        }
        
    def analyze_text(self, text_path: str) -> Dict[str, Any]:
        """Анализирует текстовый файл"""
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Классификация текста
        classification = self.text_classifier(text[:512])
        
        # Базовая статистика
        words = text.split()
        sentences = text.split('.')
        
        return {
            'classification': classification,
            'statistics': {
                'word_count': len(words),
                'sentence_count': len(sentences),
                'character_count': len(text),
                'average_word_length': sum(len(word) for word in words) / len(words),
                'average_sentence_length': sum(len(sent.split()) for sent in sentences) / len(sentences)
            }
        } 