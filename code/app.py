# app.py
from flask import Flask, request, render_template, jsonify
from hate_speech import HateSpeechDetection
from speech_processor import SpeechProcessor
import torch

app = Flask(__name__)

# Initialize the models globally
hate_speech_detector = HateSpeechDetection(data_path='hate_speech.csv')
speech_processor = SpeechProcessor()
hate_speech_detector.run()

@app.route('/')
def home():
    return render_template('index.html', 
                         cuda_available=torch.cuda.is_available())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Handle text file input
        if 'text_file' in request.files and request.files['text_file'].filename != '':
            file = request.files['text_file']
            try:
                # Read the text file
                text = file.read().decode('utf-8')
                # Analyze each line
                lines = text.splitlines()
                offensive_lines = []
                all_predictions = []
                
                for line in lines:
                    if line.strip():  # Skip empty lines
                        prediction, _ = hate_speech_detector.predict(line, None)
                        if prediction[0] in ['Offensive Language', 'Hate Speech']:
                            offensive_lines.append(line)
                        all_predictions.append(prediction[0])
                
                # Determine overall result
                if 'Hate Speech' in all_predictions:
                    result = 'Hate Speech'
                elif 'Offensive Language' in all_predictions:
                    result = 'Offensive Language'
                else:
                    result = 'Neither having Hate nor to be Offensive'
                
                return render_template('result.html',
                                    original_text=text,
                                    offensive_lines=offensive_lines,
                                    result=result,
                                    is_file=True)
            except Exception as e:
                return jsonify({'error': f'Error processing text file: {str(e)}'}), 500

        # Handle direct text input
        elif request.form.get('text'):  # Changed from 'text' in request.form
            text = request.form.get('text')
            need_translation = request.form.get('need_translation') == 'on'
            source_language = request.form.get('source_language') if need_translation else 'en'
            
            if need_translation and source_language:
                prediction, translated_text = hate_speech_detector.predict(text, source_language)
            else:
                prediction, translated_text = hate_speech_detector.predict(text, None)
            
            prediction = prediction[0]
            
            if prediction == 'Offensive Language':
                result = 'Offensive Language'
            elif prediction == 'Hate Speech':
                result = 'Hate Speech'
            else:
                result = 'Neither having Hate nor to be Offensive'
            
            return render_template('result.html', 
                                original_text=text,
                                translated_text=translated_text if text != translated_text else None,
                                original_lang=source_language if need_translation else 'en',
                                result=result)
        
        # Handle audio input
        elif 'audio_file' in request.files and request.files['audio_file'].filename != '':
            audio_file = request.files['audio_file']
            
            # Process the audio file
            transcription_result = speech_processor.transcribe_audio(audio_file)
            
            if not transcription_result['success']:
                return jsonify({'error': transcription_result.get('error', 'Unknown error')}), 500
            
            # Analyze the transcribed text
            transcribed_text = transcription_result['text']
            analysis = hate_speech_detector.process_transcription(transcribed_text)
            
            return render_template('result.html',
                                original_text=transcribed_text,
                                is_audio=True,
                                analysis_results=analysis['segments'],
                                overall_result=analysis['overall_prediction'])
        
        else:
            return jsonify({'error': 'Please provide either text input, a text file, or an audio file'}), 400
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=False, port=8002)