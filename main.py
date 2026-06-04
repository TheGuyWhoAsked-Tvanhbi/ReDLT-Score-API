from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage
from google.cloud import secretmanager
import os
from google import genai
from google.genai import types
import datetime

app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=True
)

secret_client = secretmanager.SecretManagerServiceClient()
project_id = os.environ.get('GCP_PROJECT_ID', 'debates-api-480504')
gemini_api_key_secret_name = f"projects/{project_id}/secrets/gemini-api-key/versions/latest"
GCS_BUCKET_NAME = "debates-audio-bucket"

def get_gemini_api_key():
    print(f"Attempting to retrieve secret '{gemini_api_key_secret_name}' from Secret Manager...")
    try:
        response = secret_client.access_secret_version(name=gemini_api_key_secret_name)
        api_key = response.payload.data.decode('UTF-8')
        print("Successfully retrieved Gemini API key.")
        return api_key
    except Exception as e:
        print(f"ERROR: Error accessing secret '{gemini_api_key_secret_name}': {e}")
        return None


def transcribe_with_speaker_diarization_flexible(audio_file_uri):
    print(f'Starting transcription for: {audio_file_uri}')
    try:
        client = speech.SpeechClient()

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.MP3,
            sample_rate_hertz=48000,
            language_code="vi-VN",
            use_enhanced=True,
            enable_automatic_punctuation=True,
        )

        audio = speech.RecognitionAudio(uri=audio_file_uri)

        operation = client.long_running_recognize(config=config, audio=audio)

        print("Waiting for long-running transcription operation to complete...")
        response = operation.result(timeout=900)

        res = ''
        for result in response.results:
            res += f'{result.alternatives[0].transcript} '

        print('Transcription successful.')
        return res.strip()
    except Exception as e:
        print(f"ERROR: Transcription service failed: {e}")
        return "FAILED"


def generate_ai_analysis(transcribed_text: str, topic: str, participants: int, gemini_api_key: str):
    """
    Sử dụng Gemini AI để chấm điểm cuộc tranh biện.
    """
    print('Starting Gemini AI content generation.')

    client = genai.Client(vertexai=True,api_key=gemini_api_key)

    prompt_text = f"""
    Chủ đề tranh biện: {topic}
    Số người tham gia: {participants} người.
    Nội dung của cuộc tranh biện:
    --- Bắt đầu Text Debates ---
    {transcribed_text}
    --- Kết thúc Text Debates ---
    """

    system_instruction_text = """### System Instructions ###
Bạn là một trọng tài tranh biện chuyên nghiệp trạc tuổi các thí sinh, có nhiệm vụ đánh giá và phân định đội thắng cuộc trong một cuộc tranh biện giữa phe Chính phủ và phe Phản đối. Hãy tuân thủ nghiêm ngặt các tiêu chí chấm điểm và quy tắc đã được cung cấp. 
Xưng hô là "bạn đội ủng hộ" và "bạn đội phản đối" để tạo cảm giác gần gũi.

### User Prompt ###
Đánh giá cuộc tranh biện sau đây và xác định đội thắng cuộc.

**Chủ đề tranh biện:**Chủ đề tranh biện {chu_de_tranh_bien}

**Số người tham gia:**Số người tham gia {so_nguoi_tham_gia} người

**Nội dung tranh biện:**
---Bắt đầu Text Debates--- {noi_dung_debates} ---Kết thúc Text Debates---

**Quy trình và Tiêu chí chấm điểm:**
1.  **Phân chia nội dung:** Tách nội dung tranh biện thành từng lượt nói của diễn giả. Lưu ý thứ tự nói cố định (CP1, PĐ1, CP2, PĐ2, CP3, PĐ3 - cắt giảm nếu ít hơn 6 người). Bỏ qua các phần trao đổi qua lại giữa các diễn giả.
2.  **Thang điểm diễn giả:** Từ 5 (rất tệ) đến 45 (xuất sắc), điểm trung bình 25 (chỉ số nguyên).
    *   **Lưu ý đặc biệt về điểm:**
        *   Dưới 15 hoặc trên 35: Cần giải thích lý do cụ thể.
        *   Dưới 15: Dành cho hành vi sai nghiêm trọng (lăng mạ, ngắt lời liên tục, lập luận phản cảm) và sẽ dẫn đến thua cuộc.
        *   Trên 35: Chỉ dành cho bài diễn xuất sắc nhất, cực kỳ hiếm.
3.  **Các tiêu chí đánh giá chi tiết (từ 1 đến 5):**
    1.  Warrant quality (Chất lượng lập luận / chứng cứ)
    2.  Impact quality (Chất lượng tác động / ảnh hưởng của lý lẽ)
    3.  Weighing quality (Chất lượng so sánh và ưu tiên lý lẽ)
    4.  Engagement (Tương tác / phản biện)
    5.  Argument quality (Chất lượng lập luận tổng thể)
4.  **Cách áp dụng tiêu chí theo số người tham gia:**
    *   **Nếu có 6 người tham gia (3 Chính phủ, 3 Phản đối):**
        *   Người 1 (CP & PĐ): chấm điểm theo tiêu chí [1, 2, 3].
        *   Người 2 (CP & PĐ): chấm điểm theo tiêu chí [1, 2, 3, 4].
        *   Người 3 (CP & PĐ): chấm điểm theo tiêu chí [4, 5].
        *   Tổng hợp điểm cá nhân (dựa trên thứ hạng thấp nhất cho mỗi thành viên) để xác định điểm đội, sau đó phân định đội thắng thua.
    *   **Nếu có dưới 6 người tham gia:**
        *   Chấm điểm tất cả thành viên dựa trên cả 5 tiêu chí.
        *   Tính điểm trung bình của mỗi người nói, sau đó tổng hợp kết quả cả đội để phân định thắng thua.

**Định dạng kết quả mong muốn:**
Trả về tên đội thắng cuộc trong một dòng duy nhất, sau đó xuống dòng và cung cấp lý do cho kết quả đó (không cần nêu ra số điểm dựa trên các tiêu chí chấm, chỉ cần nêu ra lí do, số điểm cho 2 đội, kết luận)."""

    model_name = "gemini-2.5-flash"

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt_text)]
        ),
]

    generate_content_config = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.95,
        max_output_tokens=5000,
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE")
        ],
        system_instruction=[types.Part.from_text(text=system_instruction_text)],
    )

    print(f'Calling Gemini API with model: {model_name}')
    text_holder = ""
    try:
        for chunk in client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=generate_content_config,
        ):
            text_holder += str(chunk.text)
        print('Gemini AI analysis successful.')
        return text_holder
    except Exception as e:
        print(f"ERROR: During Gemini content generation: {e}")
        return "FAILED_AI_ANALYSIS"

@app.route('/process_audio', methods=['POST'])
def process_audio():
    print("--- Received request for /process_audio ---")
    data = request.get_json()
    if not data:
        print("ERROR: Invalid JSON in request body.")
        return jsonify({"error": "Invalid JSON"}), 400

    file_name = data.get('file_name')
    topic = data.get('topic')
    participants = data.get('participants')

    if not all([file_name, topic, participants]):
        print(f"ERROR: Missing data: file_name={file_name}, topic={topic}, participants={participants}")
        return jsonify({"error": "Missing file_name, topic, or participants"}), 400

    print(f"Processing request for file: {file_name}, topic: {topic}, participants: {participants}")

    gcs_audio_uri = f"gs://{GCS_BUCKET_NAME}/{file_name}"

    try:
        print(f"Calling transcription service for {gcs_audio_uri}...")
        transcribed_text = transcribe_with_speaker_diarization_flexible(gcs_audio_uri)
        if transcribed_text == 'FAILED':
            print("ERROR: Transcription service returned 'FAILED'.")
            return jsonify({"error": "Transcription failed"}), 500
        print(f"Transcription result (first 100 chars): {transcribed_text[:100]}...")

        # Bước 2: Lấy Gemini API Key
        print("Retrieving Gemini API key from Secret Manager...")
        gemini_api_key = get_gemini_api_key()
        if not gemini_api_key:
            print("ERROR: Gemini API key not found or accessible.")
            return jsonify({"error": "Failed to retrieve Gemini API key"}), 500
        print("Gemini API key retrieved.")

        print("Calling Gemini AI for analysis...")
        analysis_result = generate_ai_analysis(transcribed_text, topic, participants, gemini_api_key)
        if analysis_result == 'FAILED_AI_ANALYSIS':
            print("ERROR: Gemini AI analysis returned 'FAILED_AI_ANALYSIS'.")
            return jsonify({"error": "AI analysis failed"}), 500
        print("Gemini AI analysis successful.")

        try:
            print(f"Attempting to delete file {file_name} from GCS...")
            storage_client = storage.Client(project=project_id)
            bucket = storage_client.bucket(GCS_BUCKET_NAME)
            blob = bucket.blob(file_name)
            if blob.exists():
                blob.delete()
                print(f"Successfully deleted {file_name} from GCS.")
            else:
                print(f"WARNING: File {file_name} not found in GCS for deletion.")
        except Exception as e:
            print(f"ERROR: Failed to delete file {file_name} from GCS: {e}")
            pass

        print("--- /process_audio request completed successfully ---")
        return jsonify({
            "transcribed_text": transcribed_text,
            "analysis_result": analysis_result
        }), 200
    except Exception as e:
        print(f"FATAL: An unexpected error occurred during audio processing: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))