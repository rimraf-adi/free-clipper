import unittest
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from clipper.groq_client import GroqModelPool, LLM_MODEL_POOL
from clipper.srt_utils import generate_srt_for_clip, format_timestamp
from clipper.ingest import parse_input_sources, sanitize_title
from clipper.transcribe import parse_groq_response

class TestGroqPipeline(unittest.TestCase):

    def test_sanitize_title(self):
        self.assertEqual(sanitize_title("How AI Works (Full Podcast)"), "How_AI_Works_Full_Podcast")
        self.assertEqual(sanitize_title("  My Video - Ep. 123!  "), "My_Video_Ep_123")

    def test_groq_response_with_offset(self):
        mock_word = MagicMock()
        mock_word.word = "hello"
        mock_word.start = 1.0
        mock_word.end = 1.5

        mock_seg = MagicMock()
        mock_seg.start = 1.0
        mock_seg.end = 2.0
        mock_seg.text = "hello world"

        mock_resp = MagicMock()
        mock_resp.segments = [mock_seg]
        mock_resp.words = [mock_word]

        res = parse_groq_response(mock_resp, time_offset=600.0)
        self.assertEqual(res[0]["start"], 601.0)
        self.assertEqual(res[0]["end"], 602.0)
        self.assertEqual(res[0]["words"][0]["start"], 601.0)

    def test_parse_comma_separated_sources(self):
        input_str = "https://youtube.com/watch?v=123, https://youtu.be/456, local_file.mp4"
        parsed = parse_input_sources(input_str)
        self.assertEqual(len(parsed), 3)

    def test_parse_csv_file_sources(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("https://youtube.com/watch?v=aaa,https://youtube.com/watch?v=bbb\n")
            temp_csv_path = f.name
            
        try:
            sources = parse_input_sources(temp_csv_path)
            self.assertEqual(len(sources), 2)
        finally:
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)

    def test_format_timestamp(self):
        self.assertEqual(format_timestamp(0.0), "00:00:00,000")
        self.assertEqual(format_timestamp(65.5), "00:01:05,500")

    @patch("clipper.groq_client.Groq")
    def test_groq_model_pool_rotation(self, mock_groq_class):
        mock_client = MagicMock()
        mock_groq_class.return_value = mock_client
        
        mock_choice = MagicMock()
        mock_choice.message.content = '[{"start": 0, "end": 30}]'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        pool = GroqModelPool(api_key="mock_key")
        
        res1 = pool.chat_completion(messages=[{"role": "user", "content": "hi"}])
        res2 = pool.chat_completion(messages=[{"role": "user", "content": "hi"}])

        self.assertIn(res1["model"], LLM_MODEL_POOL)
        self.assertIn(res2["model"], LLM_MODEL_POOL)

if __name__ == "__main__":
    unittest.main()
