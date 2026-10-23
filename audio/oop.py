import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import gc
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps
from pyannote.audio import Pipeline
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# --- Конфигурация ---
HF_TOKEN = "__token_is_here__"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIARIZATION = "pyannote/speaker-diarization-3.1"
WHISPER_MODEL = "large-v3-turbo"
WHISPER_COMPUTE_TYPE = "int8"


# --- Утилиты ---

def clear_memory():
    """Полная очистка памяти PyTorch и ctranslate2."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
    try:
        import ctranslate2
        ctranslate2.get_supported_compute_types("cuda")
    except ImportError:
        pass


# --- Исключения ---

class AudioProcessingError(Exception):
    pass


class TranscriptionError(Exception):
    pass


# --- Классы обработки ---

class AudioPreprocessor:
    """Подготовка аудио: конвертация в WAV и нарезка на части по VAD."""

    def __init__(self, max_chunk_size_sec: int = 600):
        self.max_chunk_size_sec = max_chunk_size_sec

    def prepare_audio(self, mp3_file: str) -> str:
        wav_file = str(Path(mp3_file).with_suffix('.wav'))
        try:
            subprocess.run(
                [
                    "ffmpeg", "-loglevel", "quiet", "-y", "-i", mp3_file,
                    "-vn", "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le", wav_file,
                ],
                check=True,
            )
        except FileNotFoundError:
            raise AudioProcessingError("FFmpeg не найден в системе.")
        except subprocess.CalledProcessError as e:
            raise AudioProcessingError(f"Ошибка конвертации ffmpeg: {e}")
        return wav_file

    def cut_audio(self, wav_file: str, clean_parts: bool = True) -> List[Dict[str, int]]:
        audio, _ = sf.read(wav_file)
        audio = audio.astype(np.float32)

        vad_options = VadOptions(
            threshold=0.5,
            min_silence_duration_ms=500,
            speech_pad_ms=5,
        )
        speech_chunks = get_speech_timestamps(audio, vad_options, sampling_rate=16000)

        chunks = []
        current_chunk = []
        for seg in speech_chunks:
            if current_chunk:
                duration = (current_chunk[-1]['end'] - current_chunk[0]['start']) / 16000
                if duration > self.max_chunk_size_sec:
                    chunks.append(current_chunk)
                    current_chunk = []
            current_chunk.append(seg)
        if current_chunk:
            chunks.append(current_chunk)

        new_chunks = [{"start": chunk[0]['start'], "end": chunk[-1]['end']} for chunk in chunks]

        base_dir = Path(wav_file).parent
        parts_dir = base_dir / "parts"

        if parts_dir.exists():
            if clean_parts:
                shutil.rmtree(parts_dir)
            else:
                # Если папка есть и не чистим, просто используем её
                pass
        parts_dir.mkdir(exist_ok=True)

        file_name = Path(wav_file).name
        for idx, part in enumerate(new_chunks):
            wrt_path = parts_dir / f"part_{idx:04}_{file_name}"
            sliced_data = audio[part['start']:part['end']]
            sf.write(wrt_path, sliced_data, 16000)
        return new_chunks


class Transcriber:
    """Транскрибация аудио с помощью Faster-Whisper."""

    def __init__(self, model_name: str = WHISPER_MODEL, device: str = DEVICE, compute_type: str = WHISPER_COMPUTE_TYPE):
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        del self.model
        clear_memory()

    def transcribe(self, wav_file: str) -> None:
        if not os.path.exists(wav_file):
            raise FileNotFoundError(f"Файл {wav_file} не найден.")

        wrt_path = f"{wav_file}.segments.json"
        try:
            segments, info = self.model.transcribe(wav_file, language=None, vad_filter=True, word_timestamps=True)
        except Exception as e:
            raise TranscriptionError(f"Ошибка транскрибации {wav_file}: {e}")

        phrases = []
        for seg in segments:
            words = [{"start": float(w.start), "end": float(w.end), "text": w.word} for w in
                     seg.words] if seg.words else []
            phrases.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
                "words": words,
                "avg_logprob": seg.avg_logprob,
                "no_speech_prob": seg.no_speech_prob,
            })

        with open(wrt_path, 'w', encoding='utf-8') as f:
            json.dump(phrases, f, ensure_ascii=False, indent=2)


class Diarizer:
    """Диаризация (определение спикеров) и сбор эмбеддингов."""

    def __init__(self, model_id: str = MODEL_DIARIZATION, token: str = HF_TOKEN, device: str = DEVICE):
        self.pipeline = Pipeline.from_pretrained(model_id, token=token)
        self.pipeline.to(torch.device(device))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        del self.pipeline
        clear_memory()

    def get_timeline(self, wav_file: str, emb_cache_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(wav_file):
            raise FileNotFoundError(f"Файл {wav_file} не найден.")

        result = []
        try:
            data, sr = sf.read(wav_file)
            waveform = torch.from_numpy(data).float().T
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)

            audio_in = {"waveform": waveform, "sample_rate": sr}
            diarization = self.pipeline(audio_in)
            annotation = diarization.exclusive_speaker_diarization
            embs = diarization.speaker_embeddings

            for turn, _, speaker in annotation.itertracks(yield_label=True):
                # speaker выглядит как 'SPEAKER_0', 'SPEAKER_1'
                emb_idx = int(speaker.split('_')[-1])
                emb = embs[emb_idx]

                result.append({
                    "start": turn.start,
                    "end": turn.end,
                    "emb_idx": len(result),
                })

                with open(emb_cache_path, 'ab') as f:
                    f.write(emb.astype(np.float32).tobytes())
        except Exception as e:
            logging.error(f"[Diarizer] Ошибка обработки {wav_file}: {e}")

        with open(f"{wav_file}.moments.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result


class SpeakerClusterer:
    """Кластеризация эмбеддингов для определения уникальных голосов."""

    def __init__(self, emb_dim: int = 256, min_speakers: int = 1, max_speakers: int = 10):
        self.emb_dim = emb_dim
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

    def cluster_speakers(self, emb_cache_path: str) -> Tuple[np.ndarray, int]:
        if not os.path.exists(emb_cache_path):
            raise FileNotFoundError(f"Кэш эмбеддингов {emb_cache_path} не найден.")

        with open(emb_cache_path, 'rb') as f:
            data = f.read()

        n_embeddings = len(data) // (self.emb_dim * 4)
        if n_embeddings == 0:
            raise ValueError("Файл эмбеддингов пуст.")

        embeddings = np.frombuffer(data, dtype=np.float32).reshape(n_embeddings, self.emb_dim)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        best_k = self.min_speakers
        best_score = -1.0

        for k in range(self.min_speakers, self.max_speakers + 1):
            clustering = AgglomerativeClustering(
                n_clusters=k,
                metric='cosine',
                linkage='average'
            )
            labels = clustering.fit_predict(embeddings)

            score = silhouette_score(embeddings, labels, metric='cosine') if k > 1 else 0.0

            logging.info(f"  k={k}, silhouette={score:.3f}")
            if score > best_score:
                best_score = score
                best_k = k

        logging.info(f"Оптимальное число спикеров: {best_k}")
        final_clustering = AgglomerativeClustering(
            n_clusters=best_k,
            metric='cosine',
            linkage='average'
        )
        labels = final_clustering.fit_predict(embeddings)
        return labels, best_k


class ResultMerger:
    """Склейка транскрибации, диаризации и присвоение меток спикеров."""

    def __init__(self, base_dir: str, file_name: str):
        self.base_dir = base_dir
        self.name = Path(file_name).stem
        self.parts_dir = Path(base_dir) / "parts"

    def assign_speaker_labels(self, labels: np.ndarray):
        emb_idx = 0
        idx = 0
        while True:
            moments_path = self.parts_dir / f"part_{idx:04}_{self.name}.wav.moments.json"
            if not moments_path.exists():
                break

            with open(moments_path, 'r', encoding='utf-8') as f:
                moments = json.load(f)

            for moment in moments:
                global_label = int(labels[emb_idx])
                moment['speaker_id'] = f"SPEAKER_{global_label:02d}"
                emb_idx += 1

            with open(moments_path, 'w', encoding='utf-8') as f:
                json.dump(moments, f, ensure_ascii=False, indent=2)
            idx += 1

    def merge_results(self, total_parts: int):
        result = []
        for idx_ in range(total_parts):
            wav_path = self.parts_dir / f"part_{idx_:04}_{self.name}.wav"
            seg_path = Path(f"{wav_path}.segments.json")
            mom_path = Path(f"{wav_path}.moments.json")

            if not seg_path.exists() or not mom_path.exists():
                continue

            with open(seg_path, "r", encoding='utf-8') as f:
                items = json.load(f)
            with open(mom_path, "r", encoding='utf-8') as f:
                moments = json.load(f)

            if not moments or not items:
                continue

            moments[-1]['end'] = items[-1]['end']
            if moments[-1]['end'] <= items[-1]['start']:
                moments.pop()
                if len(moments) == 0:
                    logging.warning(f"Empty moments for part {idx_}")
                    continue
                moments[-1]['end'] = items[-1]['end']

            moment_idx = 0
            for phrase in items:
                while moment_idx < len(moments) - 1 and moments[moment_idx]['end'] < phrase['start']:
                    moment_idx += 1

                if phrase['end'] <= moments[moment_idx]['end']:
                    result.append({
                        'start': phrase['start'],
                        'end': phrase['end'],
                        'text': phrase['text'],
                        'speaker': moments[moment_idx]['speaker_id'],
                    })
                else:
                    mid = moments[moment_idx + 1]['start']
                    if abs(phrase['start'] - mid) > abs(mid - phrase['end']):
                        result.append({
                            'start': phrase['start'],
                            'end': phrase['end'],
                            'text': phrase['text'],
                            'speaker': moments[moment_idx]['speaker_id'],
                        })
                        moment_idx += 1
                    # Иначе фраза остается за текущим спикером (overlapping)

        output_path = Path(self.base_dir) / f"{self.name}.wav.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


# --- Оркестратор ---

class PipelineOrchestrator:
    """Главный класс, управляющий всем пайплайном."""

    def __init__(self, mp3_file: str, resume: bool = True):
        self.mp3_file = mp3_file
        self.resume = resume
        self.dir_name = os.path.dirname(mp3_file)
        self.file_name = os.path.basename(mp3_file)
        self.name = Path(self.file_name).stem
        self.parts_dir = Path(self.dir_name) / "parts"
        self.emb_cache = Path(self.dir_name) / f"{self.name}.embeddings.bin"

    def _count_files(self, ext: str) -> int:
        if not self.parts_dir.exists():
            return 0
        return sum(1 for f in self.parts_dir.glob(f"*{ext}") if f.is_file())

    def run(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        if self.resume and self.parts_dir.exists():
            total = self._count_files('.wav')
            data1 = self._count_files('.segments.json')
            data2 = self._count_files('.moments.json')
            done = data2 if data1 == total else data1
        else:
            preprocessor = AudioPreprocessor()
            preprocessor.prepare_audio(self.mp3_file)
            preprocessor.cut_audio(str(Path(self.dir_name) / f"{self.name}.wav"), clean_parts=False)
            total = self._count_files('.wav')
            done = 0
            del preprocessor
            clear_memory()

        # Этап 1: Транскрибация
        logging.info(f"Starting transcription (done: {done}/{total})")
        with Transcriber() as transcriber:
            for idx_ in range(done, total):
                part_wav = self.parts_dir / f"part_{idx_:04}_{self.name}.wav"
                seg_json = Path(f"{part_wav}.segments.json")
                if self.resume and seg_json.exists():
                    continue
                logging.info(f"Transcribing part {idx_}")
                transcriber.transcribe(str(part_wav))
                clear_memory()
                time.sleep(15)

        # Этап 2: Diarization
        logging.info(f"Starting diarization (done: {done}/{total})")
        if self.emb_cache.exists():
            self.emb_cache.unlink()  # Очищаем кэш перед новым проходом

        with Diarizer() as diarizer:
            for idx_ in range(done, total):
                part_wav = self.parts_dir / f"part_{idx_:04}_{self.name}.wav"
                if self.resume and Path(f"{part_wav}.moments.json").exists():
                    continue
                logging.info(f"Diarizing part {idx_}")
                diarizer.get_timeline(str(part_wav), str(self.emb_cache))
                clear_memory()
                time.sleep(15)

        # Этап 3: Кластеризация
        logging.info("Clustering embeddings...")
        clusterer = SpeakerClusterer()
        labels, n_speakers = clusterer.cluster_speakers(str(self.emb_cache))
        logging.info(f"Found {n_speakers} unique speakers.")

        # Этап 4: Присвоение меток
        merger = ResultMerger(self.dir_name, self.file_name)
        merger.assign_speaker_labels(labels)

        # Этап 5: Склейка
        logging.info("Merging final results...")
        merger.merge_results(total)

        logging.info("Pipeline finished successfully.")


if __name__ == "__main__":
    orchestrator = PipelineOrchestrator("chess.mp3", resume=False)
    orchestrator.run()
