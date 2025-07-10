import json
import threading
from typing import List, Dict, Any
from pathlib import Path
from queue import Queue
import time
import os
import numpy as np
from datetime import datetime
from loguru import logger

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class VendiScoreLogger:
    def __init__(self, base_filename: str = "weighted_vendi_scores"):
        self.base_filename = base_filename
        self.scores_buffer: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._file_lock = threading.Lock()  # Separate lock for file operations
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        self._current_filename = None
        
    def add_score(self, score: Dict[str, Any]):
        """Add a score to the buffer"""
        with self._lock:
            # Create a copy of the score to avoid any reference issues
            score_copy = {k: float(v) if isinstance(v, np.floating) else v 
                         for k, v in score.items()}
            self.scores_buffer.append(score_copy)
            logger.debug(f"Added score to buffer. Current buffer size: {len(self.scores_buffer)}")
            
    def flush_scores(self):
        """Flush all buffered scores to file"""
        if not self.scores_buffer:
            return
            
        with self._lock:
            scores_to_write = self.scores_buffer.copy()
            self.scores_buffer.clear()
            
        with self._file_lock:
            try:
                # Create new filename with timestamp if not set
                if not self._current_filename:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self._current_filename = f"{self.base_filename}_{timestamp}.json"
                
                # Read existing scores if file exists
                existing_scores = []
                if os.path.exists(self._current_filename):
                    try:
                        with open(self._current_filename, 'r') as f:
                            existing_scores = json.load(f)
                    except json.JSONDecodeError:
                        logger.warning(f"Could not read existing scores from {self._current_filename}, starting fresh")
                
                # Combine existing scores with new scores
                all_scores = existing_scores + scores_to_write
                
                # Write all scores to file
                with open(self._current_filename, 'w') as f:
                    json.dump(all_scores, f, indent=2, cls=NumpyEncoder)
                logger.debug(f"Wrote {len(scores_to_write)} new scores to {self._current_filename} (total: {len(all_scores)})")
                
            except Exception as e:
                logger.error(f"Error saving vendi scores: {e}")
                # Put scores back in buffer if write failed
                with self._lock:
                    self.scores_buffer.extend(scores_to_write)

    def start_new_run(self):
        """Start a new run with a new file"""
        with self._lock:
            self._current_filename = None
            self.flush_scores()  # Flush any remaining scores to the old file

    def _worker(self):
        """Background worker to handle file operations"""
        while True:
            time.sleep(1)  # Check every second
            if self.scores_buffer:
                self.flush_scores()

# Global instance
vendi_logger = VendiScoreLogger() 