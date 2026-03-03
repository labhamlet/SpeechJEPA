import torch
import queue
import multiprocessing as mp 
import webdataset as wds

class RIRDataManager:
    """Manages RIR data loading with multiprocessing in the main process."""
    
    def __init__(self, rir_data_dir: str, buffer_size: int = 500, num_workers: int = 4):
        self.rir_data_dir = rir_data_dir
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        self.manager = mp.Manager()
        self.rir_queue = self.manager.Queue(maxsize=buffer_size)
        self.stop_event = self.manager.Event()
        self.processes = []
        self.started = False
        
    def _worker(self):
        """Worker process to load RIR data."""
        def to_torch(sample):
            return torch.from_numpy(sample[0]).float()
        
        shuffle_buffer = 100 
        dataset = (wds.WebDataset(self.rir_data_dir,
                                resampled=True,
                                shardshuffle=False)
                    .repeat()
                    .shuffle(shuffle_buffer)
                    .decode("pil")
                    .to_tuple("npy")
                    .map(to_torch))

        loader = iter(torch.utils.data.DataLoader(dataset,
                            num_workers=self.num_workers,
                            prefetch_factor=4,
                            batch_size=None))
        print("RIR Loader is set", flush = True)
        while not self.stop_event.is_set():
            try:
                rirs = next(loader)
                self.rir_queue.put(rirs, timeout=1.0)
            except queue.Full:
                continue

    def start(self):
        """Start the RIR loading process."""
        if not self.started:
            self.process = mp.Process(target=self._worker, daemon=False)
            self.process.start()
            self.started = True
        return self
    

    def __next__(self, timeout: float = 1.0):
        """Get RIR data from the queue."""
        while True:
            try:
                return self.rir_queue.get(timeout=timeout)
            except queue.Empty:
                continue
        
    def stop(self):
        """Stop the RIR loading process."""
        if self.started:
            self.stop_event.set()
            self.process.join(timeout=5.0)
            if self.process.is_alive():
                self.process.terminate()
            self.started = False
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        self.stop()