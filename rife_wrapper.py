import cv2
import numpy as np
import sys
import os

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class RIFEInterpolator:
    def __init__(self, model_path=None, use_gpu=True):
        self.device = None
        self.model = None
        self.use_fallback = True
        
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "RIFEv4.22", "train_log", "flownet.pkl")
        
        if TORCH_AVAILABLE and os.path.exists(model_path):
            try:
                model_dir = os.path.dirname(os.path.dirname(model_path))
                sys.path.insert(0, model_dir)
                
                from model.RIFE_HDv3 import Model
                
                self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
                self.model = Model()
                self.model.load_model(os.path.dirname(model_path), -1)
                self.model.eval()
                self.model.device()
                
                self.use_fallback = False
                sys.stderr.write(f"RIFE model loaded on {self.device}.\n")
            except Exception as e:
                sys.stderr.write(f"Model load failed: {e}. Using fallback.\n")
    
    def interpolate(self, frame1, frame2, timestep=0.5):
        if frame1 is None or frame2 is None or frame1.shape != frame2.shape:
            return None
        
        try:
            if self.use_fallback:
                return cv2.addWeighted(frame1, 1.0 - timestep, frame2, timestep, 0)
            else:
                return self._rife(frame1, frame2, timestep)
        except Exception as e:
            sys.stderr.write(f"Interpolation error: {e}\n")
            return cv2.addWeighted(frame1, 1.0 - timestep, frame2, timestep, 0)
    
    def _rife(self, frame1, frame2, timestep):
        img0 = torch.from_numpy(frame1.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0
        img1 = torch.from_numpy(frame2.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0
        
        n, c, h, w = img0.shape
        ph = ((h - 1) // 64 + 1) * 64
        pw = ((w - 1) // 64 + 1) * 64
        padding = (0, pw - w, 0, ph - h)
        img0 = torch.nn.functional.pad(img0, padding)
        img1 = torch.nn.functional.pad(img1, padding)
        
        with torch.no_grad():
            output = self.model.inference(img0, img1, timestep)
        
        output = output[:, :, :h, :w]
        output = output[0].cpu().numpy().transpose(1, 2, 0)
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        
        return output
