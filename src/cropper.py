"""
Frame Cropper Module - GPU-accelerated crop and resize
"""
import torch
import torchvision.transforms.functional as TF
import numpy as np
from typing import Tuple, Optional
from tracker import TrackedPosition


class FrameCropper:
    """
    GPU-accelerated frame cropping centered on tracked face.
    
    Features:
    - Maintains aspect ratio
    - Smooth zoom based on face size
    - Boundary clamping to prevent black bars
    - Configurable zoom level and padding
    """
    
    def __init__(
        self,
        output_size: Tuple[int, int] = (1920, 1080),
        zoom_level: float = 2.0,
        face_padding: float = 2.5,
        device: str = "cuda",
        use_gpu_resize: bool = True
    ):
        """
        Args:
            output_size: (width, height) of output frame
            zoom_level: Base zoom multiplier (2.0 = crop to half the frame)
            face_padding: Multiplier for face size to determine crop region
            device: "cuda" for GPU, "cpu" for CPU
            use_gpu_resize: Use GPU for resize (faster but more VRAM)
        """
        self.output_size = output_size
        self.zoom_level = zoom_level
        self.face_padding = face_padding
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_gpu_resize = use_gpu_resize and self.device == "cuda"
        
        # Cache for avoiding repeated tensor allocation
        self._frame_tensor = None
    
    def crop(
        self, 
        frame: np.ndarray, 
        tracked_pos: Optional[TrackedPosition]
    ) -> np.ndarray:
        """
        Crop frame around tracked position.
        
        Args:
            frame: Input BGR frame (OpenCV format)
            tracked_pos: Current tracked face position, or None for full frame
            
        Returns:
            Cropped and resized RGB frame ready for virtual camera
        """
        h, w = frame.shape[:2]
        out_w, out_h = self.output_size
        aspect = out_w / out_h
        
        if tracked_pos is None:
            # No face - return center crop maintaining aspect ratio
            return self._crop_center(frame, aspect)
        
        # Calculate crop region based on zoom level
        crop_w = w / self.zoom_level
        crop_h = crop_w / aspect  # Maintain output aspect ratio
        
        # Optionally adjust zoom based on face size
        # Larger face = zoom out a bit, smaller face = zoom in
        # face_zoom = self.face_padding * max(tracked_pos.width, tracked_pos.height)
        # crop_w = max(crop_w, face_zoom)
        # crop_h = crop_w / aspect
        
        # Center crop on face position
        x1 = tracked_pos.cx - crop_w / 2
        y1 = tracked_pos.cy - crop_h / 2
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        
        # Clamp to frame boundaries
        x1, y1, x2, y2 = self._clamp_bounds(x1, y1, x2, y2, w, h)
        
        # Convert to int
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Crop and resize
        if self.use_gpu_resize:
            return self._gpu_crop_resize(frame, x1, y1, x2, y2)
        else:
            return self._cpu_crop_resize(frame, x1, y1, x2, y2)
    
    def _crop_center(self, frame: np.ndarray, aspect: float) -> np.ndarray:
        """Center crop maintaining aspect ratio"""
        h, w = frame.shape[:2]
        
        if w / h > aspect:
            # Frame is wider - crop width
            new_w = int(h * aspect)
            x1 = (w - new_w) // 2
            cropped = frame[:, x1:x1+new_w]
        else:
            # Frame is taller - crop height
            new_h = int(w / aspect)
            y1 = (h - new_h) // 2
            cropped = frame[y1:y1+new_h, :]
        
        import cv2
        resized = cv2.resize(cropped, self.output_size)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    def _clamp_bounds(
        self, 
        x1: float, y1: float, 
        x2: float, y2: float, 
        w: int, h: int
    ) -> Tuple[float, float, float, float]:
        """Clamp crop region to frame boundaries"""
        crop_w = x2 - x1
        crop_h = y2 - y1
        
        # Shift if out of bounds
        if x1 < 0:
            x1 = 0
            x2 = crop_w
        if y1 < 0:
            y1 = 0
            y2 = crop_h
        if x2 > w:
            x2 = w
            x1 = w - crop_w
        if y2 > h:
            y2 = h
            y1 = h - crop_h
        
        # Final clamp (in case crop is larger than frame)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        return x1, y1, x2, y2
    
    def _gpu_crop_resize(
        self, 
        frame: np.ndarray, 
        x1: int, y1: int, 
        x2: int, y2: int
    ) -> np.ndarray:
        """GPU-accelerated crop and resize using torchvision"""
        out_h, out_w = self.output_size[1], self.output_size[0]
        
        # Convert to tensor (HWC -> CHW)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
        frame_tensor = frame_tensor.to(self.device)
        
        # Crop
        cropped = frame_tensor[:, y1:y2, x1:x2]
        
        # Resize with antialiasing
        resized = TF.resize(cropped, [out_h, out_w], antialias=True)
        
        # Convert back (CHW -> HWC, BGR -> RGB)
        result = resized.permute(1, 2, 0).byte().cpu().numpy()
        
        # BGR to RGB for pyvirtualcam
        result = result[:, :, ::-1].copy()
        
        return result
    
    def _cpu_crop_resize(
        self, 
        frame: np.ndarray, 
        x1: int, y1: int, 
        x2: int, y2: int
    ) -> np.ndarray:
        """CPU crop and resize using OpenCV"""
        import cv2
        
        cropped = frame[y1:y2, x1:x2]
        resized = cv2.resize(cropped, self.output_size, interpolation=cv2.INTER_LINEAR)
        
        # BGR to RGB for pyvirtualcam
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


class AdaptiveZoomCropper(FrameCropper):
    """
    Extended cropper with adaptive zoom based on face size.
    
    Automatically adjusts zoom level to keep face at consistent screen size.
    """
    
    def __init__(
        self,
        target_face_ratio: float = 0.25,
        zoom_smoothing: float = 0.05,
        min_zoom: float = 1.2,
        max_zoom: float = 4.0,
        **kwargs
    ):
        """
        Args:
            target_face_ratio: Target face width as ratio of output width (0.25 = 25%)
            zoom_smoothing: How fast zoom adjusts (lower = smoother)
            min_zoom: Minimum zoom level
            max_zoom: Maximum zoom level
        """
        super().__init__(**kwargs)
        self.target_face_ratio = target_face_ratio
        self.zoom_smoothing = zoom_smoothing
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self._current_zoom = self.zoom_level
    
    def crop(
        self, 
        frame: np.ndarray, 
        tracked_pos: Optional[TrackedPosition]
    ) -> np.ndarray:
        if tracked_pos is not None:
            # Calculate desired zoom based on face size
            h, w = frame.shape[:2]
            face_ratio = tracked_pos.width / w
            
            if face_ratio > 0:
                desired_zoom = face_ratio / self.target_face_ratio * self.zoom_level
                desired_zoom = max(self.min_zoom, min(self.max_zoom, desired_zoom))
                
                # Smooth zoom transition
                self._current_zoom += self.zoom_smoothing * (desired_zoom - self._current_zoom)
        
        # Temporarily override zoom level
        original_zoom = self.zoom_level
        self.zoom_level = self._current_zoom
        result = super().crop(frame, tracked_pos)
        self.zoom_level = original_zoom
        
        return result
