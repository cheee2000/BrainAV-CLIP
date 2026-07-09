import os
import io
import importlib
import torch
import numpy as np
import laion_clap
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoProcessor, AutoFeatureExtractor, AutoImageProcessor, BlipProcessor, BlipModel, Blip2Processor, Blip2Model, BlipImageProcessor
from tqdm import tqdm
from abc import ABC, abstractmethod

from retrieval_caption import RefDataset


# ==========================================
# Global configuration
# ==========================================
CONFIG = {
    # Data root directory
    "DATA_ROOT": "PATH/TO/Narrative_Movie_fMRI_Dataset/stimuli",
    
    # Feature output root directory
    "OUTPUT_ROOT": "PATH/TO/Narrative_Movie_fMRI_Dataset/derivatives/feat/",
    
    # Model cache root directory
    "MODEL_CACHE_ROOT": "PATH/TO/LLM_ckpt/",
    
    # Movies to process
    "VIDEO_NAMES": [
        'Breaking_Bad',
        'Dream_Girls',
        'Glee',
        'Heroes',
        'Suits',
        'The_Big_Bang_Theory',
        'The_Crown',
        'The_Mentalist'
    ]
}

# ==========================================
# Base extractor
# ==========================================
class BaseExtractor(ABC):
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.processor = None
        self.model_name = "base"
        self.category = "" # e.g., "Visual_Model", "Audio_Model", "Text_Model"
        self.modality_dir = "" # e.g., "img", "wav", "txt"
        self.feature_save_dir = "" # e.g., "clip_img"

    @property
    def model_cache_dir(self):
        """Build the concrete model cache path."""
        return os.path.join(CONFIG["MODEL_CACHE_ROOT"], self.category)

    @abstractmethod
    def load_model(self):
        """Load model and processor."""
        pass

    def count_parameters(self):
        """Print the number of model parameters."""
        if self.model:
            params = sum(p.numel() for p in self.model.parameters())
            print(f"[{self.__class__.__name__}] Model Parameters: {params / 1e6:.2f}M")

    @abstractmethod
    def extract(self, video_name):
        """Extract features for a single video/movie."""
        pass

    def get_save_path(self, video_name):
        """Get the feature save path."""
        save_dir = os.path.join(CONFIG["OUTPUT_ROOT"], self.category, self.feature_save_dir)
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, f'{video_name}.npz')

    def get_stimuli_dir(self, video_name):
        """Get the input data directory."""
        return os.path.join(CONFIG["DATA_ROOT"], self.modality_dir, video_name)


# ==========================================
# Base visual extractor
# ==========================================
class BaseVisualExtractor(BaseExtractor):
    def __init__(self, device='cuda', model_name="base", batch_size=64):
        super().__init__(device)
        self.model_name = model_name
        self.batch_size = batch_size
        self.category = "Visual_Model"
        self.modality_dir = "img"
        
    @abstractmethod
    def extract_batch_features(self, images):
        """
        Extract features from a preprocessed image batch.
        Args:
            images: List of PIL Images
        Returns:
            numpy array of features (B, D)
        """
        pass

    def extract_from_image_dir(self, image_dir, save_name):
        """
        Extract and save features from a given image directory.
        Args:
            image_dir: image directory path
            save_name: save file name (without extension)
        """
        save_path = self.get_save_path(save_name)

        if not os.path.exists(image_dir):
            print(f"Warning: directory does not exist {image_dir}")
            return

        image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]
        image_paths.sort()

        sample_size = len(image_paths)
        if sample_size == 0:
            print(f"{save_name} has no processable images.")
            return

        print(f"Processing {save_name}, num images: {sample_size}")
        feat_list = []

        for i in tqdm(range(0, sample_size, self.batch_size), desc=f"Processing {save_name}"):
            batch_paths = image_paths[i:i + self.batch_size]
            images = []
            for p in batch_paths:
                try:
                    with Image.open(p) as img:
                        images.append(img.convert("RGB"))
                except Exception as e:
                    print(f"Error loading image {p}: {e}")
                    if len(images) > 0:
                        images.append(images[-1])
                    else:
                        images.append(Image.new("RGB", (224, 224)))

            if not images:
                continue

            try:
                feats = self.extract_batch_features(images)
                feat_list.append(feats)
            except Exception as e:
                print(f"Error extracting features for batch {i}: {e}")
                continue

        if not feat_list:
            print(f"{save_name} produced no features.")
            return

        try:
            all_feats = np.vstack(feat_list)
        except ValueError as e:
            print(f"Failed to stack features: {e}")
            return

        np.savez(save_path, feature=all_feats)
        print(f"Saved: {save_path}")

    def extract_from_video_dir(self, video_dir=None, save_name=None, frame_interval=None, video_paths=None):
        """
        Extract and save features from a video directory or list of video paths (one aggregated feature per video).
        Args:
            video_dir: video directory path (mutually exclusive with video_paths)
            save_name: save file name (without extension)
            frame_interval: frame sampling interval (in frames). If None, defaults to ~1 frame per second.
            video_paths: optional list or single video path (mutually exclusive with video_dir).
        """
        has_video_dir = video_dir is not None
        has_video_paths = video_paths is not None
        if has_video_dir == has_video_paths:
            print("Error: exactly one of video_dir or video_paths must be provided.")
            return

        if not save_name:
            print("Error: save_name must not be empty.")
            return

        save_path = self.get_save_path(save_name)

        if frame_interval is not None and frame_interval < 1:
            print(f"Warning: frame_interval={frame_interval} is invalid; defaulting to 1")
            frame_interval = 1

        if has_video_dir:
            if not os.path.exists(video_dir):
                print(f"Warning: directory does not exist {video_dir}")
                return

            video_paths = [
                os.path.join(video_dir, f)
                for f in os.listdir(video_dir)
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))
            ]
            video_paths.sort(key=lambda p: os.path.basename(p))
        else:
            if isinstance(video_paths, str):
                video_paths = [video_paths]
            else:
                video_paths = list(video_paths)

            # when paths are given directly, keep only existing files
            video_paths = [p for p in video_paths if os.path.isfile(p)]

        if len(video_paths) == 0:
            print(f"{save_name} has no processable videos.")
            return

        print(f"Processing {save_name}, num videos: {len(video_paths)}")
        video_feat_list = []

        try:
            import cv2
        except Exception as e:
            print(f"Error importing cv2: {e}")
            return

        for video_path in tqdm(video_paths, desc=f"Processing videos {save_name}"):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error opening video {video_path}")
                continue

            # default to ~1 frame per second: interval = round(video fps)
            current_interval = frame_interval
            if current_interval is None:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps is None or fps <= 0:
                    current_interval = 1
                else:
                    current_interval = max(1, int(round(fps)))

            frame_idx = 0
            batch_images = []
            per_video_feat_batches = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_idx % current_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    batch_images.append(Image.fromarray(frame_rgb))

                    if len(batch_images) >= self.batch_size:
                        try:
                            feats = self.extract_batch_features(batch_images)
                            feats = np.asarray(feats)
                            if feats.ndim == 1:
                                feats = feats[None, :]
                            per_video_feat_batches.append(feats)
                        except Exception as e:
                            print(f"Error extracting features in {video_path}: {e}")
                        batch_images = []

                frame_idx += 1

            cap.release()

            if batch_images:
                try:
                    feats = self.extract_batch_features(batch_images)
                    feats = np.asarray(feats)
                    if feats.ndim == 1:
                        feats = feats[None, :]
                    per_video_feat_batches.append(feats)
                except Exception as e:
                    print(f"Error extracting tail features in {video_path}: {e}")

            if not per_video_feat_batches:
                print(f"Warning: no usable frame features in video, skipping {video_path}")
                continue

            try:
                per_video_feats = np.vstack(per_video_feat_batches)
                video_feat = per_video_feats.mean(axis=0)
                video_feat_list.append(video_feat)
            except ValueError as e:
                print(f"Error pooling video features {video_path}: {e}")
                continue

        if not video_feat_list:
            print(f"{save_name} produced no features.")
            return

        try:
            all_feats = np.vstack(video_feat_list)
        except ValueError as e:
            print(f"Failed to stack features: {e}")
            return

        print(f"feature shape: {all_feats.shape}")
        np.savez(save_path, feature=all_feats)
        print(f"Saved: {save_path}")

    def extract(self, video_name):
        stimuli_dir = self.get_stimuli_dir(video_name)
        save_path = self.get_save_path(video_name)

        # list image files
        if not os.path.exists(stimuli_dir):
            print(f"Warning: directory does not exist {stimuli_dir}")
            return

        stimuli_paths = [os.path.join(stimuli_dir, f) for f in os.listdir(stimuli_dir) if f.endswith(('.jpg', '.png'))]
        stimuli_paths.sort()
        
        # truncate so the count is divisible by 5
        sample_size = len(stimuli_paths) - len(stimuli_paths) % 5
        stimuli_paths = stimuli_paths[:sample_size]

        print(f"Processing {video_name}, num images: {sample_size}")

        feat_list = []
        
        # batch processing
        for i in tqdm(range(0, sample_size, self.batch_size), desc=f"Processing {video_name}"):
            batch_paths = stimuli_paths[i:i + self.batch_size]
            
            # load images
            images = []
            for p in batch_paths:
                try:
                    images.append(Image.open(p).convert("RGB"))
                except Exception as e:
                    print(f"Error loading image {p}: {e}")
                    # simple fallback: reuse the previous image or a black image
                    if len(images) > 0:
                        images.append(images[-1])
                    else:
                        images.append(Image.new('RGB', (224, 224)))

            if not images:
                continue

            # call the subclass-specific extraction logic
            try:
                feats = self.extract_batch_features(images)
                feat_list.append(feats)
            except Exception as e:
                print(f"Error extracting features for batch {i}: {e}")
                continue

        if not feat_list:
            print(f"{video_name} produced no features.")
            return

        # stack and post-process
        try:
            all_feats = np.vstack(feat_list)
        except ValueError as e:
            print(f"Failed to stack features: {e}")
            return
            
        # average pooling (groups of 5 frames)
        try:
            # make sure the feature dim is correct
            if all_feats.ndim == 2:
                pooled_feats = all_feats.reshape(-1, 5, all_feats.shape[-1]).mean(1)
                print(f"feature shape (pooled): {pooled_feats.shape}")
            else:
                print(f"Warning: feature dim is not 2D ({all_feats.shape}); skipping pooling")
                pooled_feats = all_feats
        except ValueError as e:
            print(f"Reshape failed: {e}")
            pooled_feats = all_feats

        # save
        features = {'feature': pooled_feats}
        np.savez(save_path, **features)
        print(f"Saved: {save_path}")


# ==========================================
# CLIP image extractor
# ==========================================
class CLIPImageExtractor(BaseVisualExtractor):
    def __init__(self, device='cuda', model_name="openai/clip-vit-base-patch32", batch_size=128):
        super().__init__(device, model_name, batch_size)
        self.model_name = model_name
        self.batch_size = batch_size
        self.category = "Visual_Model"
        self.modality_dir = "img"        # input data in stimuli/img/
        
        # auto-generate the save dir name from the model name
        lower_name = model_name.lower()
        
        version = "clip"
            
        if "base" in lower_name or "vit-b" in lower_name:
            size = "base"
        elif "large" in lower_name or "vit-l" in lower_name:
            size = "large"
        elif "huge" in lower_name or "vit-h" in lower_name:
            size = "huge"
        else:
            size = "custom"
            
        self.feature_save_dir = f"{version}_{size}_img"

    @property
    def model_cache_dir(self):
        return os.path.join(CONFIG["MODEL_CACHE_ROOT"], self.category, "CLIP")

    def load_model(self):
        print(f"[{self.__class__.__name__}] Loading model: {self.model_name} ...")
        print(f"cache path: {self.model_cache_dir}")
        
        try:
            self.model = CLIPModel.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
            self.processor = CLIPProcessor.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
            self.model.to(self.device)
            self.model.eval()
            print(f"[{self.__class__.__name__}] Model loaded.")
            self.count_parameters()
        except Exception as e:
            print(f"Model loading failed: {e}")
            raise

    def extract_batch_features(self, images):
        # preprocess
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            # extract visual features
            image_features = self.model.get_image_features(**inputs)
            
        return image_features.detach().cpu().numpy()

    def extract(self, video_name):
        # reuse the base extract method
        super().extract(video_name)


# ==========================================
# ResNet-50 image extractor
# ==========================================
class ResNet50ImageExtractor(BaseVisualExtractor):
    def __init__(self, device='cuda', model_name="microsoft/resnet-50", batch_size=128):
        super().__init__(device, model_name, batch_size)
        self.feature_save_dir = "resnet50"

    @property
    def model_cache_dir(self):
        return os.path.join(CONFIG["MODEL_CACHE_ROOT"], self.category, "ResNet")

    def load_model(self):
        print(f"[{self.__class__.__name__}] Loading model: {self.model_name} ...")
        print(f"cache path: {self.model_cache_dir}")
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_name, cache_dir=self.model_cache_dir
            )
            self.model = AutoModel.from_pretrained(
                self.model_name, cache_dir=self.model_cache_dir
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"[{self.__class__.__name__}] Model loaded.")
            self.count_parameters()
        except Exception as e:
            print(f"Model loading failed: {e}")
            raise

    def _pool_outputs(self, outputs):
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feat = outputs.pooler_output
            if feat.ndim == 4:
                if feat.shape[-1] == 1 and feat.shape[-2] == 1:
                    return feat.squeeze(-1).squeeze(-1)
                return feat.mean(dim=(-2, -1))
            if feat.ndim > 2:
                return feat.reshape(feat.shape[0], -1)
            return feat

        if hasattr(outputs, "last_hidden_state"):
            feat = outputs.last_hidden_state
            if feat.ndim == 4:
                # [B, C, H, W] -> [B, C]
                return feat.mean(dim=(-2, -1))
            if feat.ndim == 3:
                # [B, N, C] -> [B, C]
                return feat.mean(dim=1)
            if feat.ndim == 2:
                return feat
        raise RuntimeError("Unsupported ResNet output structure; cannot obtain image features.")

    def extract_batch_features(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            image_features = self._pool_outputs(outputs)

        return image_features.detach().cpu().numpy()

    def extract(self, video_name):
        super().extract(video_name)


# ==========================================
# ConvNeXt image extractor
# ==========================================
class ConvNeXtImageExtractor(BaseVisualExtractor):
    def __init__(self, device='cuda', model_name="facebook/convnext-base-224", batch_size=128):
        super().__init__(device, model_name, batch_size)
        lower_name = model_name.lower()
        if "tiny" in lower_name:
            size = "tiny"
        elif "small" in lower_name:
            size = "small"
        elif "base" in lower_name:
            size = "base"
        elif "large" in lower_name:
            size = "large"
        elif "xlarge" in lower_name:
            size = "xlarge"
        else:
            size = "custom"
        self.feature_save_dir = f"convnext_{size}"

    @property
    def model_cache_dir(self):
        return os.path.join(CONFIG["MODEL_CACHE_ROOT"], self.category, "ConvNeXt")

    def load_model(self):
        print(f"[{self.__class__.__name__}] Loading model: {self.model_name} ...")
        print(f"cache path: {self.model_cache_dir}")
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_name, cache_dir=self.model_cache_dir
            )
            self.model = AutoModel.from_pretrained(
                self.model_name, cache_dir=self.model_cache_dir
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"[{self.__class__.__name__}] Model loaded.")
            self.count_parameters()
        except Exception as e:
            print(f"Model loading failed: {e}")
            raise

    def _pool_outputs(self, outputs):
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feat = outputs.pooler_output
            if feat.ndim == 4:
                if feat.shape[-1] == 1 and feat.shape[-2] == 1:
                    return feat.squeeze(-1).squeeze(-1)
                return feat.mean(dim=(-2, -1))
            if feat.ndim > 2:
                return feat.reshape(feat.shape[0], -1)
            return feat

        if hasattr(outputs, "last_hidden_state"):
            feat = outputs.last_hidden_state
            if feat.ndim == 4:
                # [B, C, H, W] -> [B, C]
                return feat.mean(dim=(-2, -1))
            if feat.ndim == 3:
                # [B, N, C] -> [B, C]
                return feat.mean(dim=1)
            if feat.ndim == 2:
                return feat
        raise RuntimeError("Unsupported ConvNeXt output structure; cannot obtain image features.")

    def extract_batch_features(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            image_features = self._pool_outputs(outputs)

        return image_features.detach().cpu().numpy()

    def extract(self, video_name):
        super().extract(video_name)


# ==========================================
# DINOv3 image extractor
# ==========================================
class DINOv3ImageExtractor(BaseVisualExtractor):
    def __init__(
        self,
        device='cuda',
        model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        batch_size=128,
    ):
        super().__init__(device, model_name, batch_size)
        # fixed feature name
        self.feature_save_dir = "dinov3-vitb16"

    @property
    def model_cache_dir(self):
        return os.path.join(CONFIG["MODEL_CACHE_ROOT"], self.category, "DINOv3")

    def _resolve_local_model_dir(self):
        """
        Prefer local files:
        1) model_name is itself a local directory
        2) {cache_dir}/{repo_name} (e.g. .../DINOv3/dinov3-vitb16-pretrain-lvd1689m)
        3) {cache_dir} is itself the model directory
        """
        if os.path.isdir(self.model_name):
            return self.model_name

        repo_name = str(self.model_name).split("/")[-1]
        candidate_subdir = os.path.join(self.model_cache_dir, repo_name)
        if os.path.isdir(candidate_subdir):
            return candidate_subdir

        if os.path.isdir(self.model_cache_dir):
            cfg_path = os.path.join(self.model_cache_dir, "config.json")
            if os.path.isfile(cfg_path):
                return self.model_cache_dir
        return None

    def load_model(self):
        print(f"[{self.__class__.__name__}] Loading model: {self.model_name} ...")
        print(f"cache path: {self.model_cache_dir}")
        try:
            local_model_dir = self._resolve_local_model_dir()
            if local_model_dir is not None:
                print(f"Local DINOv3 directory detected, loading locally first: {local_model_dir}")
                self.processor = AutoImageProcessor.from_pretrained(
                    local_model_dir, local_files_only=True
                )
                self.model = AutoModel.from_pretrained(
                    local_model_dir, local_files_only=True
                )
            else:
                print("No complete local model directory found; falling back to loading by HuggingFace repo name.")
                self.processor = AutoImageProcessor.from_pretrained(
                    self.model_name, cache_dir=self.model_cache_dir
                )
                self.model = AutoModel.from_pretrained(
                    self.model_name, cache_dir=self.model_cache_dir
                )
            self.model.to(self.device)
            self.model.eval()
            print(f"[{self.__class__.__name__}] Model loaded.")
            self.count_parameters()
        except Exception as e:
            print(f"Model loading failed: {e}")
            raise

    def _pool_outputs(self, outputs):
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feat = outputs.pooler_output
            if feat.ndim > 2:
                feat = feat.reshape(feat.shape[0], -1)
            return feat

        if hasattr(outputs, "last_hidden_state"):
            feat = outputs.last_hidden_state
            if feat.ndim == 3:
                # for ViT-like models the first token is usually the CLS token
                return feat[:, 0, :]
            if feat.ndim == 2:
                return feat
            if feat.ndim > 3:
                return feat.reshape(feat.shape[0], -1)

        raise RuntimeError("Unsupported DINOv3 output structure; cannot obtain image features.")

    def extract_batch_features(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            image_features = self._pool_outputs(outputs)

        return image_features.detach().cpu().numpy().astype(np.float32)

    def extract(self, video_name):
        super().extract(video_name)


# ==========================================
# Gemini Embedding image extractor (API)
# ==========================================
class GeminiImageExtractor(BaseVisualExtractor):
    """
    Extract image features via the Gemini Embedding API.
    Requires GEMINI_API_KEY in the environment.
    """
    def __init__(
        self,
        device='cuda',
        model_name="gemini-embedding-2-preview",
        batch_size=1,
        api_key=None,
        output_dimensionality=None,
        timeout=60,
    ):
        # device is unused on the API path; kept for BaseExtractor compatibility
        super().__init__(device, model_name, batch_size)
        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        self.output_dimensionality = output_dimensionality
        self.timeout = timeout
        self.feature_save_dir = "gemini_embedding2_img"
        self.client = None
        self.genai_types = None

    def load_model(self):
        print(f"[{self.__class__.__name__}] Using API model: {self.model_name}")
        try:
            genai = importlib.import_module("google.genai")
            genai_types = importlib.import_module("google.genai.types")
        except Exception as e:
            raise ImportError(
                "google-genai is not installed. Please install it: pip install google-genai"
            ) from e

        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            # official style: read the API key from the environment by default
            self.client = genai.Client()
        self.genai_types = genai_types
        print(f"[{self.__class__.__name__}] API configured.")

    def _to_png_bytes(self, image):
        """Encode a PIL image to PNG bytes."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _embed_single_image(self, image_bytes):
        """Call embed_content following the official SDK."""
        part = self.genai_types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/png",
        )

        kwargs = {
            "model": self.model_name,
            "contents": [part],
        }
        if self.output_dimensionality is not None:
            # support different SDK versions: prefer the config object
            if hasattr(self.genai_types, "EmbedContentConfig"):
                kwargs["config"] = self.genai_types.EmbedContentConfig(
                    output_dimensionality=int(self.output_dimensionality)
                )
            else:
                kwargs["config"] = {"output_dimensionality": int(self.output_dimensionality)}

        result = self.client.models.embed_content(**kwargs)
        if hasattr(result, "embeddings") and result.embeddings:
            emb0 = result.embeddings[0]
            if hasattr(emb0, "values"):
                return emb0.values
        if hasattr(result, "embedding") and hasattr(result.embedding, "values"):
            return result.embedding.values
        raise ValueError(f"Cannot parse embedding from SDK response: {result}")

    def extract_batch_features(self, images):
        if self.client is None or self.genai_types is None:
            raise RuntimeError("Please call load_model() to initialize the Gemini SDK client first.")

        feat_list = []
        for idx, image in enumerate(images):
            try:
                image_bytes = self._to_png_bytes(image)
                values = self._embed_single_image(image_bytes)
                feat_list.append(np.asarray(values, dtype=np.float32))
            except Exception as e:
                raise RuntimeError(f"Gemini embedding extraction failed (batch_idx={idx}): {e}") from e

        if not feat_list:
            raise RuntimeError("No valid features obtained in the current batch.")
        return np.vstack(feat_list)

    def extract(self, video_name):
        # reuse the base extract method
        super().extract(video_name)


# ==========================================
# SigLIP image extractor
# ==========================================
class SigLIPImageExtractor(BaseVisualExtractor):
    def __init__(self, device='cuda', model_name="google/siglip-so400m-patch14-384", batch_size=64):
        super().__init__(device, model_name, batch_size)
        
        # auto-generate the save dir name from the model name
        if "siglip2" in model_name.lower():
            version = "siglip2"
        elif "siglip" in model_name.lower():
            version = "siglip"
        else:
            version = "custom"
            
        if "base" in model_name.lower():
            size = "base"
        elif "large" in model_name.lower():
            size = "large"
        elif "so400m" in model_name.lower():
            size = "so400m"
        else:
            size = "custom"
            
        self.feature_save_dir = f"{version}_{size}_img"

    @property
    def model_cache_dir(self):
        return os.path.join(CONFIG["MODEL_CACHE_ROOT"], self.category, "SigLIP")

    def load_model(self):
        print(f"[{self.__class__.__name__}] Loading model: {self.model_name} ...")
        print(f"cache path: {self.model_cache_dir}")
        
        try:
            # prefer AutoModel and AutoProcessor
            # SigLIP 2 uses GemmaTokenizer and requires AutoProcessor to load correctly
            self.model = AutoModel.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
            
            # for SigLIP 2, AutoProcessor may fail due to tokenizer issues
            # passing use_fast=True/False explicitly may help with some path issues
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
            except Exception as e_proc:
                print(f"AutoProcessor loading failed: {e_proc}")
                print("Trying to load without tokenizer (image processor only)...")
                from transformers import AutoImageProcessor, AutoTokenizer
                # only ImageProcessor is needed if just image features are required
                self.processor = AutoImageProcessor.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
                # try loading the tokenizer separately (if text features are needed)
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
                    # attach the tokenizer to the processor (mimic AutoProcessor)
                    self.processor.tokenizer = self.tokenizer
                except Exception as e_tok:
                    print(f"Tokenizer loading failed (safe to ignore if only image features are needed): {e_tok}")

            self.model.to(self.device)
            self.model.eval()
            print(f"[{self.__class__.__name__}] Model loaded.")
            self.count_parameters()
            
        except OSError as e:
            if "sentencepiece" in str(e) or "protobuf" in str(e):
                print("Error: missing required dependencies. SigLIP 2 (Gemma Tokenizer) needs sentencepiece and protobuf.")
                print("Please run: pip install sentencepiece protobuf")
            raise e
        except Exception as e:
            print(f"Model loading failed: {e}")
            print("Hint: on tokenizer-related errors, make sure sentencepiece is installed: pip install sentencepiece")
            print("Hint: on KeyError: 'siglip', upgrade transformers: pip install --upgrade transformers")
            raise

    def extract_batch_features(self, images):
        # preprocess
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            # extract visual features

            if hasattr(self.model, 'get_image_features'):
                image_features = self.model.get_image_features(**inputs)
            else:

                outputs = self.model.vision_model(**inputs)
                image_features = outputs.pooler_output



            # image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
        return image_features.detach().cpu().numpy()

    def extract(self, video_name):
        # reuse the base extract method
        super().extract(video_name)


# ==========================================
# BLIP image extractor
# ==========================================
class BLIPImageExtractor(BaseVisualExtractor):
    def __init__(self, device='cuda', model_name="Salesforce/blip-image-captioning-base", batch_size=64):
        super().__init__(device, model_name, batch_size)
        
        # auto-generate the save dir
        if "large" in model_name.lower():
            size = "large"
        else:
            size = "base"
            
        if "itm" in model_name.lower():
            type_ = "itm"
        else:
            type_ = "caption"
            
        self.feature_save_dir = f"blip_{type_}_{size}_img"

    @property
    def model_cache_dir(self):
        return os.path.join(CONFIG["MODEL_CACHE_ROOT"], self.category, "BLIP")

    def load_model(self):
        print(f"[{self.__class__.__name__}] Loading model: {self.model_name} ...")
        print(f"cache path: {self.model_cache_dir}")
        
        try:
            self.model = BlipModel.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
            self.processor = BlipProcessor.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
            self.model.to(self.device)
            self.model.eval()
            print(f"[{self.__class__.__name__}] Model loaded.")
            self.count_parameters()
        except Exception as e:
            print(f"Model loading failed: {e}")
            raise

    def extract_batch_features(self, images):
        # BLIP preprocess
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            # extract visual features
            # BlipModel.vision_model outputs last_hidden_state (B, seq_len, hidden_size)
            # pooler_output (B, hidden_size)
            outputs = self.model.vision_model(**inputs)
            image_features = outputs.pooler_output
            
        return image_features.detach().cpu().numpy()


# ==========================================
# BLIP-2 image extractor
# ==========================================
class BLIP2ImageExtractor(BaseVisualExtractor):
    def __init__(self, device='cuda', model_name="Salesforce/blip2-opt-2.7b", batch_size=32):
        super().__init__(device, model_name, batch_size)
        
        # auto-generate the save dir
        if "opt" in model_name.lower():
            llm = "opt"
        elif "flan" in model_name.lower():
            llm = "flan"
        else:
            llm = "custom"
            
        if "2.7b" in model_name.lower():
            size = "2.7b"
        elif "6.7b" in model_name.lower():
            size = "6.7b"
        elif "xxl" in model_name.lower():
            size = "xxl"
        elif "xl" in model_name.lower():
            size = "xl"
        else:
            size = "base"
            
        self.feature_save_dir = f"blip2_{llm}_{size}_img"

    @property
    def model_cache_dir(self):
        return os.path.join(CONFIG["MODEL_CACHE_ROOT"], self.category, "BLIP2")

    def load_model(self):
        print(f"[{self.__class__.__name__}] Loading model: {self.model_name} ...")
        print(f"cache path: {self.model_cache_dir}")
        
        try:
            # BLIP-2 may take a while to load and uses a lot of GPU memory
            # load the whole model even though only the vision part is used
            # if GPU memory is limited, try loading only the vision model
            self.model = Blip2Model.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
            
            # fix TypeError for num_query_tokens in Blip2Processor
            try:
                # use_fast=False to avoid the untagged-enum ModelWrapper error
                self.processor = Blip2Processor.from_pretrained(self.model_name, cache_dir=self.model_cache_dir, use_fast=False)
            except TypeError as e:
                if "num_query_tokens" in str(e):
                    print("Detected legacy BLIP-2 config; attempting to fix processor loading...")
                    # manually load the config and drop extra args
                    # BLIP-2 reuses BlipImageProcessor
                    from transformers import BlipImageProcessor, AutoTokenizer
                    
                    image_processor = BlipImageProcessor.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
                    tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_cache_dir, use_fast=False)
                    
                    self.processor = Blip2Processor(image_processor=image_processor, tokenizer=tokenizer)
                else:
                    raise e

            self.model.to(self.device)
            self.model.eval()
            print(f"[{self.__class__.__name__}] Model loaded.")
            self.count_parameters()
        except Exception as e:
            print(f"Model loading failed: {e}")
            raise

    def extract_batch_features(self, images):
        # BLIP-2 preprocess
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            # get Q-Former output features


            qformer_outputs = self.model.get_image_features(**inputs)
            
            # it is a sequence feature; average-pool to get a global vector
            # (B, 32, 768) -> (B, 768)
            # if output is BaseModelOutputWithPooling, take last_hidden_state
            if hasattr(qformer_outputs, 'last_hidden_state'):
                qformer_outputs = qformer_outputs.last_hidden_state
            
            image_features = qformer_outputs.mean(dim=1)
            
        return image_features.detach().cpu().numpy()


# ==========================================
# Base audio extractor
# ==========================================
class BaseAudioExtractor(BaseExtractor):
    def __init__(self, device='cuda', model_name="base", batch_size=128):
        super().__init__(device)
        self.model_name = model_name
        self.batch_size = batch_size
        self.category = "Audio_Model"
        self.modality_dir = "wav"

    @abstractmethod
    def extract_batch_features(self, audio_paths):
        """
        Extract features from a batch of audio paths.
        Args:
            audio_paths: List[str]
        Returns:
            numpy array of features (B, D)
        """
        pass

    def extract(self, video_name, stimuli_dir=None):
        if stimuli_dir is None:
            stimuli_dir = self.get_stimuli_dir(video_name)
        save_path = self.get_save_path(video_name)

        if not os.path.exists(stimuli_dir):
            print(f"Warning: directory does not exist {stimuli_dir}")
            return


        stimuli_paths = [
            os.path.join(stimuli_dir, f)
            for f in os.listdir(stimuli_dir)
            if f.lower().endswith((".wav", ".mp3", ".flac"))
        ]
        stimuli_paths.sort()
        sample_size = len(stimuli_paths)

        print(f"Processing {video_name}, num audio: {sample_size}")
        feat_list = []

        # batch processing
        for i in tqdm(range(0, sample_size, self.batch_size), desc=f"Processing {video_name}"):
            batch_paths = stimuli_paths[i:i + self.batch_size]
            try:
                feats = self.extract_batch_features(batch_paths)
                feats = np.asarray(feats)
                if feats.ndim == 1:
                    feats = feats[None, :]
                feat_list.append(feats)
            except Exception as e:
                print(f"Error processing batch starting at {i}: {e}")
                continue

        if not feat_list:
            print(f"{video_name} produced no features.")
            return

        # stack and post-process
        try:
            all_feats = np.vstack(feat_list)
        except ValueError as e:
            print(f"Failed to stack features: {e}")
            return

        print(f"feature shape: {all_feats.shape}")
        np.savez(save_path, feature=all_feats)
        print(f"Saved: {save_path}")


# ==========================================
# Gemini Embedding audio extractor (API)
# ==========================================
class GeminiAudioExtractor(BaseAudioExtractor):
    """
    Extract audio features via the Gemini Embedding API.
    Requires GEMINI_API_KEY or GOOGLE_API_KEY in the environment.
    """
    def __init__(
        self,
        device='cuda',
        model_name="gemini-embedding-2-preview",
        batch_size=1,
        api_key=None,
        output_dimensionality=None,
        timeout=60,
    ):
        # device is unused on the API path; kept for BaseExtractor compatibility
        super().__init__(device, model_name, batch_size)
        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        self.output_dimensionality = output_dimensionality
        self.timeout = timeout
        self.feature_save_dir = "gemini_embedding2_audio"
        self.client = None
        self.genai_types = None

    def extract(self, video_name, start=0, length=None, stimuli_dir=None):
        if stimuli_dir is None:
            stimuli_dir = self.get_stimuli_dir(video_name)
        save_path = self.get_save_path(video_name)

        if not os.path.exists(stimuli_dir):
            print(f"Warning: directory does not exist {stimuli_dir}")
            return

        stimuli_paths = [
            os.path.join(stimuli_dir, f)
            for f in os.listdir(stimuli_dir)
            if f.lower().endswith((".wav", ".mp3", ".flac"))
        ]
        stimuli_paths.sort()
        total_size = len(stimuli_paths)

        start = int(start)
        if start < 0:
            start = 0
        if start > total_size:
            start = total_size

        if length is None:
            selected_paths = stimuli_paths[start:]
        else:
            length = int(length)
            if length < 0:
                length = 0
            end = min(total_size, start + length)
            selected_paths = stimuli_paths[start:end]

        selected_size = len(selected_paths)
        if selected_size != total_size:
            end_index = start + selected_size
            save_path = self.get_save_path(f"{video_name}_{start}_{end_index}")

        print(
            f"Processing {video_name}, total num audio: {total_size}, "
            f"range: [{start}, {start + selected_size}), num extracted: {selected_size}"
        )
        feat_list = []

        for i in tqdm(range(0, selected_size, self.batch_size), desc=f"Processing {video_name}"):
            batch_paths = selected_paths[i:i + self.batch_size]
            try:
                feats = self.extract_batch_features(batch_paths)
                feats = np.asarray(feats)
                if feats.ndim == 1:
                    feats = feats[None, :]
                feat_list.append(feats)
            except Exception as e:
                print(f"Error processing batch starting at {i}: {e}")
                continue

        if not feat_list:
            print(f"{video_name} produced no features.")
            return

        try:
            all_feats = np.vstack(feat_list)
        except ValueError as e:
            print(f"Failed to stack features: {e}")
            return

        print(f"feature shape: {all_feats.shape}")
        np.savez(save_path, feature=all_feats)
        print(f"Saved: {save_path}")

    def load_model(self):
        print(f"[{self.__class__.__name__}] Using API model: {self.model_name}")
        try:
            genai = importlib.import_module("google.genai")
            genai_types = importlib.import_module("google.genai.types")
        except Exception as e:
            raise ImportError(
                "google-genai is not installed. Please install it: pip install google-genai"
            ) from e

        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            # official style: read the API key from the environment by default
            self.client = genai.Client()
        self.genai_types = genai_types
        print(f"[{self.__class__.__name__}] API configured.")

    def _infer_audio_mime_type(self, audio_path):
        ext = os.path.splitext(audio_path)[1].lower()
        mime_map = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".flac": "audio/flac",
            ".m4a": "audio/mp4",
            ".aac": "audio/aac",
            ".ogg": "audio/ogg",
        }
        return mime_map.get(ext, "audio/wav")

    def _embed_single_audio(self, audio_path):
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        part = self.genai_types.Part.from_bytes(
            data=audio_bytes,
            mime_type=self._infer_audio_mime_type(audio_path),
        )

        kwargs = {
            "model": self.model_name,
            "contents": [part],
        }
        if self.output_dimensionality is not None:
            # support different SDK versions: prefer the config object
            if hasattr(self.genai_types, "EmbedContentConfig"):
                kwargs["config"] = self.genai_types.EmbedContentConfig(
                    output_dimensionality=int(self.output_dimensionality)
                )
            else:
                kwargs["config"] = {"output_dimensionality": int(self.output_dimensionality)}

        result = self.client.models.embed_content(**kwargs)
        if hasattr(result, "embeddings") and result.embeddings:
            emb0 = result.embeddings[0]
            if hasattr(emb0, "values"):
                return emb0.values
        if hasattr(result, "embedding") and hasattr(result.embedding, "values"):
            return result.embedding.values
        raise ValueError(f"Cannot parse embedding from SDK response: {result}")

    def extract_batch_features(self, audio_paths):
        if self.client is None or self.genai_types is None:
            raise RuntimeError("Please call load_model() to initialize the Gemini SDK client first.")

        feat_list = []
        for idx, audio_path in enumerate(audio_paths):
            try:
                values = self._embed_single_audio(audio_path)
                feat_list.append(np.asarray(values, dtype=np.float32))
            except Exception as e:
                raise RuntimeError(f"Gemini audio embedding extraction failed (batch_idx={idx}, path={audio_path}): {e}") from e

        if not feat_list:
            raise RuntimeError("No valid features obtained in the current batch.")
        return np.vstack(feat_list)


# ==========================================
# CLAP audio extractor
# ==========================================
class CLAPAudioExtractor(BaseAudioExtractor):
    def __init__(self, device='cuda', model_name="630k-audioset-best.pt", batch_size=128):
        super().__init__(device, model_name, batch_size)
        self.feature_save_dir = "clap_audio" # features saved under feat/Audio_Model/clap_audio/

    def load_model(self):
        print(f"[{self.__class__.__name__}] Loading model: {self.model_name} ...")
        print(f"cache path: {self.model_cache_dir}")
        
        try:
            # initialize the CLAP model (HTSAT-base is common)
            # enable_fusion=False means audio encoder only
            self.model = laion_clap.CLAP_Module(enable_fusion=False)
            
            # if defaults do not match weights, try specifying 'HTSAT-tiny' etc.
            # self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')

            # self.model.load_ckpt()
            # try loading the checkpoint
            ckpt_path = os.path.join(self.model_cache_dir, self.model_name)

            print(f"Loading checkpoint from {ckpt_path}")
            self.model.load_ckpt(ckpt_path)

            self.model.to(self.device)
            self.model.eval()
            print(f"[{self.__class__.__name__}] Model loaded.")
            self.count_parameters()
        except Exception as e:
            print(f"Model loading failed: {e}")
            raise

    def extract_batch_features(self, audio_paths):
        with torch.no_grad():
            # use_tensor=True is unsupported in some versions; returns numpy by default
            audio_features = self.model.get_audio_embedding_from_filelist(x=audio_paths)

        if isinstance(audio_features, torch.Tensor):
            return audio_features.detach().cpu().numpy()
        return audio_features


# ==========================================
# AST audio extractor (Audio Spectrogram Transformer)
# ==========================================
class ASTAudioExtractor(BaseAudioExtractor):
    def __init__(
        self,
        device='cuda',
        model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
        batch_size=16,
        sample_rate=16000,
    ):
        super().__init__(device, model_name, batch_size)
        self.sample_rate = sample_rate
        self.feature_save_dir = "ast"
        self.audio_backend = None

    @property
    def model_cache_dir(self):
        return os.path.join(CONFIG["MODEL_CACHE_ROOT"], self.category, "AST")

    def load_model(self):
        print(f"[{self.__class__.__name__}] Loading model: {self.model_name} ...")
        print(f"cache path: {self.model_cache_dir}")
        try:
            self.audio_backend = importlib.import_module("torchaudio")
        except Exception as e:
            raise ImportError(
                "torchaudio is not installed. Please install a torchaudio matching your torch version."
            ) from e

        try:
            self.processor = AutoFeatureExtractor.from_pretrained(
                self.model_name, cache_dir=self.model_cache_dir
            )
            self.model = AutoModel.from_pretrained(
                self.model_name, cache_dir=self.model_cache_dir
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"[{self.__class__.__name__}] Model loaded.")
            self.count_parameters()
        except Exception as e:
            print(f"Model loading failed: {e}")
            raise

    def _load_audio_mono(self, audio_path):
        waveform, sr = self.audio_backend.load(audio_path)  # [C, T]
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)
        if sr != self.sample_rate:
            waveform = self.audio_backend.functional.resample(
                waveform, orig_freq=sr, new_freq=self.sample_rate
            )
        return waveform.detach().cpu().numpy()

    def extract_batch_features(self, audio_paths):
        if self.model is None or self.processor is None or self.audio_backend is None:
            raise RuntimeError("Please call load_model() to initialize the AST extractor first.")

        waveforms = []
        for path in audio_paths:
            try:
                waveforms.append(self._load_audio_mono(path))
            except Exception as e:
                raise RuntimeError(f"AST failed to read audio: {path}, error={e}") from e

        inputs = self.processor(
            waveforms,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            audio_features = outputs.last_hidden_state.mean(dim=1)

        return audio_features.detach().cpu().numpy()


# ==========================================
# PANNs audio extractor
# ==========================================
class PANNsAudioExtractor(BaseAudioExtractor):
    def __init__(
        self,
        device='cuda',
        model_name="Cnn14",
        batch_size=32,
        sample_rate=32000,
    ):
        super().__init__(device, model_name, batch_size)
        self.sample_rate = sample_rate
        self.feature_save_dir = "panns"
        self.audio_backend = None
        self.audio_tagger = None

    @property
    def model_cache_dir(self):
        return os.path.join(CONFIG["MODEL_CACHE_ROOT"], self.category, "PANNs")

    def load_model(self):
        print(f"[{self.__class__.__name__}] Loading model: {self.model_name} ...")
        print(f"cache path: {self.model_cache_dir}")
        try:
            self.audio_backend = importlib.import_module("torchaudio")
        except Exception as e:
            raise ImportError(
                "torchaudio is not installed. Please install a torchaudio matching your torch version."
            ) from e

        try:
            panns_module = importlib.import_module("panns_inference")
        except Exception as e:
            raise ImportError(
                "panns-inference is not installed. Please install it: pip install panns-inference librosa soundfile"
            ) from e

        try:
            self.audio_tagger = panns_module.AudioTagging(checkpoint_path=None, device=self.device)
            if hasattr(self.audio_tagger, "model"):
                self.model = self.audio_tagger.model
                self.model.eval()
            print(f"[{self.__class__.__name__}] Model loaded.")
            self.count_parameters()
        except Exception as e:
            print(f"Model loading failed: {e}")
            raise

    def _load_audio_mono(self, audio_path):
        waveform, sr = self.audio_backend.load(audio_path)  # [C, T]
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)
        if sr != self.sample_rate:
            waveform = self.audio_backend.functional.resample(
                waveform, orig_freq=sr, new_freq=self.sample_rate
            )
        return waveform.detach().cpu().numpy().astype(np.float32)

    def extract_batch_features(self, audio_paths):
        if self.audio_tagger is None:
            raise RuntimeError("Please call load_model() to initialize the PANNs extractor first.")

        feat_list = []
        for path in audio_paths:
            try:
                waveform = self._load_audio_mono(path)
                if waveform.ndim != 1:
                    waveform = waveform.reshape(-1)
                with torch.no_grad():
                    _, embedding = self.audio_tagger.inference(waveform[None, :])
                emb = embedding
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu().numpy()
                emb = np.asarray(emb).squeeze(0).astype(np.float32)
                feat_list.append(emb)
            except Exception as e:
                raise RuntimeError(f"PANNs extraction failed: {path}, error={e}") from e

        if not feat_list:
            raise RuntimeError("No valid PANNs features obtained in the current batch.")
        return np.vstack(feat_list)

# ==========================================
# Main
# ==========================================
def main():

    extractors = [
        # CLAPAudioExtractor(device="cuda", batch_size=1),
        # ASTAudioExtractor(device="cuda", batch_size=8),
        # PANNsAudioExtractor(device="cuda", batch_size=16),


        # GeminiImageExtractor(),
        # GeminiAudioExtractor(),

        # ResNet50ImageExtractor(device="cuda", model_name="microsoft/resnet-50", batch_size=128),
        # ConvNeXtImageExtractor(device="cuda", model_name="facebook/convnext-base-224", batch_size=128),
        # DINOv3ImageExtractor(device="cuda", model_name="facebook/dinov3-vitb16-pretrain-lvd1689m", batch_size=128),

        # # --- CLIP Models ---

        # CLIPImageExtractor(device="cuda", model_name="openai/clip-vit-base-patch32", batch_size=1024),
        #

        # CLIPImageExtractor(device="cuda", model_name="openai/clip-vit-large-patch14", batch_size=512),
        #

        # # CLIPImageExtractor(device="cuda", model_name="laion/CLIP-ViT-H-14-laion2B-s32B-b79K", batch_size=256),
        #
        # # SigLIP 1 (Base & Large)
        # SigLIPImageExtractor(device="cuda", model_name="google/siglip-base-patch16-224", batch_size=1024),
        # SigLIPImageExtractor(device="cuda", model_name="google/siglip-large-patch16-256", batch_size=512),
        #
        # # SigLIP 2 (Base & Large)
        # SigLIPImageExtractor(device="cuda", model_name="google/siglip2-base-patch16-224", batch_size=1024),
        # SigLIPImageExtractor(device="cuda", model_name="google/siglip2-large-patch16-256", batch_size=512),
        #
        # # SigLIP 1 SO400M (Recommended)
        # SigLIPImageExtractor(device="cuda", model_name="google/siglip-so400m-patch14-384", batch_size=256),
        #
        # # --- BLIP Models ---
        # # BLIP Base (Captioning)
        # BLIPImageExtractor(device="cuda", model_name="Salesforce/blip-image-captioning-base", batch_size=256),
        # # BLIP Large (Captioning)
        # BLIPImageExtractor(device="cuda", model_name="Salesforce/blip-image-captioning-large", batch_size=128),
        
        # --- BLIP-2 Models ---

        # BLIP2ImageExtractor(device="cuda", model_name="Salesforce/blip2-opt-2.7b", batch_size=128),

        # BLIP2ImageExtractor(device="cuda", model_name="Salesforce/blip2-opt-6.7b", batch_size=64),
        # BLIP-2 Flan T5 XL (3B)
        # BLIP2ImageExtractor(device="cuda", model_name="Salesforce/blip2-flan-t5-xl", batch_size=32),
    ]
    
    if not extractors:
        print("Please uncomment at least one extractor in main() to run.")
        return
    
    for extractor in extractors:
        print(f"\n{'='*20}\nRunning {extractor.__class__.__name__}\n{'='*20}")
        extractor.load_model()

        for video_name in CONFIG["VIDEO_NAMES"]:
            extractor.extract(video_name)
            # extractor.extract(video_name, start=1000)


if __name__ == "__main__":
    main()
