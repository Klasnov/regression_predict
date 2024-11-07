import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from sklearn.decomposition import TruncatedSVD
import re
from typing import List, Union
from tqdm.auto import tqdm

class SpecificProcessor:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_length: int = 128,
        batch_size: int = 128,
        n_components: int = 32,
        device: str = None
    ):
        self.max_length = max_length
        self.batch_size = batch_size
        self.n_components = n_components
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize Sentence-BERT model
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        
        # Initialize TruncatedSVD for dimensionality reduction
        self.svd = TruncatedSVD(n_components=n_components)
        
        # Precompile regex patterns for faster matching
        self.patterns = {
            "accident_free": re.compile(r"(?i)\b(accident[\s-]*free|no[\s-]*accident)\b"),
            "well_maintained": re.compile(r"(?i)\b(well[\s-]*maintained|regularly[\s-]*serviced)\b"),
            "pristine": re.compile(r"(?i)\b(pristine|excellent|superb|perfect)\s+condition\b"),
            "wear_tear": re.compile(r"(?i)(wear[\s-]*and[\s-]*tear|repairs)\s+(done|needed)"),
            "loan": re.compile(r"(?i)\b(loan|finance|banking)\b"),
            "trade_in": re.compile(r"(?i)\b(trade[\s-]*in)\b"),
            "owner_count": re.compile(r"(?i)(\d+)\s+owner"),
            "service_history": re.compile(r"(?i)(service[\s-]*history|maintained[\s-]*at|serviced[\s-]*at)"),
            "appointment": re.compile(r"(?i)\b(appointment|viewing|call[\s-]*now)\b"),
            "features": re.compile(r"(?i)\b(sunroof|leather|navigation|camera|sensors)\b"),
            "mileage": re.compile(r"(?i)(\d+[,.]?\d*)\s*km"),
            "year": re.compile(r"(?i)(19|20)\d{2}"),
            "warranty": re.compile(r"(?i)\b(warranty|guaranteed)\b"),
            "engine": re.compile(r"(?i)(\d+[.,]\d+)[l\s]*(engine|cc)"),
            "transmission": re.compile(r"(?i)\b(auto|manual|amt|cvt|dct)\b"),
            "fuel": re.compile(r"(?i)\b(petrol|diesel|electric|hybrid)\b")
        }

    def clean_text(self, text: str) -> str:
        """Basic cleaning to preserve important information."""
        if pd.isna(text):
            return ""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        return text

    def extract_features(self, text: str) -> dict:
        """Extract features using regex patterns."""
        if pd.isna(text):
            return {k: 0 for k in self.patterns.keys()}
        features = {}
        for name, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if name in ['mileage', 'year', 'engine']:
                # Parse numerical values
                features[name] = float(matches[0][0]) if matches else 0
            else:
                features[name] = len(matches)
        # Add general text metrics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        return features

    @torch.no_grad()
    def get_embeddings(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Generate embeddings in batches."""
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=self.batch_size,
                normalize_embeddings=normalize
            )
            embeddings.append(batch_embeddings)
        return np.vstack(embeddings)
    
    def fit_transform(self, descriptions: Union[pd.Series, List[str]]) -> pd.DataFrame:
        """ Fit SVD on embeddings and transform descriptions. """
        # Clean text data
        cleaned_texts = [self.clean_text(text) for text in descriptions]
        # Generate embeddings and fit SVD
        embeddings = self.get_embeddings(cleaned_texts)
        reduced_embeddings = self.svd.fit_transform(embeddings)
        # Extract regex-based features
        pattern_features = pd.DataFrame([
            self.extract_features(text) for text in tqdm(cleaned_texts, desc="Extracting features")
        ])
        # Combine embeddings and pattern features
        embedding_df = pd.DataFrame(
            reduced_embeddings,
            columns=[f'emb_{i}' for i in range(self.n_components)]
        )
        return pd.concat([embedding_df, pattern_features], axis=1)
    
    def transform(self, descriptions: Union[pd.Series, List[str]]) -> pd.DataFrame:
        """ Transform new descriptions based on fitted SVD. """
        cleaned_texts = [self.clean_text(text) for text in descriptions]
        # Generate embeddings and apply SVD transform
        embeddings = self.get_embeddings(cleaned_texts)
        reduced_embeddings = self.svd.transform(embeddings)
        # Extract regex-based features
        pattern_features = pd.DataFrame([
            self.extract_features(text) for text in cleaned_texts
        ])
        # Combine embeddings and pattern features
        embedding_df = pd.DataFrame(
            reduced_embeddings,
            columns=[f'emb_{i}' for i in range(self.n_components)]
        )
        return pd.concat([embedding_df, pattern_features], axis=1)

    def get_explained_variance_ratio(self) -> np.ndarray:
        """ Get the explained variance ratio of the fitted SVD. """
        return self.svd.explained_variance_ratio_
