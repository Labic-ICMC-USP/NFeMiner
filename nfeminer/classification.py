import json
import multiprocessing
import numpy as np
import os
import pandas as pd
import pickle
import sys
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

class NFeMinerModelCreator:
    """
    NFeModelCreator is a class designed to build multiple text-based models for product description data.
    
    It supports different vectorization strategies and embedding models to create:
    - A string match dictionary model for exact matching of unique product descriptions per GTIN.
    - A vectorizer-based 1-NN classification model with PyTorch tensors.
    - A Sentence-BERT embedding model with tensor representations for semantic similarity.
    
    Attributes:
        dataset (pd.DataFrame): Input dataset containing GTIN and original product descriptions.
        trusted_records (int): Threshold for minimum record count to consider a GTIN reliable.
        vectorizer (sklearn Pipeline or Vectorizer): Text vectorizer for feature extraction.
        embedding (SentenceTransformer): Sentence transformer model for embedding text.
    
    Raises:
        ValueError: If the input dataset is empty or if unsupported vectorizer or embedding is specified.
    """
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 trusted_records: int = 100, 
                 vectorizer: str = 'TFIDF_CHAR_NGRAM',
                 embedding: str = 'SBERT',
                 basedir='./'
        ):
        """
        Initialize the model creator with dataset, vectorizer, and embedding options.
        
        Args:
            data (pd.DataFrame): Dataframe with columns 'gtin' and 'original'.
            trusted_records (int): Minimum count of records per GTIN to be trusted.
            vectorizer (str): Vectorizer key to select predefined vectorization pipeline.
            embedding (str): Embedding key to select pretrained embedding model.
        
        Raises:
            ValueError: If data is empty or vectorizer/embedding not supported.
        """
        # Check if input dataset is empty, raise error if true
        if data.empty:
            raise ValueError("Input dataset is empty. Please provide valid data.")
        
        # Store input dataset and trusted records threshold in instance variables
        self.dataset = data
        self.trusted_records = trusted_records
        self.basedir = f'{basedir}/models'

        # Predefine multiple vectorizer configurations for text feature extraction
        self.vectorizer_configs = {
            'BOW_WORD': make_pipeline(
                CountVectorizer(analyzer='word', max_features=1000),
                Normalizer(norm='l2')
            ),
            'BOW_CHAR': make_pipeline(
                CountVectorizer(analyzer='char', max_features=1000),
                Normalizer(norm='l2')
            ),
            'BOW_CHAR_NGRAM': make_pipeline(
                CountVectorizer(ngram_range=(1, 5), analyzer='char', max_features=1000),
                Normalizer(norm='l2')
            ),
            'TFIDF_WORD': TfidfVectorizer(analyzer='word', max_features=1000, norm='l2'),
            'TFIDF_CHAR': TfidfVectorizer(analyzer='char', max_features=1000, norm='l2'),
            'TFIDF_CHAR_NGRAM': TfidfVectorizer(ngram_range=(1, 5), analyzer='char', max_features=1000, norm='l2')
        }

        # Validate vectorizer choice, raise error if not supported
        if vectorizer not in self.vectorizer_configs:
            raise ValueError(f"Vectorizer '{vectorizer}' not supported. Choose from: {list(self.vectorizer_configs.keys())}")

        # Assign selected vectorizer pipeline/model to instance variable
        self.vectorizer = self.vectorizer_configs[vectorizer]

        # Predefine embedding configurations - only one here: Sentence-BERT all-MiniLM-L6-v2
        self.embedding_configs = {
            'SBERT': SentenceTransformer('all-MiniLM-L6-v2')
        }

        # Validate embedding choice, raise error if not supported
        if embedding not in self.embedding_configs:
            raise ValueError(f"Embedding '{embedding}' not supported. Choose from: {list(self.embedding_configs.keys())}")

        # Assign selected embedding model to instance variable
        self.embedding = self.embedding_configs[embedding]

        # Build all models upon initialization
        self.create_string_match_model()
        self.create_vectorizer_tensors_model()
        self.create_embedding_tensors_model()

    def create_string_match_model(self):
        """
        Create a dictionary-based string match model.
        
        This model identifies unique 'original' product descriptions per GTIN that are
        frequent enough (above trusted_records threshold) and exclusive (do not appear in other GTINs).
        
        The result is saved as a pickle file mapping unique descriptions to their GTIN.
        """
        
        print('Creating String Match model...', flush=True)
        
        # Group dataset by GTIN and 'original' description, counting occurrences
        data_count = (
            self.dataset
            .groupby(['gtin', 'original'])
            .size()
            .reset_index(name='count')
            .sort_values(by='count', ascending=False)
        )

        # Filter to keep only descriptions with counts above the trusted_records threshold
        data_filtered = data_count[data_count['count'] >= self.trusted_records]

        # Group by GTIN, aggregating original descriptions into sets
        model = (
            data_filtered
            .groupby('gtin')['original']
            .agg(set)
            .reset_index()
            # For each GTIN row, filter descriptions unique to that GTIN
            .assign(string_match=lambda df: df.apply(lambda row: self.get_unique_descriptions(row, df), axis=1) if not df.empty else {})
            # Keep only rows where there are unique descriptions
            .loc[lambda df: df['string_match'].apply(bool)]
            # Convert sets back to lists for easier processing
            .assign(string_match=lambda df: df['string_match'].apply(list))
            # Explode lists into separate rows
            .explode('string_match')
            # Convert to dictionary mapping description -> GTIN
            .pipe(lambda df: dict(zip(df['string_match'], df['gtin'])))
        )

        # Save the resulting string match model dictionary to disk
        print('Saving String Match model...')
        path_file = self.create_file_path(filename='model_string_match.pkl')
        with open(path_file, 'wb') as f:
            pickle.dump(model, f)
        print('String Match model saved.\n\n', flush=True)

    def create_vectorizer_tensors_model(self):
        """
        Create a vectorizer-based model for 1-NN classification.
        
        Steps:
        - Filter dataset to remove null descriptions.
        - Group by GTIN, selecting most frequent original description.
        - Train selected vectorizer on these descriptions.
        - Save vectorizer model.
        - Convert feature vectors to PyTorch tensors.
        - Store tensor vectors with GTIN and original descriptions in a DataFrame.
        - Save DataFrame as a pickle file.
        """
        
        # Filter out rows with null in 'original' column
        df_grouped = self.dataset[self.dataset['original'].notnull()]

        # Group by GTIN and select the most frequent 'original' description per GTIN
        df_grouped = (
            df_grouped
            .groupby('gtin')['original']
            .agg(lambda x: x.value_counts().idxmax())
            .reset_index()
        )

        # Train vectorizer pipeline/model on the selected original descriptions
        print('Creating vectorizer model...')
        model_knn = self.vectorizer.fit_transform(df_grouped['original'])

        # Save the vectorizer pipeline/model to disk for future use
        print('Saving vectorizer model...')
        path_file = self.create_file_path(filename='model_vectorizer.pkl')
        with open(path_file, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print('Vectorizer model saved.\n\n')

        # Convert sparse matrix to dense numpy array
        print('Creating tensor model for 1-NN classification...', flush=True)
        array_vectors = model_knn.toarray()

        # Convert each numpy vector to a PyTorch tensor of float32 dtype
        tensor_vectors = [torch.tensor(vec, dtype=torch.float32) for vec in array_vectors]

        # Create DataFrame holding tensor vectors, original descriptions, and GTINs
        df_vectors = pd.DataFrame({
            'vector': tensor_vectors,
            'original': df_grouped['original'],
            'gtin': df_grouped['gtin'].values
        }).reset_index(drop=True)

        # Save this DataFrame of tensors for 1-NN classification
        print('Saving tensor model for 1-NN classification...')
        path_file = self.create_file_path(filename='model_1_NN.pkl')
        with open(path_file, 'wb') as f:
            pickle.dump(df_vectors, f)
        print('Tensor model for 1-NN classification saved.\n\n', flush=True)

    def create_embedding_tensors_model(self):
        """
        Create an embedding-based tensor model using Sentence-BERT.
        
        Steps:
        - Filter dataset to remove null descriptions.
        - Group by GTIN, selecting most frequent original description.
        - Encode descriptions into embeddings (tensor).
        - Store tensor embeddings with GTIN and original descriptions in a DataFrame.
        - Save DataFrame as a pickle file.
        """
        
        # Filter dataset for non-null original descriptions
        df_grouped = self.dataset[self.dataset['original'].notnull()]

        # Group by GTIN and select the most frequent original description per GTIN
        df_grouped = (
            df_grouped
            .groupby('gtin')['original']
            .agg(lambda x: x.value_counts().idxmax())
            .reset_index()
        )

        # Print status
        print('Creating embedding model...', flush=True)

        # Encode descriptions as tensor embeddings using the selected embedding model
        embedded = self.embedding.encode(
            df_grouped['original'].tolist(), 
            convert_to_tensor=True
        )
        
        # Create DataFrame holding embedding tensors, original descriptions, and GTINs
        df_vectors = pd.DataFrame({
            'vector': list(embedded.cpu()),
            'original': df_grouped['original'],
            'gtin': df_grouped['gtin'].values
        }).reset_index(drop=True)

        # Save the embedding DataFrame to disk
        print('Saving embedding tensor model...')
        path_file = self.create_file_path(filename='model_embedding.pkl')
        with open(path_file, 'wb') as f:
            pickle.dump(df_vectors, f)
        print('Embedding tensor model saved.\n\n', flush=True)

    def get_unique_descriptions(self, row, all_data) -> set:
        """
        Helper function to identify unique 'original' descriptions for a given GTIN.
        
        Args:
            row (pd.Series): Row containing 'gtin' and set of 'original' descriptions.
            all_data (pd.DataFrame): DataFrame with all grouped data.
        
        Returns:
            set: Set of descriptions unique to the GTIN in the given row.
        """
        # Extract current GTIN and its set of descriptions
        current_gtin = row['gtin']
        current_set = row['original']

        # Collect sets of descriptions from other GTINs (excluding current)
        other_sets = all_data[all_data['gtin'] != current_gtin]['original']

        if other_sets.empty:
            return {}
        
        # Union all descriptions from other GTINs
        combined_others = set().union(*other_sets)

        # Return only descriptions unique to current GTIN (difference set)
        return current_set - combined_others

    def create_file_path(self, filename: str = 'filename.pickle') -> str:
        """
        Helper function to create directory and file path for saving models.
        
        Args:
            dir (str): Directory where model files are stored.
            filename (str): Name of the file to save.
        
        Returns:
            str: Full file path to save the model.
        """
        # Create directory if it doesn't exist
        os.makedirs(self.basedir, exist_ok=True)

        # Construct and return full path string
        return os.path.join(self.basedir, filename)

class NFeMinerGTINEstimator:
    """
    A class to estimate GTIN codes from product descriptions using 
    multiple matching models (string matching, n-gram vectorization, and sentence embeddings).
    """

    def __init__(self, 
                 batch: pd.DataFrame = None, 
                 model: str = "all-MiniLM-L6-v2",
                 n_threads: int = 0,
                 force_cpu: bool = False,
                 min_ngram_confidence: float = 0.5,
                 min_embedding_confidence: float = 0.4,
                 basedir='./'
        ):
        """
        Initializes the estimator, loads models, and performs classification.

        Args:
            batch (pd.DataFrame): Input DataFrame with at least 'original' and 'gtin' columns.
            model (str): SentenceTransformer model to be used for embeddings.
            n_threads (int): Number of threads used for processing data (GPU disable).
            force_cpu (bool): If True, forces computation on CPU (ignoring GPU availability).
            min_ngram_confidence (float): Minimum confidence score required for an n-gram to be considered valid.
            min_embedding_confidence (float): Minimum confidence score required for an embedding to be considered valid.
        """
        if batch is None:
            raise ValueError("Input DataFrame is required to perform estimation.")

        # Store input as working DataFrame
        self.results = batch.copy()

        # Store confidences.
        self.ngram_confidence = 1 - min_ngram_confidence
        self.embedding_confidence = 1 - min_embedding_confidence

        # Create default columns for results
        self.results['gtin'] = None
        self.results['similarity'] = None
        self.results['description_ref'] = None
        self.results['method'] = None

        # Set number of threads based on available CPU cores
        n_cores = multiprocessing.cpu_count()
        self.num_cores = (
            n_cores - n_threads 
            if n_threads != 0 and n_cores - n_threads > 0
            else n_cores - 1
        )

        torch.set_num_threads(
            n_cores
            if self.num_cores <= 0 
            else self.num_cores
        )

        # Set device to GPU if available, otherwise CPU
        self.device = torch.device(
            'cuda' 
            if torch.cuda.is_available() and not force_cpu 
            else 'cpu'
        )

        # Load SentenceTransformer model for embeddings
        self.model = SentenceTransformer(model).to(self.device)

        # Load string-matching dictionary model
        try:
            path = os.path.join(basedir, "models", "model_string_match.pkl")
            self.string_matching_ref = pd.read_pickle(path)
        except Exception as e:
            raise FileNotFoundError(f"Error loading string-matching model: {e}")

        # Load n-gram vectorizer
        try:
            path = os.path.join(basedir, "models", "model_vectorizer.pkl")
            self.vectorizer = pd.read_pickle(path)
        except Exception as e:
            raise FileNotFoundError(f"Error loading vectorizer model: {e}")

        # Load n-gram vector reference data
        try:
            path = os.path.join(basedir, "models", "model_1_NN.pkl")
            self.ngram_tensor_ref = pd.read_pickle(path)
        except Exception as e:
            raise FileNotFoundError(f"Error loading n-gram model: {e}")

        # Load embedding vector reference data
        try:
            path = os.path.join(basedir, "models", "model_embedding.pkl")
            self.embedding_tensor_ref = pd.read_pickle(path)
        except Exception as e:
            raise FileNotFoundError(f"Error loading embedding model: {e}")

        # Run classification in three stages
        self.string_matching_classifier()
        self.ngram_classifier()
        self.embedding_classifier()

        # Ensure unclassified records are properly marked
        self.results.loc[self.results['gtin'].isnull(), 'gtin'] = None
        self.results.loc[self.results['gtin'].isnull(), 'similarity'] = 0

    def string_matching_classifier(self):
        """
        Classifies using direct string-matching dictionary.
        """
        # Apply dictionary mapping from 'original' to 'gtin'
        self.results['gtin'] = self.results['original'].map(self.string_matching_ref)

        # Identify classified rows
        mask = self.results['gtin'].notnull()

        # Fill additional result columns
        self.results.loc[mask, 'similarity'] = 1.0
        self.results.loc[mask, 'description_ref'] = self.results.loc[mask, 'original']
        self.results.loc[mask, 'method'] = 'string_match'

        print('String-Matching: finished', flush=True)

    def ngram_classifier(self):
        """
        Classifies using n-gram vectorization and 1-NN with Euclidean distance.
        """
        # Select unclassified rows
        mask = self.results['gtin'].isna()
        if mask.sum() == 0:
            return

        # Extract raw text to classify
        texts_to_classify = self.results.loc[mask, 'original'].tolist()

        # Vectorize text using pre-trained vectorizer
        try:
            vectors = self.vectorizer.transform(texts_to_classify)
            vectors_tensor = torch.tensor(vectors.toarray(), dtype=torch.float32, device=self.device)
        except Exception as e:
            print(f"Error vectorizing texts for n-gram classification: {e}")
            return

        # Prepare reference vectors
        ref_vectors = torch.stack(self.ngram_tensor_ref['vector'].tolist()).to(self.device)

        # Compute distances between input and reference vectors
        dists = torch.cdist(vectors_tensor, ref_vectors, p=2)
        min_dists, min_indices = torch.min(dists, dim=1)
        matched_mask = min_dists < self.ngram_confidence

        # Get indices of confident matches
        idxs_classified = [i for i, match in enumerate(matched_mask) if match]
        if not idxs_classified:
            return

        # Map local indices to DataFrame indices
        global_indices = self.results.loc[mask].index.to_list()

        for i in idxs_classified:
            global_idx = global_indices[i]
            ref_idx = min_indices[i].item()

            self.results.at[global_idx, 'gtin'] = self.ngram_tensor_ref.at[ref_idx, 'gtin']
            self.results.at[global_idx, 'similarity'] = 1 - min_dists[i].item()
            self.results.at[global_idx, 'description_ref'] = self.ngram_tensor_ref.at[ref_idx, 'original']
            self.results.at[global_idx, 'method'] = 'n_gram_match'

        print('NGRAM-Match: finished', flush=True)


    def embedding_classifier(self):
        """
        Classifies using sentence embeddings and 1-NN with Euclidean distance.
        """
        # Select unclassified rows
        mask = self.results['gtin'].isna()
        if mask.sum() == 0:
            return

        # Extract raw text to classify
        texts_to_classify = self.results.loc[mask, 'original'].tolist()

        try:
            # Encode input texts into embeddings
            with torch.no_grad():
                embeddings_tensor = self.model.encode(texts_to_classify, device=self.device)
                if not isinstance(embeddings_tensor, torch.Tensor):
                    embeddings_tensor = torch.tensor(embeddings_tensor, device=self.device, dtype=torch.float32)
        except Exception as e:
            print(f"Error generating sentence embeddings: {e}")
            return

        # Prepare reference embeddings
        ref_vectors = torch.stack(self.embedding_tensor_ref['vector'].tolist()).to(self.device)

        # Compute distances
        dists = torch.cdist(embeddings_tensor, ref_vectors, p=2)
        min_dists, min_indices = torch.min(dists, dim=1)
        matched_mask = min_dists < self.embedding_confidence
        idxs_classified = [i for i, matched in enumerate(matched_mask) if matched]
        if not idxs_classified:
            return

        # Map local to global DataFrame indices
        global_indices = self.results.loc[mask].index.to_list()

        for i in idxs_classified:
            global_idx = global_indices[i]
            ref_idx = min_indices[i].item()

            self.results.at[global_idx, 'gtin'] = self.embedding_tensor_ref.at[ref_idx, 'gtin']
            self.results.at[global_idx, 'similarity'] = 1 - min_dists[i].item()
            self.results.at[global_idx, 'description_ref'] = self.embedding_tensor_ref.at[ref_idx, 'original']
            self.results.at[global_idx, 'method'] = 'embedding_match'

        print('Embedding-Match: finished', flush=True)

    def accuracy_measure(self, mask):
        """
        Measures and prints accuracy for a subset of classified rows.

        Args:
            mask (Union[list, pd.Series]): Boolean mask or list of row indices.
        """
        accuracy = accuracy_score(self.results.loc[mask, 'gtin'], self.results.loc[mask, 'gtin'])
        print(f"Accuracy: {accuracy:.4f}")

    def report(self):
        """Generates a report of the DataFrame results, including GTIN coverage and a summary of unique combinations.

        This method checks if the results DataFrame is empty or None. If not, it prints:
          - Total number of rows in the dataset
          - Total number of rows with non-null GTIN
          - Percentage of rows with GTIN
        Then, it returns a DataFrame summarizing unique combinations of
        'original', 'description_ref', 'gtin', 'similarity', and 'method',
        including their counts, sorted by similarity.

        Returns:
            pd.DataFrame: A DataFrame containing unique combinations of 
                          'original', 'description_ref', 'gtin', 'similarity', and 'method',
                          with a 'count' column and sorted by 'similarity'.
        """
        if self.results is None or self.results.empty:
            print("The DataFrame is None or empty")
        else:
            print("Dataset report:")

            # Total rows with non-null GTIN
            total_with_gtin = self.results['gtin'].notna().sum()

            # Total rows in the DataFrame
            total_rows = len(self.results)

            # Percentage of rows with GTIN
            percentage = (total_with_gtin / total_rows) * 100

            print(f"Total de registros no dataset: {total_rows}")
            print(f"Total de registros classificados: {total_with_gtin}")
            print(f"Porcentagem de classificação: {percentage:.2f}%")
            print(f"Report:")

            return (
                self.results[['original', 'description_ref', 'gtin', 'similarity', 'method']]
                .value_counts()                          # count unique combinations
                .reset_index(name='count')               # convert to DataFrame and name count column
                .rename(columns={0: 'count'})            # ensure count column name (depending on pandas version)
                .sort_values(by='similarity', ascending=False)  # sort by similarity
            )

if __name__ == "__main__":
    # Path to feather dataset file
    path = sys.argv[1]

    # Load feather dataset into pandas DataFrame
    nfe = pd.read_feather(path)

    # Create empty DataFrame for simplified data extraction
    df = pd.DataFrame()

    # Extract GTIN from nested JSON in 'json' column
    df['gtin'] = nfe['json'].apply(lambda item: json.loads(item)['produto']['gtin'])

    # Extract 'original' product description from nested JSON
    df['original'] = nfe['json'].apply(lambda item: json.loads(item)['produto']['descricao']['original'])

    # Display first two rows to verify extraction
    print(df.head(2))

    # Instantiate the model creator with extracted data
    model_creator = NFeMinerModelCreator(data=df)

    # Configure pandas display options for easier reading
    pd.set_option('display.max_columns', None)

    print('#' * 60)
    print('TESTING MODELS')
    
    # Test loading string match model
    path = 'models/model_string_match.pkl'
    test = pd.read_pickle(path)
    print(test)

    # Test loading 1-NN tensor model and print first 10 rows
    path = 'models/model_1_NN.pkl'
    test = pd.read_pickle(path)
    print(test.head(10))    

    # Test loading saved vectorizer model and check transformation consistency
    path = 'models/model_vectorizer.pkl'
    test = pd.read_pickle(path)
    vec1 = test.transform(['HAMBURGUER DE TESTE']).toarray()
    vec2 = model_creator.vectorizer.transform(['HAMBURGUER DE TESTE']).toarray()
    print('TRUE' if np.array_equal(vec1, vec2) else 'FALSE')

    # Test loading embedding tensor model and print first 10 rows
    path = 'models/model_embedding.pkl'
    test = pd.read_pickle(path)
    print(test.head(10))

    # Build DataFrame from parsed fields
    df = pd.DataFrame()
    df['label'] = nfe['json'].apply(lambda item: json.loads(item)['produto']['gtin'])
    df['original'] = nfe['json'].apply(lambda item: json.loads(item)['produto']['descricao']['original'])

    # Instantiate estimator
    estimator = NFeMinerGTINEstimator(batch=df)

    # Print sample and diagnostics
    print(estimator.results.head(2))
    print(accuracy_score(estimator.results['gtin'], estimator.results['gtin'].fillna('Sem Valor')))
    print(estimator.results.loc[estimator.results['gtin'].isnull()])
    print(estimator.results.head())
    print(estimator.device)
    print(estimator.string_matching_ref)
    print(estimator.vectorizer.get_params())
    print(estimator.ngram_tensor_ref.head(2))
    print(estimator.embedding_tensor_ref.head(2))