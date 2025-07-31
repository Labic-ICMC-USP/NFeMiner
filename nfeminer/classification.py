import os, pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class NFeMinerGTINEstimator:
    """ 
    NFeMinerGTINEstimator is responsible for carrying out the complete flow of analysis and classification of textual data based on multiple similarity techniques.

    The algorithm operates as follows:

    1) Receiving input data: the class receives the raw data to be analyzed, containing information that will be used for comparison and classification.

    2) Extraction of relevant features: the data is preprocessed to identify and extract textual attributes of interest (such as the original description from NFe).

    3) Sequential processing with similarity techniques: to determine the degree of similarity between the input description and the classification model, the following techniques are used:

        - string_match: direct matching of strings.

        - n-grams match: analysis of the TF-IDF vector formed with n-grams from the description (with n = 1 to 5).

        - description similarity: comparison based on transformer embeddings.

    4) Final classification: if a classification within the thresholds is reached and it is unique, a GTIN is estimated for the NFe.
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        """ 
        Initialization constructor.

        Parameters: 
            input(DataFrame): receives input data as Pandas DataFrame.
            model (str): receives the model used by the transformer to generate the vector representation of the description. By default, it uses the "all-MiniLM-L6-v2" model.
        """

        # Transformer model to embedding classification
        self.model = SentenceTransformer(model)

        # Basedir
        basedir = '..'

        # Load similarity reference
        model_path = os.path.join(basedir, "models", "model_simil.pkl")
        with open(model_path, 'rb') as f:
            self.ref = pickle.load(f)

        # Load TF-IDF model to vectorize N-GRAMS
        model_path = os.path.join(basedir, "models", 'vectorizer.pkl')
        with open(model_path, 'rb') as f:
            self.vectorizer = pickle.load(f)

        # Create string match reference
        self.__create_string_match_ref()

        # Create N-GRAM reference
        self.__create_ngrams_ref()

        # Create Embedding reference
        self.__create_embbedings_ref()
        
    def __create_string_match_ref(self):
        """ 
        The create_string_match_ref method is used to create a data structure that assists in the classification process using string_match similarity.
        
        In short, the method uses the classification model to generate a dictionary that maps descriptions to GTINs. Example: { "GROUND BEEF": '00001', ..., "BEEF": '000050' }.
        
        Parameters: 
            No input parameters.
        """

        # Data transformation
        self.string_match_ref = self.ref\
            .loc[self.ref['string_match'], ['gtin', 'examples']]\
            .explode('examples') \
            .set_index('examples')['gtin']\
            .to_dict()

    def __create_ngrams_ref(self):
        """ 
        The create_ngrams_ref method is used to create a data structure that supports the classification process using n_grams similarity.

        In summary, it generates a structured dataframe with three (3) columns: a reliable description from the electronic invoice (reference GTIN description), the GTIN itself (classification reference), and the embedding of the reference string (string transformed into n-gram embeddings by a TF-IDF vectorizer with n = 1 to 5).
        
        Parameters: 
            No input parameters.
        """

        # Deep copy
        ngrams_ref_temporary = self.ref[['gtin', 'n_gram_string']].copy(deep=True)

        # Generate N-GRAMs reference
        ngrams_ref_temporary.loc[:, 'embbedings'] = ngrams_ref_temporary['n_gram_string'].apply(
            lambda x: self.vectorizer.transform([x]).toarray()
        )
        
        # Stores data for internal-use variable
        self.ngrams_ref = ngrams_ref_temporary.copy()

    def __create_embbedings_ref(self):
        """ 
        The create_embeddings_ref method is used to create a data structure that supports the classification process through description-based similarity comparison.
        
        In summary, it generates a structured dataframe with two columns: a) reference GTIN; b) reference example of the GTIN.
        
        Parameters: 
            No input parameters.
        """

        # Loads realiable GTINs
        self.embeddings_ref = self.ref[['gtin', 'examples']].explode('examples')

    def ngram_classifier(self, description):
        """ 
        The ngram_classifier method is a 1NN classification algorithm that uses N-GRAM vectors with TF-IDF to calculate similarity, applying a minimum threshold between N-GRAM vectors.
        Parameters:
            description (str): receives the description string of the GTIN to be estimated.
        
        Return: 
            If only one possible classification is found:
                outcome (str): estimated GTIN
                similarity (float): the similarity score
                string_ref (str): reference used
            Otherwise:
                None values.
        """

        # Perform deep copy to ensure safe data processing
        ref = self.ngrams_ref.copy()

        # Calculates the TF-IDF vector representing the string
        string_vetorizada = self.vectorizer.transform([description]).toarray()

        # Calculate cosine similarity
        ref['result'] = ref['embbedings'].apply(
            lambda x: cosine_similarity( x, string_vetorizada )[0][0]
        )

        # Filter
        results = ref.loc[ref['result'] > 0.88, ['gtin', 'result', 'n_gram_string']]
        
        # Return only one value
        if results.shape[0] == 0 or results.shape[0] > 1:
            return None, None, None
        else:
            return str(results.iloc[0, 0]), results.iloc[0, 1], results.iloc[0, 2]

    def embedding_classifier(self, description):
        """ 
        The embedding_classifier method is a 1NN classifier that uses similarity with a minimum threshold by comparing the input embeddings with those of the original reference description of the GTIN.
        Parameters: 
            description (str): receives the description string of the GTIN to be estimated.
        
        Return: 
            If only one possible classification is found:
                outcome (str): estimated GTIN
                similarity (float): the similarity score
                string_ref (str): reference used
            Otherwise:
                None values.
        """

        # Perform deep copy to ensure safe data processing
        ref = self.embeddings_ref.copy().reset_index()
        
        # Calculate similarity
        string_vector = self.model.encode([description])
        examples_vectors = self.model.encode(ref['examples'])
        ref['result'] = self.model.similarity(examples_vectors, string_vector)

        # Filter
        filtered = ref.loc[ref['result'] > 0.86, ['gtin', 'result', 'examples']]
        idx_max = filtered.groupby('gtin')['result'].idxmax()
        results = filtered.loc[idx_max].reset_index(drop=True)

        # Return only one value
        if results.shape[0] == 0 or results.shape[0] > 1:
            return None, None, None
        else:
            return str(results.iloc[0, 0]), results.iloc[0, 1], results.iloc[0,2]

    def gtin_classifier(self, description):
        """ 
        The gtin_classifier method receives the string of an input row to perform similarity-based classification: total similarity (string_match), similarity by n-grams (cosine of n-gram vectors with a minimum threshold), and description similarity (cosine of description vectors).
        
        Parameters: 
            string (str): receives the description string of the GTIN to be estimated.

        Return: dict{
                gtin (str): classification estimate or none. 
                similarity (float): the similarity score or none.
                description_ref (str): the reference used or none. 
                method (str): and the classification method or none.
            }
        """

        # string classification
        outcome = self.string_match_ref.get(description, None)
        if outcome is not None:
            return {'gtin': outcome, 'similarity': 1.0, 'description_ref': description, 'method': 'string_match'}

        # N-GRAM Classification
        outcome, similarity, string_ref = self.ngram_classifier(description)
        if outcome is not None:
            return {'gtin': outcome, 'similarity': similarity, 'description_ref': string_ref, 'method': 'n_gram_match'}

        # Embeddings Classification
        outcome, similarity, string_ref = self.embedding_classifier(description)
        if outcome is not None:
            return {'gtin': outcome, 'similarity': similarity, 'description_ref': string_ref, 'method': 'embedding_match'}

        # Not classified
        return {'gtin': None, 'similarity': 0, 'description_ref': None, 'method': None}

if __name__ == "__main__":
    estimator = NFeMinerGTINEstimator()
    
    for description in ['CARNE BOLVINA', 'ARROZ AGULHA', 'CREME DENTAL']:
        print(estimator.gtin_classifier(description))