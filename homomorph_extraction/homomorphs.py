import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)


class HomomorphismExtractor:
    """
    Class to extract homomorphisms between nationality embeddings.
    """
    
    def __init__(self, k_homomorphs=10, include_self_matches=False):
        """
        Initialize the homomorphism extractor.
        
        Args:
            k_homomorphs (int): Number of top homomorphs to extract per word
            include_self_matches (bool): Whether to include self-matches in results
        """
        self.k_homomorphs = k_homomorphs
        self.include_self_matches = include_self_matches
    
    def compute_similarity_matrix(self, emb1, emb2):
        """
        Compute cosine similarity matrix between two sets of embeddings.
        
        Args:
            emb1 (np.array): First set of embeddings (n1 x dim)
            emb2 (np.array): Second set of embeddings (n2 x dim)
            
        Returns:
            np.array: Similarity matrix (n1 x n2)
        """
        logging.info("Computing similarity matrix...")
        return cosine_similarity(emb1, emb2)
    
    def extract_homomorphisms_to_rows(self, similarity_matrix, types1, types2, nat1, nat2):
        """
        Extract homomorphisms and return as list of dictionaries for CSV export.
        
        Args:
            similarity_matrix (np.array): Cosine similarity matrix
            types1 (list): Word types from first nationality
            types2 (list): Word types from second nationality
            nat1 (str): First nationality name
            nat2 (str): Second nationality name
            
        Returns:
            list: List of dictionaries with homomorphism data
        """
        homomorphism_rows = []
        
        # Extract homomorphisms nat1 -> nat2
        for i, word in enumerate(types1):
            if i < len(similarity_matrix):
                top_indices = np.argsort(similarity_matrix[i])[::-1][:self.k_homomorphs]
                homomorphs = [(types2[j], similarity_matrix[i][j]) 
                             for j in top_indices if j < len(types2)]
                
                # Option to include or exclude self-matches
                if not self.include_self_matches:
                    # Only include if the word doesn't appear in top 3 matches
                    top_words = [h[0] for h in homomorphs[:3]]
                    if word in top_words:
                        continue
                
                # Create row for CSV
                homomorph_words = [h[0] for h in homomorphs]
                similarities = [float(h[1]) for h in homomorphs]
                
                homomorphism_rows.append({
                    'word': word,
                    'source_nationality': nat1,
                    'target_nationality': nat2,
                    'homomorphs': homomorph_words,
                    'similarities': similarities
                })
        
        # Extract homomorphisms nat2 -> nat1  
        for i, word in enumerate(types2):
            if i < similarity_matrix.shape[1]:
                top_indices = np.argsort(similarity_matrix.T[i])[::-1][:self.k_homomorphs]
                homomorphs = [(types1[j], similarity_matrix.T[i][j]) 
                             for j in top_indices if j < len(types1)]
                
                # Option to include or exclude self-matches
                if not self.include_self_matches:
                    # Only include if the word doesn't appear in top 3 matches
                    top_words = [h[0] for h in homomorphs[:3]]
                    if word in top_words:
                        continue
                
                # Create row for CSV
                homomorph_words = [h[0] for h in homomorphs]
                similarities = [float(h[1]) for h in homomorphs]
                
                homomorphism_rows.append({
                    'word': word,
                    'source_nationality': nat2,
                    'target_nationality': nat1,
                    'homomorphs': homomorph_words,
                    'similarities': similarities
                })
        
        return homomorphism_rows
    
    def process_nationality_pair(self, embedding_gen, story_dict, nat1, nat2, batch_size=32):
        """
        Process a single pair of nationalities and extract homomorphisms.
        
        Args:
            embedding_gen: EmbeddingGenerator instance
            story_dict (dict): nationality -> stories mapping
            nat1 (str): First nationality
            nat2 (str): Second nationality
            batch_size (int): Batch size for embedding computation
            
        Returns:
            list: List of homomorphism rows for CSV
        """
        logging.info(f"Processing pair: {nat1} vs {nat2}")
        
        # Generate embeddings for both nationalities
        nat1_types, nat1_embs = embedding_gen.generate_nationality_embeddings(
            story_dict, nat1, batch_size)  
        
        nat2_types, nat2_embs = embedding_gen.generate_nationality_embeddings(
            story_dict, nat2, batch_size)
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(nat1_embs, nat2_embs)
        
        # Extract homomorphisms
        homomorphism_rows = self.extract_homomorphisms_to_rows(
            similarity_matrix, nat1_types, nat2_types, nat1, nat2)
        
        logging.info(f"Extracted {len(homomorphism_rows)} homomorphism mappings for {nat1}-{nat2}")
        
        return homomorphism_rows
    
    def save_homomorphisms_to_csv(self, homomorphism_rows, output_file):
        """
        Save homomorphisms to CSV file with lists as string representations.
        
        Args:
            homomorphism_rows (list): List of homomorphism dictionaries
            output_file (str): Output CSV file path
        """
        # Convert list fields to string representations for CSV compatibility
        processed_rows = []
        for row in homomorphism_rows:
            processed_row = row.copy()
            # Convert lists to string representations that can be evaluated
            processed_row['homomorphs'] = str(processed_row['homomorphs'])
            processed_row['similarities'] = str([round(s, 6) for s in processed_row['similarities']])
            processed_rows.append(processed_row)
        
        # Create DataFrame and save
        df = pd.DataFrame(processed_rows)
        df.to_csv(output_file, index=False)
        logging.info(f"Saved {len(processed_rows)} homomorphisms to {output_file}")
    
    def load_homomorphisms_from_csv(self, input_file):
        """
        Load homomorphisms from CSV file and convert string representations back to lists.
        
        Args:
            input_file (str): Input CSV file path
            
        Returns:
            list: List of homomorphism dictionaries with lists restored
        """
        import ast
        
        df = pd.read_csv(input_file)
        
        homomorphism_rows = []
        for _, row in df.iterrows():
            processed_row = {
                'word': row['word'],
                'source_nationality': row['source_nationality'],
                'target_nationality': row['target_nationality'],
                'homomorphs': ast.literal_eval(row['homomorphs']),
                'similarities': ast.literal_eval(row['similarities'])
            }
            homomorphism_rows.append(processed_row)
        
        return homomorphism_rows