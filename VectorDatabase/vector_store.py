"""
Implementing vector database from scratch using numpy.
The purpose is to make myself familiar with the broader understanding of vector database.
"""

import numpy as np

class VectorStore:
    """
    Implementing vector store an important part of vector database.

    Methods:
        add_vector:  Add a new vector to the existing database.
        get_vector: Retrieve the vector corresponding to the given id.
    """
    def __init__(self) -> None:
        """
        Constructor for the class. Initializes the vector_data and vector_index dictionaries.
        """
        self.vector_data = {}   # A dictionary to store vectors.
        self.vector_index = {}  # A dictionary for indexing structure for retrieval

    def add_vector(self, vector_id: str, vector: np.ndarray):
        """
        Add a vector to the vector store.

        Args:
            vector_id: An unique identity for the vector.
            vector: The vector to be stored.
        """
        self.vector_data[vector_id] = vector
        self._update_index(vector_id, vector)

    def get_vector(self, vector_id: str) -> np.ndarray:
        """
        Retrive a vector from the vector store.

        Args: 
            vector_id: Unique id corresponding to a vector.

        Return:
            Vector corresponding to the the id provided.
        """
        out = self.vector_data.get(vector_id)
        return out

    def _update_index(self, vector_id: str, vector: np.ndarray):
        """
        Update the index of an existing vector.

        Args:
            vector_id: Unique id corresponding the given vector.
            vector: The stored vector.
        """
        vector_unit = vector / np.linalg.norm(vector)
        for existing_vector_id, existing_vector in self.vector_data.items():
            existing_vector_unit = existing_vector / np.linalg.norm(existing_vector)
            similarity_score = np.dot(existing_vector_unit, vector_unit)

            if existing_vector_id not in self.vector_index:
                self.vector_index[existing_vector_id] = {}
            self.vector_index[existing_vector_id][vector_id] = similarity_score

    def get_similar_vectors(self, query_vector: np.ndarray, num_similar: int = 5) -> list:
        """
        To return similar vectors to the given vector.

        Args:
            query_vector: The vector corresponding to which similar vectors are to be returned.
            num_similar: Number of similar vectors to be returned.

        Returns:
            List of similar vectors to the given vector.
        """
        query_vector_unit = query_vector / np.linalg.norm(query_vector)
        out_results = []

        # For each vector in the index, calculate the similarity score.
        for vector_id, vector in self.vector_data.items():
            vector_unit = vector / np.linalg.norm(vector)
            similarity_score = np.dot(query_vector_unit, vector_unit)
            out_results.append((vector_id, similarity_score))

        # Sort the results based on the similarity score.
        out_results.sort(key=lambda x: x[1], reverse=True)
        out = out_results[:num_similar]
        return out
