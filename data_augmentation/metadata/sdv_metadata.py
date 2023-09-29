from typing import Dict

import pandas as pd
from sdv.metadata import SingleTableMetadata


class SDVMetadata(object):

    def __init__(self, metadata = None):
        if not metadata:
            metadata = SingleTableMetadata()
        self.metadata = metadata

    def create_from_df(
        self, dataframe: pd.DataFrame, dtypes: Dict[str, str] = {}
    ):
        '''
            Creates a metadata object from a dataframe.
            If dtypes are provided, updates the type of the given columns. 
        '''
        self.metadata.detect_from_dataframe(data=dataframe)

        if dtypes:
            for column, type_ in dtypes.items():
                self.metadata.update_column(
                    column_name=column, sdtype=type_
                )

    def get_metadata(self):
        return self.metadata
    
    def get_metadata_dict(self):
        return self.metadata.to_dict()
    
    def validate(self):
        self.metadata.validate()

    def save(self, path: str):
        self.metadata.save_to_json(filepath=path)

    @staticmethod
    def load_from_json(path: str):
        metadata = SingleTableMetadata.load_from_json(filepath=path)
        return SDVMetadata(metadata)

    @staticmethod
    def load_from_dict(metadata_dict: dict):
        metadata = SingleTableMetadata.load_from_dict(metadata_dict)
        return SDVMetadata(metadata)