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
