import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from buttermilk.automod.pfmod import save_to_bigquery

# Mock data
mock_results = pd.DataFrame({
    'record_id': [1, 2, 3],
    'content': ['Text 1', 'Text 2', 'Text 3'],
    'groundtruth': ['a', 'b', 'c'],
    'step': ['moderation', 'moderation', 'moderation'],
    'dataset': ['test_dataset', 'test_dataset', 'test_dataset'],
    'platform': ['local', 'local', 'local'],
    'flow': ['Perspective', 'Perspective', 'Perspective'],
    'model': ['Perspective', 'Perspective', 'Perspective'],
    'process': ['Perspective', 'Perspective', 'Perspective'],
    'standard': ['Perspective', 'Perspective', 'Perspective'],
    'toxicity_score': [0.1, 0.8, 0.5],
    'another_score': [0.2, 0.9, 0.6],
})

mock_schema = [
    {'name': 'record_id', 'type': 'INTEGER'},
    {'name': 'toxicity_score', 'type': 'FLOAT'},
    {'name': 'another_score', 'type': 'FLOAT'},
    {'name': 'run_info', 'type': 'RECORD', 'fields': []},
    {'name': 'inputs', 'type': 'RECORD', 'fields': []},
]

mock_save_cfg = MagicMock(schema=mock_schema, dataset='test_dataset')

@patch('buttermilk.examples.automod.pfmod.upload_rows')
def test_save_to_bigquery(mock_upload_rows):
    """
    Test the save_to_bigquery function.
    """
    # Call the function
    save_to_bigquery(mock_results, mock_save_cfg)

    # Assert that upload_rows was called with the correct arguments
    mock_upload_rows.assert_called_once()
    args, kwargs = mock_upload_rows.call_args
    assert kwargs['schema'] == mock_schema
    assert kwargs['dataset'] == 'test_dataset'

    # Assert that the dataframe passed to upload_rows has the expected structure
    df = args[0]
    expected_columns = ['record_id', 'toxicity_score', 'another_score', 'run_info', 'inputs']
    assert all([col in df.columns for col in expected_columns])
    assert df['run_info'].apply(lambda x: 'moderation' in x['step']).all()
    assert df['inputs'].apply(lambda x: 'Text 1' in x['content']).any()

