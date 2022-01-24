from src.preparation.text_processing import DataFrameTextProcessing
import unittest
import pandas as pd


class TestPreprocessing(unittest.TestCase):

    def test_normalization(self):
        text_processor = DataFrameTextProcessing('english')
        mock_input = [
            {'text': 'I thought this was a great course. I knew a bit about ML before taking it, but this course '
                     'helped fill in a lot of gaps in my knowledge. The pacing was good and course material was '
                     'taught in a logical order. There are good examples of ML applications and motivational '
                     'problems. So there is a nice balance between theory and practical application.'}
        ]
        mock_response = 'think great course know bit ml take course help fill lot gap knowledge pacing good course ' \
                        'material teach logical order good example ml application motivational problem nice balance ' \
                        'theory practical application'
        df_mock = pd.DataFrame(mock_input)
        processed_text_column = text_processor.normalize_df(df_mock, 'text')
        assert mock_response == processed_text_column.loc[0]


if __name__ == '__main__':
    unittest.main()