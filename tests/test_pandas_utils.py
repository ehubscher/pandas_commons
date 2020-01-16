from pandas import DataFrame, Index, RangeIndex
import unittest


class TestPandasUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pass

    def test_index_is_not_default(self, index: Index):
        self.assertNotIsInstance(
            obj=index,
            cls=RangeIndex,
            msg='The DataFrame\'s index is the default.'
        )

    def test_index_has_no_duplicates(self, index: Index):
        self.assertEqual(
            first=index.duplicated().any(),
            second=False,
            msg='The index contains duplicate values.'
        )

    def test_index_has_no_nans(self, index: Index):
        self.assertEqual(
            first=index.contains(float('nan')),
            second=False,
            msg='The index contains NaN value(s).'
        )

    def test_drop_rows_with(self):
        pass
