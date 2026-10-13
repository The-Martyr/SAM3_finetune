import json
import tempfile
import unittest
from pathlib import Path
from sam3_sft.data import QueryJsonLoader


class DataTest(unittest.TestCase):
    def load(self, query_ids):
        payload = {'format':'sam3_sft_queries_v1','images':[{
            'id':1,'file_name':'synthetic.png','width':2,'height':2,
            'objects':[{'id':7,'bbox':[0,0,2,2],'segmentation':{'size':[2,2],'counts':[0,4]}}],
            'queries':[{'text':'square','object_ids':query_ids,'is_exhaustive':True},
                       {'text':'circle','object_ids':[],'is_exhaustive':True}]}]}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.json';path.write_text(json.dumps(payload))
            return QueryJsonLoader(path)

    def test_positive_negative_and_normalization(self):
        q,a=self.load([7]).loadQueriesAndAnnotationsFromDatapoint(0)
        self.assertEqual(q[0]['object_ids_output'],[7])
        self.assertEqual(q[1]['object_ids_output'],[])
        self.assertEqual(a[0]['bbox'],[0,0,1,1])

    def test_missing_reference_fails(self):
        with self.assertRaises(ValueError): self.load([99])

if __name__ == '__main__': unittest.main()
