import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('export',Path(__file__).parents[2]/'scripts/revision/export_tables.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class TableTests(unittest.TestCase):
    def test_each_row_has_tex_line_terminator(self):
        text=m.table('Caption','tab:test','lr','Name & Value',['A & 1','B & 2'],'Description')
        for row in ['Name & Value','A & 1','B & 2']:
            self.assertIn(row+' '+chr(92)*2+'\n',text)
if __name__=='__main__':unittest.main()
