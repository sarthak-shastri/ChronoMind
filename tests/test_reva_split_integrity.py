import json
from scripts.check_reva_split_integrity import audit

def row(qid,path='a.mp4',answer='A'):
 return {'qa_id':qid,'video_path':path,'source_sequence_id':'src-'+path,'options':{'A':'a','B':'b','C':'c','D':'d'},'correct_answer':answer}
def test_clean_disjoint_splits_pass(tmp_path):
 files={}
 for s, rr in [('train',[row('t','t.mp4')]),('val',[row('v','v.mp4')]),('test',[row('x','x.mp4',None)])]:
  f=tmp_path/f'{s}.json'; f.write_text(json.dumps({'QA':rr})); files[s]=f
 assert audit(files)==[]
def test_exact_path_overlap_fails(tmp_path):
 assert any('video-path overlap' in e for e in audit({'train': [{'qa_id':'t','video_path':'X/a.mp4','options':{'A':1,'B':2,'C':3,'D':4},'correct_answer':'A'}], 'val':[{'qa_id':'v','video_path':'x\\a.mp4','options':{'A':1,'B':2,'C':3,'D':4},'correct_answer':'B'}]}))
def test_source_overlap_and_labels_and_duplicate_fail():
 a=row('dup','a.mp4'); b=row('dup','b.mp4'); a['source_sequence_id']='same'; b['source_sequence_id']='same'; b['correct_answer']='Z'
 issues=audit({'train':[a], 'val':[b]})
 assert any('source-sequence overlap' in x for x in issues)
 assert any('duplicate qa_id' in x for x in issues)
 assert any('invalid/missing answer' in x for x in issues)
def test_missing_video_fails(tmp_path):
 r=row('t','not-there.mp4'); r['source_sequence_id']=None
 assert any('missing video' in e for e in audit({'train':[r]},tmp_path))
