"""Fetch official PCB-IND and maintainer-hosted NEU-DET archives; record checksums."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,zipfile,requests,gdown
ROOT=Path(__file__).resolve().parents[1]
def download(dataset):
 out=ROOT/'data/raw'/dataset;out.mkdir(parents=True,exist_ok=True)
 if dataset=='pcb_ind':
  r=requests.get('https://zenodo.org/api/records/19723114',timeout=60);r.raise_for_status();record=r.json();entry=record['files'][0];p=out/entry['key']
  if not p.exists() or hashlib.md5(p.read_bytes()).hexdigest()!=entry['checksum'].split(':')[1]:
   with requests.get(entry['links']['self'],stream=True,timeout=120) as response:
    response.raise_for_status()
    with p.open('wb') as f:
     for b in response.iter_content(1<<20):f.write(b)
  if hashlib.md5(p.read_bytes()).hexdigest()!=entry['checksum'].split(':')[1]:raise ValueError('PCB archive checksum mismatch')
  info={'record':'https://zenodo.org/records/19723114','file':entry,'download_date':'2026-11-05'}
 else:
  p=out/'NEU-DET.zip'
  if not p.exists():gdown.download(id='1qrdZlaDi272eA79b0uCwwqPrm2Q_WI3k',output=str(p),quiet=True)
  info={'url':'https://drive.google.com/open?id=1qrdZlaDi272eA79b0uCwwqPrm2Q_WI3k','source':'https://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm','download_date':'2026-11-05'}
 info['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();(out/'provenance.json').write_text(json.dumps(info,indent=2))
 with zipfile.ZipFile(p) as z:
  if z.testzip() is not None:raise ValueError('Corrupt zip')
  for name in z.namelist():
   if not (out/'extracted'/name).resolve().is_relative_to((out/'extracted').resolve()):raise ValueError('Unsafe archive member')
  z.extractall(out/'extracted')
 print(dataset,'ready',flush=True)
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(download,['pcb_ind','neu_det']))
